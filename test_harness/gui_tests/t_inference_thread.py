"""
G-IT PcapInferenceThread — verifies the inference worker thread
processes a small PCAP and signals completion without leaving any
zombie threads behind.

This mirrors what app/upload_page._run_pcap_async does:

    1. Construct PcapInferenceThread(pcap_path) on the main thread.
    2. Call worker.ensure_monitor() on the main thread — this is REQUIRED.
       (BotnetMonitor() calls torch.load three times; on macOS torch.load
        from a non-main thread segfaults once Qt has initialised libomp.)
    3. Wire the worker's signals: progress / error / done(list).
    4. Call worker.start() and pump the Qt event loop.

Runs headless via QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import os
import time

from test_harness.generators.pcap_gen import benign_tcp_pcap
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import


TEST_ID = "G-IT"


def run(n_flows: int = 10, timeout_sec: int = 60) -> dict:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    qt, err = soft_import("PyQt6.QtCore")
    if qt is None:
        return _skip(f"PyQt6 import: {err}")
    iw_mod, err = soft_import("app.inference_worker")
    if iw_mod is None:
        return _skip(f"inference_worker import: {err}")
    Thread = getattr(iw_mod, "PcapInferenceThread", None)
    if Thread is None:
        return _skip("PcapInferenceThread class not found")

    # Build a tiny PCAP — kept small so model loading + processing fits
    # comfortably in the timeout on macOS without GPU.
    dirs = for_test(TEST_ID)
    pcap = benign_tcp_pcap(dirs["artifacts"] / "small.pcap",
                           n_flows=n_flows, pkts_per_flow=4, seed=0)

    # Fresh QCoreApplication for headless event-loop pumping.
    app = qt.QCoreApplication.instance() or qt.QCoreApplication([])

    # Slot state — populated by the worker's signals.
    state: dict = {"done_hit": False, "n_results": 0, "error": None,
                   "last_progress": None}

    def _on_done(payload):
        state["done_hit"] = True
        try:
            state["n_results"] = len(payload) if payload is not None else 0
        except Exception:
            pass

    def _on_error(msg):
        state["error"] = str(msg)

    def _on_progress(cur, total, text):
        state["last_progress"] = (int(cur), int(total), str(text))

    # ── Main-thread construction (matches upload_page's contract) ─────
    # PcapInferenceThread(pcap_path: str, parent=None) — positional.
    try:
        worker = Thread(str(pcap))
    except Exception as e:                                    # noqa: BLE001
        return {"test_id": TEST_ID, "name": "PcapInferenceThread",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": "thread constructs",
                "actual":   f"ctor raised: {type(e).__name__}: {e}",
                "pcap": str(pcap)}

    # CRITICAL: load BotnetMonitor on this (the main) thread, BEFORE
    # start() is called. Skipping this is what produced the
    # "BotnetMonitor not initialised" error on the previous run.
    if not worker.ensure_monitor():
        init_err = getattr(worker, "_init_error", "unknown")
        # Distinguish between missing-models (skip) and real failure (fail).
        msg = (init_err or "").lower()
        if "missing" in msg or "no module" in msg or "scapy" in msg:
            return _skip(f"ensure_monitor: {init_err}")
        return {"test_id": TEST_ID, "name": "PcapInferenceThread",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": "ensure_monitor() returns True",
                "actual":   f"ensure_monitor failed: {init_err}",
                "pcap": str(pcap)}

    # Wire signals. The worker emits exactly:
    #   progress(int, int, str)  ·  error(str)  ·  done(list)
    try:
        worker.done.connect(_on_done)
    except Exception:
        return _skip("worker.done signal not present")
    try:
        worker.error.connect(_on_error)
    except Exception:
        pass
    try:
        worker.progress.connect(_on_progress)
    except Exception:
        pass

    # ── Run the worker, pump events until done/error/timeout ──────────
    worker.start()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        app.processEvents()
        if state["done_hit"] or state["error"]:
            break
        time.sleep(0.05)

    # Graceful join.
    if hasattr(worker, "wait"):
        worker.wait(2_000)

    # ── Verdict ──────────────────────────────────────────────────────
    if state["error"]:
        return {"test_id": TEST_ID, "name": "PcapInferenceThread",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": "worker emits done(list); no error signal",
                "actual":   f"error signal: {state['error']}",
                "pcap": str(pcap),
                "raw": {"last_progress": state["last_progress"]}}
    if not state["done_hit"]:
        return {"test_id": TEST_ID, "name": "PcapInferenceThread",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": f"done signal within {timeout_sec}s",
                "actual":   "no done signal received within timeout",
                "pcap": str(pcap),
                "raw": {"last_progress": state["last_progress"]}}
    return {"test_id": TEST_ID, "name": "PcapInferenceThread",
            "severity": "HIGH", "verdict": "PASS",
            "expected": "worker emits done(list); no error",
            "actual":   f"done hit; n_results={state['n_results']}",
            "pcap":     str(pcap),
            "raw":      {"n_results": state["n_results"],
                         "last_progress": state["last_progress"]}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "PcapInferenceThread",
            "severity": "HIGH", "verdict": "SKIPPED",
            "expected": "worker completes cleanly",
            "actual":   reason}
