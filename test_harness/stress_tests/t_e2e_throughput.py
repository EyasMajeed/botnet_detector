"""
ST-2b End-to-end throughput (ingestion + flow flush + Stage-2 + XAI).

Replays the same UDP burst as ST-2a, then force-flushes all open flows
(matching what PcapInferenceThread does for uploaded files). This
measures the full system: every flow goes through Stage-1 routing,
Stage-2 LSTM inference, and the per-flow XAI explainer.

The test reports a 3-way breakdown:
  ingest_pps    — process_packet throughput before any flow closes
  flush_seconds — total time for flush_idle_flows() (Stage-2 + XAI)
  e2e_pps       — n_packets / (ingest + flush) — the number that
                  actually matters for live-monitoring sustainability

Pass target: end-to-end ≥ 200 pps. Failing this test does NOT mean the
hot path is slow (ST-2a covers that); it means flow finalisation
(Stage-2 inference + XAI) cannot keep up with the rate at which flows
complete. Common cause: XAI runs on every flow regardless of label.
"""

from __future__ import annotations

import time

from test_harness.utils.project_imports import (
    soft_import, models_state, ensure_on_path,
)


TEST_ID = "ST-2b"


def run(n_packets: int = 10_000, target_e2e_pps: float = 200.0) -> dict:
    ensure_on_path()
    ms = models_state()
    if not all(ms.values()):
        missing = [k for k, v in ms.items() if not v]
        return _skip(f"missing models: {missing}")

    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return _skip(f"monitoring import: {err}")

    try:
        bm = monitoring.BotnetMonitor()
    except Exception as e:                                    # noqa: BLE001
        return _skip(f"BotnetMonitor init: {e}")

    base_ts = time.time()
    src_ip, dst_ip = "10.0.0.55", "10.0.0.1"

    # ── Phase 1: ingest ──────────────────────────────────────────────
    t0 = time.monotonic()
    for i in range(n_packets):
        bm.process_packet(base_ts + i * 1e-4, src_ip, dst_ip,
                          40000 + (i % 20000), 53, 17, 100, 64, "", 0)
    ingest_dur = time.monotonic() - t0
    ingest_pps = n_packets / max(ingest_dur, 1e-6)

    # ── Phase 2: force-flush every open flow ─────────────────────────
    flushed_count = 0
    flush_dur = 0.0
    flush_error: str | None = None
    try:
        if hasattr(bm, "aggregator"):
            bm.aggregator._idle = -1e9
        if hasattr(bm, "flush_idle_flows"):
            t1 = time.monotonic()
            flushed = bm.flush_idle_flows() or []
            flush_dur = time.monotonic() - t1
            flushed_count = len(flushed)
    except Exception as e:                                    # noqa: BLE001
        flush_error = f"{type(e).__name__}: {e}"

    if flush_error:
        return {"test_id": TEST_ID, "name": "End-to-end throughput",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "flush completes",
                "actual":   f"flush raised: {flush_error}",
                "raw":      {"ingest_pps": round(ingest_pps, 1),
                             "ingest_dur_sec": round(ingest_dur, 3)}}

    e2e_total = ingest_dur + flush_dur
    e2e_pps = n_packets / max(e2e_total, 1e-6)
    flow_per_sec = flushed_count / max(flush_dur, 1e-6) if flush_dur > 0 else 0.0

    if e2e_pps >= target_e2e_pps:
        verdict = "PASS"
        actual = (f"end-to-end {e2e_pps:.0f} pps "
                  f"(ingest {ingest_pps:.0f} pps, "
                  f"flush {flushed_count} flows in {flush_dur:.1f}s = "
                  f"{flow_per_sec:.0f} flow/s)")
    else:
        verdict = "FAIL"
        actual = (f"end-to-end only {e2e_pps:.0f} pps "
                  f"(ingest is fine at {ingest_pps:.0f} pps, but "
                  f"flush of {flushed_count} flows took {flush_dur:.1f}s "
                  f"= {flow_per_sec:.0f} flow/s). The bottleneck is "
                  "flow finalisation: Stage-2 inference + per-flow XAI.")

    return {"test_id": TEST_ID, "name": "End-to-end throughput",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"end-to-end ≥ {target_e2e_pps:.0f} pps",
            "actual":   actual,
            "raw": {
                "ingest_pps":      round(ingest_pps, 1),
                "ingest_dur_sec":  round(ingest_dur, 3),
                "flush_dur_sec":   round(flush_dur, 3),
                "flushed_flows":   flushed_count,
                "flow_per_sec":    round(flow_per_sec, 1),
                "e2e_pps":         round(e2e_pps, 1),
                "n_packets":       n_packets,
            }}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "End-to-end throughput",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "end-to-end ≥ 200 pps",
            "actual":   reason}
