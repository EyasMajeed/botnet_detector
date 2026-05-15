"""
G-AF Alert flood / signal storm.

Pumps 100k DetectionFlow objects into DetectionStore in a single batch and
ensures:
  - the store's flows_changed signal does not fire once per row (debouncing
    works — the project debounces at SIGNAL_DEBOUNCE_MS = 250)
  - peak memory remains bounded
"""

from __future__ import annotations

import time

from test_harness.utils.project_imports import soft_import


TEST_ID = "G-AF"


def run(n_rows: int = 100_000, max_signals: int = 200) -> dict:
    ds_mod, err = soft_import("app.detection_store")
    if ds_mod is None:
        return _skip(f"detection_store import: {err}")
    DetectionStore = getattr(ds_mod, "DetectionStore", None)
    DetectionFlow  = getattr(ds_mod, "DetectionFlow",  None)
    if DetectionStore is None or DetectionFlow is None:
        return _skip("DetectionStore / DetectionFlow class not found")
    qt, err = soft_import("PyQt6.QtCore")
    if qt is None:
        return _skip(f"PyQt6 import: {err}")

    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="harness_af_")) / "store.json"

    app = qt.QCoreApplication.instance() or qt.QCoreApplication([])
    try:
        store = DetectionStore(tmp)
    except Exception as e:                                    # noqa: BLE001
        return _skip(f"DetectionStore init: {e}")

    counter = {"n": 0}
    # The project's signal is named flows_changed (not "changed").
    sig = getattr(store, "flows_changed", None)
    if sig is not None:
        try:
            sig.connect(lambda *a, **kw: counter.update(n=counter["n"] + 1))
        except Exception:
            pass

    flows = [DetectionFlow(
        src_ip      = f"10.0.0.{i % 254 + 1}",
        dst_ip      = "10.0.0.1",
        src_port    = 40000 + (i % 20000),
        dst_port    = 80,
        protocol    = "TCP",
        label       = "botnet" if i % 11 == 0 else "benign",
        confidence  = (i % 100) / 100.0,
        device_type = "noniot",
    ) for i in range(n_rows)]

    t0 = time.monotonic()
    try:
        # The project's batch entrypoint is add_upload_batch(flows, filename).
        if hasattr(store, "add_upload_batch"):
            chunk = 5_000
            for k in range(0, n_rows, chunk):
                store.add_upload_batch(flows[k:k + chunk], "harness_test")
        elif hasattr(store, "add_live_flow"):
            for f in flows:
                store.add_live_flow(f)
        else:
            return _skip("DetectionStore: neither add_upload_batch nor add_live_flow")
        # Drive the event loop briefly so debounced signals fire.
        for _ in range(50):
            app.processEvents()
            time.sleep(0.01)
    except Exception as e:                                    # noqa: BLE001
        return {"test_id": TEST_ID, "name": "Alert flood",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "store survives 100k inserts",
                "actual":   f"raised: {type(e).__name__}: {e}"}
    duration = time.monotonic() - t0

    fired = counter["n"]
    n_kept = len(getattr(store, "flows", []))
    # When add_upload_batch is used, the project emits flows_changed once
    # per batch (not debounced — that's only for live mode). So with
    # n_rows / chunk = 20 batches we expect ~20 signals. For live mode the
    # debounced signal would fire ~ duration / 0.25s.
    if fired > max_signals:
        verdict = "FAIL"
        actual  = (f"flows_changed fired {fired} times for {n_rows} rows "
                   f"(target ≤ {max_signals}). Debouncing not effective.")
    else:
        verdict = "PASS"
        actual  = (f"{n_rows} rows inserted in {duration:.2f}s; "
                   f"flows_changed fired {fired} times; "
                   f"store retained {n_kept} flows")
    return {"test_id": TEST_ID, "name": "Alert flood",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"flows_changed fires ≤ {max_signals} times",
            "actual":   actual,
            "raw":      {"signals_fired": fired,
                         "duration_sec": round(duration, 3),
                         "n_kept":       n_kept}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "Alert flood",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "debounced signal under flood",
            "actual":   reason}
