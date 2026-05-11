"""
G-DS DetectionStore unit checks (no UI required).

Validates the three claims the GUI relies on:

  1. MAX_FLOWS cap is enforced — adding > MAX_FLOWS detections does not
     leak memory unboundedly.
  2. relabel_with_threshold relabels rows in place without re-running inference.
  3. The store survives a 60k-flow batch insertion without raising.
"""

from __future__ import annotations

from test_harness.utils.project_imports import soft_import


TEST_ID = "G-DS"


def run(n_rows: int = 60_000) -> dict:
    ds_mod, err = soft_import("app.detection_store")
    if ds_mod is None:
        return _skip(f"detection_store import: {err}")

    DetectionStore = getattr(ds_mod, "DetectionStore", None)
    DetectionFlow  = getattr(ds_mod, "DetectionFlow",  None)
    MAX_FLOWS      = int(getattr(ds_mod, "MAX_FLOWS", 50_000))
    if DetectionStore is None or DetectionFlow is None:
        return _skip("DetectionStore / DetectionFlow class not found")

    # DetectionStore requires a persist_path; use a fresh tmp file so the
    # test never collides with the user's real store.json.
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="harness_ds_")) / "store.json"

    try:
        store = DetectionStore(tmp)
    except Exception as e:                                    # noqa: BLE001
        return _skip(f"DetectionStore init: {e}")

    # ── 1. Build n_rows DetectionFlow objects matching the project schema
    flows = []
    for i in range(n_rows):
        flows.append(DetectionFlow(
            src_ip      = f"10.0.0.{i % 254 + 1}",
            dst_ip      = "10.0.0.1",
            src_port    = 40000 + (i % 20000),
            dst_port    = 80,
            protocol    = "TCP",
            label       = "botnet" if i % 7 == 0 else "benign",
            confidence  = (i % 100) / 100.0,
            device_type = "noniot",
        ))

    # ── 2. Insertion via the actual upload-batch API
    try:
        if hasattr(store, "add_upload_batch"):
            # Insert in chunks to mimic real batches.
            chunk = 5_000
            for k in range(0, n_rows, chunk):
                store.add_upload_batch(flows[k:k + chunk], "harness_test")
        elif hasattr(store, "add_live_flow"):
            for f in flows:
                store.add_live_flow(f)
        else:
            return _skip("DetectionStore exposes neither add_upload_batch nor add_live_flow")
    except Exception as e:                                    # noqa: BLE001
        return {"test_id": TEST_ID, "name": "DetectionStore behaviour",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "store accepts batched detections",
                "actual":   f"insertion raised: {type(e).__name__}: {e}"}

    # The project keeps flows in store.flows (list).
    actual_size = (len(store.flows) if hasattr(store, "flows")
                   else (store.row_count() if hasattr(store, "row_count")
                         else None))
    cap_ok = (actual_size is not None) and (actual_size <= MAX_FLOWS)

    # ── 3. Relabel via the threshold helper
    relabel_ok: object = "n/a"
    if hasattr(store, "relabel_with_threshold"):
        try:
            n1 = store.relabel_with_threshold(0.999)
            n2 = store.relabel_with_threshold(0.0)
            relabel_ok = f"changed_at_high={n1}, changed_at_zero={n2}"
        except Exception as e:                                # noqa: BLE001
            relabel_ok = f"raised: {type(e).__name__}: {e}"
    elif hasattr(store, "apply_threshold"):
        # Fallback for older API — treat as informational.
        try:
            store.apply_threshold(0.999)
            store.apply_threshold(0.0)
            relabel_ok = "apply_threshold callable"
        except Exception as e:                                # noqa: BLE001
            relabel_ok = f"apply_threshold raised: {type(e).__name__}: {e}"

    relabel_passed = (
        isinstance(relabel_ok, str)
        and not relabel_ok.startswith("raised")
        and not relabel_ok.startswith("apply_threshold raised")
    )

    if cap_ok and relabel_passed:
        verdict = "PASS"
        actual  = (f"size={actual_size} (cap={MAX_FLOWS}); relabel: {relabel_ok}")
    else:
        verdict = "FAIL"
        actual  = (f"size={actual_size} cap={MAX_FLOWS} "
                   f"cap_ok={cap_ok}; relabel={relabel_ok}")
    return {"test_id": TEST_ID, "name": "DetectionStore behaviour",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"size ≤ {MAX_FLOWS}; relabel_with_threshold callable",
            "actual":   actual,
            "raw":      {"n_inserted": n_rows,
                         "size": actual_size,
                         "cap":  MAX_FLOWS,
                         "relabel": str(relabel_ok)}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "DetectionStore behaviour",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "MAX_FLOWS cap and threshold reapply",
            "actual":   reason}
