"""
_inference.py — Drive a CSV through the project's inference bridge.

Returns predictions per row aligned to the input order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from test_harness.utils.project_imports import (
    soft_import, ensure_on_path, models_state,
)


def predict_csv(csv_path: str) -> dict:
    ensure_on_path()
    ms = models_state()
    if not ms.get("models/stage1/rf_model.pkl") and \
       not (Path(csv_path).parent / "rf_model.json").exists():
        return {"skipped": True, "skip_reason": "Stage-1 RF model missing",
                "models_state": ms}

    bridge, err = soft_import("inference_bridge")
    if bridge is None:
        # The GUI ships it under app/, but inference_bridge.py is sometimes
        # discoverable at the project root via sys.path injection in the
        # original module. Try the alternative.
        bridge, err2 = soft_import("app.inference_bridge")
        if bridge is None:
            return {"skipped": True,
                    "skip_reason": f"inference_bridge: {err} | app.inference_bridge: {err2}"}

    # Resolve file_handler the same way inference_bridge does internally.
    # `app/` is on sys.path (added by ensure_on_path), so `import file_handler`
    # works; we also fall back to `app.file_handler` for safety.
    fh, _ = soft_import("file_handler")
    if fh is None:
        fh, _ = soft_import("app.file_handler")
    if fh is not None:
        try:
            info = fh.load_file(csv_path)
            if not info.is_valid:
                return {"skipped": True, "skip_reason": f"file_handler: {info.error}"}
        except Exception:
            class _Stub:
                pass
            info = _Stub()
            info.path = csv_path
            info.format = None
    else:
        class _Stub:
            pass
        info = _Stub()
        info.path = csv_path
        info.format = None

    try:
        results = bridge.run_file_inference(info)
    except Exception as e:                                    # noqa: BLE001
        return {"skipped": False, "error": f"{type(e).__name__}: {e}",
                "results": []}
    return {"skipped": False, "results": results, "n": len(results)}
