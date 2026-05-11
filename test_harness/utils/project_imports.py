"""
project_imports.py — Defensively import the project's modules.

The harness must keep running even if a project module or model file is
absent — the test that depended on it gets recorded as SKIPPED with a
clear reason.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

from .paths import PROJECT_ROOT


def ensure_on_path() -> None:
    """
    Put the project on sys.path. We add THREE roots:
        1. The project root          → enables `import monitoring`,
                                         `import app.foo`
        2. <project_root>/app/       → enables `from file_handler import …`,
                                         `from inference_bridge import …`,
                                         which the project's own modules use
                                         when launched from app/ (e.g.
                                         `python -m app.gui_main`)
        3. <project_root>/src/       → enables `from xai import …` paths
                                         that some helpers use directly
    """
    p_root = str(PROJECT_ROOT)
    p_app  = str(PROJECT_ROOT / "app")
    p_src  = str(PROJECT_ROOT / "src")
    for p in (p_root, p_app, p_src):
        if p not in sys.path:
            sys.path.insert(0, p)


def soft_import(module: str) -> tuple[Optional[Any], Optional[str]]:
    """Return (module_or_None, error_str_or_None)."""
    ensure_on_path()
    try:
        return importlib.import_module(module), None
    except Exception as e:                                 # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def model_present(rel_path: str) -> bool:
    """Quick check for a model artifact without importing torch."""
    return (PROJECT_ROOT / rel_path).exists()


def models_state() -> dict:
    """Single dict reporting which model artifacts exist."""
    candidates = [
        "models/stage1/rf_model.pkl",
        "models/stage1/s1_scaler.json",
        "models/stage2/iot_cnn_lstm.pt",
        "models/stage2/iot_scaler.json",
        "models/stage2/noniot_cnn_lstm.pt",
        "models/stage2/noniot_scaler.json",
    ]
    return {p: (PROJECT_ROOT / p).exists() for p in candidates}
