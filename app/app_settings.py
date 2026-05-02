"""
app_settings.py — Persistent application settings.

Single instance owned by MainWindow. Backed by data/state/settings.json.
Pages read settings via attribute access; setters auto-persist + emit
settings_changed(key).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except Exception:
    class QObject:
        def __init__(self, *a, **kw): pass
    def pyqtSignal(*a, **kw):
        class _S:
            def connect(self, *a, **k): pass
            def emit(self, *a, **k): pass
        return _S()


DEFAULTS: Dict[str, Any] = {
    "confidence_threshold": 0.50,
    "xai_enabled":          True,
    "real_time_alerts":     True,
    "auto_export_reports":  False,
    "output_dir":           str(Path.home() / "Desktop" / "botnet_reports"),
    "table_row_limit":      1000,    # 0 = unlimited
}


class AppSettings(QObject):
    """JSON-backed settings with one signal: settings_changed(str)."""

    settings_changed = pyqtSignal(str)   # emits the changed key

    def __init__(self, persist_path: Path, parent=None):
        super().__init__(parent)
        self.persist_path: Path = Path(persist_path)
        self._d: Dict[str, Any] = dict(DEFAULTS)
        self.load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default if default is not None else DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        if self._d.get(key) == value:
            return
        self._d[key] = value
        self.save()
        self.settings_changed.emit(key)

    @property
    def confidence_threshold(self) -> float:
        return float(self._d.get("confidence_threshold", DEFAULTS["confidence_threshold"]))

    @property
    def xai_enabled(self) -> bool:
        return bool(self._d.get("xai_enabled", DEFAULTS["xai_enabled"]))

    @property
    def real_time_alerts(self) -> bool:
        return bool(self._d.get("real_time_alerts", DEFAULTS["real_time_alerts"]))

    @property
    def auto_export_reports(self) -> bool:
        return bool(self._d.get("auto_export_reports", DEFAULTS["auto_export_reports"]))

    @property
    def output_dir(self) -> str:
        return str(self._d.get("output_dir", DEFAULTS["output_dir"]))

    @property
    def table_row_limit(self) -> int:
        return int(self._d.get("table_row_limit", DEFAULTS["table_row_limit"]))

    def save(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".json.tmp")
            with open(tmp, "w") as fp:
                json.dump(self._d, fp, indent=2)
            tmp.replace(self.persist_path)
        except Exception as e:
            print(f"[AppSettings] save failed: {e!r}")

    def load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path) as fp:
                loaded = json.load(fp)
            for k, v in loaded.items():
                self._d[k] = v
        except Exception as e:
            print(f"[AppSettings] load failed: {e!r}")