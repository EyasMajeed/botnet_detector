"""
detection_store.py — Central data store for the BotSense GUI.

A single instance of DetectionStore is owned by MainWindow and shared
across all pages. It receives flows from MonitorPage (live capture) and
UploadPage (batch inference), groups them into Reports, and persists
everything to disk so the dashboard / reports list survive app restarts.

Pages subscribe to two QObject signals:
    flows_changed()    — list of flows changed (new flow, batch added, cleared)
    reports_changed()  — reports list changed (new session/report, counters update)

Storage format:
    data/state/store.json
        { "flows": [...], "reports": [...], "next_report_n": <int> }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from PyQt6.QtCore import QObject, QTimer, pyqtSignal
    _QT = True
except Exception:
    _QT = False
    class QObject:
        def __init__(self, *a, **kw): pass
    def pyqtSignal(*a, **kw):
        class _S:
            def connect(self, *a, **k): pass
            def emit(self, *a, **k): pass
        return _S()
    QTimer = None


# ── Constants ────────────────────────────────────────────────────────────────
MAX_FLOWS          = 50_000   # cap to keep RAM and JSON file bounded
SIGNAL_DEBOUNCE_MS = 250      # coalesce flows_changed emissions to ~4Hz


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionFlow:
    """One completed flow + its detection result (live or upload)."""
    src_ip:        str   = ""
    dst_ip:        str   = ""
    src_port:      int   = 0
    dst_port:      int   = 0
    protocol:      str   = "—"
    label:         str   = "benign"      # "botnet" | "benign" | "unknown"
    confidence:    float = 0.0           # Stage-2 confidence (botnet probability)
    device_type:   str   = "noniot"      # "iot" | "noniot"
    s1_confidence: float = 0.0
    suspicion:     float = 0.0
    latency_ms:    float = 0.0
    alerted:       bool  = False
    timestamp:     float = field(default_factory=lambda: datetime.now().timestamp())
    # Provenance
    source:        str   = "live"        # "live" | "upload"
    source_file:   str   = ""            # filename when source == "upload"
    report_id:     str   = ""


@dataclass
class Report:
    """One scan session — either a live capture or a single uploaded file."""
    report_id:     str
    filename:      str
    created_at:    str       # "YYYY-MM-DD HH:MM:SS"
    n_flows:       int   = 0
    n_botnet:      int   = 0
    n_benign:      int   = 0
    n_iot:         int   = 0
    n_noniot:      int   = 0
    duration_sec:  float = 0.0
    source:        str   = "live"   # "live" | "upload"


# ═════════════════════════════════════════════════════════════════════════════
# DetectionStore
# ═════════════════════════════════════════════════════════════════════════════

class DetectionStore(QObject):
    """
    Central in-memory + on-disk store. Thread-safety is NOT required because
    every mutator is called from the Qt main thread (signals from QThreads
    are queued by default).
    """

    flows_changed   = pyqtSignal()
    reports_changed = pyqtSignal()

    def __init__(self, persist_path: Path, parent=None):
        super().__init__(parent)
        self.persist_path: Path                 = Path(persist_path)
        self.flows:        List[DetectionFlow]  = []
        self.reports:      List[Report]         = []
        self._next_report_n: int                = 1
        self._active_live: Optional[Report]     = None
        self._live_started_at: float            = 0.0

        # Debounced flows_changed timer (so a 100-flow burst causes ~1 redraw)
        self._dirty_flows = False
        if _QT and QTimer is not None:
            self._debounce = QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.timeout.connect(self._emit_flows_changed_now)
        else:
            self._debounce = None

        self.load()

    # ── Live session lifecycle (called by MonitorPage) ──────────────────────
    def start_live_session(self) -> str:
        """Begin a new live-capture report. Idempotent."""
        if self._active_live is not None:
            return self._active_live.report_id
        rid = self._next_id()
        self._active_live = Report(
            report_id  = rid,
            filename   = "<live capture>",
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source     = "live",
        )
        self._live_started_at = time.time()
        self.reports.append(self._active_live)
        self.reports_changed.emit()
        return rid

    def end_live_session(self) -> None:
        """Close the active live session and persist. No-op if none active."""
        if self._active_live is None:
            return
        self._active_live.duration_sec = round(time.time() - self._live_started_at, 1)
        self._active_live = None
        self.save()
        self.reports_changed.emit()

    def add_live_flow(self, f: DetectionFlow) -> None:
        """Append a flow from MonitorPage's BotnetMonitorThread."""
        if self._active_live is None:
            self.start_live_session()
        f.report_id = self._active_live.report_id
        f.source    = "live"
        self._append_flow(f)
        self._update_counters(self._active_live, f)
        self._mark_flows_dirty()

    # ── Upload batch (called from MainWindow once UploadPage emits) ─────────
    def add_upload_batch(self, flows: List[DetectionFlow], filename: str) -> str:
        """Register an uploaded file's results as a new completed report."""
        rid = self._next_id()
        report = Report(
            report_id  = rid,
            filename   = filename,
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source     = "upload",
        )
        for f in flows:
            f.report_id   = rid
            f.source      = "upload"
            f.source_file = filename
            self._append_flow(f)
            self._update_counters(report, f)
        self.reports.append(report)
        self.save()
        self._emit_flows_changed_now()
        self.reports_changed.emit()
        return rid

    # ── Read-only API for pages ─────────────────────────────────────────────
    def stats(self) -> dict:
        n  = len(self.flows)
        nb = sum(1 for f in self.flows if f.label == "botnet")
        ng = sum(1 for f in self.flows if f.label == "benign")
        ni = sum(1 for f in self.flows if f.device_type == "iot")
        nn = n - ni
        devs = len({f.src_ip for f in self.flows if f.src_ip})
        return {
            "total_flows": n, "n_botnet": nb, "n_benign": ng,
            "n_iot": ni, "n_noniot": nn, "devices": devs,
            "n_alerts": sum(1 for f in self.flows if f.alerted),
        }

    def botnet_per_minute(self, n_buckets: int = 20) -> List[int]:
        """Return botnet count per 1-minute bucket for the last n_buckets minutes."""
        now = time.time()
        bsz = 60.0
        buckets = [0] * n_buckets
        for f in self.flows:
            if f.label != "botnet":
                continue
            age = now - f.timestamp
            idx = n_buckets - 1 - int(age // bsz)
            if 0 <= idx < n_buckets:
                buckets[idx] += 1
        return buckets

    def flows_for_report(self, report_id: str) -> List[DetectionFlow]:
        return [f for f in self.flows if f.report_id == report_id]

    def recent_flows(self, n: int = 6) -> List[DetectionFlow]:
        return list(self.flows[-n:][::-1])     # newest first

    def avg_confidence(self) -> float:
        if not self.flows:
            return 0.0
        return float(sum(f.confidence for f in self.flows) / len(self.flows))

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "flows":         [asdict(f) for f in self.flows],
                "reports":       [asdict(r) for r in self.reports],
                "next_report_n": self._next_report_n,
            }
            tmp = self.persist_path.with_suffix(".json.tmp")
            with open(tmp, "w") as fp:
                json.dump(data, fp)
            tmp.replace(self.persist_path)
        except Exception as e:
            print(f"[DetectionStore] save failed: {e!r}")

    def load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path) as fp:
                data = json.load(fp)
            self.flows   = [DetectionFlow(**fd) for fd in data.get("flows", [])]
            self.reports = [Report(**rd)        for rd in data.get("reports", [])]
            self._next_report_n = int(data.get("next_report_n", len(self.reports) + 1))
        except Exception as e:
            print(f"[DetectionStore] load failed: {e!r}")
            self.flows   = []
            self.reports = []
            self._next_report_n = 1

    def clear(self) -> None:
        """Wipe everything. Used by Settings → 'Clear all data'."""
        self.flows.clear()
        self.reports.clear()
        self._next_report_n = 1
        self._active_live   = None
        self.save()
        self._emit_flows_changed_now()
        self.reports_changed.emit()

    # ── Internals ───────────────────────────────────────────────────────────
    def _next_id(self) -> str:
        rid = f"RPT-{self._next_report_n:03d}"
        self._next_report_n += 1
        return rid

    def _append_flow(self, f: DetectionFlow) -> None:
        self.flows.append(f)
        if len(self.flows) > MAX_FLOWS:
            del self.flows[: len(self.flows) - MAX_FLOWS]

    @staticmethod
    def _update_counters(r: Report, f: DetectionFlow) -> None:
        r.n_flows += 1
        if f.label == "botnet":
            r.n_botnet += 1
        elif f.label == "benign":
            r.n_benign += 1
        if f.device_type == "iot":
            r.n_iot += 1
        else:
            r.n_noniot += 1

    def _mark_flows_dirty(self) -> None:
        """Coalesce many add_live_flow calls into ~4Hz UI refreshes."""
        self._dirty_flows = True
        if self._debounce is not None and not self._debounce.isActive():
            self._debounce.start(SIGNAL_DEBOUNCE_MS)
        elif self._debounce is None:
            self._emit_flows_changed_now()

    def _emit_flows_changed_now(self) -> None:
        if self._dirty_flows:
            self._dirty_flows = False
        self.flows_changed.emit()