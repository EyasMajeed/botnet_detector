"""
monitor_page.py  —  Live Monitoring Page (upgraded from mockApp stub)
======================================================================
Drop-in replacement for the MonitorPage class in mockApp.py.

What changed vs the mock:
    - QTimer random demo  →  LiveCaptureThread (real OR demo-simulated)
    - No scoring          →  SuspicionScorer   (Section 7.3.4 thresholds)
    - No ML               →  inference_bridge  (stub today, real tomorrow)
    - Rows are colour-coded by SUSPICION SCORE, not just label
    - Suspicious flows (score ≥ 2) show a 🔴 badge; mild (score 1) show 🟡
    - Status bar shows sniff-trigger count and inference latency

How to use in mockApp.py:
    # Replace the MonitorPage class import / definition with:
    from app.monitor_page import MonitorPage

Or copy-paste the class body directly into mockApp.py.

Design tokens (BG, CARD, ERR, OK, etc.) are imported from the caller's
namespace at runtime — pass them as a dict to MonitorPage.__init__ if
you want to keep mockApp.py self-contained.  The file below assumes
the same globals as mockApp.py are in scope.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore    import Qt, QTimer, pyqtSlot
from PyQt6.QtGui     import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QCheckBox,
)


from live_capture      import LiveCaptureThread, get_interfaces, SCAPY_AVAILABLE
from suspicion_scorer  import SuspicionScorer
from inference_bridge  import run_inference

try:
    from monitor_bridge import BotnetMonitorThread, MONITOR_AVAILABLE
except Exception:
    BotnetMonitorThread = None
    MONITOR_AVAILABLE   = False

from detection_store import DetectionFlow

# ── Design tokens (must match mockApp.py) ─────────────────────────────────────
BG   = "#1E1E2F";  CARD = "#16161F";  BDR = "#374151"
TW   = "#FFFFFF";  TG   = "#9CA3AF";  TD  = "#6B7280"
ACC  = "#3A7AFE";  OK   = "#10B981";  ERR = "#EF4444"
WARN = "#F97316";  YEL  = "#EAB308";  FNT = "Segoe UI"

MAX_ROWS = 200   # keep last N flows in the table


def _lbl(text: str, size: int = 11, bold: bool = False,
         color: str = TW, mono: bool = False) -> QLabel:
    w = QLabel(text)
    f = QFont("Courier New" if mono else FNT, size)
    f.setBold(bold)
    w.setFont(f)
    w.setStyleSheet(f"color:{color};background:transparent;")
    return w


def _btn(text: str, style: str = "primary", small: bool = False) -> QPushButton:
    b = QPushButton(text)
    h = 30 if small else 36
    b.setFixedHeight(h)
    if style == "primary":
        css = (f"QPushButton{{background:{ACC};color:white;border:none;border-radius:8px;"
               f"padding:0 16px;font-size:{'11' if small else '13'}px;font-weight:bold;}}"
               f"QPushButton:hover{{background:#2A5FD4;}}")
    else:
        css = (f"QPushButton{{background:transparent;color:{TG};border:1px solid {BDR};"
               f"border-radius:8px;padding:0 12px;font-size:{'11' if small else '13'}px;}}"
               f"QPushButton:hover{{color:{TW};border-color:{ACC};}}")
    b.setStyleSheet(css)
    return b


def _table_css() -> str:
    return (f"QTableWidget{{background:{CARD};color:{TW};border:none;"
            f"font-size:12px;gridline-color:transparent;}}"
            f"QTableWidget::item{{padding:4px 10px;border-bottom:1px solid {BDR};}}"
            f"QTableWidget::item:selected{{background:{ACC};color:white;}}"
            f"QHeaderView::section{{background:{BG};color:{TG};border:none;"
            f"font-size:11px;padding:6px 10px;border-bottom:1px solid {BDR};}}")


# ══════════════════════════════════════════════════════════════════════════════
# MonitorPage
# ══════════════════════════════════════════════════════════════════════════════

class MonitorPage(QWidget):
    """
    Real-time network monitoring page.

    Lifecycle:
        __init__   — build UI, create thread + scorer (not started yet)
        showEvent  — auto-start capture when page becomes visible
        hideEvent  — auto-stop capture when page is hidden
    """

    def __init__(self, store=None, settings=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")

        # Shared services (may be None when run standalone for dev)
        self.store    = store
        self.settings = settings

        # State
        self._running      = False     # was True — auto-start logic depends on this
        self._t0           = datetime.now()
        self._total_flows  = 0
        self._alert_count  = 0
        self._sniff_count  = 0
        self._last_latency = 0.0

        # Core components (not started yet)
        self._scorer       = SuspicionScorer()
        self._model_path   = self._resolve_detector_model()
        # Prefer the full Stage-1 + Stage-2 pipeline when monitoring.py and
        # all model artifacts are loadable; fall back to the legacy
        # LiveCaptureThread (DETECTOR/LIVE/DEMO) otherwise.
        self._use_monitor  = MONITOR_AVAILABLE and BotnetMonitorThread is not None
        if self._use_monitor:
            self._thread = BotnetMonitorThread()
        else:
            self._thread = LiveCaptureThread(model_path=self._model_path)
        self._thread.flow_ready.connect(self._on_flow)
        self._thread.stats.connect(self._on_stats)
        self._thread.error.connect(self._on_error)

        # Uptime clock
        self._uptime_timer = QTimer(self)
        self._uptime_timer.timeout.connect(self._tick_uptime)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        # ── Header row ────────────────────────────────────────────────────────
        ch = QHBoxLayout()
        ch.addWidget(_lbl("Live Traffic Monitor", 18, bold=True))
        ch.addStretch()

        # Interface selector
        self._iface_cb = QComboBox()
        self._iface_cb.setFixedHeight(34)
        self._iface_cb.setMinimumWidth(220)
        cb_css = (f"QComboBox{{background:{CARD};color:{TW};border:1px solid {BDR};"
                  f"border-radius:8px;padding:0 10px;font-size:12px;}}"
                  f"QComboBox::drop-down{{border:none;}}"
                  f"QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}")
        self._iface_cb.setStyleSheet(cb_css)
        self._populate_interfaces()
        self._iface_cb.currentIndexChanged.connect(self._on_iface_change)
        ch.addWidget(self._iface_cb)
        ch.addSpacing(8)

        self._clr_btn = _btn("Clear", "outline", small=True)
        self._clr_btn.clicked.connect(self._clear_table)
        ch.addWidget(self._clr_btn)
        ch.addSpacing(8)

        self._tog_btn = _btn("▶  Start", "primary")
        self._tog_btn.clicked.connect(self._toggle)
        ch.addWidget(self._tog_btn)
        root.addLayout(ch)

        # ── Stat strip ────────────────────────────────────────────────────────
        sr = QHBoxLayout()
        sr.setSpacing(10)
        self._stat_labels: dict[str, QLabel] = {}
        for title, val, color in [
            ("Flows/sec",    "0",          ACC),
            ("Bandwidth",    "0 KB/s",     OK),
            ("Alerts",       "0",          ERR),
            ("Sniff Triggers","0",         WARN),
            ("Uptime",       "00:00:00",   TG),
        ]:
            f = QFrame()
            f.setStyleSheet(f"background:{CARD};border:none;border-radius:10px;")
            fv = QVBoxLayout(f)
            fv.setContentsMargins(14, 8, 14, 8)
            fv.setSpacing(2)
            vl = _lbl(val, 16, bold=True, color=color, mono=True)
            fv.addWidget(_lbl(title, 10, color=TD))
            fv.addWidget(vl)
            self._stat_labels[title] = vl
            sr.addWidget(f)
        root.addLayout(sr)

        # ── Table card ────────────────────────────────────────────────────────
        tc = QFrame()
        tc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        tv = QVBoxLayout(tc)
        tv.setContentsMargins(0, 0, 0, 0)

        # Table header bar
        tb = QWidget()
        tb.setFixedHeight(44)
        tb.setStyleSheet(
            f"background:{BG};border-radius:12px 12px 0 0;"
            f"border-bottom:1px solid {BDR};"
        )
        th = QHBoxLayout(tb)
        th.setContentsMargins(16, 0, 16, 0)
        th.addWidget(_lbl("Real-time Flow Feed", 13, bold=True))
        th.addStretch()

        # Filter
        self._filter_cb = QComboBox()
        self._filter_cb.addItems(["All", "Botnet", "Benign", "Suspicious (score≥2)"])
        self._filter_cb.setFixedHeight(28)
        self._filter_cb.setStyleSheet(
            f"QComboBox{{background:{CARD};color:{TG};border:1px solid {BDR};"
            f"border-radius:6px;padding:0 8px;font-size:11px;}}"
            f"QComboBox::drop-down{{border:none;}}"
            f"QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}"
        )
        th.addWidget(self._filter_cb)
        th.addSpacing(12)

        self._flow_count_lbl = _lbl("0 flows", 11, color=TD)
        th.addWidget(self._flow_count_lbl)
        tv.addWidget(tb)

        # Table
        cols = ["Time", "Src IP", "Dst IP", "Protocol",
                "Label", "Confidence", "Device", "Score", "Sniff?"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet(_table_css())

        # Per-column sizing so short fields (Time, Protocol, Score…) don't get
        # the same width as long IP:port pairs. Only Src IP and Dst IP stretch
        # to absorb extra horizontal space — every other column is fixed-width.
        h = self._table.horizontalHeader()
        h.setStretchLastSection(False)
        for col, mode, w in [
            (0, QHeaderView.ResizeMode.Fixed,    80),   # Time      08:14:22
            (1, QHeaderView.ResizeMode.Stretch,   0),   # Src IP    192.168.100.151:54012
            (2, QHeaderView.ResizeMode.Stretch,   0),   # Dst IP    8.8.8.8:443
            (3, QHeaderView.ResizeMode.Fixed,    70),   # Protocol  TCP / UDP / ICMP
            (4, QHeaderView.ResizeMode.Fixed,    80),   # Label     Botnet / Benign
            (5, QHeaderView.ResizeMode.Fixed,    90),   # Confidence 12.34%
            (6, QHeaderView.ResizeMode.Fixed,    80),   # Device    NONIOT / IOT
            (7, QHeaderView.ResizeMode.Fixed,    70),   # Score     🔴 0
            (8, QHeaderView.ResizeMode.Fixed,    80),   # Sniff?    🔴 Yes / —
        ]:
            h.setSectionResizeMode(col, mode)
            if w:
                self._table.setColumnWidth(col, w)
        self._table.verticalHeader().setDefaultSectionSize(34)

        # Right-align numeric columns so digits line up vertically.
        # (We do this once on the header here; cell-level alignment is set in
        #  _add_row when each item is created.)
        tv.addWidget(self._table)
        root.addWidget(tc)

        # ── Inference latency footer ──────────────────────────────────────────
        self._status_lbl = _lbl("Idle — press Start to begin monitoring.", 10, color=TD)
        root.addWidget(self._status_lbl)

    # ── Public control ────────────────────────────────────────────────────────

    def start_capture(self):
        if self._running:
            return

        # If using BotnetMonitorThread, build the BotnetMonitor in THIS (main)
        # thread before start(). torch.load() segfaults from worker threads on
        # macOS when PyQt6 is loaded — main-thread construction is the only
        # reliable path. First click blocks for ~3-5s while 3 models load.
        if hasattr(self._thread, "ensure_monitor"):
            self._tog_btn.setEnabled(False)
            self._status_lbl.setText("● Loading detection models…")
            self._status_lbl.setStyleSheet(f"color:{TG};background:transparent;")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()                # paint the status text
            ok = self._thread.ensure_monitor()
            self._tog_btn.setEnabled(True)
            if not ok:
                self._on_error(getattr(self._thread, "_init_error",
                                       "BotnetMonitor init failed"))
                return

        self._running = True
        self._t0      = datetime.now()
        self._scorer.reset_baseline()
        self._thread.start()
        self._uptime_timer.start(1000)
        self._tog_btn.setText("⏸  Pause")
        self._status_lbl.setText("● Capturing…")
        self._status_lbl.setStyleSheet(f"color:{OK};background:transparent;")

    def stop_capture(self):
        if not self._running:
            return
        self._running = False
        self._thread.stop()
        self._uptime_timer.stop()
        self._tog_btn.setText("▶  Resume")
        self._status_lbl.setText("Paused.")
        self._status_lbl.setStyleSheet(f"color:{TG};background:transparent;")

    # ── Qt lifecycle ──────────────────────────────────────────────────────────
    ''' uncomment the following if u want the live monitoring stops when u switch to another page,
     and restarts when u switch back.'''


    # def showEvent(self, event):
    #     super().showEvent(event)
    #     # Auto-start when page becomes visible the first time
    #     if not self._running and self._total_flows == 0:
    #         self.start_capture()

    # def hideEvent(self, event):
    #     super().hideEvent(event)
    #     self.stop_capture()


    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_flow(self, flow: dict):
        """Receive one flow from LiveCaptureThread, score + infer, add to table."""
        if not self._running:
            return

        # Ignore detector heartbeat frames that keep the UI alive
        # if flow.get("_heartbeat"):
        #     return

        # 1. Suspicion scoring (fast, no ML)
        score_result = self._scorer.score(flow)
        score        = score_result["score"]
        trigger_sniff= score_result["trigger_sniff"]

        # 2. ML inference / detector mode
        if "_label" in flow:
            label      = flow.get("_label", "benign")
            confidence = float(flow.get("_botnet_prob", 0.0))
            # Trust _device_type from the bridge; fall back to alert-flag heuristic
            # only for legacy LiveCaptureThread DETECTOR-mode flows that lack it.
            device     = flow.get(
                "_device_type",
                "iot" if flow.get("_alert") else "noniot",
            )
            self._last_latency = float(flow.get("_latency_ms", 0.0))
        else:
            infer = run_inference(flow)
            self._last_latency = infer["latency_ms"]
            label      = infer["label"]
            confidence = infer["confidence"]
            device     = infer["device_type"]

        # 2b. Apply user's confidence threshold from Settings (functional).
        # The pre-trained models have internal thresholds (IoT 0.52, NonIoT
        # 0.7656); this lets the user move the operating point on the ROC
        # curve without retraining. 'unknown' labels are preserved.
        if self.settings is not None and label != "unknown":
            from detection_store import apply_threshold
            label = apply_threshold(label, confidence, self.settings.confidence_threshold)

        # 3. Update counters
        self._total_flows += 1
        if label == "botnet":
            self._alert_count += 1
        if trigger_sniff:
            self._sniff_count += 1

        # 3b. Push to the shared store so Dashboard / Results / Reports update.
        if self.store is not None:
            try:
                self.store.add_live_flow(DetectionFlow(
                    src_ip        = str(flow.get("src_ip", "")),
                    dst_ip        = str(flow.get("dst_ip", "")),
                    src_port      = int(flow.get("src_port", 0) or 0),
                    dst_port      = int(flow.get("dst_port", 0) or 0),
                    protocol      = str(flow.get("protocol", "—")),
                    label         = label,
                    confidence    = confidence,
                    device_type   = device,
                    s1_confidence = float(flow.get("_s1_confidence", 0.0)),
                    suspicion     = float(flow.get("_suspicion", score)),
                    latency_ms    = self._last_latency,
                    alerted       = bool(flow.get("_alert", False)),
                ))
            except Exception as _e:
                # Never let a store hiccup kill live capture.
                pass

        # 4. Apply filter
        filt = self._filter_cb.currentText()
        if filt == "Botnet"             and label != "botnet":      return
        if filt == "Benign"             and label != "benign":      return
        if filt == "Suspicious (score≥2)" and score < 2:            return

        # 5. Append row to table
        self._add_row(flow, label, confidence, device, score, trigger_sniff)

        # 6. Status bar
        self._status_lbl.setText(
            f"● Flows: {self._total_flows}  |  "
            f"Last inference: {self._last_latency:.1f} ms  |  "
            f"Sniff triggers: {self._sniff_count}"
        )
        self._flow_count_lbl.setText(f"{self._table.rowCount()} rows")

    @pyqtSlot(dict)
    def _on_stats(self, stats: dict):
        self._stat_labels["Flows/sec"].setText(str(stats.get("flows_per_sec", 0)))
        bw = stats.get("bandwidth_kbps", 0)
        self._stat_labels["Bandwidth"].setText(
            f"{bw} KB/s" if bw < 1024 else f"{bw/1024:.1f} MB/s"
        )
        self._stat_labels["Alerts"].setText(str(self._alert_count))
        self._stat_labels["Sniff Triggers"].setText(str(self._sniff_count))

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._status_lbl.setText(f"⚠ {msg}")
        self._status_lbl.setStyleSheet(f"color:{ERR};background:transparent;")

    def _toggle(self):
        if self._running:
            self.stop_capture()
        else:
            self.start_capture()

    def _clear_table(self):
        self._table.setRowCount(0)
        self._total_flows = 0
        self._alert_count = 0
        self._sniff_count = 0
        self._scorer.reset_baseline()
        self._flow_count_lbl.setText("0 rows")
        for lbl, val in [("Flows/sec","0"),("Bandwidth","0 KB/s"),
                          ("Alerts","0"),("Sniff Triggers","0")]:
            self._stat_labels[lbl].setText(val)

    def _populate_interfaces(self):
            """Fill the interface dropdown with real interfaces + a Demo option."""
            self._iface_cb.blockSignals(True)
            self._iface_cb.clear()
            self._iface_cb.addItem("🔴  Demo Mode (simulated)", userData=None)

            ifaces = get_interfaces()
            for name, ip in ifaces:
                self._iface_cb.addItem(f"📡  {name}  ({ip})", userData=name)

            if not ifaces and not SCAPY_AVAILABLE:
                self._iface_cb.addItem("⚠  Scapy not installed — Demo only", userData=None)
            elif not ifaces:
                self._iface_cb.addItem("⚠  No interfaces found — run setup_live_capture.py", userData=None)
            else:
                # Default to first real interface (index 1, since index 0 is Demo)
                self._iface_cb.setCurrentIndex(1)
            # ── BUG FIX ────────────────────────────────────────────────────────
            # blockSignals(False) was previously orphaned inside
            # _resolve_detector_model() after a return — unreachable. The combo
            # box stayed permanently muted, so changing the interface did nothing.
            self._iface_cb.blockSignals(False)

    def _resolve_detector_model(self) -> Optional[str]:
        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "stage2" / "iot_cnn_lstm.pt"
        return str(model_path) if model_path.exists() else None

    def _on_iface_change(self, idx: int):
        """User selected a different interface / demo mode."""
        iface = self._iface_cb.itemData(idx)   # None → demo mode
        was_running = self._running
        if was_running:
            self.stop_capture()
        self._thread.flow_ready.disconnect()
        self._thread.stats.disconnect()
        self._thread.error.disconnect()
        if iface is None:
            # Demo mode is always served by LiveCaptureThread
            self._thread = LiveCaptureThread(demo_mode=True)
        elif self._use_monitor:
            # Full Stage-1 + Stage-2 pipeline on a real interface
            self._thread = BotnetMonitorThread(interface=iface)
        else:
            # Legacy fallback (single-stage detector or plain scapy)
            self._thread = LiveCaptureThread(
                interface=iface,
                model_path=self._model_path,
                demo_mode=False,
            )
        self._thread.flow_ready.connect(self._on_flow)
        self._thread.stats.connect(self._on_stats)
        self._thread.error.connect(self._on_error)
        if was_running:
            self.start_capture()

    def _tick_uptime(self):
        up = str(datetime.now() - self._t0).split(".")[0]
        self._stat_labels["Uptime"].setText(up)

    # ── Table helpers ─────────────────────────────────────────────────────────

    def _add_row(self, flow: dict, label: str, conf: float,
                 device: str, score: int, sniff: bool):
        is_botnet = (label == "botnet")
        ts = datetime.now().strftime("%H:%M:%S")

        # Row background tint by severity
        if is_botnet or score >= 2:
            row_bg = "#2A1515"      # deep red tint
        elif score == 1:
            row_bg = "#1F1A0E"      # amber tint
        else:
            row_bg = ""             # default

        sniff_icon = "🔴 Yes" if sniff else "—"
        score_str  = f"{'🔴' if score >= 2 else '🟡' if score == 1 else '🟢'} {score}"
        label_str  = label.capitalize()
        conf_str   = f"{conf:.2%}"
        dev_str    = device.upper()
        proto      = str(flow.get("protocol", "—"))
        src        = f"{flow.get('src_ip','?')}:{flow.get('src_port','?')}"
        dst        = f"{flow.get('dst_ip','?')}:{flow.get('dst_port','?')}"

        values = [ts, src, dst, proto, label_str, conf_str, dev_str, score_str, sniff_icon]
        colors = [TW, TW, TW, TG,
                  ERR if is_botnet else OK,    # Label
                  TW,                          # Confidence
                  ACC,                         # Device
                  ERR if score >= 2 else YEL if score == 1 else OK,   # Score
                  ERR if sniff else TG]        # Sniff

        r = self._table.rowCount()
        self._table.insertRow(r)
        # Per-column horizontal alignment. Long text (Src/Dst IP) stays left;
        # short categorical / numeric columns centre so the visual grid stays
        # aligned even with mixed-width emoji prefixes.
        ALIGN_CENTER = (Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        ALIGN_LEFT   = (Qt.AlignmentFlag.AlignLeft   | Qt.AlignmentFlag.AlignVCenter)
        col_align = {
            0: ALIGN_CENTER,  # Time
            1: ALIGN_CENTER,  # Src IP
            2: ALIGN_CENTER,  # Dst IP
            3: ALIGN_CENTER,  # Protocol
            4: ALIGN_CENTER,  # Label
            5: ALIGN_CENTER,  # Confidence
            6: ALIGN_CENTER,  # Device
            7: ALIGN_CENTER,  # Score
            8: ALIGN_CENTER,  # Sniff?
        }
        for j, (val, col) in enumerate(zip(values, colors)):
            item = QTableWidgetItem(val)
            item.setForeground(QColor(col))
            item.setTextAlignment(col_align.get(j, ALIGN_LEFT))
            if j in (4, 7):
                f = QFont(FNT, 11)
                f.setBold(True)
                item.setFont(f)
            if j in (0, 5):
                item.setFont(QFont("Courier New", 11))
            if row_bg:
                item.setBackground(QColor(row_bg))
            self._table.setItem(r, j, item)

        # Enforce MAX_ROWS
        if self._table.rowCount() > MAX_ROWS:
            self._table.removeRow(0)

        self._table.scrollToBottom()