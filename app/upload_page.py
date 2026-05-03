"""
upload_page.py  —  Upload & Analyze Page (real file handling)
=============================================================
Drop-in replacement for the UploadPage class in mockApp.py.

What's upgraded vs the mock:
    - Drag-and-drop actually works (QDragEnterEvent / QDropEvent)
    - Browse dialog filters to supported extensions
    - file_handler.load_file() validates + detects format instantly
    - File info card shows: format badge, size, estimated rows, columns
    - Warnings panel appears when the handler flags issues
    - "Run Detection" calls inference_bridge.run_file_inference() (Stage-1 ready)
    - A progress overlay appears while processing
    - Emits  file_ready(FileInfo)  signal so other pages can react

How to use in mockApp.py:
    from app.upload_page import UploadPage
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from PyQt6.QtCore    import Qt, pyqtSignal, QThread, pyqtSlot, QMimeData
from PyQt6.QtGui     import QColor, QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QFileDialog, QScrollArea, QProgressBar,
    QSizePolicy, QMessageBox,
)

from file_handler   import load_file, FileInfo, FileFormat
from inference_bridge import run_inference, run_file_inference

# ── Design tokens (single source of truth: app/theme.py) ─────────────────────
# The page used to hardcode its own slightly-off palette
# (BG=#1E1E2F vs theme BG=#0A0E1A, ACC=#3A7AFE vs theme ACC=#3FA9F5),
# which is why this page looked subtly different from Dashboard / Results /
# Reports / Settings. Import the same tokens those pages do.
from theme import (
    BG, CARD, BDR, TW, TG, TD,
    ACC, OK, ERR, WARN, YEL, FNT,
)

# We only advertise extensions the inference pipeline actually handles end-to-end:
#   .pcap / .pcapng → Stage-1 + Stage-2 IoT + NonIoT (full pipeline via BotnetMonitor)
#   .csv            → Stage-1 + Stage-2 NonIoT (IoT rows return 'unknown' because
#                     Stage-2 IoT needs raw packet sequences, not flow summaries)
# Other formats (.binetflow / .nfcapd / .nfdump) require pre-processing through
# data_processing/process_*.py before they can be uploaded.
ACCEPT_FILTER = (
    "Supported Traffic Files (*.pcap *.pcapng *.csv);;"
    "PCAP Files (*.pcap *.pcapng);;"
    "CSV Files (*.csv);;"
    "All Files (*)"
)

FORMAT_COLORS: dict[FileFormat, str] = {
    FileFormat.PCAP:         ACC,
    FileFormat.PCAPNG:       ACC,
    FileFormat.CSV_UNIFIED:  OK,
    FileFormat.CSV_CICFLOW:  OK,
    FileFormat.CSV_CTU13:    OK,
    FileFormat.CSV_UNSW:     OK,
    FileFormat.CSV_GENERIC:  YEL,
    FileFormat.NETFLOW_BIN:  "#A855F7",
    FileFormat.NETFLOW_CSV:  "#A855F7",
    FileFormat.UNKNOWN:      ERR,
}


# ── Tiny widget helpers ────────────────────────────────────────────────────────

def _lbl(text: str, size: int = 11, bold: bool = False,
         color: str = TW, mono: bool = False) -> QLabel:
    w = QLabel(text)
    f = QFont("Courier New" if mono else FNT, size)
    f.setBold(bold)
    w.setFont(f)
    w.setStyleSheet(f"color:{color};background:transparent;")
    return w


def _btn(text: str, style: str = "primary", small: bool = False,
         width: Optional[int] = None) -> QPushButton:
    b = QPushButton(text)
    h = 30 if small else 38
    b.setFixedHeight(h)
    if width:
        b.setFixedWidth(width)
    if style == "primary":
        css = (f"QPushButton{{background:{ACC};color:white;border:none;"
               f"border-radius:9px;padding:0 20px;"
               f"font-size:{'11' if small else '13'}px;font-weight:bold;}}"
               f"QPushButton:hover{{background:#2A5FD4;}}"
               f"QPushButton:disabled{{background:#2A3A5A;color:#6B7280;}}")
    elif style == "danger":
        css = (f"QPushButton{{background:transparent;color:{ERR};"
               f"border:1px solid {ERR};border-radius:9px;padding:0 14px;"
               f"font-size:11px;}}"
               f"QPushButton:hover{{background:#3A1515;}}")
    else:
        css = (f"QPushButton{{background:transparent;color:{TG};"
               f"border:1px solid {BDR};border-radius:9px;padding:0 14px;"
               f"font-size:{'11' if small else '13'}px;}}"
               f"QPushButton:hover{{color:{TW};border-color:{ACC};}}")
    b.setStyleSheet(css)
    return b


def _badge(text: str, color: str) -> QLabel:
    w = QLabel(text)
    w.setFont(QFont(FNT, 10, QFont.Weight.Bold))
    w.setStyleSheet(
        f"color:white;background:{color};border-radius:6px;"
        f"padding:2px 8px;"
    )
    w.setFixedHeight(22)
    return w


def _divider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setStyleSheet(f"color:{BDR};background:{BDR};")
    d.setFixedHeight(1)
    return d


def _card(radius: int = 12) -> QFrame:
    f = QFrame()
    f.setStyleSheet(
        f"QFrame{{background:{CARD};border:none;border-radius:{radius}px;}}"
    )
    return f


# ══════════════════════════════════════════════════════════════════════════════
# DropZone — accepts drag-and-drop
# ══════════════════════════════════════════════════════════════════════════════

class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    _NORMAL_CSS = (
        f"QFrame{{background:#16161F;border:2px dashed #374151;"
        f"border-radius:12px;}}"
        f"QFrame:hover{{border-color:#3A7AFE;}}"
    )
    _HOVER_CSS = (
        f"QFrame{{background:#1A2040;border:2px dashed #3A7AFE;"
        f"border-radius:12px;}}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(190)
        self.setStyleSheet(self._NORMAL_CSS)

        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(8)

        self._icon = _lbl("⬆", 36, color=ACC)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._icon)

        self._title = _lbl("Click to upload or drag & drop", 14, bold=True)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._title)

        self._sub = _lbl(
            "PCAP · PCAPng · CSV   —  Max 500 MB", 11, color=TD
        )
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._sub)

        v.addSpacing(6)
        self._browse_btn = _btn("Browse Files", "primary", width=150)
        v.addWidget(self._browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Drag & drop events ────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._HOVER_CSS)
            self._title.setText("Release to load file")

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._NORMAL_CSS)
        self._title.setText("Click to upload or drag & drop")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._NORMAL_CSS)
        self._title.setText("Click to upload or drag & drop")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.file_dropped.emit(path)


# ══════════════════════════════════════════════════════════════════════════════
# FileInfoCard — shown after successful validation
# ══════════════════════════════════════════════════════════════════════════════

class FileInfoCard(QFrame):
    cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{CARD};border:none;border-radius:12px;}}"
        )
        self.hide()

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(18, 14, 18, 14)
        self._root.setSpacing(10)

        # ── Top row: icon + name + badge + clear ──────────────────────────
        top = QHBoxLayout()
        self._icon_lbl  = _lbl("📦", 22)
        self._name_lbl  = _lbl("—", 13, bold=True)
        self._fmt_badge = _badge("—", ACC)
        self._clear_btn = _btn("✕", "danger", small=True, width=32)
        self._clear_btn.clicked.connect(self._on_clear)

        top.addWidget(self._icon_lbl)
        top.addSpacing(8)
        top.addWidget(self._name_lbl)
        top.addSpacing(10)
        top.addWidget(self._fmt_badge)
        top.addStretch()
        top.addWidget(self._clear_btn)
        self._root.addLayout(top)

        self._root.addWidget(_divider())

        # ── Meta grid ────────────────────────────────────────────────────
        meta = QHBoxLayout()
        meta.setSpacing(30)

        self._size_lbl   = self._meta_cell("Size",      "—")
        self._rows_lbl   = self._meta_cell("Rows (est)","—")
        self._cols_lbl   = self._meta_cell("Columns",   "—")
        self._path_lbl   = self._meta_cell("Path",      "—")

        for w in [self._size_lbl, self._rows_lbl, self._cols_lbl, self._path_lbl]:
            meta.addWidget(w)
        meta.addStretch()
        self._root.addLayout(meta)

        # ── Column preview ────────────────────────────────────────────────
        self._col_preview = _lbl("", 10, color=TD, mono=True)
        self._col_preview.setWordWrap(True)
        self._root.addWidget(self._col_preview)

        # ── Warnings ──────────────────────────────────────────────────────
        self._warn_lbl = _lbl("", 11, color=WARN)
        self._warn_lbl.setWordWrap(True)
        self._warn_lbl.hide()
        self._root.addWidget(self._warn_lbl)

    def _meta_cell(self, label: str, value: str) -> QWidget:
        w  = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        vl.addWidget(_lbl(label, 10, color=TD))
        val = _lbl(value, 12, bold=True, mono=True)
        vl.addWidget(val)
        w._val_lbl = val   # store ref for updates
        return w

    def populate(self, info: FileInfo):
        fmt_color = FORMAT_COLORS.get(info.format, ACC)

        self._icon_lbl.setText(info.icon)
        self._name_lbl.setText(info.filename)
        self._fmt_badge.setText(info.format_label)
        self._fmt_badge.setStyleSheet(
            f"color:white;background:{fmt_color};"
            f"border-radius:6px;padding:2px 8px;"
        )

        self._size_lbl._val_lbl.setText(info.size_label)

        if info.row_count is not None:
            self._rows_lbl._val_lbl.setText(f"~{info.row_count:,}")
        else:
            self._rows_lbl._val_lbl.setText("N/A (binary)")

        self._cols_lbl._val_lbl.setText(
            str(info.col_count) if info.col_count else "—"
        )
        # Truncate path for display
        path_display = info.path
        if len(path_display) > 55:
            path_display = "…" + path_display[-52:]
        self._path_lbl._val_lbl.setText(path_display)
        self._path_lbl._val_lbl.setToolTip(info.path)

        if info.columns:
            preview = "  ·  ".join(info.columns[:12])
            if len(info.columns) > 12:
                preview += f"  …+{len(info.columns)-12} more"
            self._col_preview.setText(preview)
        else:
            self._col_preview.setText("")

        if info.warnings:
            self._warn_lbl.setText("⚠  " + "  ·  ".join(info.warnings))
            self._warn_lbl.show()
        else:
            self._warn_lbl.hide()

        self.show()

    def _on_clear(self):
        self.hide()
        self.cleared.emit()


# ══════════════════════════════════════════════════════════════════════════════
# UploadPage — main page widget
# ══════════════════════════════════════════════════════════════════════════════

class UploadPage(QWidget):
    """
    Upload & Analyze page.

    Signals:
        file_ready(FileInfo)   — emitted after successful validation
        analysis_done(list)    — emitted after Run Detection completes
                                 (list of result dicts from inference_bridge)
    """

    file_ready    = pyqtSignal(object)   # FileInfo
    analysis_done = pyqtSignal(list)     # list[dict] from inference_bridge

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")
        self._current_file: Optional[FileInfo] = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Scroll wrapper
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")

        inner = QWidget()
        inner.setStyleSheet(f"background:{BG};")
        root = QVBoxLayout(inner)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        # ── Page title (18pt bold — matches every other page) ─────────────
        root.addWidget(_lbl("Upload & Analyze", 18, bold=True))

        # ── Drop zone ─────────────────────────────────────────────────────
        self._drop_zone = DropZone()
        self._drop_zone._browse_btn.clicked.connect(self._browse)
        self._drop_zone.file_dropped.connect(self._load_file)
        self._drop_zone.mousePressEvent = lambda _: self._browse()
        root.addWidget(self._drop_zone)

        # ── File info card ────────────────────────────────────────────────
        self._file_card = FileInfoCard()
        self._file_card.cleared.connect(self._on_file_cleared)
        root.addWidget(self._file_card)

        # ── Error label ───────────────────────────────────────────────────
        self._error_lbl = _lbl("", 11, color=ERR)
        self._error_lbl.setWordWrap(True)
        self._error_lbl.hide()
        root.addWidget(self._error_lbl)

        # ── Detection settings card ───────────────────────────────────────
        root.addWidget(self._build_settings_card())

        # ── Run button + progress ─────────────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = _btn("▶  Run Detection", "primary", width=180)
        self._run_btn.setEnabled(False)
        # Single permanent connection — _on_run_button_clicked dispatches based
        # on the current run state (start vs cancel). Avoids the disconnect/
        # reconnect dance that's easy to get wrong with Qt's signal API.
        self._run_btn.clicked.connect(self._on_run_button_clicked)

        # Async PCAP worker handle (None when no PCAP is being processed).
        self._pcap_worker = None
        self._status_lbl = _lbl("Load a file to begin.", 11, color=TD)
        run_row.addWidget(self._run_btn)
        run_row.addSpacing(16)
        run_row.addWidget(self._status_lbl)
        run_row.addStretch()
        root.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{BDR};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{ACC};border-radius:2px;}}"
        )
        self._progress.hide()
        root.addWidget(self._progress)

        root.addStretch()

        scroll.setWidget(inner)
        ol = QVBoxLayout(self)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)

    def _build_settings_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(12)
        v.addWidget(_lbl("Detection Configuration", 14, bold=True))
        v.addWidget(_lbl(
            "PCAP files run the full Stage-1 + Stage-2 IoT/NonIoT pipeline. "
            "CSV files run Stage-1 + Stage-2 NonIoT only — IoT rows are flagged "
            "as 'unknown' because Stage-2 IoT requires raw packet sequences.",
            11, color=TD
        ))
        v.addWidget(_divider())

        # Model info row
        models_row = QHBoxLayout()
        for label, value, color in [
            ("Stage-1 Classifier", "Random Forest",          ACC),
            ("Stage-2 IoT",        "CNN-LSTM (PCAP only)",   OK),
            ("Stage-2 Non-IoT",    "CNN-LSTM (PCAP + CSV)",  OK),
            ("XAI Method",         "Permutation Importance", YEL),
        ]:
            cell = QWidget()
            cell.setStyleSheet("background:transparent;")
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(3)
            cv.addWidget(_lbl(label, 10, color=TD))
            cv.addWidget(_lbl(value, 12, bold=True, color=color))
            models_row.addWidget(cell)
        models_row.addStretch()
        v.addLayout(models_row)

        return card

    # ── File loading ──────────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Traffic File", "", ACCEPT_FILTER
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        """Validate + detect format, update UI."""
        self._error_lbl.hide()
        self._file_card.hide()
        self._status_lbl.setText("Validating…")
        self._run_btn.setEnabled(False)

        info = load_file(path)

        if not info.is_valid:
            self._show_error(info.error)
            self._current_file = None
            return

        self._current_file = info
        self._file_card.populate(info)
        self._run_btn.setEnabled(True)
        self._status_lbl.setText(
            f"✔  {info.format_label} detected — ready to run detection."
        )
        self._status_lbl.setStyleSheet(f"color:{OK};background:transparent;")
        self.file_ready.emit(info)

    def _show_error(self, msg: str):
        self._error_lbl.setText(f"⚠  {msg}")
        self._error_lbl.show()
        self._status_lbl.setText("File rejected.")
        self._status_lbl.setStyleSheet(f"color:{ERR};background:transparent;")

    def _on_file_cleared(self):
        self._current_file = None
        self._run_btn.setEnabled(False)
        self._error_lbl.hide()
        self._status_lbl.setText("Load a file to begin.")
        self._status_lbl.setStyleSheet(f"color:{TD};background:transparent;")

    # ── Run detection ─────────────────────────────────────────────────────────

    # ── Run button dispatcher ─────────────────────────────────────────────
    def _on_run_button_clicked(self):
        """Single permanent slot — dispatches to start or cancel."""
        if self._pcap_worker is not None and self._pcap_worker.isRunning():
            self._on_pcap_cancel()
        else:
            self._run_detection()

    def _run_detection(self):
        if not self._current_file:
            return
        # PCAP / PCAPNG → async worker (see _run_pcap_async).
        # Everything else → synchronous bridge call (fast enough).
        if self._current_file.format in (FileFormat.PCAP, FileFormat.PCAPNG):
            self._run_pcap_async()
        else:
            self._run_csv_sync()

    # ── Synchronous CSV path (fast — typically <1s) ───────────────────────
    def _run_csv_sync(self):
        self._run_btn.setEnabled(False)
        self._progress.setRange(0, 0)               # indeterminate
        self._progress.show()
        self._status_lbl.setText("Running detection pipeline…")
        self._status_lbl.setStyleSheet(f"color:{ACC};background:transparent;")
        # Force a paint of the status text BEFORE blocking on inference,
        # otherwise the user sees nothing until the call returns.
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            results = run_file_inference(self._current_file)
        except Exception as e:
            results = []
            self._show_error(f"Inference error: {e}")
        finally:
            self._progress.hide()
            self._run_btn.setEnabled(True)

        if results:
            self._status_lbl.setText(
                f"✔  Detection complete — {len(results)} result(s) ready. "
                "Navigate to Results page to view."
            )
            self._status_lbl.setStyleSheet(f"color:{OK};background:transparent;")
            self.analysis_done.emit(results)

    # ── Async PCAP path (slow — minutes possible for large files) ─────────
    def _run_pcap_async(self):
        """
        Phase 1 (main thread, blocks ~3-5s):  load BotnetMonitor's models.
        Phase 2 (worker thread):              process every packet, emit
                                              progress as it goes. UI stays
                                              responsive so the user can
                                              switch tabs / cancel / etc.
        """
        from inference_worker import PcapInferenceThread

        # Switch UI to "loading models" state
        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Loading detection models…")
        self._status_lbl.setStyleSheet(f"color:{ACC};background:transparent;")
        self._progress.setRange(0, 0)               # indeterminate during load
        self._progress.show()
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()                # paint status text now

        # Phase 1: construct BotnetMonitor on this (main) thread.
        worker = PcapInferenceThread(self._current_file.path, parent=self)
        if not worker.ensure_monitor():
            self._show_error(
                f"Model loading failed: {worker._init_error or 'unknown'}"
            )
            self._reset_pcap_run_ui()
            return

        # Phase 2: wire signals, repurpose Run button as Cancel, start worker.
        worker.progress.connect(self._on_pcap_progress)
        worker.error.connect(self._on_pcap_error)
        worker.done.connect(self._on_pcap_done)
        self._pcap_worker = worker

        self._run_btn.setText("✕  Cancel")
        self._run_btn.setEnabled(True)
        worker.start()

    # ── PCAP worker signal slots ──────────────────────────────────────────
    def _on_pcap_progress(self, current: int, total: int, text: str):
        if total > 0:
            if self._progress.maximum() != total:
                self._progress.setRange(0, total)
            self._progress.setValue(current)
        else:
            self._progress.setRange(0, 0)           # indeterminate
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color:{ACC};background:transparent;")

    def _on_pcap_error(self, msg: str):
        # Cancellation is a normal exit — show as muted info, not red error.
        if "Cancelled" in msg:
            self._status_lbl.setText("Cancelled.")
            self._status_lbl.setStyleSheet(f"color:{TG};background:transparent;")
        else:
            self._show_error(f"PCAP inference error: {msg}")
        self._reset_pcap_run_ui()

    def _on_pcap_done(self, results: list):
        n = len(results)
        if n > 0:
            self._status_lbl.setText(
                f"✔  Detection complete — {n} result(s) ready. "
                "Navigate to Results page to view."
            )
            self._status_lbl.setStyleSheet(f"color:{OK};background:transparent;")
            self.analysis_done.emit(results)
        else:
            self._status_lbl.setText(
                "Detection finished — no flows completed in this PCAP."
            )
            self._status_lbl.setStyleSheet(f"color:{WARN};background:transparent;")
        self._reset_pcap_run_ui()

    def _on_pcap_cancel(self):
        if self._pcap_worker is not None:
            self._pcap_worker.cancel()
            self._status_lbl.setText("Cancelling…")
            self._status_lbl.setStyleSheet(f"color:{WARN};background:transparent;")
            self._run_btn.setEnabled(False)         # block double-click while shutting down

    def _reset_pcap_run_ui(self):
        """Restore the Run button + progress bar to their idle state."""
        self._progress.hide()
        self._progress.setRange(0, 0)               # back to indeterminate default
        self._run_btn.setText("▶  Run Detection")
        self._run_btn.setEnabled(self._current_file is not None)
        self._pcap_worker = None
