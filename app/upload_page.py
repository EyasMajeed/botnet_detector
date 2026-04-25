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

# ── Design tokens (must match mockApp.py) ─────────────────────────────────────
BG   = "#1E1E2F";  CARD = "#16161F";  BDR = "#374151"
TW   = "#FFFFFF";  TG   = "#9CA3AF";  TD  = "#6B7280"
ACC  = "#3A7AFE";  OK   = "#10B981";  ERR = "#EF4444"
WARN = "#F97316";  YEL  = "#EAB308";  FNT = "Segoe UI"

ACCEPT_FILTER = (
    "Network Traffic Files "
    "(*.pcap *.pcapng *.csv *.txt *.log *.binetflow *.nfcapd *.nfdump);;"
    "PCAP Files (*.pcap *.pcapng);;"
    "CSV / Flow Files (*.csv *.txt *.log *.binetflow);;"
    "NetFlow Files (*.nfcapd *.nfdump);;"
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
            "PCAP · PCAPng · CSV · NetFlow  —  Max 500 MB", 11, color=TD
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

        # ── Page title ────────────────────────────────────────────────────
        root.addWidget(_lbl("Upload & Analyze", 20, bold=True))
        root.addWidget(_lbl(
            "Upload a traffic capture file for AI-based botnet detection.",
            12, color=TD
        ))

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
        self._run_btn.clicked.connect(self._run_detection)
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
            "These settings will be passed to the inference pipeline "
            "when your teammates' models are ready.",
            11, color=TD
        ))
        v.addWidget(_divider())

        # Model info row
        models_row = QHBoxLayout()
        for label, value, color in [
            ("Stage-1 Classifier", "Random Forest", ACC),
            ("Stage-2 IoT",        "CNN-LSTM",      OK),
            ("Stage-2 Non-IoT",    "CNN-LSTM",      OK),
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

    def _run_detection(self):
        if not self._current_file:
            return

        self._run_btn.setEnabled(False)
        self._progress.show()
        self._status_lbl.setText("Running detection pipeline…")
        self._status_lbl.setStyleSheet(f"color:{ACC};background:transparent;")

        # Hand off to inference bridge (stub today, real pipeline tomorrow).
        # The bridge receives the FileInfo object — when the real pipeline is
        # ready, teammates implement inference_bridge.run_file_inference(info).
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
