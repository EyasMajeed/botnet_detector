"""
theme.py — BotSense design tokens and global stylesheet
========================================================
Single source of truth for colors, typography, spacing, and the application-wide
Qt stylesheet. Other GUI files import constants from here instead of redefining
their own — this is what keeps the look consistent across pages.

Usage:
    from theme import (BG, SURFACE, ACCENT, TEXT_PRIMARY, FONT_FAMILY,
                       qss, icon, brand_mark)
    QApplication.instance().setStyleSheet(qss())
    btn.setIcon(icon("play"))

Pulled from the BotSense logo:
    accent       = #3FA9F5  (cyan node/edge in the hex mark)
    surface base = #0A0E1A  (logo background navy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor
from PyQt6.QtSvg import QSvgRenderer


# ══════════════════════════════════════════════════════════════════════════════
# COLOR TOKENS  (named for semantic role, never for what they look like)
# ══════════════════════════════════════════════════════════════════════════════

# Background layers — go from outer to inner; use the right one for each depth.
BG                = "#0A0E1A"   # window background (matches logo backdrop)
SURFACE           = "#11172A"   # card/panel surface
SURFACE_ELEVATED  = "#161D33"   # hover state, modals, popovers
SURFACE_INPUT     = "#0E1322"   # text input fill (slightly inset feel)

# Border & dividers — subtle, never compete with content.
BORDER            = "rgba(255, 255, 255, 0.06)"
BORDER_STRONG     = "rgba(255, 255, 255, 0.12)"
DIVIDER           = "rgba(255, 255, 255, 0.04)"

# Text hierarchy — three levels, no in-betweens.
TEXT_PRIMARY      = "#E8ECF4"   # headings, primary values
TEXT_SECONDARY    = "#9BA4B8"   # labels, captions
TEXT_MUTED        = "#5C6680"   # placeholders, disabled

# Brand & accent (from the BotSense mark).
ACCENT            = "#3FA9F5"   # primary actions, active nav, key data
ACCENT_HOVER      = "#5AB8F8"
ACCENT_PRESSED    = "#2A8DD9"
ACCENT_FAINT      = "rgba(63, 169, 245, 0.12)"   # backgrounds for active items

# Status colors — muted, harmonized.
SUCCESS           = "#22C55E"   # benign labels
SUCCESS_FAINT     = "rgba(34, 197, 94, 0.15)"
WARNING           = "#F59E0B"   # caution states
WARNING_FAINT     = "rgba(245, 158, 11, 0.15)"
DANGER            = "#EF4444"   # botnet labels, alerts
DANGER_FAINT      = "rgba(239, 68, 68, 0.15)"
INFO              = "#60A5FA"   # informational pills
INFO_FAINT        = "rgba(96, 165, 250, 0.15)"

# Aliases preserved for files that haven't been migrated yet.
# Migrate call sites away from these to the semantic names above.
CARD   = SURFACE
HOVER  = SURFACE_ELEVATED
ACC    = ACCENT
ACC2   = ACCENT_PRESSED
BDR    = BORDER_STRONG.replace("rgba(255, 255, 255, 0.12)", "#374151")  # legacy fallback
TW     = TEXT_PRIMARY
TM     = "#D1D5DB"
TG     = TEXT_SECONDARY
TD     = TEXT_MUTED
OK     = SUCCESS
ERR    = DANGER
WARN   = WARNING
YEL    = "#EAB308"


# ══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════

# Inter is free, ships with a system-ui fallback chain, reads cleanly on dark.
# Qt resolves the first available family in the comma-separated list.
FONT_FAMILY      = "Inter, -apple-system, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif"
FONT_FAMILY_MONO = "'JetBrains Mono', 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace"

# Legacy alias (mockApp.py reads FNT in many places).
FNT = "Inter"


# ══════════════════════════════════════════════════════════════════════════════
# SPACING & RADIUS  (use scale tokens, never raw numbers)
# ══════════════════════════════════════════════════════════════════════════════

SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_6 = 24
SPACE_8 = 32

RADIUS_SM = 6        # buttons, inputs
RADIUS_MD = 10       # cards, panels
RADIUS_PILL = 999    # pills, badges


# ══════════════════════════════════════════════════════════════════════════════
# ICON ASSET HELPER
# ══════════════════════════════════════════════════════════════════════════════

_ICON_DIR  = Path(__file__).resolve().parent / "assets" / "icons"
_BRAND_DIR = Path(__file__).resolve().parent / "assets" / "branding"


def _render_svg_with_color(svg_bytes: bytes, color: str, size: int) -> QPixmap:
    """
    Render an SVG to a QPixmap, replacing `currentColor` with the requested hex.
    Lucide icons use stroke="currentColor", so this re-tints them cleanly.
    """
    text = svg_bytes.decode("utf-8").replace("currentColor", color)
    renderer = QSvgRenderer(text.encode("utf-8"))

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(img)


def icon(name: str, color: Optional[str] = None, size: int = 20) -> QIcon:
    """
    Load a Lucide-style icon from assets/icons/<name>.svg, tinted to `color`.

    Args:
        name:  filename without `.svg` extension (e.g. "play", "shield-alert")
        color: any CSS color string accepted by Qt; defaults to TEXT_PRIMARY
        size:  pixel size of the rasterized icon (icon will scale on HiDPI)

    Returns:
        QIcon. If the SVG file is missing, returns an empty QIcon (no crash).
    """
    if color is None:
        color = TEXT_PRIMARY

    path = _ICON_DIR / f"{name}.svg"
    if not path.is_file():
        return QIcon()

    svg = path.read_bytes()
    pix = _render_svg_with_color(svg, color, size)
    return QIcon(pix)


def brand_mark(size: int = 28) -> QIcon:
    """The BotSense hexagonal mesh mark, sized for the sidebar header."""
    path = _BRAND_DIR / "botsense_mark.svg"
    if not path.is_file():
        return QIcon()
    pix = _render_svg_with_color(path.read_bytes(), ACCENT, size)
    return QIcon(pix)


def brand_mark_pixmap(size: int = 28) -> QPixmap:
    """Same as brand_mark() but returns a QPixmap (for QLabel.setPixmap)."""
    path = _BRAND_DIR / "botsense_mark.svg"
    if not path.is_file():
        return QPixmap()
    return _render_svg_with_color(path.read_bytes(), ACCENT, size)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLESHEET  —  applied once at QApplication level
# ══════════════════════════════════════════════════════════════════════════════

def qss() -> str:
    """
    Application-wide Qt stylesheet. Apply via:
        QApplication.instance().setStyleSheet(qss())
    Per-widget styles in monitor_page.py / upload_page.py / mockApp.py override
    when needed but inherit these defaults.
    """
    return f"""
    /* ── Window & default text ────────────────────────────────────────── */
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        font-size: 13px;
    }}

    QToolTip {{
        background-color: {SURFACE_ELEVATED};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_STRONG};
        border-radius: {RADIUS_SM}px;
        padding: 6px 10px;
        font-size: 12px;
    }}

    /* ── Push buttons (default = neutral; primary set per-widget) ──────── */
    QPushButton {{
        background-color: transparent;
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_STRONG};
        border-radius: {RADIUS_SM}px;
        padding: 8px 14px;
        font-weight: 500;
    }}
    QPushButton:hover  {{ background-color: {SURFACE_ELEVATED}; }}
    QPushButton:pressed {{ background-color: {SURFACE}; }}
    QPushButton:disabled {{
        color: {TEXT_MUTED};
        border-color: {BORDER};
        background-color: transparent;
    }}

    /* Primary button (set objectName='primary' on the button) */
    QPushButton#primary {{
        background-color: {ACCENT};
        color: white;
        border: 1px solid {ACCENT};
    }}
    QPushButton#primary:hover   {{ background-color: {ACCENT_HOVER}; }}
    QPushButton#primary:pressed {{ background-color: {ACCENT_PRESSED}; }}
    QPushButton#primary:disabled {{
        background-color: {SURFACE_ELEVATED};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
    }}

    /* Danger button */
    QPushButton#danger {{
        background-color: {DANGER};
        color: white;
        border: 1px solid {DANGER};
    }}
    QPushButton#danger:hover {{ background-color: #F87171; }}

    /* ── Inputs & line edits ──────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {SURFACE_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM}px;
        padding: 7px 10px;
        selection-background-color: {ACCENT_FAINT};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT};
    }}
    QLineEdit::placeholder {{ color: {TEXT_MUTED}; }}

    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE_ELEVATED};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_STRONG};
        border-radius: {RADIUS_SM}px;
        selection-background-color: {ACCENT_FAINT};
        selection-color: {TEXT_PRIMARY};
        padding: 4px;
    }}

    /* ── Tables ──────────────────────────────────────────────────────── */
    QTableWidget, QTableView {{
        background-color: transparent;
        color: {TEXT_PRIMARY};
        gridline-color: {DIVIDER};
        border: none;
        selection-background-color: {ACCENT_FAINT};
        selection-color: {TEXT_PRIMARY};
        alternate-background-color: rgba(255, 255, 255, 0.015);
    }}
    QTableWidget::item, QTableView::item {{
        padding: 8px 12px;
        border: none;
    }}
    QHeaderView::section {{
        background-color: transparent;
        color: {TEXT_SECONDARY};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 10px 12px;
        font-weight: 500;
        font-size: 12px;
    }}
    QTableCornerButton::section {{
        background-color: transparent;
        border: none;
    }}

    /* ── Scrollbars ──────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TEXT_MUTED}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {TEXT_MUTED}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    /* ── Frames acting as cards (set objectName='card') ──────────────── */
    QFrame#card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_MD}px;
    }}

    /* ── Dialogs ─────────────────────────────────────────────────────── */
    QDialog, QFileDialog, QMessageBox {{
        background-color: {BG};
        color: {TEXT_PRIMARY};
    }}

    /* ── Progress bars ───────────────────────────────────────────────── */
    QProgressBar {{
        background-color: {SURFACE_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM}px;
        text-align: center;
        height: 8px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: {RADIUS_SM}px;
    }}

    /* ── Checkbox & radio (kept simple, recolored) ───────────────────── */
    QCheckBox, QRadioButton {{
        color: {TEXT_PRIMARY};
        spacing: 8px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {BORDER_STRONG};
        background-color: {SURFACE_INPUT};
        border-radius: 4px;
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QRadioButton::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}

    /* ── Headings (use objectName='h1' / 'h2' / 'h3') ────────────────── */
    QLabel#h1 {{
        color: {TEXT_PRIMARY};
        font-size: 24px;
        font-weight: 600;
    }}
    QLabel#h2 {{
        color: {TEXT_PRIMARY};
        font-size: 18px;
        font-weight: 600;
    }}
    QLabel#h3 {{
        color: {TEXT_PRIMARY};
        font-size: 15px;
        font-weight: 600;
    }}
    QLabel#caption {{
        color: {TEXT_SECONDARY};
        font-size: 12px;
    }}
    QLabel#muted {{
        color: {TEXT_MUTED};
        font-size: 12px;
    }}
    """
