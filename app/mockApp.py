"""
Botnet Detection System — SOC Dashboard  v2
Group 07 · CPCS499
Design: Professional dark SOC dashboard, sidebar nav, 6 pages.
"""
import sys, os, random, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QStackedWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QLineEdit, QScrollArea,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QMessageBox,
    QProgressBar, QSizePolicy,
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSignal
from PyQt6.QtGui   import (
    QFont, QColor, QPalette, QPainter, QBrush, QPen,
    QLinearGradient, QPolygonF, QCursor,
)
from PyQt6.QtCore import QPointF, QRectF

# ── Design tokens ─────────────────────────────────────────────────────────────
# Single source of truth lives in app/theme.py. The legacy short names (BG, CARD,
# HOVER, ACC, ACC2, BDR, TW, TG, TD, TM, OK, ERR, WARN, INFO, YEL, FNT) are kept
# as aliases over there so the rest of this file does not need touching.
from theme import (
    BG, CARD, HOVER, ACC, ACC2, BDR, TW, TG, TD, TM,
    OK, ERR, WARN, INFO, YEL, FNT,
    ACCENT, ACCENT_HOVER, ACCENT_PRESSED,
    SURFACE, SURFACE_ELEVATED, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    qss, icon, brand_mark_pixmap,
)

# Shared services
from pathlib import Path
from detection_store import DetectionStore, DetectionFlow
from app_settings    import AppSettings

DEMO = [
    {"id":"FL-001","src":"192.168.1.5", "dst":"91.108.4.15",   "proto":"TCP",   "lbl":"Botnet","conf":0.92,"dev":"IoT",    "bytes":"14,200","dur":"5.3s","port":4444},
    {"id":"FL-002","src":"10.0.0.10",   "dst":"8.8.8.8",       "proto":"UDP",   "lbl":"Benign","conf":0.97,"dev":"Non-IoT","bytes":"320",   "dur":"0.1s","port":53},
    {"id":"FL-003","src":"192.168.1.5", "dst":"91.108.4.15",   "proto":"TCP",   "lbl":"Botnet","conf":0.78,"dev":"IoT",    "bytes":"9,180", "dur":"3.1s","port":4444},
    {"id":"FL-004","src":"10.0.0.3",    "dst":"172.16.0.1",    "proto":"TCP",   "lbl":"Benign","conf":0.91,"dev":"Non-IoT","bytes":"5,200", "dur":"1.2s","port":80},
    {"id":"FL-005","src":"192.168.1.7", "dst":"185.220.101.5", "proto":"TCP",   "lbl":"Botnet","conf":0.85,"dev":"IoT",    "bytes":"18,920","dur":"6.4s","port":9999},
    {"id":"FL-006","src":"10.0.0.8",    "dst":"142.250.80.46", "proto":"HTTPS", "lbl":"Benign","conf":0.98,"dev":"Non-IoT","bytes":"7,640", "dur":"2.1s","port":443},
    {"id":"FL-007","src":"10.0.0.2",    "dst":"8.8.4.4",       "proto":"UDP",   "lbl":"Benign","conf":0.89,"dev":"Non-IoT","bytes":"150",   "dur":"0.1s","port":53},
    {"id":"FL-008","src":"192.168.1.5", "dst":"91.108.4.15",   "proto":"TCP",   "lbl":"Botnet","conf":0.91,"dev":"IoT",    "bytes":"14,200","dur":"5.3s","port":4444},
    {"id":"FL-009","src":"10.0.0.5",    "dst":"216.58.208.14", "proto":"HTTPS", "lbl":"Benign","conf":0.96,"dev":"Non-IoT","bytes":"4,480", "dur":"1.5s","port":443},
    {"id":"FL-010","src":"192.168.1.9", "dst":"91.108.4.15",   "proto":"TCP",   "lbl":"Botnet","conf":0.80,"dev":"IoT",    "bytes":"22,300","dur":"7.0s","port":4444},
    {"id":"FL-011","src":"10.0.0.12",   "dst":"104.18.10.1",   "proto":"HTTPS", "lbl":"Benign","conf":0.94,"dev":"Non-IoT","bytes":"3,200", "dur":"0.9s","port":443},
    {"id":"FL-012","src":"192.168.1.3", "dst":"45.33.32.156",  "proto":"TCP",   "lbl":"Botnet","conf":0.87,"dev":"IoT",    "bytes":"11,500","dur":"4.2s","port":23},
]

XAI_BOT = {"flow_pkts_per_sec":0.91,"flag_SYN_count":0.74,"bytes_per_sec_window":0.61,"periodicity_score":0.55,"flow_duration":0.45,"total_fwd_packets":0.38}
XAI_BEN = {"flow_duration":0.55,"total_fwd_packets":0.40,"bytes_per_sec_window":0.20,"flow_pkts_per_sec":0.15,"flag_SYN_count":0.08,"periodicity_score":0.05}

# ── Helpers ───────────────────────────────────────────────────────────────────
def L(text, sz=13, bold=False, color=TW, mono=False, wrap=False):
    l = QLabel(text)
    f = QFont("Courier New" if mono else FNT, sz)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color:{color};background:transparent;")
    if wrap: l.setWordWrap(True)
    return l

def SEP():
    s = QFrame(); s.setFrameShape(QFrame.Shape.HLine)
    s.setFixedHeight(1); s.setStyleSheet(f"background:{BDR};border:none;")
    return s

def BTN(text, kind="outline", small=False):
    b = QPushButton(text)
    sz = 11 if small else 13
    b.setFont(QFont(FNT, sz, QFont.Weight.Bold if kind=="primary" else QFont.Weight.Normal))
    pad = "5px 12px" if small else "8px 20px"
    cfg = {
        "primary": (ACC,  ACC2,   "white"),
        "outline": ("transparent", HOVER, TG),
        "success": (OK,   "#0D9668","white"),
        "danger":  (ERR,  "#CC2222","white"),
    }.get(kind, ("transparent", HOVER, TG))
    b.setStyleSheet(f"""
        QPushButton{{background:{cfg[0]};color:{cfg[2]};
            border:{"none" if kind!="outline" else f"1px solid {BDR}"};
            border-radius:8px;padding:{pad};}}
        QPushButton:hover{{background:{cfg[1]};color:white;}}
    """)
    b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    return b

def BADGE(text, kind="info"):
    c = {
        "danger": (ERR,"#3D1515"),
        "success":(OK, "#0F2E24"),
        "warning":(WARN,"#2E1D0F"),
        "info":   (ACC,"#0F1E3D"),
    }.get(kind,(ACC,"#0F1E3D"))
    l = QLabel(text)
    l.setFont(QFont(FNT,11,QFont.Weight.Bold))
    l.setStyleSheet(f"color:{c[0]};background:{c[1]};border:1px solid {c[0]}44;border-radius:5px;padding:2px 10px;")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return l

def TABLE_CSS():
    return f"""
        QTableWidget{{background:{CARD};color:{TW};border:none;outline:none;font-size:12px;}}
        QTableWidget::item{{padding:4px 10px;border-bottom:1px solid {BDR};}}
        QTableWidget::item:selected{{background:{ACC}44;}}
        QHeaderView::section{{background:{BG};color:{TG};font-size:11px;
            padding:6px 10px;border:none;border-bottom:1px solid {BDR};}}
        QScrollBar:vertical{{background:{BG};width:6px;}}
        QScrollBar::handle:vertical{{background:{BDR};border-radius:3px;}}
    """

# ── Stat Card ─────────────────────────────────────────────────────────────────
class StatCard(QFrame):
    """
    Stat card: SVG icon badge on the left, title + value + sub-line on the right.
    `icon_name` must match a file in app/assets/icons/ (without `.svg`).
    Renamed parameter from `icon` to `icon_name` so it doesn't shadow the
    imported theme.icon() helper.
    """
    def __init__(self, icon_name, title, value, sub="", color=ACC):
        super().__init__()
        self.setStyleSheet(f"QFrame{{background:{CARD};border:none;border-radius:12px;}}")
        h = QHBoxLayout(self); h.setContentsMargins(18,14,18,14); h.setSpacing(14)
        # Icon badge: tinted square with the SVG icon centered in it.
        ic = QLabel(); ic.setFixedSize(46, 46)
        ic.setPixmap(icon(icon_name, color=color, size=22).pixmap(22, 22))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"background:{color}22;border:1px solid {color}44;border-radius:10px;")
        h.addWidget(ic)
        col = QVBoxLayout(); col.setSpacing(2)
        self.vl = L(value, 20, bold=True, color=color)
        self.sl = L(sub, 10, color=TD)             # store ref so we can update
        col.addWidget(L(title, 11, color=TG))
        col.addWidget(self.vl)
        col.addWidget(self.sl)
        h.addLayout(col); h.addStretch()

    def set_val(self, v): self.vl.setText(v)
    def set_sub(self, v): self.sl.setText(v)

# ── Pie Chart ─────────────────────────────────────────────────────────────────
class PieChart(QWidget):
    def __init__(self, benign=124, botnet=18):
        super().__init__(); self.b=benign; self.n=botnet
        self.setMinimumSize(180,180); self.setStyleSheet("background:transparent;")
    def set_data(self,b,n): self.b=b; self.n=n; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W,H=self.width(),self.height(); t=self.b+self.n
        if t==0: return
        sz=min(W,H)-16; x=(W-sz)//2; y=(H-sz)//2
        rect=QRectF(x,y,sz,sz)
        ba=int(360*16*self.b/t); na=int(360*16*self.n/t)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(OK)); p.drawPie(rect,90*16,ba)
        p.setBrush(QColor(ERR)); p.drawPie(rect,90*16+ba,na)
        hs=sz*0.56; hx=(W-hs)/2; hy=(H-hs)/2
        p.setBrush(QColor(CARD)); p.drawEllipse(QRectF(hx,hy,hs,hs))
        # Center label: large percentage on top, small "botnet" caption below.
        # (Previously crammed into one drawText with embedded \n.)
        pct=int(100*self.n/t)
        cx, cy = W/2, H/2
        p.setPen(QColor(TW)); p.setFont(QFont(FNT, 18, QFont.Weight.Bold))
        p.drawText(QRectF(0, cy-22, W, 26),
                   Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom, f"{pct}%")
        p.setPen(QColor(TG)); p.setFont(QFont(FNT, 10))
        p.drawText(QRectF(0, cy+2, W, 16),
                   Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop, "botnet")
        p.end()

# ── Sparkline ─────────────────────────────────────────────────────────────────
class Spark(QWidget):
    def __init__(self, vals=None, color=ACC):
        super().__init__(); self.vals=vals or []; self.color=color
        self.setMinimumHeight(110); self.setStyleSheet("background:transparent;")
    def paintEvent(self,e):
        if len(self.vals)<2: return
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W,H=self.width(),self.height(); pad=10
        mn,mx=min(self.vals),max(self.vals); rng=mx-mn or 1
        def pt(i,v): return (pad+(W-2*pad)*i/(len(self.vals)-1),
                              H-pad-(H-2*pad)*(v-mn)/rng)
        # Dashed horizontal gridlines for context.
        p.setPen(QPen(QColor(BDR),1,Qt.PenStyle.DashLine))
        for f in [.25,.5,.75]:
            yy=int(pad+(H-2*pad)*(1-f)); p.drawLine(pad,yy,W-pad,yy)
        # Line only — gradient fill removed, the muddy yellow under-fill was
        # competing with the red line for attention.
        p.setPen(QPen(QColor(self.color),2))
        for i in range(len(self.vals)-1):
            x1,y1=pt(i,self.vals[i]); x2,y2=pt(i+1,self.vals[i+1])
            p.drawLine(int(x1),int(y1),int(x2),int(y2))
        p.end()

# ── Horizontal Bar Chart ──────────────────────────────────────────────────────
class HBar(QWidget):
    def __init__(self, data=None):
        super().__init__(); self.data=data or {}
        self.setMinimumHeight(200); self.setStyleSheet("background:transparent;")
    def paintEvent(self,e):
        if not self.data: return
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W,H=self.width(),self.height()
        lw=160; bw=W-lw-55; mx=max(self.data.values()) or 1
        rh=H//len(self.data)
        for i,(name,val) in enumerate(self.data.items()):
            cy=i*rh+rh//2
            p.setPen(QColor(TG)); p.setFont(QFont(FNT,10))
            p.drawText(0,cy-8,lw-8,20,
                Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter, name)
            p.setPen(Qt.PenStyle.NoPen)
            # Bars halved: was 8px tall (cy-4..cy+4), now 4px (cy-2..cy+2).
            p.setBrush(QColor(BDR)); p.drawRoundedRect(lw,cy-2,bw,4,2,2)
            fw=int(bw*val/mx)
            if fw>0:
                c=ERR if val>.7 else (WARN if val>.4 else ACC)
                p.setBrush(QColor(c)); p.drawRoundedRect(lw,cy-2,fw,4,2,2)
            p.setPen(QColor(TM)); p.setFont(QFont("Courier New",10))
            p.drawText(lw+bw+5,cy-8,48,20,
                Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,f"{val:.2f}")
        p.end()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
class Sidebar(QWidget):
    switched = pyqtSignal(int)
    # (icon_name from assets/icons/, label, page_index)
    NAV = [("layout-grid",  "Dashboard",        0),
           ("upload-cloud", "Upload && Analyze", 1),
           ("activity",     "Monitoring",       2),
           ("list-checks",  "Results",          3),
           ("file-text",    "Reports",          4),
           ("settings",     "Settings",         5)]
    def __init__(self):
        super().__init__(); self.setFixedWidth(220)
        self.setStyleSheet(f"background:{CARD};")
        self._btns=[]; self._icon_names=[]
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        # Logo header
        logo=QWidget(); logo.setFixedHeight(72)
        logo.setStyleSheet(f"background:{CARD};")
        lh=QHBoxLayout(logo); lh.setContentsMargins(16,0,16,0); lh.setSpacing(12)
        ic=QLabel(); ic.setPixmap(brand_mark_pixmap(28))
        ic.setStyleSheet("background:transparent;")
        lh.addWidget(ic)
        tc=QVBoxLayout(); tc.setSpacing(0)
        tc.addWidget(L("BotSense",14,bold=True))
        tc.addWidget(L("AI Botnet Detection",9,color=TD))
        lh.addLayout(tc); lh.addStretch(); root.addWidget(logo)
        # Nav
        nav=QWidget(); nav.setStyleSheet("background:transparent;")
        nl=QVBoxLayout(nav); nl.setContentsMargins(10,14,10,14); nl.setSpacing(4)
        for icon_name, name, idx in self.NAV:
            b=QPushButton(f"   {name}"); b.setFont(QFont(FNT,12))
            b.setIcon(icon(icon_name, color=TG, size=18))
            from PyQt6.QtCore import QSize
            b.setIconSize(QSize(18, 18))
            b.setFixedHeight(40); b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.clicked.connect(lambda _,i=idx: self._sel(i))
            self._btns.append(b); self._icon_names.append(icon_name); nl.addWidget(b)
        nl.addStretch(); root.addWidget(nav)
        # Footer
        ft=QWidget(); ft.setStyleSheet(f"background:transparent;")
        fl=QVBoxLayout(ft); fl.setContentsMargins(16,10,16,14); fl.setSpacing(2)
        fl.addWidget(L("v1.0.0  ·  Group 07",10,color=TD))
        fl.addWidget(L("CPCS499  ·  2026",10,color=TD))
        root.addWidget(ft)
        self._sel(0)

    def _sel(self,idx):
        for i,b in enumerate(self._btns):
            if i==idx:
                b.setIcon(icon(self._icon_names[i], color="white", size=18))
                b.setStyleSheet(f"QPushButton{{background:{ACC};color:white;border:none;"
                    f"border-radius:8px;text-align:left;padding-left:14px;font-weight:600;}}")
            else:
                b.setIcon(icon(self._icon_names[i], color=TG, size=18))
                b.setStyleSheet(f"QPushButton{{background:transparent;color:{TG};border:none;"
                    f"border-radius:8px;text-align:left;padding-left:14px;}}"
                    f"QPushButton:hover{{background:{HOVER};color:white;}}")
        self.switched.emit(idx)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
class Header(QWidget):
    def __init__(self):
        super().__init__(); self.setFixedHeight(64)
        self.setStyleSheet(f"background:{CARD};border-bottom:1px solid {BDR};")
        h=QHBoxLayout(self); h.setContentsMargins(24,0,24,0)
        self.tl=L("Dashboard",18,bold=True); h.addWidget(self.tl); h.addStretch()
        # System Active indicator. The wallclock timestamp that used to live
        # here was decorative — removed in Pass 3 to reduce visual noise.
        dot=QLabel(); dot.setFixedSize(10,10)
        dot.setStyleSheet(f"background:{OK};border-radius:5px;")
        self.sl=L("System Active",13,color=TG)
        h.addWidget(dot); h.addSpacing(6); h.addWidget(self.sl)

    def set_title(self,t): self.tl.setText(t)

# ══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════
class StatusBar(QWidget):
    def __init__(self):
        super().__init__(); self.setFixedHeight(32)
        self.setStyleSheet(f"background:{CARD};border-top:1px solid {BDR};")
        h=QHBoxLayout(self); h.setContentsMargins(24,0,24,0)
        self.ml=L("Ready — No file loaded.",11,color=TD); h.addWidget(self.ml); h.addStretch()
        h.addWidget(L("CNN-LSTM  ·  Stage-1 + Stage-2 Active",11,color=TD))
    def set(self,t,c=TD): self.ml.setText(t); self.ml.setStyleSheet(f"color:{c};background:transparent;")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
class DashPage(QWidget):
    """Real-data dashboard. Subscribes to DetectionStore.flows_changed."""

    def __init__(self, store: DetectionStore, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.store    = store
        self.settings = settings
        self.setStyleSheet(f"background:{BG};")
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        root  = QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)

        # ── Stat cards ───────────────────────────────────────────────────────
        g = QGridLayout(); g.setSpacing(14)
        self.sc_t = StatCard("radio",        "Total Flows",  "0", "—", ACC)
        self.sc_b = StatCard("shield-alert", "Botnet Flows", "0", "—", ERR)
        self.sc_g = StatCard("shield-check", "Benign Flows", "0", "—", OK)
        self.sc_d = StatCard("cpu",          "Devices",      "0", "—", INFO)
        for i, c in enumerate([self.sc_t, self.sc_b, self.sc_g, self.sc_d]):
            g.addWidget(c, 0, i)
        root.addLayout(g)

        # ── Charts row (Pie + Sparkline) ─────────────────────────────────────
        cr = QHBoxLayout(); cr.setSpacing(14)
        pc = QFrame(); pc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        pv = QVBoxLayout(pc); pv.setContentsMargins(18,14,18,14)
        pv.addWidget(L("Traffic Distribution", 14, bold=True))
        pv.addWidget(L("Benign vs Botnet",     11, color=TD)); pv.addSpacing(6)
        self.pie = PieChart(0, 0); pv.addWidget(self.pie)
        lr = QHBoxLayout()
        self.pie_lbl_b = L("Benign  0.0%", 11, color=TG)
        self.pie_lbl_n = L("Botnet  0.0%", 11, color=TG)
        for col, lbl in [(OK, self.pie_lbl_b), (ERR, self.pie_lbl_n)]:
            d = QLabel("●"); d.setStyleSheet(f"color:{col};background:transparent;")
            lr.addWidget(d); lr.addWidget(lbl); lr.addSpacing(8)
        lr.addStretch(); pv.addLayout(lr); cr.addWidget(pc, 1)

        skc = QFrame(); skc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        sv  = QVBoxLayout(skc); sv.setContentsMargins(18,14,18,14)
        sv.addWidget(L("Botnet Activity – last 20 min", 14, bold=True))
        sv.addWidget(L("Botnet flows per minute",       11, color=TD)); sv.addSpacing(6)
        self.spark = Spark([0]*20, ERR); sv.addWidget(self.spark)
        cr.addWidget(skc, 2); root.addLayout(cr)

        # ── Recent detections table ──────────────────────────────────────────
        rc = QFrame(); rc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        rv = QVBoxLayout(rc); rv.setContentsMargins(18,14,18,14)
        rh = QHBoxLayout(); rh.addWidget(L("Recent Detections", 14, bold=True)); rh.addStretch()
        self.view_all_btn = BTN("View All →", "outline", small=True)
        rh.addWidget(self.view_all_btn); rv.addLayout(rh); rv.addSpacing(10)
        self.recent_tbl = QTableWidget(0, 6)
        self.recent_tbl.setHorizontalHeaderLabels(
            ["Flow ID","Src IP","Dst IP","Protocol","Label","Confidence"])
        self.recent_tbl.verticalHeader().setVisible(False); self.recent_tbl.setShowGrid(False)
        self.recent_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recent_tbl.setFixedHeight(230); self.recent_tbl.setStyleSheet(TABLE_CSS())
        self.recent_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.recent_tbl.verticalHeader().setDefaultSectionSize(34)
        rv.addWidget(self.recent_tbl); root.addWidget(rc)

        sc.setWidget(inner); ol = QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

        # Subscribe + initial paint
        self.store.flows_changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        s   = self.store.stats()
        n   = s["total_flows"]
        nb  = s["n_botnet"]; ng = s["n_benign"]
        ni  = s["n_iot"];    nn = s["n_noniot"]
        # Stat cards
        self.sc_t.set_val(str(n))
        self.sc_b.set_val(str(nb))
        self.sc_g.set_val(str(ng))
        self.sc_d.set_val(str(s["devices"]))
        self.sc_t.set_sub(f"{s['n_alerts']} alert(s)" if n else "No flows yet")
        self.sc_b.set_sub(f"{nb/n*100:.1f}% of traffic" if n else "—")
        self.sc_g.set_sub(f"{ng/n*100:.1f}% of traffic" if n else "—")
        self.sc_d.set_sub(f"{ni} IoT · {nn} Non-IoT" if (ni or nn) else "—")
        # Pie
        self.pie.set_data(ng, nb)
        if (ng + nb) > 0:
            tot = ng + nb
            self.pie_lbl_b.setText(f"Benign  {ng/tot*100:.1f}%")
            self.pie_lbl_n.setText(f"Botnet  {nb/tot*100:.1f}%")
        else:
            self.pie_lbl_b.setText("Benign  0.0%")
            self.pie_lbl_n.setText("Botnet  0.0%")
        # Sparkline
        self.spark.vals = self.store.botnet_per_minute(20)
        self.spark.update()
        # Recent detections table
        recent = self.store.recent_flows(6)
        self.recent_tbl.setRowCount(len(recent))
        for i, f in enumerate(recent):
            ib = f.label == "botnet"
            row_id = f"{f.report_id or '—'}/{i+1:02d}"
            cells  = [row_id, f.src_ip or "—", f.dst_ip or "—",
                      f.protocol or "—", f.label.capitalize(),
                      f"{f.confidence:.0%}"]
            for j, v in enumerate(cells):
                it = QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j == 4:
                    it.setForeground(QColor(ERR if ib else OK))
                    it.setFont(QFont(FNT, 11, QFont.Weight.Bold))
                if j in (0, 3, 5):
                    it.setFont(QFont("Courier New", 11))
                self.recent_tbl.setItem(i, j, it)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
from upload_page import UploadPage


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
from monitor_page import MonitorPage

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
class ResultsPage(QWidget):
    """Real-data results table. XAI panel keeps mock weights as requested."""

    def __init__(self, store: DetectionStore, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.store    = store
        self.settings = settings
        self._displayed: list[DetectionFlow] = []      # newest-first
        self._report_filter: str | None       = None
        self.setStyleSheet(f"background:{BG};")
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── Left: search + table ─────────────────────────────────────────────
        lw = QWidget(); lw.setStyleSheet(f"background:{BG};")
        lv = QVBoxLayout(lw); lv.setContentsMargins(24,24,12,24); lv.setSpacing(12)
        hh = QHBoxLayout(); hh.addWidget(L("Detection Results", 18, bold=True)); hh.addStretch()
        self.badge_bot = BADGE("0 Botnet", "danger")
        self.badge_ben = BADGE("0 Benign", "success")
        hh.addWidget(self.badge_bot); hh.addSpacing(6); hh.addWidget(self.badge_ben)
        lv.addLayout(hh)

        # Filter chip (visible only when filtering by a single Report)
        self.fchip_wrap = QFrame(); self.fchip_wrap.setStyleSheet(f"background:transparent;")
        fcl = QHBoxLayout(self.fchip_wrap); fcl.setContentsMargins(0,0,0,0); fcl.setSpacing(8)
        self.fchip_lbl = L("", 11, color=ACC)
        self.fchip_lbl.setStyleSheet(
            f"color:{ACC};background:{ACC}22;border:1px solid {ACC}44;"
            f"border-radius:10px;padding:4px 10px;font-size:11px;font-weight:600;")
        clr_btn = QPushButton("✕  Clear filter")
        clr_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        clr_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TG};border:1px solid {BDR};"
            f"border-radius:8px;padding:3px 10px;font-size:11px;}}"
            f"QPushButton:hover{{color:{TW};border-color:{ACC};}}")
        clr_btn.clicked.connect(self.clear_report_filter)
        fcl.addWidget(self.fchip_lbl); fcl.addWidget(clr_btn); fcl.addStretch()
        self.fchip_wrap.hide()
        lv.addWidget(self.fchip_wrap)

        fr = QHBoxLayout()
        self.srch = QLineEdit(); self.srch.setPlaceholderText("Search IP, protocol, device…")
        self.srch.setFixedHeight(34)
        self.srch.setStyleSheet(
            f"QLineEdit{{background:{CARD};color:{TW};border:1px solid {BDR};"
            f"border-radius:8px;padding:0 12px;font-size:12px;}}"
            f"QLineEdit:focus{{border-color:{ACC};}}")
        self.fcb = QComboBox()
        self.fcb.addItems(["All","Botnet only","Benign only","IoT only","Non-IoT only"])
        self.fcb.setFixedHeight(34)
        self.fcb.setStyleSheet(
            f"QComboBox{{background:{CARD};color:{TW};border:1px solid {BDR};"
            f"border-radius:8px;padding:0 10px;font-size:12px;}}"
            f"QComboBox::drop-down{{border:none;}}"
            f"QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}")
        fr.addWidget(self.srch); fr.addWidget(self.fcb); lv.addLayout(fr)

        self.rt = QTableWidget(0, 7)
        self.rt.setHorizontalHeaderLabels(
            ["#","Src IP","Dst IP","Protocol","Label","Confidence","Device"])
        self.rt.verticalHeader().setVisible(False); self.rt.setShowGrid(False)
        self.rt.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rt.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rt.setStyleSheet(TABLE_CSS())
        h2 = self.rt.horizontalHeader()
        for col, mode, w in [
            (0, QHeaderView.ResizeMode.Fixed, 50),
            (1, QHeaderView.ResizeMode.Stretch, 0),
            (2, QHeaderView.ResizeMode.Stretch, 0),
            (3, QHeaderView.ResizeMode.Fixed, 80),
            (4, QHeaderView.ResizeMode.Fixed, 90),
            (5, QHeaderView.ResizeMode.Fixed, 100),
            (6, QHeaderView.ResizeMode.Fixed, 90),
        ]:
            h2.setSectionResizeMode(col, mode)
            if w: self.rt.setColumnWidth(col, w)
        self.rt.verticalHeader().setDefaultSectionSize(36)
        self.rt.itemSelectionChanged.connect(self._sel)
        lv.addWidget(self.rt); root.addWidget(lw, 3)

        # ── Right: details panel ─────────────────────────────────────────────
        rw = QWidget(); rw.setFixedWidth(300)
        rw.setStyleSheet(f"background:{CARD};border-left:1px solid {BDR};")
        rv = QVBoxLayout(rw); rv.setContentsMargins(18,18,18,18); rv.setSpacing(12)
        rv.addWidget(L("Flow Details", 14, bold=True)); rv.addWidget(SEP())
        self.df: dict[str, QLabel] = {}
        for lbl_, key in [
            ("Flow ID","id"), ("Src IP","src"), ("Dst IP","dst"),
            ("Protocol","proto"), ("Duration","dur"), ("Bytes","bytes"),
            ("Device","dev"), ("Prediction","lbl"), ("Confidence","conf"),
        ]:
            r  = QHBoxLayout(); r.addWidget(L(lbl_, 11, color=TD)); r.addStretch()
            vl = L("—", 11, mono=(key in ("src","dst","id","conf")))
            self.df[key] = vl; r.addWidget(vl); rv.addLayout(r)
        rv.addWidget(SEP())
        # XAI panel — wrapped in self.xai_card so it can be hidden via Settings.
        self.xai_card = QFrame(); self.xai_card.setStyleSheet("background:transparent;")
        xl = QVBoxLayout(self.xai_card); xl.setContentsMargins(0,0,0,0); xl.setSpacing(8)
        xl.addWidget(L("Top Features (XAI)", 13, bold=True))
        self.xai = HBar(); xl.addWidget(self.xai)
        rv.addWidget(self.xai_card)
        self.xai_card.setVisible(self.settings.xai_enabled)
        rv.addWidget(SEP()); rv.addWidget(L("Explanation", 13, bold=True))
        self.expl = L("Select a flow to view.", 11, color=TG, wrap=True)
        self.expl.setStyleSheet(
            f"color:{TG};background:{BG};border-left:3px solid {BDR};"
            f"border-radius:4px;padding:8px 10px;")
        rv.addWidget(self.expl)
        self.rec_chip = QLabel()
        self.rec_chip.setStyleSheet(
            f"color:{ERR};background:{ERR}22;border:1px solid {ERR}44;"
            f"border-radius:10px;padding:4px 10px;font-size:11px;font-weight:600;")
        self.rec_chip.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.rec_chip.hide()
        rv.addWidget(self.rec_chip, alignment=Qt.AlignmentFlag.AlignLeft)
        rv.addStretch(); root.addWidget(rw)

        # Wire signals
        self.srch.textChanged.connect(self._refilter)
        self.fcb.currentIndexChanged.connect(self._refilter)
        self.store.flows_changed.connect(self._refilter)
        self.settings.settings_changed.connect(self._on_settings)
        self._refilter()

    # ── Public API used by ReportsPage's View button ────────────────────────
    def filter_by_report(self, report_id: str):
        self._report_filter = report_id
        rep = next((r for r in self.store.reports if r.report_id == report_id), None)
        if rep:
            self.fchip_lbl.setText(f"  Filtered: {rep.report_id}  ·  {rep.filename}")
            self.fchip_wrap.show()
        self._refilter()

    def clear_report_filter(self):
        self._report_filter = None
        self.fchip_wrap.hide()
        self._refilter()

    # ── Filtering / rendering ───────────────────────────────────────────────
    def _refilter(self):
        flows = self.store.flows
        if self._report_filter:
            flows = [f for f in flows if f.report_id == self._report_filter]
        q = self.srch.text().strip().lower()
        if q:
            flows = [f for f in flows
                     if q in (f.src_ip or "").lower()
                     or q in (f.dst_ip or "").lower()
                     or q in (f.protocol or "").lower()
                     or q in (f.device_type or "").lower()]
        sel = self.fcb.currentText()
        if   sel == "Botnet only":  flows = [f for f in flows if f.label == "botnet"]
        elif sel == "Benign only":  flows = [f for f in flows if f.label == "benign"]
        elif sel == "IoT only":     flows = [f for f in flows if f.device_type == "iot"]
        elif sel == "Non-IoT only": flows = [f for f in flows if f.device_type == "noniot"]
        cap = self.settings.table_row_limit
        if cap and cap > 0 and len(flows) > cap:
            flows = flows[-cap:]
        self._displayed = flows[::-1]   # newest first
        nb = sum(1 for f in self._displayed if f.label == "botnet")
        ng = sum(1 for f in self._displayed if f.label == "benign")
        self.badge_bot.setText(f"{nb} Botnet")
        self.badge_ben.setText(f"{ng} Benign")
        self._render_table()

    def _render_table(self):
        self.rt.setRowCount(len(self._displayed))
        for i, f in enumerate(self._displayed):
            ib = f.label == "botnet"
            cells = [str(i+1), f.src_ip or "—", f.dst_ip or "—",
                     f.protocol or "—", f.label.capitalize(),
                     f"{f.confidence:.2f}", f.device_type.upper()]
            for j, v in enumerate(cells):
                it = QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j == 4:
                    it.setForeground(QColor(ERR if ib else OK))
                    it.setFont(QFont(FNT, 11, QFont.Weight.Bold))
                if j in (0, 3, 5):
                    it.setFont(QFont("Courier New", 11))
                self.rt.setItem(i, j, it)
        if self._displayed:
            self.rt.selectRow(0)
        else:
            self._clear_details()

    def _sel(self):
        idx = self.rt.currentRow()
        if idx < 0 or idx >= len(self._displayed):
            return
        f  = self._displayed[idx]
        ib = f.label == "botnet"
        self.df["id"].setText(f.report_id or "—")
        self.df["src"].setText(f.src_ip or "—")
        self.df["dst"].setText(f.dst_ip or "—")
        self.df["proto"].setText(f.protocol or "—")
        self.df["dur"].setText("—")     # not tracked at flow-result level today
        self.df["bytes"].setText("—")   # not tracked at flow-result level today
        self.df["dev"].setText((f.device_type or "—").upper())
        self.df["lbl"].setText(f.label.capitalize())
        self.df["lbl"].setStyleSheet(
            f"color:{ERR if ib else OK};background:transparent;font-weight:bold;")
        self.df["conf"].setText(f"{f.confidence:.2f}")
        # XAI panel: per requirement, weights remain mock.
        if self.settings.xai_enabled:
            self.xai.data = XAI_BOT if ib else XAI_BEN
            self.xai.update()
        ex = ("High packet rate and repeated SYN packets detected. Periodic "
              "beaconing consistent with C&C. Recommend blocking source IP."
              if ib else
              "Flow appears normal. Packet rates and timing within baseline. "
              "No suspicious patterns.")
        self.expl.setText(ex)
        self.expl.setStyleSheet(
            f"color:{TM};background:{BG};border-left:3px solid "
            f"{ERR if ib else OK};border-radius:4px;padding:8px 10px;")
        if ib:
            self.rec_chip.setText("Recommended action: block source IP")
            self.rec_chip.show()
        else:
            self.rec_chip.hide()

    def _clear_details(self):
        for k, w in self.df.items():
            w.setText("—")
            if k == "lbl":
                w.setStyleSheet(f"color:{TG};background:transparent;font-weight:bold;")
        self.expl.setText("Select a flow to view.")
        self.expl.setStyleSheet(
            f"color:{TG};background:{BG};border-left:3px solid {BDR};"
            f"border-radius:4px;padding:8px 10px;")
        self.rec_chip.hide()

    def _on_settings(self, key: str):
        if key == "xai_enabled":
            self.xai_card.setVisible(self.settings.xai_enabled)
        elif key == "table_row_limit":
            self._refilter()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
class ReportsPage(QWidget):
    """Real reports list. Each Report = one live session or one upload batch."""

    view_report = pyqtSignal(str)        # emits report_id

    def __init__(self, store: DetectionStore, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.store    = store
        self.settings = settings
        self.setStyleSheet(f"background:{BG};")
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        root  = QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)

        hh = QHBoxLayout(); hh.addWidget(L("Reports", 18, bold=True)); hh.addStretch()
        ec = BTN("Export CSV", "outline", small=True)
        ep = BTN("Export PDF", "primary", small=True)
        ec.clicked.connect(lambda: self._exp("CSV"))
        ep.clicked.connect(lambda: self._exp("PDF"))
        hh.addWidget(ec); hh.addSpacing(6); hh.addWidget(ep); root.addLayout(hh)

        # ── Stat cards ───────────────────────────────────────────────────────
        g = QGridLayout(); g.setSpacing(12)
        self.sc_total = StatCard("archive",     "Total Reports", "0", "All time", ACC)
        self.sc_last  = StatCard("clock",       "Last Scan",     "0", "Botnet flows", ERR)
        self.sc_conf  = StatCard("target",      "Avg Confidence","0%","All flows",   OK)
        self.sc_files = StatCard("folder-open", "Files Analyzed","0", "Upload reports", INFO)
        for i, c in enumerate([self.sc_total, self.sc_last, self.sc_conf, self.sc_files]):
            g.addWidget(c, 0, i)
        root.addLayout(g)

        # ── History table ────────────────────────────────────────────────────
        hc = QFrame(); hc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        hv = QVBoxLayout(hc); hv.setContentsMargins(18,14,18,14)
        hv.addWidget(L("Report History", 14, bold=True)); hv.addSpacing(8)
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["Report ID","Filename","Date","Botnet","Actions"])
        self.tbl.verticalHeader().setVisible(False); self.tbl.setShowGrid(False)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl.setFixedHeight(360)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.verticalHeader().setDefaultSectionSize(36)
        self.tbl.setStyleSheet(TABLE_CSS())
        hv.addWidget(self.tbl); root.addWidget(hc); root.addStretch()

        sc.setWidget(inner); ol = QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

        # Subscribe + paint
        self.store.reports_changed.connect(self.refresh)
        self.store.flows_changed.connect(self.refresh)   # avg-confidence drifts as flows arrive
        self.refresh()

    def refresh(self):
        reports = list(self.store.reports)[::-1]    # newest first
        # Stat cards
        self.sc_total.set_val(str(len(reports)))
        last_n = reports[0].n_botnet if reports else 0
        self.sc_last.set_val(str(last_n))
        self.sc_last.set_sub(f"Report {reports[0].report_id}" if reports else "—")
        self.sc_conf.set_val(f"{self.store.avg_confidence()*100:.1f}%")
        self.sc_conf.set_sub(f"{self.store.stats()['total_flows']} flows total")
        n_uploads = sum(1 for r in self.store.reports if r.source == "upload")
        self.sc_files.set_val(str(n_uploads))

        # Table
        self.tbl.setRowCount(len(reports))
        for i, r in enumerate(reports):
            for j, v in enumerate([r.report_id, r.filename, r.created_at, str(r.n_botnet)]):
                it = QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j == 0:
                    it.setFont(QFont("Courier New", 11)); it.setForeground(QColor(ACC))
                if j == 3:
                    it.setForeground(QColor(ERR if r.n_botnet > 0 else TM))
                self.tbl.setItem(i, j, it)
            view_btn = QPushButton("View")
            view_btn.setFixedHeight(26)
            view_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            view_btn.setStyleSheet(
                f"QPushButton{{background:transparent;color:{ACC};border:1px solid {ACC}55;"
                f"border-radius:6px;padding:0 12px;font-size:11px;font-weight:500;}}"
                f"QPushButton:hover{{background:{ACC}22;border-color:{ACC};}}")
            view_btn.clicked.connect(lambda _checked=False, rid=r.report_id:
                                     self.view_report.emit(rid))
            cell = QWidget(); cl = QHBoxLayout(cell)
            cl.setContentsMargins(8,4,8,4); cl.addWidget(view_btn); cl.addStretch()
            self.tbl.setCellWidget(i, 4, cell)

    def _exp(self, fmt: str):
        if not self.store.flows:
            QMessageBox.information(
                self, "Nothing to export",
                "No detection results yet. Run a live capture or upload a file first.")
            return
        out_dir = self.settings.output_dir
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = ""
        default = (str(Path(out_dir) /
                       f"botnet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt.lower()}")
                   if out_dir else
                   f"botnet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt.lower()}")
        p, _ = QFileDialog.getSaveFileName(self, f"Export {fmt}", default,
                                           f"{fmt} Files (*.{fmt.lower()})")
        if not p:
            return
        if fmt == "CSV":
            with open(p, "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["#","Report","Source","Src IP","Dst IP","Protocol",
                            "Label","Confidence","Device","S1 Conf","Latency ms",
                            "Suspicion","Alerted","Timestamp"])
                for i, f in enumerate(self.store.flows, 1):
                    w.writerow([i, f.report_id, f.source, f.src_ip, f.dst_ip,
                                f.protocol, f.label, f"{f.confidence:.4f}",
                                f.device_type, f"{f.s1_confidence:.4f}",
                                f"{f.latency_ms:.2f}", f"{f.suspicion:.2f}",
                                f.alerted,
                                datetime.fromtimestamp(f.timestamp).strftime("%Y-%m-%d %H:%M:%S")])
            QMessageBox.information(self, "Exported", f"CSV saved:\n{p}")
        else:
            QMessageBox.information(
                self, "PDF",
                "PDF export not yet wired — run Export CSV for now.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
class SettingsPage(QWidget):
    """Persistent settings backed by AppSettings."""

    def __init__(self, store: DetectionStore, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.store    = store
        self.settings = settings
        self.setStyleSheet(f"background:{BG};")
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        root  = QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)
        root.addWidget(L("Settings", 18, bold=True))

        def mk_combo(opts, cur=0):
            cb = QComboBox(); cb.addItems(opts); cb.setCurrentIndex(cur)
            cb.setFixedWidth(200); cb.setFixedHeight(32)
            cb.setStyleSheet(
                f"QComboBox{{background:{BG};color:{TW};border:1px solid {BDR};"
                f"border-radius:6px;padding:0 10px;font-size:12px;}}"
                f"QComboBox::drop-down{{border:none;}}"
                f"QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}")
            return cb

        def mk_tog(on=True):
            b = QPushButton("ON" if on else "OFF"); b.setFixedSize(64,28)
            b.setCheckable(True); b.setChecked(on)
            def upd(c):
                b.setText("ON" if c else "OFF")
                b.setStyleSheet(
                    f"background:{OK if c else BDR};color:white;border:none;"
                    f"border-radius:14px;font-size:11px;font-weight:bold;")
            b.toggled.connect(upd); upd(on); return b

        def scard(title, rows):
            f = QFrame(); f.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
            v = QVBoxLayout(f); v.setContentsMargins(18,14,18,14); v.setSpacing(10)
            v.addWidget(L(title, 14, bold=True)); v.addWidget(SEP())
            for (t1, t2), w in rows:
                r = QHBoxLayout(); c = QVBoxLayout(); c.setSpacing(2)
                c.addWidget(L(t1, 12, color=TW)); c.addWidget(L(t2, 10, color=TD))
                r.addLayout(c); r.addStretch(); r.addWidget(w); v.addLayout(r)
            return f

        # ── Detection Settings ───────────────────────────────────────────────
        thresh_opts = ["0.40","0.45","0.50","0.55","0.60","0.65","0.70"]
        cur_t = self.settings.confidence_threshold
        try:    cur_t_idx = thresh_opts.index(f"{cur_t:.2f}")
        except: cur_t_idx = 2
        self._thresh_cb = mk_combo(thresh_opts, cur=cur_t_idx)
        self._thresh_cb.currentTextChanged.connect(
            lambda t: self.settings.set("confidence_threshold", float(t)))

        self._xai_tog = mk_tog(self.settings.xai_enabled)
        self._xai_tog.toggled.connect(lambda c: self.settings.set("xai_enabled", bool(c)))

        self._alert_tog = mk_tog(self.settings.real_time_alerts)
        self._alert_tog.toggled.connect(lambda c: self.settings.set("real_time_alerts", bool(c)))

        self._auto_tog = mk_tog(self.settings.auto_export_reports)
        self._auto_tog.toggled.connect(lambda c: self.settings.set("auto_export_reports", bool(c)))

        root.addWidget(scard("Detection Settings", [
            (("Confidence Threshold",
              "Re-labels every flow: botnet if Stage-2 confidence ≥ threshold"),
             self._thresh_cb),
            (("Explainable AI (XAI)","Show feature importance panel in Results"),
             self._xai_tog),
            (("Real-time Alerts","Highlight botnet rows in Monitoring page"),
             self._alert_tog),
            (("Auto-export Reports","Save CSV after each scan to output folder"),
             self._auto_tog),
        ]))

        # ── System Preferences ───────────────────────────────────────────────
        self._row_cb = mk_combo(["100","500","1000","Unlimited"])
        cur_l = self.settings.table_row_limit
        for i, lbl in enumerate(["100","500","1000","Unlimited"]):
            if (lbl == "Unlimited" and cur_l == 0) or (lbl != "Unlimited" and cur_l == int(lbl)):
                self._row_cb.setCurrentIndex(i); break
        self._row_cb.currentTextChanged.connect(self._on_row_limit)

        # Output dir as label with a Choose button
        out_w = QWidget(); out_w.setStyleSheet("background:transparent;")
        out_h = QHBoxLayout(out_w); out_h.setContentsMargins(0,0,0,0); out_h.setSpacing(8)
        self._out_lbl = L(self.settings.output_dir, 11, color=TD, mono=True)
        choose = QPushButton("Choose…"); choose.setFixedHeight(28)
        choose.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        choose.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TG};border:1px solid {BDR};"
            f"border-radius:8px;padding:0 12px;font-size:11px;}}"
            f"QPushButton:hover{{color:{TW};border-color:{ACC};}}")
        choose.clicked.connect(self._pick_dir)
        out_h.addWidget(self._out_lbl); out_h.addWidget(choose); out_h.addStretch()

        root.addWidget(scard("System Preferences", [
            (("Output Directory","Where exported reports are saved"),  out_w),
            (("Table Row Limit","Max rows shown in Results table"),    self._row_cb),
        ]))

        # ── Data card ───────────────────────────────────────────────────────
        clr = QPushButton("Clear All Data")
        clr.setFixedHeight(34); clr.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        clr.setStyleSheet(
            f"QPushButton{{background:transparent;color:{ERR};border:1px solid {ERR}55;"
            f"border-radius:8px;padding:0 14px;font-size:12px;font-weight:600;}}"
            f"QPushButton:hover{{background:{ERR}22;border-color:{ERR};}}")
        clr.clicked.connect(self._clear_all)
        root.addWidget(scard("Data", [
            (("Reset detection store",
              "Wipes all flows + reports from disk. Cannot be undone."), clr),
        ]))

        root.addStretch()
        sc.setWidget(inner); ol = QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

    def _on_row_limit(self, txt: str):
        self.settings.set("table_row_limit", 0 if txt == "Unlimited" else int(txt))

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory", self.settings.output_dir)
        if d:
            self.settings.set("output_dir", d)
            self._out_lbl.setText(d)

    def _clear_all(self):
        ans = QMessageBox.question(
            self, "Clear all data?",
            "This will permanently remove every stored flow and report.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            self.store.clear()
            QMessageBox.information(self, "Done", "Detection store cleared.")
# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
TITLES=["Dashboard","Upload & Analyze","Monitoring","Results","Reports","Settings"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BotSense — AI Botnet Detection  ·  Group 07")
        self.setMinimumSize(1200, 720); self.resize(1440, 880)
        self.setStyleSheet(f"background:{BG};")

        # ── Shared services ──────────────────────────────────────────────────
        # Persisted under <project_root>/data/state/  next to data/raw/exports/etc.
        state_dir = Path(__file__).resolve().parents[1] / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.store    = DetectionStore(state_dir / "store.json", parent=self)
        self.settings = AppSettings   (state_dir / "settings.json", parent=self)

        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.hdr = Header(); root.addWidget(self.hdr)
        body = QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        self.sb = Sidebar(); self.sb.switched.connect(self._go); body.addWidget(self.sb)

        self.stk = QStackedWidget(); self.stk.setStyleSheet(f"background:{BG};")
        # All pages now receive the shared store + settings.
        self.dash_page    = DashPage    (self.store, self.settings)
        self.upload_page  = UploadPage  ()                                   # unchanged signature
        self.monitor_page = MonitorPage (self.store, self.settings)
        self.results_page = ResultsPage (self.store, self.settings)
        self.reports_page = ReportsPage (self.store, self.settings)
        self.settings_page = SettingsPage(self.store, self.settings)
        self.pages = [self.dash_page, self.upload_page, self.monitor_page,
                      self.results_page, self.reports_page, self.settings_page]
        for p in self.pages:
            self.stk.addWidget(p)
        body.addWidget(self.stk)
        bw = QWidget(); bw.setLayout(body); root.addWidget(bw)
        self.sbar = StatusBar(); root.addWidget(self.sbar)

        # ── Cross-page wiring ────────────────────────────────────────────────
        # Upload finishes → push results into the store as an upload Report.
        self.upload_page.analysis_done.connect(self._on_upload_done)
        # Reports "View" button → switch to Results filtered by that report.
        self.reports_page.view_report.connect(self._on_view_report)
        # Dashboard "View All" → switch to Results, no filter.
        self.dash_page.view_all_btn.clicked.connect(self._on_view_all)
        # Settings → threshold change → retroactively re-label all stored flows.
        self.settings.settings_changed.connect(self._on_settings_changed)

    # ── Slots ────────────────────────────────────────────────────────────────
    def _go(self, idx):
        self.stk.setCurrentIndex(idx); self.hdr.set_title(TITLES[idx])
        self.sbar.set(f"Viewing: {TITLES[idx]}")

    def _on_upload_done(self, results: list):
        """
        UploadPage.analysis_done emits a list of result dicts from
        inference_bridge.run_file_inference. We translate them into
        DetectionFlows and register a new upload-source Report in the store.

        Note: the current run_file_inference is Stage-1-only — IoT rows are
        labelled 'unknown' and Non-IoT rows are labelled 'benign' as a fallback.
        Wire real Stage-2 batch inference later in inference_bridge — no changes
        to this slot will be needed.
        """
        from detection_store import apply_threshold
        thresh = self.settings.confidence_threshold
        flows = []
        for r in results or []:
            raw_label = str(r.get("label", "benign"))
            conf      = float(r.get("confidence", 0.0))
            # Apply user threshold — preserves 'unknown' for IoT-from-CSV rows.
            flows.append(DetectionFlow(
                src_ip        = str(r.get("src_ip", "") or ""),
                dst_ip        = str(r.get("dst_ip", "") or ""),
                src_port      = int(r.get("src_port", 0) or 0),
                dst_port      = int(r.get("dst_port", 0) or 0),
                protocol      = str(r.get("protocol", "—")),
                label         = apply_threshold(raw_label, conf, thresh),
                confidence    = conf,
                device_type   = str(r.get("device_type", "noniot")),
                s1_confidence = float(r.get("stage1_conf", 0.0)),
                latency_ms    = float(r.get("latency_ms", 0.0)),
            ))
        if not flows:
            return
        # Best-effort filename — UploadPage internals are not exposed.
        fname = "<uploaded file>"
        try:
            ci = getattr(self.upload_page, "_current_file", None)
            if ci is not None:
                fname = getattr(ci, "name", None) or os.path.basename(getattr(ci, "path", "")) or fname
        except Exception:
            pass
        rid = self.store.add_upload_batch(flows, fname)
        # Auto-export if the user opted in.
        if self.settings.auto_export_reports:
            try:
                out = Path(self.settings.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"{rid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(p, "w", newline="") as fp:
                    w = csv.writer(fp)
                    w.writerow(["#","Src IP","Dst IP","Protocol","Label",
                                "Confidence","Device","S1 Conf","Latency ms"])
                    for i, f in enumerate(flows, 1):
                        w.writerow([i, f.src_ip, f.dst_ip, f.protocol, f.label,
                                    f"{f.confidence:.4f}", f.device_type,
                                    f"{f.s1_confidence:.4f}", f"{f.latency_ms:.2f}"])
            except Exception as e:
                print(f"[auto_export] failed: {e!r}")

    def _on_view_report(self, report_id: str):
        self.results_page.filter_by_report(report_id)
        self.sb._sel(3)   # navigate to Results

    def _on_view_all(self):
        self.results_page.clear_report_filter()
        self.sb._sel(3)

    def _on_settings_changed(self, key: str):
        """React to settings updates that affect already-stored data."""
        if key == "confidence_threshold":
            t       = self.settings.confidence_threshold
            changed = self.store.relabel_with_threshold(t)
            self.sbar.set(
                f"Confidence threshold = {t:.2f} — re-labelled {changed} flow(s)"
                if changed else
                f"Confidence threshold = {t:.2f}  (no flows changed)"
            )
    
    def closeEvent(self, event):
        """Stop background work cleanly so threads don't outlive the GUI."""
        # 1. Live capture (sniff loop)
        try:
            if hasattr(self, "monitor_page") and getattr(self.monitor_page, "_running", False):
                self.monitor_page.stop_capture()
        except Exception:
            pass
        # 2. PCAP inference worker (if user closes mid-job)
        try:
            worker = getattr(self.upload_page, "_pcap_worker", None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(2000)   # give it up to 2s to exit cleanly
        except Exception:
            pass
        # 3. Final persist (defensive — store auto-persists on every batch)
        try:
            self.store.save()
            self.settings.save()
        except Exception:
            pass
        super().closeEvent(event)

if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion")
    pal=QPalette()
    pal.setColor(QPalette.ColorRole.Window,        QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText,    QColor(TW))
    pal.setColor(QPalette.ColorRole.Base,          QColor(CARD))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(HOVER))
    pal.setColor(QPalette.ColorRole.Text,          QColor(TW))
    pal.setColor(QPalette.ColorRole.ButtonText,    QColor(TW))
    pal.setColor(QPalette.ColorRole.Highlight,     QColor(ACC))
    pal.setColor(QPalette.ColorRole.HighlightedText,QColor("#FFFFFF"))
    app.setPalette(pal)
    # Apply the BotSense global stylesheet (theme.py is the single source of truth).
    app.setStyleSheet(qss())
    win=MainWindow(); win.show(); sys.exit(app.exec())