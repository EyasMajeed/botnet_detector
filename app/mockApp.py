"""
Botnet Detection System — SOC Dashboard  v2
Group 07 · CPCS499
Design: Professional dark SOC dashboard, sidebar nav, 6 pages.
"""
import sys, os, random, csv
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
        col.addWidget(L(title, 11, color=TG))
        col.addWidget(self.vl)
        col.addWidget(L(sub, 10, color=TD))
        h.addLayout(col); h.addStretch()

    def set_val(self, v): self.vl.setText(v)

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
           ("upload-cloud", "Upload & Analyze", 1),
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
        fl.addWidget(L("CPCS499  ·  2025",10,color=TD))
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
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        sc=QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner=QWidget(); inner.setStyleSheet(f"background:{BG};")
        root=QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)
        # Stat cards
        g=QGridLayout(); g.setSpacing(14)
        # Icon names map to app/assets/icons/<name>.svg.  Each is semantically
        # tied to what the card actually represents, not just decoration.
        self.sc_t=StatCard("radio",       "Total Flows", "142","Last scan: 2 min ago",ACC)
        self.sc_b=StatCard("shield-alert","Botnet Flows","18", "12.7% of traffic",ERR)
        self.sc_g=StatCard("shield-check","Benign Flows","124","87.3% of traffic",OK)
        self.sc_d=StatCard("cpu",         "Devices",     "34", "12 IoT · 22 Non-IoT",INFO)
        for i,c in enumerate([self.sc_t,self.sc_b,self.sc_g,self.sc_d]): g.addWidget(c,0,i)
        root.addLayout(g)
        # Charts row
        cr=QHBoxLayout(); cr.setSpacing(14)
        # Pie
        pc=QFrame(); pc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        pv=QVBoxLayout(pc); pv.setContentsMargins(18,14,18,14)
        pv.addWidget(L("Traffic Distribution",14,bold=True))
        pv.addWidget(L("Benign vs Botnet",11,color=TD)); pv.addSpacing(6)
        self.pie=PieChart(124,18); pv.addWidget(self.pie)
        lr=QHBoxLayout()
        for col,lbl_ in [(OK,"Benign  87.3%"),(ERR,"Botnet  12.7%")]:
            d=QLabel("●"); d.setStyleSheet(f"color:{col};background:transparent;")
            lr.addWidget(d); lr.addWidget(L(lbl_,11,color=TG)); lr.addSpacing(8)
        lr.addStretch(); pv.addLayout(lr); cr.addWidget(pc,1)
        # Spark
        skc=QFrame(); skc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        sv=QVBoxLayout(skc); sv.setContentsMargins(18,14,18,14)
        sv.addWidget(L("Botnet Activity – last 20 min",14,bold=True))
        sv.addWidget(L("Flows per minute",11,color=TD)); sv.addSpacing(6)
        sv.addWidget(Spark([1,0,2,1,3,2,5,4,6,8,7,9,6,10,8,12,9,11,8,13],ERR))
        cr.addWidget(skc,2); root.addLayout(cr)
        # Recent detections table
        rc=QFrame(); rc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        rv=QVBoxLayout(rc); rv.setContentsMargins(18,14,18,14)
        rh=QHBoxLayout(); rh.addWidget(L("Recent Detections",14,bold=True)); rh.addStretch()
        rh.addWidget(BTN("View All →","outline",small=True)); rv.addLayout(rh); rv.addSpacing(10)
        t=QTableWidget(6,6); t.setHorizontalHeaderLabels(["Flow ID","Src IP","Dst IP","Protocol","Label","Confidence"])
        t.verticalHeader().setVisible(False); t.setShowGrid(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setFixedHeight(230); t.setStyleSheet(TABLE_CSS())
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setDefaultSectionSize(34)
        for i,row in enumerate(DEMO[:6]):
            ib=row["lbl"]=="Botnet"
            # Confidence column rendered as % to match the Monitoring page
            # (where flow confidence shows as 59.47%, not 0.5947).
            for j,v in enumerate([row["id"],row["src"],row["dst"],row["proto"],row["lbl"],f"{row['conf']:.0%}"]):
                it=QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j==4: it.setForeground(QColor(ERR if ib else OK)); it.setFont(QFont(FNT,11,QFont.Weight.Bold))
                if j in(0,3,5): it.setFont(QFont("Courier New",11))
                t.setItem(i,j,it)
        rv.addWidget(t); root.addWidget(rc)
        sc.setWidget(inner); ol=QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

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
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        root=QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        # Left
        lw=QWidget(); lw.setStyleSheet(f"background:{BG};")
        lv=QVBoxLayout(lw); lv.setContentsMargins(24,24,12,24); lv.setSpacing(12)
        hh=QHBoxLayout(); hh.addWidget(L("Detection Results",18,bold=True)); hh.addStretch()
        nb=sum(1 for r in DEMO if r["lbl"]=="Botnet"); ng=len(DEMO)-nb
        hh.addWidget(BADGE(f"{nb} Botnet","danger")); hh.addSpacing(6); hh.addWidget(BADGE(f"{ng} Benign","success"))
        lv.addLayout(hh)
        fr=QHBoxLayout()
        self.srch=QLineEdit(); self.srch.setPlaceholderText("Search IP, protocol, device...")
        self.srch.setFixedHeight(34)
        self.srch.setStyleSheet(f"QLineEdit{{background:{CARD};color:{TW};border:1px solid {BDR};border-radius:8px;padding:0 12px;font-size:12px;}}QLineEdit:focus{{border-color:{ACC};}}")
        self.fcb=QComboBox(); self.fcb.addItems(["All","Botnet only","Benign only","IoT only","Non-IoT only"])
        self.fcb.setFixedHeight(34)
        self.fcb.setStyleSheet(f"QComboBox{{background:{CARD};color:{TW};border:1px solid {BDR};border-radius:8px;padding:0 10px;font-size:12px;}}QComboBox::drop-down{{border:none;}}QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}")
        fr.addWidget(self.srch); fr.addWidget(self.fcb); lv.addLayout(fr)
        self.rt=QTableWidget(len(DEMO),7)
        self.rt.setHorizontalHeaderLabels(["#","Src IP","Dst IP","Protocol","Label","Confidence","Device"])
        self.rt.verticalHeader().setVisible(False); self.rt.setShowGrid(False)
        self.rt.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rt.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rt.setStyleSheet(TABLE_CSS())
        h2=self.rt.horizontalHeader()
        for col,mode,w in [(0,QHeaderView.ResizeMode.Fixed,50),(1,QHeaderView.ResizeMode.Stretch,0),
                           (2,QHeaderView.ResizeMode.Stretch,0),(3,QHeaderView.ResizeMode.Fixed,80),
                           (4,QHeaderView.ResizeMode.Fixed,90),(5,QHeaderView.ResizeMode.Fixed,100),
                           (6,QHeaderView.ResizeMode.Fixed,90)]:
            h2.setSectionResizeMode(col,mode)
            if w: self.rt.setColumnWidth(col,w)
        self.rt.verticalHeader().setDefaultSectionSize(36)
        self.rt.itemSelectionChanged.connect(self._sel)
        for i,row in enumerate(DEMO):
            ib=row["lbl"]=="Botnet"
            for j,v in enumerate([str(i+1),row["src"],row["dst"],row["proto"],row["lbl"],f"{row['conf']:.2f}",row["dev"]]):
                it=QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j==4: it.setForeground(QColor(ERR if ib else OK)); it.setFont(QFont(FNT,11,QFont.Weight.Bold))
                if j in(0,3,5): it.setFont(QFont("Courier New",11))
                self.rt.setItem(i,j,it)
        lv.addWidget(self.rt); root.addWidget(lw,3)
        # Right panel
        rw=QWidget(); rw.setFixedWidth(300)
        rw.setStyleSheet(f"background:{CARD};border-left:1px solid {BDR};")
        rv=QVBoxLayout(rw); rv.setContentsMargins(18,18,18,18); rv.setSpacing(12)
        rv.addWidget(L("Flow Details",14,bold=True)); rv.addWidget(SEP())
        self.df={}
        for lbl_,key in [("Flow ID","id"),("Src IP","src"),("Dst IP","dst"),("Protocol","proto"),
                         ("Duration","dur"),("Bytes","bytes"),("Device","dev"),("Prediction","lbl"),("Confidence","conf")]:
            r=QHBoxLayout(); r.addWidget(L(lbl_,11,color=TD)); r.addStretch()
            vl=L("—",11,mono=(key in("src","dst","id","conf"))); self.df[key]=vl; r.addWidget(vl); rv.addLayout(r)
        rv.addWidget(SEP()); rv.addWidget(L("Top Features (XAI)",13,bold=True))
        self.xai=HBar(); rv.addWidget(self.xai)
        rv.addWidget(SEP()); rv.addWidget(L("Explanation",13,bold=True))
        self.expl=L("Select a flow to view.",11,color=TG,wrap=True)
        self.expl.setStyleSheet(f"color:{TG};background:{BG};border-left:3px solid {BDR};border-radius:4px;padding:8px 10px;")
        rv.addWidget(self.expl)
        # Recommendation chip — small action pill, only shown for botnet flows.
        # Hidden by default and toggled in _sel().
        self.rec_chip = QLabel()
        self.rec_chip.setStyleSheet(
            f"color:{ERR};background:{ERR}22;border:1px solid {ERR}44;"
            f"border-radius:10px;padding:4px 10px;font-size:11px;font-weight:600;"
        )
        self.rec_chip.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.rec_chip.hide()
        rv.addWidget(self.rec_chip, alignment=Qt.AlignmentFlag.AlignLeft)
        rv.addStretch(); root.addWidget(rw)
        self.rt.selectRow(0)

    def _sel(self):
        idx=self.rt.currentRow()
        if idx<0 or idx>=len(DEMO): return
        row=DEMO[idx]; ib=row["lbl"]=="Botnet"
        for key,wid in self.df.items():
            v=row.get(key,"—")
            if key=="conf": v=f"{v:.2f}"
            wid.setText(str(v))
            if key=="lbl": wid.setStyleSheet(f"color:{ERR if ib else OK};background:transparent;font-weight:bold;")
        self.xai.data=XAI_BOT if ib else XAI_BEN; self.xai.update()
        ex=("High packet rate and repeated SYN packets detected. Periodic beaconing consistent with C&C. "
            "Recommend blocking source IP." if ib else
            "Flow appears normal. Packet rates and timing within baseline. No suspicious patterns.")
        self.expl.setText(ex)
        self.expl.setStyleSheet(f"color:{TM};background:{BG};border-left:3px solid {ERR if ib else OK};border-radius:4px;padding:8px 10px;")
        # Recommendation chip: only meaningful for botnet flows.
        if ib:
            self.rec_chip.setText("Recommended action: block source IP")
            self.rec_chip.show()
        else:
            self.rec_chip.hide()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
class ReportsPage(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        sc=QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner=QWidget(); inner.setStyleSheet(f"background:{BG};")
        root=QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)
        hh=QHBoxLayout(); hh.addWidget(L("Reports",18,bold=True)); hh.addStretch()
        ec=BTN("Export CSV","outline",small=True); ep=BTN("Export PDF","primary",small=True)
        ec.clicked.connect(lambda:self._exp("CSV")); ep.clicked.connect(lambda:self._exp("PDF"))
        hh.addWidget(ec); hh.addSpacing(6); hh.addWidget(ep); root.addLayout(hh)
        g=QGridLayout(); g.setSpacing(12)
        # Each card icon ties to what it actually represents:
        # archive (history), clock (recency), target (accuracy), folder-open (files).
        for i,(t,v,s,c) in enumerate([("Total Reports","12","All time",ACC),("Last Scan","18","Botnet flows",ERR),
                                       ("Avg Accuracy","94%","All scans",OK),("Files Analyzed","47","PCAP+CSV",INFO)]):
            g.addWidget(StatCard(["archive","clock","target","folder-open"][i],t,v,s,c),0,i)
        root.addLayout(g)
        hc=QFrame(); hc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        hv=QVBoxLayout(hc); hv.setContentsMargins(18,14,18,14); hv.addWidget(L("Report History",14,bold=True)); hv.addSpacing(8)
        t=QTableWidget(8,5); t.setHorizontalHeaderLabels(["Report ID","Filename","Date","Botnet","Actions"])
        t.verticalHeader().setVisible(False); t.setShowGrid(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setFixedHeight(300); t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setDefaultSectionSize(36); t.setStyleSheet(TABLE_CSS())
        data=[("RPT-012","capture_final.pcap","2025-05-10","18","View"),
              ("RPT-011","iot_traffic.pcap","2025-05-09","7","View"),
              ("RPT-010","lab_capture.csv","2025-05-08","0","View"),
              ("RPT-009","network_dump.pcap","2025-05-07","12","View"),
              ("RPT-008","flows_may6.csv","2025-05-06","3","View"),
              ("RPT-007","monitor_export.csv","2025-05-05","22","View"),
              ("RPT-006","pcap_test2.pcap","2025-05-04","5","View"),
              ("RPT-005","live_capture.pcap","2025-05-03","9","View")]
        for i,(rid,fn,dt,bn,ac) in enumerate(data):
            # Columns 0-3 are data items (Report ID / Filename / Date / Botnet).
            # The Botnet column is intentionally neutral — these are past
            # results, not active threats; saturated red on every value made
            # the table feel like everything was on fire.
            for j,v in enumerate([rid,fn,dt,bn]):
                it=QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j==3: it.setForeground(QColor(TM))   # neutral past-data
                if j==0: it.setFont(QFont("Courier New",11)); it.setForeground(QColor(ACC))
                t.setItem(i,j,it)
            # Column 4: actual button widget so all rows look identical.
            view_btn = QPushButton("View")
            view_btn.setFixedHeight(26)
            view_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            view_btn.setStyleSheet(
                f"QPushButton{{background:transparent;color:{ACC};border:1px solid {ACC}55;"
                f"border-radius:6px;padding:0 12px;font-size:11px;font-weight:500;}}"
                f"QPushButton:hover{{background:{ACC}22;border-color:{ACC};}}"
            )
            cell_w = QWidget(); cell_l = QHBoxLayout(cell_w)
            cell_l.setContentsMargins(8, 4, 8, 4); cell_l.addWidget(view_btn); cell_l.addStretch()
            t.setCellWidget(i, 4, cell_w)
        hv.addWidget(t); root.addWidget(hc); root.addStretch()
        sc.setWidget(inner); ol=QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

    def _exp(self,fmt):
        p,_=QFileDialog.getSaveFileName(self,f"Export {fmt}",
            f"botnet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt.lower()}",
            f"{fmt} Files (*.{fmt.lower()})")
        if not p: return
        if fmt=="CSV":
            with open(p,"w",newline="") as f:
                w=csv.writer(f); w.writerow(["#","Src IP","Dst IP","Protocol","Label","Confidence","Device"])
                for i,r in enumerate(DEMO): w.writerow([i+1,r["src"],r["dst"],r["proto"],r["lbl"],f"{r['conf']:.2f}",r["dev"]])
            QMessageBox.information(self,"Exported",f"CSV saved:\n{p}")
        else:
            QMessageBox.information(self,"PDF","PDF export coming in Day 10–11 using reportlab!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
class SettingsPage(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        sc=QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner=QWidget(); inner.setStyleSheet(f"background:{BG};")
        root=QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)
        root.addWidget(L("Settings",18,bold=True))

        def mk_combo(opts,cur=0):
            cb=QComboBox(); cb.addItems(opts); cb.setCurrentIndex(cur)
            cb.setFixedWidth(200); cb.setFixedHeight(32)
            cb.setStyleSheet(f"QComboBox{{background:{BG};color:{TW};border:1px solid {BDR};border-radius:6px;padding:0 10px;font-size:12px;}}QComboBox::drop-down{{border:none;}}QComboBox QAbstractItemView{{background:{CARD};color:{TW};border:1px solid {BDR};}}")
            return cb

        def mk_path_combo(opts, cur=0):
            """Path-style combo: folder-open SVG icon prefix + combo box.
            Used to visually differentiate file/path inputs from numeric ones.
            """
            w = QWidget(); w.setStyleSheet("background:transparent;")
            h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)
            ic = QLabel(); ic.setPixmap(icon("folder-open", color=TG, size=16).pixmap(16, 16))
            ic.setStyleSheet("background:transparent;")
            cb = mk_combo(opts, cur)
            h.addWidget(ic); h.addWidget(cb); h.addStretch()
            return w

        def mk_tog(on=True):
            b=QPushButton("ON" if on else "OFF"); b.setFixedSize(64,28); b.setCheckable(True); b.setChecked(on)
            def upd(c): b.setText("ON" if c else "OFF"); b.setStyleSheet(f"background:{OK if c else BDR};color:white;border:none;border-radius:14px;font-size:11px;font-weight:bold;")
            b.toggled.connect(upd); upd(on); return b

        def scard(title,rows):
            f=QFrame(); f.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
            v=QVBoxLayout(f); v.setContentsMargins(18,14,18,14); v.setSpacing(10)
            v.addWidget(L(title,14,bold=True)); v.addWidget(SEP())
            for (t1,t2),w in rows:
                r=QHBoxLayout(); c=QVBoxLayout(); c.setSpacing(2)
                c.addWidget(L(t1,12,color=TW)); c.addWidget(L(t2,10,color=TD))
                r.addLayout(c); r.addStretch(); r.addWidget(w); v.addLayout(r)
            return f

        root.addWidget(scard("Detection Settings",[
            (("Confidence Threshold","Flag as botnet if above this value"),mk_combo(["0.40","0.45","0.50","0.55","0.60"],cur=2)),
            (("Explainable AI (XAI)","Show feature importance per detection"),mk_tog(True)),
            (("Real-time Alerts","Notify on botnet detections"),mk_tog(True)),
            (("Auto-export Reports","Save report after each scan"),mk_tog(False)),
        ]))
        root.addWidget(scard("System Preferences",[
            (("Output Directory","Where exported reports are saved"),mk_path_combo(["Desktop/botnet_reports","Documents/reports"])),
            (("Table Row Limit","Max rows in results table"),mk_combo(["100","500","1000","Unlimited"])),
        ]))
        root.addStretch()
        sc.setWidget(inner); ol=QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
TITLES=["Dashboard","Upload & Analyze","Monitoring","Results","Reports","Settings"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BotSense — AI Botnet Detection  ·  Group 07")
        self.setMinimumSize(1200,720); self.resize(1440,880)
        self.setStyleSheet(f"background:{BG};")
        cw=QWidget(); self.setCentralWidget(cw)
        root=QVBoxLayout(cw); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.hdr=Header(); root.addWidget(self.hdr)
        body=QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        self.sb=Sidebar(); self.sb.switched.connect(self._go); body.addWidget(self.sb)
        self.stk=QStackedWidget(); self.stk.setStyleSheet(f"background:{BG};")
        self.pages=[DashPage(),UploadPage(),MonitorPage(),ResultsPage(),ReportsPage(),SettingsPage()]
        for p in self.pages: self.stk.addWidget(p)
        body.addWidget(self.stk)
        bw=QWidget(); bw.setLayout(body); root.addWidget(bw)
        self.sbar=StatusBar(); root.addWidget(self.sbar)

    def _go(self,idx):
        self.stk.setCurrentIndex(idx); self.hdr.set_title(TITLES[idx])
        self.sbar.set(f"Viewing: {TITLES[idx]}")

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