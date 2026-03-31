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
BG      = "#1E1E2F"
CARD    = "#16161F"
HOVER   = "#252535"
ACC     = "#3A7AFE"
ACC2    = "#2A5FD4"
BDR     = "#374151"
TW      = "#FFFFFF"
TG      = "#9CA3AF"
TD      = "#6B7280"
TM      = "#D1D5DB"
OK      = "#10B981"
ERR     = "#EF4444"
WARN    = "#F97316"
INFO    = "#A855F7"
YEL     = "#EAB308"
FNT     = "Segoe UI"

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
    def __init__(self, icon, title, value, sub="", color=ACC):
        super().__init__()
        self.setStyleSheet(f"QFrame{{background:{CARD};border:none;border-radius:12px;}}")
        h = QHBoxLayout(self); h.setContentsMargins(18,14,18,14); h.setSpacing(14)
        ic = QLabel(icon); ic.setFixedSize(46,46)
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"background:{color}22;border:1px solid {color}44;border-radius:10px;font-size:20px;")
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
        p.setPen(QColor(TW)); p.setFont(QFont(FNT,12,QFont.Weight.Bold))
        pct=int(100*self.n/t)
        p.drawText(rect,Qt.AlignmentFlag.AlignCenter,f"{pct}%\nbotnet")
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
        p.setPen(QPen(QColor(BDR),1,Qt.PenStyle.DashLine))
        for f in [.25,.5,.75]:
            yy=int(pad+(H-2*pad)*(1-f)); p.drawLine(pad,yy,W-pad,yy)
        poly=QPolygonF()
        poly.append(QPointF(pad,H-pad))
        for i,v in enumerate(self.vals): x,y=pt(i,v); poly.append(QPointF(x,y))
        poly.append(QPointF(W-pad,H-pad))
        g=QLinearGradient(0,0,0,H)
        g.setColorAt(0,QColor(self.color+"55")); g.setColorAt(1,QColor(self.color+"00"))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g)); p.drawPolygon(poly)
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
            p.setBrush(QColor(BDR)); p.drawRoundedRect(lw,cy-4,bw,8,4,4)
            fw=int(bw*val/mx)
            if fw>0:
                c=ERR if val>.7 else (WARN if val>.4 else ACC)
                p.setBrush(QColor(c)); p.drawRoundedRect(lw,cy-4,fw,8,4,4)
            p.setPen(QColor(TM)); p.setFont(QFont("Courier New",10))
            p.drawText(lw+bw+5,cy-8,48,20,
                Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,f"{val:.2f}")
        p.end()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
class Sidebar(QWidget):
    switched = pyqtSignal(int)
    NAV = [("🛡","Dashboard",0),("📤","Upload & Analyze",1),
           ("📡","Monitoring",2),("📊","Results",3),
           ("📋","Reports",4),("⚙","Settings",5)]
    def __init__(self):
        super().__init__(); self.setFixedWidth(220)
        self.setStyleSheet(f"background:{CARD};")
        self._btns=[]; root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        # Logo
        logo=QWidget(); logo.setFixedHeight(64)
        logo.setStyleSheet(f"background:{CARD};")
        lh=QHBoxLayout(logo); lh.setContentsMargins(16,0,16,0); lh.setSpacing(10)
        ic=QLabel("🛡"); ic.setFont(QFont(FNT,22)); ic.setStyleSheet(f"color:{ACC};background:transparent;")
        lh.addWidget(ic)
        tc=QVBoxLayout(); tc.setSpacing(0)
        tc.addWidget(L("BotDetect",13,bold=True)); tc.addWidget(L("AI Security Suite",9,color=TD))
        lh.addLayout(tc); lh.addStretch(); root.addWidget(logo)
        # Nav
        nav=QWidget(); nav.setStyleSheet("background:transparent;")
        nl=QVBoxLayout(nav); nl.setContentsMargins(10,14,10,14); nl.setSpacing(4)
        for icon,name,idx in self.NAV:
            b=QPushButton(f"  {icon}   {name}"); b.setFont(QFont(FNT,12))
            b.setFixedHeight(42); b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.clicked.connect(lambda _,i=idx: self._sel(i))
            self._btns.append(b); nl.addWidget(b)
        nl.addStretch(); root.addWidget(nav)
        # Footer
        ft=QWidget(); ft.setStyleSheet(f"background:transparent;")
        fl=QVBoxLayout(ft); fl.setContentsMargins(16,10,16,14); fl.setSpacing(2)
        fl.addWidget(L("v1.0.0  ·  Group 07",10,color=TD)); fl.addWidget(L("CPCS499  ·  2025",10,color=TD))
        root.addWidget(ft)
        self._sel(0)

    def _sel(self,idx):
        for i,b in enumerate(self._btns):
            if i==idx:
                b.setStyleSheet(f"QPushButton{{background:{ACC};color:white;border:none;"
                    f"border-radius:8px;text-align:left;padding-left:14px;font-weight:bold;}}")
            else:
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
        dot=QLabel(); dot.setFixedSize(10,10)
        dot.setStyleSheet(f"background:{OK};border-radius:5px;")
        self.sl=L("System Active",13,color=TG)
        self.cl=L("",12,color=TD,mono=True)
        h.addWidget(dot); h.addSpacing(6); h.addWidget(self.sl); h.addSpacing(20); h.addWidget(self.cl)
        t=QTimer(self); t.timeout.connect(lambda: self.cl.setText(datetime.now().strftime("%H:%M:%S"))); t.start(1000)
        self.cl.setText(datetime.now().strftime("%H:%M:%S"))

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
        self.sc_t=StatCard("📡","Total Flows","142","Last scan: 2 min ago",ACC)
        self.sc_b=StatCard("⚠","Botnet Flows","18","12.7% of traffic",ERR)
        self.sc_g=StatCard("✓","Benign Flows","124","87.3% of traffic",OK)
        self.sc_d=StatCard("📱","Devices","34","12 IoT · 22 Non-IoT",INFO)
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
            for j,v in enumerate([row["id"],row["src"],row["dst"],row["proto"],row["lbl"],f"{row['conf']:.2f}"]):
                it=QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j==4: it.setForeground(QColor(ERR if ib else OK)); it.setFont(QFont(FNT,11,QFont.Weight.Bold))
                if j in(0,3,5): it.setFont(QFont("Courier New",11))
                t.setItem(i,j,it)
        rv.addWidget(t); root.addWidget(rc)
        sc.setWidget(inner); ol=QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
class UploadPage(QWidget):
    file_loaded=pyqtSignal(str)
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        sc=QScrollArea(); sc.setWidgetResizable(True)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner=QWidget(); inner.setStyleSheet(f"background:{BG};")
        root=QVBoxLayout(inner); root.setContentsMargins(24,24,24,24); root.setSpacing(18)
        # Drop zone
        dz=QFrame(); dz.setFixedHeight(200)
        dz.setStyleSheet(f"QFrame{{background:{CARD};border:2px dashed {BDR};border-radius:12px;}}QFrame:hover{{border-color:{ACC};}}")
        dz.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        dv=QVBoxLayout(dz); dv.setAlignment(Qt.AlignmentFlag.AlignCenter); dv.setSpacing(8)
        up=L("⬆",36,color=ACC); up.setAlignment(Qt.AlignmentFlag.AlignCenter); dv.addWidget(up)
        dv.addWidget(L("Click to upload or drag & drop",14,bold=True,color=TW))
        self.hint=L("PCAP · CSV · NetFlow — Max 500 MB",12,color=TD); self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dv.addWidget(self.hint); dv.addSpacing(8)
        ub=BTN("Browse Files","primary"); ub.setFixedWidth(150); dv.addWidget(ub,alignment=Qt.AlignmentFlag.AlignCenter)
        ub.clicked.connect(self._browse); dz.mousePressEvent=lambda e:self._browse(); root.addWidget(dz)
        # File info card
        self.fic=QFrame()
        self.fic.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        self.fic.hide()
        fh=QHBoxLayout(self.fic); fh.setContentsMargins(18,14,18,14)
        fh.addWidget(L("📄",22))
        fc=QVBoxLayout(); self.fn=L("file.pcap",13,bold=True); self.fm=L("—",11,color=TD)
        fc.addWidget(self.fn); fc.addWidget(self.fm); fh.addLayout(fc); fh.addStretch()
        self.rb=BTN("▶  Run Detection","primary"); self.rb.setFixedWidth(160)
        self.rb.clicked.connect(self._run); fh.addWidget(self.rb); root.addWidget(self.fic)
        # Config
        cr=QHBoxLayout(); cr.setSpacing(14)
        mc=QFrame(); mc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        mv=QVBoxLayout(mc); mv.setContentsMargins(18,14,18,14); mv.setSpacing(10)
        mv.addWidget(L("AI Model Configuration",14,bold=True)); mv.addWidget(SEP())
        for lbl_,val in [("Stage-1","Random Forest (IoT vs Non-IoT)"),
                         ("Stage-2 IoT","CNN-LSTM (IoT-23)"),("Stage-2 Non-IoT","CNN-LSTM (CTU-13)"),
                         ("XAI","Feature Perturbation")]:
            r=QHBoxLayout(); r.addWidget(L(lbl_,11,color=TG)); r.addStretch(); r.addWidget(L(val,11,color=TW))
            mv.addLayout(r)
        cr.addWidget(mc,1)
        fc2=QFrame(); fc2.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        fv=QVBoxLayout(fc2); fv.setContentsMargins(18,14,18,14); fv.setSpacing(10)
        fv.addWidget(L("Feature Extraction",14,bold=True)); fv.addWidget(SEP())
        for nm,on in [("Flow-level statistical features",True),("Time-window temporal features",True),
                      ("Packet-level features (PCAP)",True),("TLS handshake metadata",False)]:
            cb=QCheckBox(nm); cb.setChecked(on); cb.setFont(QFont(FNT,12))
            cb.setStyleSheet(f"QCheckBox{{color:{TW};background:transparent;spacing:8px;}}"
                f"QCheckBox::indicator{{width:16px;height:16px;border-radius:4px;border:1px solid {BDR};background:{BG};}}"
                f"QCheckBox::indicator:checked{{background:{ACC};border-color:{ACC};}}")
            fv.addWidget(cb)
        cr.addWidget(fc2,1); root.addLayout(cr)
        # Threshold
        tc=QFrame(); tc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        tv=QVBoxLayout(tc); tv.setContentsMargins(18,14,18,14); tv.setSpacing(10)
        tv.addWidget(L("Detection Sensitivity",14,bold=True))
        tv.addWidget(L("Lower = higher recall (catches more botnets, more false alarms)",11,color=TD))
        tr=QHBoxLayout(); tr.addWidget(L("Threshold:",12,color=TG))
        self.sl2=QSlider(Qt.Orientation.Horizontal); self.sl2.setRange(10,90); self.sl2.setValue(50)
        self.sl2.setStyleSheet(f"QSlider::groove:horizontal{{height:6px;background:{BDR};border-radius:3px;}}"
            f"QSlider::handle:horizontal{{background:{ACC};width:18px;height:18px;margin:-6px 0;border-radius:9px;}}"
            f"QSlider::sub-page:horizontal{{background:{ACC};border-radius:3px;}}")
        self.tv2=L("0.50",13,bold=True,color=ACC,mono=True)
        self.sl2.valueChanged.connect(lambda v:self.tv2.setText(f"{v/100:.2f}"))
        tr.addWidget(self.sl2); tr.addWidget(self.tv2); tv.addLayout(tr); root.addWidget(tc)
        root.addStretch()
        sc.setWidget(inner); ol=QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(sc)

    def _browse(self):
        p,_=QFileDialog.getOpenFileName(self,"Open Traffic File","","Network Files (*.pcap *.csv *.netflow *.txt);;All Files (*)")
        if p:
            n=os.path.basename(p); s=os.path.getsize(p)
            ss=f"{s/1024:.1f} KB" if s<1048576 else f"{s/1048576:.1f} MB"
            self.fn.setText(n); self.fm.setText(f"{ss}  ·  Ready"); self.fic.show(); self.file_loaded.emit(p)

    def _run(self):
        QMessageBox.information(self,"Detection",
            "In CPCS499 this will call your teammates'\npreprocess_and_predict() function.\n\nShowing demo results on the Results page.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
class MonitorPage(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{BG};")
        self._cnt=0; self._alerts=0; self._running=True; self._t0=datetime.now()
        root=QVBoxLayout(self); root.setContentsMargins(24,24,24,24); root.setSpacing(14)
        # Controls
        ch=QHBoxLayout(); ch.addWidget(L("Live Traffic Monitor",18,bold=True)); ch.addStretch()
        self.tog=BTN("⏸  Pause","primary"); self.tog.clicked.connect(self._toggle); ch.addWidget(self.tog)
        ch.addSpacing(8); ch.addWidget(BTN("Clear","outline",small=True)); root.addLayout(ch)
        # Live stat strip
        sr=QHBoxLayout(); sr.setSpacing(10); self._ls={}
        for t,v,c in [("Flows/sec","0",ACC),("Bandwidth","0 KB/s",OK),("Alerts","0",ERR),("Uptime","00:00:00",TG)]:
            f=QFrame(); f.setStyleSheet(f"background:{CARD};border:none;border-radius:10px;")
            fv=QVBoxLayout(f); fv.setContentsMargins(14,8,14,8); fv.setSpacing(2)
            vl=L(v,16,bold=True,color=c,mono=True); fv.addWidget(L(t,10,color=TD)); fv.addWidget(vl)
            self._ls[t]=vl; sr.addWidget(f)
        root.addLayout(sr)
        # Table card
        tc=QFrame(); tc.setStyleSheet(f"background:{CARD};border:none;border-radius:12px;")
        tv=QVBoxLayout(tc); tv.setContentsMargins(0,0,0,0)
        tb=QWidget(); tb.setFixedHeight(44)
        tb.setStyleSheet(f"background:{BG};border-radius:12px 12px 0 0;border-bottom:1px solid {BDR};")
        th=QHBoxLayout(tb); th.setContentsMargins(16,0,16,0)
        th.addWidget(L("Real-time Flow Feed",13,bold=True)); th.addStretch()
        self.lc=L("0 flows",11,color=TD); th.addWidget(self.lc); tv.addWidget(tb)
        self.lt=QTableWidget(0,7)
        self.lt.setHorizontalHeaderLabels(["Timestamp","Src IP","Dst IP","Protocol","Label","Confidence","Device"])
        self.lt.verticalHeader().setVisible(False); self.lt.setShowGrid(False)
        self.lt.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lt.setStyleSheet(TABLE_CSS())
        self.lt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.lt.verticalHeader().setDefaultSectionSize(34); tv.addWidget(self.lt)
        root.addWidget(tc)
        self._timer=QTimer(self); self._timer.timeout.connect(self._add); self._timer.start(1600)

    def _toggle(self):
        if self._running: self._timer.stop(); self.tog.setText("▶  Resume")
        else: self._timer.start(1600); self.tog.setText("⏸  Pause")
        self._running=not self._running

    def _add(self):
        row=random.choice(DEMO); ib=row["lbl"]=="Botnet"; ts=datetime.now().strftime("%H:%M:%S")
        r=self.lt.rowCount(); self.lt.insertRow(r)
        for j,v in enumerate([ts,row["src"],row["dst"],row["proto"],row["lbl"],f"{row['conf']:.2f}",row["dev"]]):
            it=QTableWidgetItem(v); it.setForeground(QColor(TW))
            if j==4: it.setForeground(QColor(ERR if ib else OK)); it.setFont(QFont(FNT,11,QFont.Weight.Bold))
            self.lt.setItem(r,j,it)
        if ib: self._alerts+=1
        if self.lt.rowCount()>50: self.lt.removeRow(0)
        self.lt.scrollToBottom(); self._cnt+=1
        up=str(datetime.now()-self._t0).split(".")[0]
        self._ls["Flows/sec"].setText(str(random.randint(2,12)))
        self._ls["Bandwidth"].setText(f"{random.randint(50,400)} KB/s")
        self._ls["Alerts"].setText(str(self._alerts))
        self._ls["Uptime"].setText(up)
        self.lc.setText(f"{self._cnt} flows")

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
        rv.addWidget(self.expl); rv.addStretch(); root.addWidget(rw)
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
        for i,(t,v,s,c) in enumerate([("Total Reports","12","All time",ACC),("Last Scan","18","Botnet flows",ERR),
                                       ("Avg Accuracy","94%","All scans",OK),("Files Analyzed","47","PCAP+CSV",INFO)]):
            g.addWidget(StatCard(["📋","⚠","✓","📁"][i],t,v,s,c),0,i)
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
            for j,v in enumerate([rid,fn,dt,bn,ac]):
                it=QTableWidgetItem(v); it.setForeground(QColor(TW))
                if j==3 and int(bn)>0: it.setForeground(QColor(ERR)); it.setFont(QFont(FNT,11,QFont.Weight.Bold))
                if j==0: it.setFont(QFont("Courier New",11)); it.setForeground(QColor(ACC))
                t.setItem(i,j,it)
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
            (("Output Directory","Where exported reports are saved"),mk_combo(["Desktop/botnet_reports","Documents/reports"])),
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
        self.setWindowTitle("BotDetect — AI Security Suite  ·  Group 07")
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
    win=MainWindow(); win.show(); sys.exit(app.exec())