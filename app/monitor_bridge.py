"""
monitor_bridge.py — Qt thread wrapper around the two-stage BotnetMonitor.

This is the integration layer between monitoring.py (CLI pipeline) and
the PyQt GUI. The design avoids two macOS-specific pitfalls:

  1. PyTorch's libomp/MKL conflicts with Qt's threading runtime, causing
     torch.load() to SIGSEGV when called from a non-main thread. We solve
     this by constructing BotnetMonitor (which calls torch.load() three
     times) in the MAIN thread via ensure_monitor(), then handing the
     pre-loaded instance to the QThread for sniff/inference only.

  2. The construction is LAZY — triggered by the first Start click — so
     app startup doesn't block for 3-5s while three models are loaded.

Signal contract (matches LiveCaptureThread for drop-in compatibility):
    flow_ready(dict) — fired for every completed flow's DetectionResult
    stats(dict)      — periodic {flows_per_sec, bandwidth_kbps}
    error(str)       — capture/init failures shown in the page status bar
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# ── Locate monitoring.py at the project root and add it to sys.path ──────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force torch single-threaded BEFORE the monitoring import pulls torch in.
# This is defence-in-depth on top of the OMP_NUM_THREADS env var set in
# mockApp.py — covers the case where this module is loaded standalone for
# unit tests with no shell preamble.
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Lazy/optional imports — never crash the GUI if dependencies are missing.
try:
    from monitoring import BotnetMonitor  # noqa: E402
    MONITOR_AVAILABLE = True
    _IMPORT_ERROR: str = ""
except Exception as _e:
    MONITOR_AVAILABLE = False
    _IMPORT_ERROR = repr(_e)

try:
    from scapy.all import sniff, Ether, IP, TCP, UDP, ICMP, conf as scapy_conf
    scapy_conf.verb = 0
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

from PyQt6.QtCore import QThread, pyqtSignal


# ── Constants ────────────────────────────────────────────────────────────────
SNIFF_TIMEOUT_SEC = 1.0
IDLE_FLUSH_EVERY  = 5.0
STATS_EMIT_EVERY  = 1.0


# ═════════════════════════════════════════════════════════════════════════════
# BotnetMonitorThread
# ═════════════════════════════════════════════════════════════════════════════

class BotnetMonitorThread(QThread):
    """
    QThread wrapping the full two-stage BotnetMonitor pipeline.

    Lifecycle:
        __init__()          — cheap; only stores config. Does NOT touch torch.
        ensure_monitor()    — constructs BotnetMonitor in the CALLING thread.
                              MUST be called from the main thread (i.e. from
                              MonitorPage.start_capture()) before start().
        start() / run()     — runs the sniff loop using the pre-loaded monitor.
        stop()              — signals the loop to exit within ~1s.
    """

    flow_ready = pyqtSignal(dict)
    stats      = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, interface: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.interface = interface
        self._running       = False
        self._monitor       = None        # filled by ensure_monitor()
        self._init_error    = None
        # Rolling windows for flows/sec & bandwidth_kbps
        self._byte_window: deque = deque(maxlen=4096)
        self._flow_window: deque = deque(maxlen=4096)
        self._last_stats_emit: float = 0.0
        self._last_idle_flush: float = 0.0

    # ── Main-thread construction (called from MonitorPage.start_capture) ────
    def ensure_monitor(self) -> bool:
        """
        Construct BotnetMonitor in the CALLING thread (main thread).
        Idempotent. Returns True on success, False with self._init_error set.
        """
        if self._monitor is not None:
            return True
        if self._init_error is not None:
            return False
        if not MONITOR_AVAILABLE:
            self._init_error = (
                f"monitoring.py not importable: {_IMPORT_ERROR}. "
                "Verify models/stage1/rf_model.pkl, models/stage2/iot_cnn_lstm.pt, "
                "models/stage2/noniot_cnn_lstm.pt are present."
            )
            return False
        if not SCAPY_AVAILABLE:
            self._init_error = "scapy not installed — pip install scapy"
            return False
        try:
            self._monitor = BotnetMonitor()
            return True
        except FileNotFoundError as e:
            self._init_error = f"BotnetMonitor: missing artifact — {e}"
        except Exception as e:
            self._init_error = f"BotnetMonitor init failed: {e!r}"
        return False

    # ── Public control ───────────────────────────────────────────────────────
    def stop(self) -> None:
        self._running = False

    # ── QThread entry point ──────────────────────────────────────────────────
    def run(self) -> None:
        if self._monitor is None:
            self.error.emit(
                self._init_error
                or "BotnetMonitor not initialised — call ensure_monitor() first."
            )
            return

        self._running         = True
        self._last_stats_emit = time.time()
        self._last_idle_flush = time.time()

        try:
            while self._running:
                try:
                    sniff(
                        iface       = self.interface,
                        prn         = self._on_packet,
                        store       = False,
                        timeout     = SNIFF_TIMEOUT_SEC,
                        promisc     = False,
                        filter      = "ip",
                        stop_filter = lambda _p: not self._running,
                    )
                except PermissionError:
                    self.error.emit(
                        "Permission denied — run as root (macOS/Linux) or "
                        "Administrator (Windows) to capture live packets."
                    )
                    return
                except OSError as e:
                    self.error.emit(f"Interface error: {e}")
                    return
                except Exception as e:
                    self.error.emit(f"sniff error: {e}")
                    return

                now = time.time()

                if now - self._last_idle_flush >= IDLE_FLUSH_EVERY:
                    try:
                        for r in self._monitor.flush_idle_flows():
                            self._emit_result(r)
                    except Exception as e:
                        self.error.emit(f"flush_idle_flows error: {e}")
                    self._last_idle_flush = now

                if now - self._last_stats_emit >= STATS_EMIT_EVERY:
                    self._emit_stats(now)
                    self._last_stats_emit = now
        finally:
            try:
                if self._monitor is not None:
                    for r in self._monitor.flush_idle_flows():
                        self._emit_result(r)
                    self._monitor.print_summary(final=True)
            except Exception:
                pass

    # ── Scapy callback (runs in this QThread) ───────────────────────────────
    def _on_packet(self, pkt) -> None:
        if IP not in pkt:
            return
        ip   = pkt[IP]
        ts   = float(pkt.time)
        mac  = pkt[Ether].src if Ether in pkt else ip.src
        plen = len(pkt)

        self._byte_window.append((ts, plen))

        try:
            if TCP in pkt:
                tcp = pkt[TCP]
                result = self._monitor.process_packet(
                    ts, ip.src, ip.dst, int(tcp.sport), int(tcp.dport),
                    6, plen, int(ip.ttl), mac, int(tcp.flags),
                )
            elif UDP in pkt:
                udp = pkt[UDP]
                result = self._monitor.process_packet(
                    ts, ip.src, ip.dst, int(udp.sport), int(udp.dport),
                    17, plen, int(ip.ttl), mac, 0,
                )
            elif ICMP in pkt:
                result = self._monitor.process_packet(
                    ts, ip.src, ip.dst, 0, 0,
                    1, plen, int(ip.ttl), mac, 0,
                )
            else:
                return
        except Exception as e:
            self.error.emit(f"process_packet error: {e}")
            return

        if result is not None:
            self._emit_result(result)

    # ── Translate DetectionResult → dict for MonitorPage ────────────────────
    def _emit_result(self, r) -> None:
        self._flow_window.append(time.time())

        src_port = dst_port = 0
        proto    = "—"
        try:
            left, _, rest = r.flow_id.partition("<->")
            right, _, p   = rest.partition("/")
            _, _, sp = left.rpartition(":")
            _, _, dp = right.rpartition(":")
            src_port = int(sp)
            dst_port = int(dp)
            proto    = {"6": "TCP", "17": "UDP", "1": "ICMP"}.get(p, p)
        except Exception:
            pass

        self.flow_ready.emit({
            "src_ip":   r.src_ip,
            "dst_ip":   r.dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
            "total_fwd_bytes":    0, "total_bwd_bytes":   0,
            "total_fwd_packets":  0, "total_bwd_packets": 0,
            "flow_duration":      0.0,
            "flow_pkts_per_sec":  0.0,
            "flow_bytes_per_sec": 0.0,
            "flag_SYN": 0, "flag_ACK": 0, "flag_FIN": 0, "flag_RST": 0,
            "_label":         r.label,
            "_botnet_prob":   float(r.s2_confidence),
            "_alert":         bool(r.alerted),
            "_device_type":   r.device_type,
            "_s1_confidence": float(r.s1_confidence),
            "_latency_ms":    float(r.latency_ms),
            "_suspicion":     float(r.suspicion_score),
        })

    def _emit_stats(self, now: float) -> None:
        recent_flows = sum(1 for t in self._flow_window if now - t <= 1.0)
        recent_bytes = sum(b for t, b in self._byte_window if now - t <= 1.0)
        self.stats.emit({
            "flows_per_sec":  recent_flows,
            "bandwidth_kbps": int(recent_bytes / 1024),
        })