"""
live_capture.py  —  Real-time packet capture thread
=====================================================
Captures live packets via Scapy, aggregates them into flow records
(same format as pcap_parser.aggregate_packets_to_flows), then emits
each flow as a Qt signal so MonitorPage can consume it safely.

Architecture:
    LiveCaptureThread (QThread)
        ├── sniff() runs in background thread
        ├── Every EMIT_INTERVAL seconds, aggregates buffered packets → flows
        └── Emits flow_ready(dict) signal → MonitorPage._on_flow(dict)

Modes:
    DETECTOR — full CNN-LSTM pipeline via LiveDetector (requires model_path +
               scapy + root). Emits a flow dict on every botnet alert.
               Also emits periodic benign-status flows every EMIT_INTERVAL.
    LIVE     — plain scapy capture with packet→flow aggregation (no ML).
               Used when scapy is available but no model_path is provided.
    DEMO     — realistic simulated flows. Falls back automatically if scapy
               is unavailable or no usable interface is found.

Usage:
    # Demo mode (no args)
    thread = LiveCaptureThread()

    # Live mode (plain scapy, no model)
    thread = LiveCaptureThread(interface="eth0")

    # Detector mode (full CNN-LSTM pipeline)
    thread = LiveCaptureThread(
        interface  = "en0",
        model_path = "models/stage2/iot_cnn_lstm.pt",
    )

    thread.flow_ready.connect(my_slot)
    thread.error.connect(lambda msg: print(msg))
    thread.stats.connect(lambda s: print(s))
    thread.start()
    ...
    thread.stop()
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from collections import defaultdict
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, QTimer

# ── Scapy import (graceful fallback to demo mode) ─────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, conf as scapy_conf, get_if_list, get_if_addr
    scapy_conf.verb = 0          # silence scapy noise
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ── LiveDetector import (graceful fallback to plain-scapy / demo mode) ────────
try:
    from src.live.live_detector import LiveDetector
    LIVE_DETECTOR_AVAILABLE = True
except ImportError:
    LIVE_DETECTOR_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
EMIT_INTERVAL  = 2.0    # seconds between flow emissions in live/demo mode
PACKET_BUFFER  = 50     # max packets to buffer before forcing a flush
PREF_IFACES    = ("wi-fi", "wlan", "en0", "en1", "eth0", "ethernet")


# ══════════════════════════════════════════════════════════════════════════════
# Interface helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_interfaces() -> list[tuple[str, str]]:
    """
    Return a list of (interface_name, ip_address) tuples for all
    non-loopback interfaces that have a real IP assigned.

    Used by MonitorPage to populate the interface selector dropdown.
    Returns [] if Scapy is not installed.

    Example:
        [("Wi-Fi", "192.168.1.10"), ("Ethernet", "10.0.0.5")]
    """
    if not SCAPY_AVAILABLE:
        return []
    result = []
    for iface in get_if_list():
        try:
            ip = get_if_addr(iface)
        except Exception:
            ip = "0.0.0.0"
        is_loopback = any(x in iface.lower() for x in ["lo", "loopback", "npcap loopback"])
        if ip and ip != "0.0.0.0" and not is_loopback:
            result.append((iface, ip))
    return result


def auto_select_interface() -> Optional[str]:
    """
    Pick the 'best' interface automatically:
      - Prefers Wi-Fi / wlan / en0 / Ethernet names
      - Falls back to first non-loopback interface with a real IP
    Returns None if nothing suitable is found.
    """
    if not SCAPY_AVAILABLE:
        return None
    available = get_interfaces()
    if not available:
        return None
    for name, _ in available:
        if any(p in name.lower() for p in PREF_IFACES):
            return name
    return available[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# Demo flows (used when scapy / model unavailable)
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_FLOWS: list[dict] = [
    {"src_ip":"192.168.1.5",  "dst_ip":"8.8.8.8",           "src_port":53012,"dst_port":53,   "protocol":"UDP",   "total_fwd_bytes":150,   "total_bwd_bytes":220,  "flow_duration":0.08, "flow_pkts_per_sec":25,  "flow_bytes_per_sec":4625,  "flag_SYN":0,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":1, "total_bwd_packets":1},
    {"src_ip":"10.0.0.15",    "dst_ip":"10.0.0.1",           "src_port":41000,"dst_port":22,   "protocol":"TCP",   "total_fwd_bytes":3200,  "total_bwd_bytes":4800, "flow_duration":12.0, "flow_pkts_per_sec":10,  "flow_bytes_per_sec":667,   "flag_SYN":1,"flag_ACK":1,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":30,"total_bwd_packets":45},
    {"src_ip":"192.168.1.12", "dst_ip":"185.100.87.41",      "src_port":60012,"dst_port":6667, "protocol":"TCP",   "total_fwd_bytes":2048,  "total_bwd_bytes":512,  "flow_duration":0.4,  "flow_pkts_per_sec":300, "flow_bytes_per_sec":6400,  "flag_SYN":1,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":8, "total_bwd_packets":2},
    {"src_ip":"10.0.0.22",    "dst_ip":"216.58.208.14",      "src_port":49100,"dst_port":443,  "protocol":"HTTPS", "total_fwd_bytes":4200,  "total_bwd_bytes":31000,"flow_duration":3.5,  "flow_pkts_per_sec":22,  "flow_bytes_per_sec":10057, "flag_SYN":1,"flag_ACK":1,"flag_FIN":1,"flag_RST":0,"total_fwd_packets":15,"total_bwd_packets":50},
]


# ══════════════════════════════════════════════════════════════════════════════
# LiveCaptureThread
# ══════════════════════════════════════════════════════════════════════════════

class LiveCaptureThread(QThread):
    """
    Background thread that captures packets and emits flow dicts.

    Signals:
        flow_ready(dict)  — one flow record ready for processing
        error(str)        — something went wrong (shown in status bar)
        stats(dict)       — periodic bandwidth/pps stats for the stat strip
    """

    flow_ready = pyqtSignal(dict)
    error      = pyqtSignal(str)
    stats      = pyqtSignal(dict)

    def __init__(
        self,
        interface:     Optional[str] = None,
        model_path:    Optional[str] = "models/stage2/iot_cnn_lstm.pt",   # path to iot_cnn_lstm.pt
        demo_mode:     bool          = False,
        emit_interval: float         = EMIT_INTERVAL,
    ) -> None:
        super().__init__()

        # If scapy is missing, force demo mode regardless of what caller asked
        if not SCAPY_AVAILABLE:
            demo_mode = True

        # Auto-detect best interface when not explicitly provided
        if not demo_mode and interface is None:
            interface = auto_select_interface()
            if interface is None:
                demo_mode = True   # no usable interface found

        self.interface     = interface
        self.model_path    = model_path
        self.demo_mode     = demo_mode
        self.emit_interval = emit_interval

        self._running      = False
        self._detector: Optional[LiveDetector] = None   # set in _run_detector
        self._pkt_buffer: list[dict] = []
        self._last_emit    = time.monotonic()
        self._total_flows  = 0
        self._total_pkts   = 0

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        if self.demo_mode:
            self._run_demo()
        elif self.model_path and LIVE_DETECTOR_AVAILABLE:
            self._run_detector()
        else:
            self._run_live()

    def stop(self) -> None:
        """Signal all capture loops to halt."""
        self._running = False
        if self._detector is not None:
            self._detector.stop()

    # ── Detector mode (LiveDetector → CNN-LSTM inference) ─────────────────────

    def _run_detector(self) -> None:
        """
        Delegate packet capture and inference entirely to LiveDetector.

        LiveDetector.start() is blocking; it runs Scapy's sniff() loop in
        this thread and calls on_alert() (and optionally on_packet()) from
        within that same thread.  Qt signals are thread-safe, so emitting
        from here is fine.

        Flow dict emitted on every BOTNET alert:
            _label          : "botnet"
            _botnet_prob    : float confidence (0-1)
            _alert          : True
            src_ip          : IP of suspected device
            + zero-filled flow statistics (no aggregation done in detector mode)

        A periodic benign status ping is emitted every EMIT_INTERVAL seconds
        so MonitorPage stat counters keep ticking even during quiet periods.
        """

        def _on_alert(src_ip: str, prob: float, t: float) -> None:
            """Called by LiveDetector for every botnet-confidence packet."""
            flow: dict = {
                # ── Identity ──────────────────────────────────────────────
                "src_ip":             src_ip,
                "dst_ip":             "unknown",
                "src_port":           0,
                "dst_port":           0,
                "protocol":           "unknown",
                # ── Flow statistics (unavailable at alert time) ────────────
                "total_fwd_bytes":    0,
                "total_bwd_bytes":    0,
                "total_fwd_packets":  0,
                "total_bwd_packets":  0,
                "flow_duration":      0.0,
                "flow_pkts_per_sec":  0.0,
                "flow_bytes_per_sec": 0.0,
                "flag_SYN": 0, "flag_ACK": 0,
                "flag_FIN": 0, "flag_RST": 0,
                # ── Detection result ──────────────────────────────────────
                "_label":             "botnet",
                "_botnet_prob":       round(prob, 4),
                "_alert":             True,
            }
            self.flow_ready.emit(flow)
            self._total_flows += 1
            self.stats.emit({
                "flows_per_sec":  1,
                "bandwidth_kbps": 0,
                "total_flows":    self._total_flows,
            })

        def _on_packet_status(detector: "LiveDetector") -> None:
            """
            Emit a lightweight benign ping every EMIT_INTERVAL seconds so
            MonitorPage doesn't appear frozen between alerts.
            Called from a QTimer in the main thread — but since we're in a
            QThread, we use a plain time-based loop instead.
            """
            pass  # handled below in the polling loop

        try:
            self._detector = LiveDetector(
                model_path = self.model_path,
                interface  = self.interface,
                on_alert   = _on_alert,
                verbose    = True,
            )
        except Exception as e:
            self.error.emit(f"LiveDetector init failed: {e}")
            return

        # Run in a separate daemon thread so we can do our own polling loop
        import threading
        capture_thread = threading.Thread(
            target=self._detector.start,
            daemon=True,
        )
        capture_thread.start()

        # ── Polling loop: emit status pings + watch for stop signal ──────────
        while self._running and capture_thread.is_alive():
            time.sleep(self.emit_interval)

            status = self._detector.get_status()
            pkts   = status.get("packets", 0)

            # Emit a benign heartbeat flow so the UI stat strip keeps updating
            heartbeat: dict = {
                "src_ip":             "—",
                "dst_ip":             "—",
                "src_port":           0,
                "dst_port":           0,
                "protocol":           "—",
                "total_fwd_bytes":    0,
                "total_bwd_bytes":    0,
                "total_fwd_packets":  0,
                "total_bwd_packets":  0,
                "flow_duration":      0.0,
                "flow_pkts_per_sec":  0.0,
                "flow_bytes_per_sec": 0.0,
                "flag_SYN": 0, "flag_ACK": 0,
                "flag_FIN": 0, "flag_RST": 0,
                "_label":             "benign",
                "_botnet_prob":       0.0,
                "_alert":             False,
                "_heartbeat":         True,
                "_packets_seen":      pkts,
            }
            self.flow_ready.emit(heartbeat)
            self.stats.emit({
                "flows_per_sec":  0,
                "bandwidth_kbps": 0,
                "total_flows":    self._total_flows,
                "packets_seen":   pkts,
                "devices":        status.get("devices", 0),
                "alerts":         status.get("alerts", 0),
            })

        # Ensure capture thread is stopped
        if self._detector:
            self._detector.stop()
        capture_thread.join(timeout=3.0)

    # ── Live mode (plain scapy — no model) ────────────────────────────────────

    def _run_live(self) -> None:
        """
        Sniff real packets on self.interface.
        Packets are buffered and flushed as aggregated flow dicts every
        EMIT_INTERVAL seconds (or when PACKET_BUFFER is reached).

        Requires: pip install scapy  +  root/admin privileges.
        """
        try:
            sniff(
                iface       = self.interface,
                prn         = self._handle_packet,
                store       = False,
                stop_filter = lambda _: not self._running,
            )
        except PermissionError:
            self.error.emit(
                "Permission denied — run as root/admin to capture live packets."
            )
        except OSError as e:
            self.error.emit(f"Interface error: {e}")
        except Exception as e:
            self.error.emit(f"Capture error: {e}")

    def _handle_packet(self, pkt) -> None:
        """Scapy callback — called for every captured packet in live mode."""
        if IP not in pkt:
            return

        rec: dict = {
            "frame.time_epoch": str(float(pkt.time)),
            "ip.src":           pkt[IP].src,
            "ip.dst":           pkt[IP].dst,
            "ip.proto":         str(pkt[IP].proto),
            "frame.len":        str(len(pkt)),
            "ip.ttl":           str(pkt[IP].ttl),
        }
        if TCP in pkt:
            rec["tcp.srcport"] = str(pkt[TCP].sport)
            rec["tcp.dstport"] = str(pkt[TCP].dport)
            rec["tcp.flags"]   = str(int(pkt[TCP].flags))
        elif UDP in pkt:
            rec["udp.srcport"] = str(pkt[UDP].sport)
            rec["udp.dstport"] = str(pkt[UDP].dport)

        self._pkt_buffer.append(rec)
        self._total_pkts += 1

        now = time.monotonic()
        if (len(self._pkt_buffer) >= PACKET_BUFFER or
                now - self._last_emit >= self.emit_interval):
            self._flush_buffer()
            self._last_emit = now

    def _flush_buffer(self) -> None:
        """Aggregate buffered packets into flow dicts and emit each one."""
        if not self._pkt_buffer:
            return

        flows = _aggregate_packets_to_flows(self._pkt_buffer)
        self._pkt_buffer.clear()

        total_bps = 0.0
        for flow in flows:
            self.flow_ready.emit(flow)
            self._total_flows += 1
            total_bps += float(flow.get("flow_bytes_per_sec", 0))

        self.stats.emit({
            "flows_per_sec":  len(flows),
            "bandwidth_kbps": int(total_bps / 1024),
            "total_flows":    self._total_flows,
        })

    # ── Demo mode ─────────────────────────────────────────────────────────────

    def _run_demo(self) -> None:
        """
        Emit realistic-looking flows at ~EMIT_INTERVAL seconds.
        Used during development and when scapy is unavailable.
        """
        while self._running:
            batch_size = random.randint(1, 3)
            for _ in range(batch_size):
                flow = dict(random.choice(_DEMO_FLOWS))
                # Add minor jitter so each emission looks unique
                flow["total_fwd_bytes"]    = int(flow["total_fwd_bytes"] * random.uniform(0.85, 1.20))
                flow["flow_pkts_per_sec"]  = flow["flow_pkts_per_sec"]  * random.uniform(0.8, 1.3)
                flow["flow_bytes_per_sec"] = flow["flow_bytes_per_sec"] * random.uniform(0.8, 1.3)
                flow["flow_duration"]      = max(0.01, flow["flow_duration"] * random.uniform(0.9, 1.1))
                self.flow_ready.emit(flow)
                self._total_flows += 1

            self.stats.emit({
                "flows_per_sec":  batch_size,
                "bandwidth_kbps": random.randint(80, 600),
                "total_flows":    self._total_flows,
            })
            time.sleep(self.emit_interval)


# ── Packet → flow aggregation (mirrors pcap_parser logic) ─────────────────────

# REPLACEMENT for the _aggregate_packets_to_flows() function in app/live_capture.py
# (everything from `def _aggregate_packets_to_flows` to end of file)

def _aggregate_packets_to_flows(packets: list[dict]) -> list[dict]:
    """
    Group raw packet dicts into bidirectional 5-tuple flows.

    Phase-A enrichment vs. the original version:
      Old: 13 fields (basic counters + 4 boolean flags).
      New: ~30 fields including per-direction packet-length stats (min/max/mean/std),
           inter-arrival-time stats per direction (mean/std/min/max), header lengths,
           and integer protocol number compatible with Stage-1's `protocol` feature.

    Field-name conventions match the unified 56-feature schema in
    models/stage1/classifier.py → ALL_FEATURES, so the bridge's _align() can map
    them directly.
    """
    import time
    import statistics
    from collections import defaultdict

    # ── Pass 1: group packets per 5-tuple, accumulate per-direction lists ──────
    raw: dict[tuple, dict] = {}

    for pkt in packets:
        src       = pkt.get("ip.src", "")
        dst       = pkt.get("ip.dst", "")
        sp        = int(pkt.get("tcp.srcport") or pkt.get("udp.srcport") or 0)
        dp        = int(pkt.get("tcp.dstport") or pkt.get("udp.dstport") or 0)
        proto_num = int(pkt.get("ip.proto", 0))

        # Canonical 5-tuple key, sorted so both directions share one entry.
        key = tuple(sorted([(src, sp), (dst, dp)])) + (proto_num,)

        size  = int(pkt.get("frame.len", 0))
        ts    = float(pkt.get("frame.time_epoch", time.time()))
        flags = int(pkt.get("tcp.flags", 0))

        # Approximate header length: scapy gives full frame.len; we don't have
        # the parsed L3+L4 header sizes here, so use a typical value (Ethernet 14
        # + IPv4 20 + TCP 20 = 54). This is a coarse but bounded estimate.
        hdr_len = 54

        if key not in raw:
            raw[key] = {
                "src_ip": src, "dst_ip": dst,
                "src_port": sp, "dst_port": dp,
                "protocol_num": proto_num,
                "fwd_sizes": [],   "bwd_sizes": [],
                "fwd_times": [],   "bwd_times": [],
                "all_times": [],
                "fwd_hdr_total": 0, "bwd_hdr_total": 0,
                "flag_FIN_count": 0, "flag_SYN_count": 0,
                "flag_RST_count": 0, "flag_PSH_count": 0,
                "flag_ACK_count": 0, "flag_URG_count": 0,
            }

        f = raw[key]
        is_fwd = (src == f["src_ip"])
        if is_fwd:
            f["fwd_sizes"].append(size)
            f["fwd_times"].append(ts)
            f["fwd_hdr_total"] += hdr_len
        else:
            f["bwd_sizes"].append(size)
            f["bwd_times"].append(ts)
            f["bwd_hdr_total"] += hdr_len
        f["all_times"].append(ts)

        if flags & 0x01: f["flag_FIN_count"] += 1
        if flags & 0x02: f["flag_SYN_count"] += 1
        if flags & 0x04: f["flag_RST_count"] += 1
        if flags & 0x08: f["flag_PSH_count"] += 1
        if flags & 0x10: f["flag_ACK_count"] += 1
        if flags & 0x20: f["flag_URG_count"] += 1

    # ── Pass 2: derive features per flow ───────────────────────────────────────
    def _stats(xs: list[float]) -> tuple[float, float, float, float]:
        """Return (min, max, mean, std). Empty → all zeros."""
        if not xs:
            return 0.0, 0.0, 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), float(xs[0]), float(xs[0]), 0.0
        return (float(min(xs)), float(max(xs)),
                float(statistics.mean(xs)), float(statistics.pstdev(xs)))

    def _iat_stats(times: list[float]) -> tuple[float, float, float, float]:
        """Inter-arrival times → (mean, std, min, max). <2 packets → zeros."""
        if len(times) < 2:
            return 0.0, 0.0, 0.0, 0.0
        sorted_t = sorted(times)
        iats = [sorted_t[i] - sorted_t[i - 1] for i in range(1, len(sorted_t))]
        return (float(statistics.mean(iats)),
                float(statistics.pstdev(iats)) if len(iats) > 1 else 0.0,
                float(min(iats)),
                float(max(iats)))

    result: list[dict] = []
    for f in raw.values():
        n_fwd      = len(f["fwd_sizes"])
        n_bwd      = len(f["bwd_sizes"])
        n_total    = n_fwd + n_bwd
        bytes_fwd  = sum(f["fwd_sizes"])
        bytes_bwd  = sum(f["bwd_sizes"])
        all_times  = f["all_times"]

        first_ts = min(all_times) if all_times else 0.0
        last_ts  = max(all_times) if all_times else 0.0
        duration = max(last_ts - first_ts, 1e-6)

        fwd_min, fwd_max, fwd_mean, fwd_std = _stats(f["fwd_sizes"])
        bwd_min, bwd_max, bwd_mean, bwd_std = _stats(f["bwd_sizes"])

        flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max = _iat_stats(all_times)
        fwd_iat_mean,  fwd_iat_std,  fwd_iat_min,  fwd_iat_max  = _iat_stats(f["fwd_times"])
        bwd_iat_mean,  bwd_iat_std,  bwd_iat_min,  bwd_iat_max  = _iat_stats(f["bwd_times"])

        result.append({
            # ── Identity ────────────────────────────────────────────────────────
            "src_ip":   f["src_ip"],
            "dst_ip":   f["dst_ip"],
            "src_port": f["src_port"],
            "dst_port": f["dst_port"],
            "protocol": f["protocol_num"],   # NUMERIC (was string in v1) ← Phase-A fix

            # ── Flow-level totals & rates (5 features) ──────────────────────────
            "flow_duration":      round(duration, 6),
            "total_fwd_packets":  n_fwd,
            "total_bwd_packets":  n_bwd,
            "total_fwd_bytes":    bytes_fwd,
            "total_bwd_bytes":    bytes_bwd,

            # ── Per-direction packet-length stats (8 features) ─ NEW in Phase A ─
            "fwd_pkt_len_min":  fwd_min,
            "fwd_pkt_len_max":  fwd_max,
            "fwd_pkt_len_mean": fwd_mean,
            "fwd_pkt_len_std":  fwd_std,
            "bwd_pkt_len_min":  bwd_min,
            "bwd_pkt_len_max":  bwd_max,
            "bwd_pkt_len_mean": bwd_mean,
            "bwd_pkt_len_std":  bwd_std,

            # ── Flow rates (2 features) ─────────────────────────────────────────
            "flow_bytes_per_sec": round((bytes_fwd + bytes_bwd) / duration, 2),
            "flow_pkts_per_sec":  round(n_total / duration, 2),

            # ── IAT stats (12 features) ────────────────────────── NEW in Phase A
            "flow_iat_mean": flow_iat_mean, "flow_iat_std": flow_iat_std,
            "flow_iat_min":  flow_iat_min,  "flow_iat_max":  flow_iat_max,
            "fwd_iat_mean":  fwd_iat_mean,  "fwd_iat_std":   fwd_iat_std,
            "fwd_iat_min":   fwd_iat_min,   "fwd_iat_max":   fwd_iat_max,
            "bwd_iat_mean":  bwd_iat_mean,  "bwd_iat_std":   bwd_iat_std,
            "bwd_iat_min":   bwd_iat_min,   "bwd_iat_max":   bwd_iat_max,

            # ── Header lengths (2 features) ─────────────────────── NEW in Phase A
            "fwd_header_length": f["fwd_hdr_total"],
            "bwd_header_length": f["bwd_hdr_total"],

            # ── TCP flag counts (6 features) ──────── NEW: counts (was 0/1 in v1)
            "flag_FIN": f["flag_FIN_count"],
            "flag_SYN": f["flag_SYN_count"],
            "flag_RST": f["flag_RST_count"],
            "flag_PSH": f["flag_PSH_count"],
            "flag_ACK": f["flag_ACK_count"],
            "flag_URG": f["flag_URG_count"],
        })

    return result
