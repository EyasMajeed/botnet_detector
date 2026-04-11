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
    LIVE  — real network interface (requires root/admin + scapy installed)
    DEMO  — realistic simulated flows using DEMO_FLOWS from mockApp
            Falls back automatically if scapy is not available.

Usage:
    thread = LiveCaptureThread(interface="eth0")
    thread.flow_ready.connect(my_slot)
    thread.error.connect(lambda msg: print(msg))
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
    ifaces = get_interfaces()
    if not ifaces:
        return None
    preferred = ["wi-fi", "wlan", "en0", "eth0", "ethernet", "wlp", "ens"]
    for name, _ip in ifaces:
        if any(p in name.lower() for p in preferred):
            return name
    return ifaces[0][0]   # fallback: first available

# ── Tuning ────────────────────────────────────────────────────────────────────
EMIT_INTERVAL   = 1.5    # seconds between flow-batch emissions
PACKET_BUFFER   = 30     # aggregate every N packets (whichever comes first)
MAX_TABLE_ROWS  = 200    # used by caller; stored here as a single config

# ── Demo traffic (fallback when scapy unavailable / demo mode) ────────────────
_DEMO_FLOWS: list[dict] = [
    {"src_ip":"192.168.1.5",  "dst_ip":"91.108.4.15",   "src_port":51234,"dst_port":4444,  "protocol":"TCP",   "total_fwd_bytes":14200, "total_bwd_bytes":800,  "flow_duration":5.3,  "flow_pkts_per_sec":48,  "flow_bytes_per_sec":2830,  "flag_SYN":1,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":48,"total_bwd_packets":12},
    {"src_ip":"10.0.0.10",    "dst_ip":"8.8.8.8",        "src_port":52100,"dst_port":53,    "protocol":"UDP",   "total_fwd_bytes":320,   "total_bwd_bytes":480,  "flow_duration":0.1,  "flow_pkts_per_sec":20,  "flow_bytes_per_sec":8000,  "flag_SYN":0,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":2, "total_bwd_packets":2},
    {"src_ip":"192.168.1.7",  "dst_ip":"185.220.101.5",  "src_port":50021,"dst_port":9999,  "protocol":"TCP",   "total_fwd_bytes":18920, "total_bwd_bytes":1200, "flow_duration":6.4,  "flow_pkts_per_sec":61,  "flow_bytes_per_sec":3145,  "flag_SYN":1,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":61,"total_bwd_packets":18},
    {"src_ip":"10.0.0.8",     "dst_ip":"142.250.80.46",  "src_port":49823,"dst_port":443,   "protocol":"HTTPS", "total_fwd_bytes":7640,  "total_bwd_bytes":22400,"flow_duration":2.1,  "flow_pkts_per_sec":30,  "flow_bytes_per_sec":14304, "flag_SYN":1,"flag_ACK":1,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":20,"total_bwd_packets":40},
    {"src_ip":"10.0.0.3",     "dst_ip":"172.16.0.1",     "src_port":44012,"dst_port":80,    "protocol":"TCP",   "total_fwd_bytes":5200,  "total_bwd_bytes":18000,"flow_duration":1.2,  "flow_pkts_per_sec":25,  "flow_bytes_per_sec":19333, "flag_SYN":1,"flag_ACK":1,"flag_FIN":1,"flag_RST":0,"total_fwd_packets":18,"total_bwd_packets":32},
    {"src_ip":"192.168.1.5",  "dst_ip":"91.108.4.15",    "src_port":51300,"dst_port":4444,  "protocol":"TCP",   "total_fwd_bytes":9180,  "total_bwd_bytes":600,  "flow_duration":3.1,  "flow_pkts_per_sec":52,  "flow_bytes_per_sec":3155,  "flag_SYN":1,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":52,"total_bwd_packets":10},
    {"src_ip":"10.0.0.2",     "dst_ip":"8.8.4.4",        "src_port":53012,"dst_port":53,    "protocol":"UDP",   "total_fwd_bytes":150,   "total_bwd_bytes":220,  "flow_duration":0.08, "flow_pkts_per_sec":25,  "flow_bytes_per_sec":4625,  "flag_SYN":0,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":1, "total_bwd_packets":1},
    {"src_ip":"10.0.0.15",    "dst_ip":"10.0.0.1",       "src_port":41000,"dst_port":22,    "protocol":"TCP",   "total_fwd_bytes":3200,  "total_bwd_bytes":4800, "flow_duration":12.0, "flow_pkts_per_sec":10,  "flow_bytes_per_sec":667,   "flag_SYN":1,"flag_ACK":1,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":30,"total_bwd_packets":45},
    {"src_ip":"192.168.1.12", "dst_ip":"185.100.87.41",  "src_port":60012,"dst_port":6667,  "protocol":"TCP",   "total_fwd_bytes":2048,  "total_bwd_bytes":512,  "flow_duration":0.4,  "flow_pkts_per_sec":300, "flow_bytes_per_sec":6400,  "flag_SYN":1,"flag_ACK":0,"flag_FIN":0,"flag_RST":0,"total_fwd_packets":8, "total_bwd_packets":2},
    {"src_ip":"10.0.0.22",    "dst_ip":"216.58.208.14",  "src_port":49100,"dst_port":443,   "protocol":"HTTPS", "total_fwd_bytes":4200,  "total_bwd_bytes":31000,"flow_duration":3.5,  "flow_pkts_per_sec":22,  "flow_bytes_per_sec":10057, "flag_SYN":1,"flag_ACK":1,"flag_FIN":1,"flag_RST":0,"total_fwd_packets":15,"total_bwd_packets":50},
]


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
        self.demo_mode     = demo_mode
        self.emit_interval = emit_interval

        self._running     = False
        self._pkt_buffer: list[dict] = []
        self._last_emit   = time.monotonic()
        self._total_flows = 0
        self._total_pkts  = 0

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        if self.demo_mode:
            self._run_demo()
        else:
            self._run_live()

    def stop(self) -> None:
        self._running = False

    # ── Demo mode ─────────────────────────────────────────────────────────────

    def _run_demo(self) -> None:
        """
        Emit realistic-looking flows at ~EMIT_INTERVAL seconds.
        Used during development and when scapy is unavailable.
        """
        while self._running:
            # Pick 1-3 random demo flows and add minor jitter so they look live
            batch_size = random.randint(1, 3)
            for _ in range(batch_size):
                flow = dict(random.choice(_DEMO_FLOWS))
                # Jitter values so each emission looks unique
                flow["total_fwd_bytes"]   = int(flow["total_fwd_bytes"] * random.uniform(0.85, 1.20))
                flow["flow_pkts_per_sec"] = flow["flow_pkts_per_sec"] * random.uniform(0.8, 1.3)
                flow["flow_bytes_per_sec"]= flow["flow_bytes_per_sec"] * random.uniform(0.8, 1.3)
                flow["flow_duration"]     = max(0.01, flow["flow_duration"] * random.uniform(0.9, 1.1))
                self.flow_ready.emit(flow)
                self._total_flows += 1

            self.stats.emit({
                "flows_per_sec": batch_size,
                "bandwidth_kbps": random.randint(80, 600),
                "total_flows": self._total_flows,
            })
            time.sleep(self.emit_interval)

    # ── Live mode (scapy) ─────────────────────────────────────────────────────

    def _run_live(self) -> None:
        """
        Sniff real packets on `self.interface`.
        Requires: pip install scapy  +  root/admin privileges.
        """
        try:
            sniff(
                iface=self.interface,
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _: not self._running,
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
        """Scapy callback — called for every captured packet."""
        if not IP in pkt:
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
            "flows_per_sec": len(flows),
            "bandwidth_kbps": int(total_bps / 1024),
            "total_flows": self._total_flows,
        })


# ── Packet → flow aggregation (mirrors pcap_parser logic) ─────────────────────

def _aggregate_packets_to_flows(packets: list[dict]) -> list[dict]:
    """
    Group raw packet dicts into bidirectional 5-tuple flows.
    Produces the same keys as pcap_parser.aggregate_packets_to_flows.
    """
    flow_buckets: dict[tuple, dict] = defaultdict(lambda: {
        "packets": [], "fwd_bytes": 0, "bwd_bytes": 0,
        "fwd_pkts": 0, "bwd_pkts": 0,
        "start_ts": None, "end_ts": None,
        "flags": set(),
        "src_ip": "", "dst_ip": "", "src_port": 0, "dst_port": 0,
        "protocol": "TCP",
    })

    for p in packets:
        try:
            ts    = float(p.get("frame.time_epoch", 0))
            proto = p.get("ip.proto", "6")
            sip   = p.get("ip.src", "0.0.0.0")
            dip   = p.get("ip.dst", "0.0.0.0")
            sport = int(p.get("tcp.srcport") or p.get("udp.srcport") or 0)
            dport = int(p.get("tcp.dstport") or p.get("udp.dstport") or 0)
            size  = int(p.get("frame.len", 0))
            flags = int(p.get("tcp.flags", 0) or 0)

            # Canonical 5-tuple (bidirectional)
            fwd_key = (sip, dip, sport, dport, proto)
            bwd_key = (dip, sip, dport, sport, proto)

            if bwd_key in flow_buckets:
                key      = bwd_key
                is_fwd   = False
            else:
                key      = fwd_key
                is_fwd   = True

            b = flow_buckets[key]
            if b["start_ts"] is None:
                b["start_ts"] = ts
                b["src_ip"]   = sip
                b["dst_ip"]   = dip
                b["src_port"] = sport
                b["dst_port"] = dport
                b["protocol"] = _proto_name(proto)
            b["end_ts"] = ts

            if is_fwd:
                b["fwd_bytes"] += size
                b["fwd_pkts"]  += 1
            else:
                b["bwd_bytes"] += size
                b["bwd_pkts"]  += 1

            # Collect TCP flags
            if flags & 0x02: b["flags"].add("SYN")
            if flags & 0x10: b["flags"].add("ACK")
            if flags & 0x01: b["flags"].add("FIN")
            if flags & 0x04: b["flags"].add("RST")
            if flags & 0x08: b["flags"].add("PSH")
            if flags & 0x20: b["flags"].add("URG")

        except (ValueError, TypeError):
            continue

    # Convert buckets → flow dicts
    flows = []
    for _, b in flow_buckets.items():
        if b["start_ts"] is None:
            continue
        dur = max((b["end_ts"] or b["start_ts"]) - b["start_ts"], 1e-6)
        total_bytes = b["fwd_bytes"] + b["bwd_bytes"]
        total_pkts  = b["fwd_pkts"]  + b["bwd_pkts"]
        flows.append({
            "src_ip":              b["src_ip"],
            "dst_ip":              b["dst_ip"],
            "src_port":            b["src_port"],
            "dst_port":            b["dst_port"],
            "protocol":            b["protocol"],
            "flow_duration":       round(dur, 6),
            "total_fwd_packets":   b["fwd_pkts"],
            "total_bwd_packets":   b["bwd_pkts"],
            "total_fwd_bytes":     b["fwd_bytes"],
            "total_bwd_bytes":     b["bwd_bytes"],
            "flow_bytes_per_sec":  round(total_bytes / dur, 2),
            "flow_pkts_per_sec":   round(total_pkts  / dur, 2),
            "flag_SYN": int("SYN" in b["flags"]),
            "flag_ACK": int("ACK" in b["flags"]),
            "flag_FIN": int("FIN" in b["flags"]),
            "flag_RST": int("RST" in b["flags"]),
            "flag_PSH": int("PSH" in b["flags"]),
            "flag_URG": int("URG" in b["flags"]),
        })

    return flows


def _proto_name(proto_num: str) -> str:
    return {"6": "TCP", "17": "UDP", "1": "ICMP"}.get(str(proto_num), proto_num)