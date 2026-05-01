"""
live_capture.py  —  Real-time packet capture thread (Phase B unified pipeline)
==================================================================================
Captures live packets via Scapy, runs Kitsune feature extraction per packet,
aggregates flows, and emits flow_ready signals enriched with both Stage-1
device classification and (when buffer is ready) Stage-2 IoT botnet probability.

Architecture (single unified path):

    LiveCaptureThread (QThread)
        ┌── Per packet ───────────────────────────────────────────────┐
        │  1. Update Kitsune extractor (115 N-BaIoT statistics)       │
        │  2. Scale via iot_scaler.json → append to per-src_ip deque  │
        │  3. Buffer raw packet for flow aggregation                  │
        └─────────────────────────────────────────────────────────────┘
        ┌── Every EMIT_INTERVAL seconds ──────────────────────────────┐
        │  4. Aggregate buffered packets → 5-tuple flow dicts         │
        │  5. For each flow:                                          │
        │       a. Attach Kitsune sequence (if src_ip buffer is full) │
        │       b. Emit flow_ready(dict) → MonitorPage                │
        │       c. MonitorPage calls inference_bridge.run_inference()  │
        │          which routes to Stage-1 RF and (for IoT) Stage-2   │
        └─────────────────────────────────────────────────────────────┘

Modes:
    LIVE — real packet capture + full Stage-1 + Stage-2 IoT pipeline.
           Requires scapy + sufficient privileges on the chosen interface.
    DEMO — synthetic flows for development without root or scapy.
           Falls back automatically when scapy is missing or no interface found.

Removed in Phase B:
    DETECTOR mode (delegated to LiveDetector). Live mode now subsumes it.
    LiveDetector remains as a standalone CLI tool in src/live/live_detector.py.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# ── Make project root importable so `src.live.kitsune_extractor` resolves
#    regardless of where the GUI was launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Scapy import (graceful fallback to demo mode) ─────────────────────────────
try:
    from scapy.all import (
        sniff, IP, TCP, UDP, Ether,
        conf as scapy_conf, get_if_list, get_if_addr,
    )
    scapy_conf.verb = 0
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ── Kitsune extractor (graceful fallback: live mode runs without Stage-2) ─────
try:
    from src.live.kitsune_extractor import KitsuneExtractor
    KITSUNE_AVAILABLE = True
except ImportError:
    KITSUNE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
EMIT_INTERVAL    = 2.0     # seconds between flow flushes
PACKET_BUFFER    = 50      # max packets buffered before forced flush
KITSUNE_SEQ_LEN  = 20      # rows per src_ip buffer (matches iot_cnn_lstm.pt training)
PREF_IFACES      = ("wi-fi", "wlan", "en0", "en1", "eth0", "ethernet")

IOT_SCALER_PATH = ROOT / "models" / "stage2" / "iot_scaler.json"


# ══════════════════════════════════════════════════════════════════════════════
# Interface helpers (unchanged from Phase A)
# ══════════════════════════════════════════════════════════════════════════════

def get_interfaces() -> list[tuple[str, str]]:
    """Return [(iface_name, ip_address), ...] for interfaces with a real IP."""
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
    if not result:
        for iface in get_if_list():
            if not any(x in iface.lower() for x in ["lo", "loopback", "npcap loopback"]):
                result.append((iface, "0.0.0.0"))
    return result


def auto_select_interface() -> Optional[str]:
    """Return the most likely user-facing interface name, or None."""
    if not SCAPY_AVAILABLE:
        return None
    candidates = get_interfaces()
    if not candidates:
        return None
    for pref in PREF_IFACES:
        for iface, _ in candidates:
            if pref in iface.lower():
                return iface
    return candidates[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# Kitsune scaler — loaded once at thread init
# ══════════════════════════════════════════════════════════════════════════════

class _KitsuneScaler:
    """
    Wraps iot_scaler.json for fast vector-level scaling.

    The training pipeline applied MinMaxScaler to raw Kitsune output before
    saving the IoT model. The scaler.json stores the raw min/max so we can
    reproduce the same transformation at inference time.

    Returns clipped [0, 1] float32 vectors.
    """

    def __init__(self, scaler_path: Path):
        with open(scaler_path) as f:
            data = json.load(f)
        self.feat_min = np.array(data["min"], dtype=np.float32)
        self.feat_max = np.array(data["max"], dtype=np.float32)
        self.range    = self.feat_max - self.feat_min
        # Guard against zero-range features (constant during training)
        self.range[self.range == 0.0] = 1.0

        # Sanity check: if max ≤ 1.0 across the board, the scaler was exported
        # from already-normalized data and applying it will collapse live values.
        if self.feat_max.max() < 2.0:
            raise ValueError(
                f"{scaler_path} has all max values <= 1.0 — looks like it was "
                "exported from the already-normalized CSV. Re-run "
                "preprocess_nbaiot.py to regenerate the raw-range scaler."
            )

    def scale(self, raw_vec: np.ndarray) -> np.ndarray:
        scaled = (raw_vec - self.feat_min) / self.range
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Demo flows (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_FLOWS = [
    {"src_ip":"192.168.1.10","dst_ip":"8.8.8.8","src_port":54321,"dst_port":443,
     "protocol":6,"flow_duration":0.42,"total_fwd_packets":12,"total_bwd_packets":10,
     "total_fwd_bytes":1500,"total_bwd_bytes":12000,
     "flow_pkts_per_sec":52.4,"flow_bytes_per_sec":32100,
     "flag_SYN":1,"flag_ACK":18,"flag_FIN":1,"flag_RST":0,"flag_PSH":4,"flag_URG":0},
    {"src_ip":"192.168.1.42","dst_ip":"185.199.108.153","src_port":48923,"dst_port":443,
     "protocol":6,"flow_duration":1.85,"total_fwd_packets":48,"total_bwd_packets":52,
     "total_fwd_bytes":7400,"total_bwd_bytes":62000,
     "flow_pkts_per_sec":54.0,"flow_bytes_per_sec":37500,
     "flag_SYN":1,"flag_ACK":98,"flag_FIN":1,"flag_RST":0,"flag_PSH":15,"flag_URG":0},
    {"src_ip":"192.168.1.103","dst_ip":"172.217.16.142","src_port":33567,"dst_port":80,
     "protocol":6,"flow_duration":0.18,"total_fwd_packets":4,"total_bwd_packets":3,
     "total_fwd_bytes":400,"total_bwd_bytes":1200,
     "flow_pkts_per_sec":38.9,"flow_bytes_per_sec":8888,
     "flag_SYN":1,"flag_ACK":6,"flag_FIN":1,"flag_RST":0,"flag_PSH":2,"flag_URG":0},
]


# ══════════════════════════════════════════════════════════════════════════════
# LiveCaptureThread — Phase B unified pipeline
# ══════════════════════════════════════════════════════════════════════════════

class LiveCaptureThread(QThread):
    flow_ready = pyqtSignal(dict)   # one completed flow dict (with optional _kitsune_seq)
    error      = pyqtSignal(str)
    stats      = pyqtSignal(dict)

    def __init__(
        self,
        interface:        Optional[str] = None,
        model_path:       Optional[str] = None,   # kept for backward-compat; now unused
        demo_mode:        bool          = False,
        emit_interval:    float         = EMIT_INTERVAL,
        enable_kitsune:   bool          = True,
    ) -> None:
        super().__init__()

        if not SCAPY_AVAILABLE:
            demo_mode = True

        if not demo_mode and interface is None:
            interface = auto_select_interface()
            if interface is None:
                demo_mode = True

        self.interface     = interface
        self.demo_mode     = demo_mode
        self.emit_interval = emit_interval

        # Phase B state — Kitsune extraction and per-src_ip buffer
        self._kitsune: Optional[KitsuneExtractor] = None
        self._scaler:  Optional[_KitsuneScaler]   = None
        self._kit_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=KITSUNE_SEQ_LEN)
        )
        self._kitsune_enabled = False

        if enable_kitsune and not demo_mode and KITSUNE_AVAILABLE:
            try:
                if not IOT_SCALER_PATH.exists():
                    raise FileNotFoundError(
                        f"IoT scaler not found at {IOT_SCALER_PATH}"
                    )
                self._scaler = _KitsuneScaler(IOT_SCALER_PATH)
                self._kitsune = KitsuneExtractor()
                self._kitsune_enabled = True
            except Exception as e:
                # Don't fail thread startup — Stage-2 IoT just won't fire.
                # Stage-1 still runs and the user sees flows.
                self._kitsune_enabled = False
                self._kitsune_init_error = str(e)

        self._running     = False
        self._pkt_buffer: list[dict] = []
        self._last_emit   = time.monotonic()
        self._total_flows = 0
        self._total_pkts  = 0

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        # Surface Kitsune init error once on startup so the user knows
        # Stage-2 IoT won't fire even though the thread is running.
        if (not self.demo_mode) and (not self._kitsune_enabled) and KITSUNE_AVAILABLE:
            err = getattr(self, "_kitsune_init_error", "Kitsune disabled")
            self.error.emit(
                f"Stage-2 IoT disabled: {err}. "
                "Stage-1 device classification will still run."
            )

        if self.demo_mode:
            self._run_demo()
        else:
            self._run_live()

    def stop(self) -> None:
        """Signal the capture loop to halt."""
        self._running = False

    # ── Live mode (unified pipeline) ──────────────────────────────────────────

    def _run_live(self) -> None:
        """
        Sniff real packets on self.interface, run Kitsune per packet,
        and flush flows every EMIT_INTERVAL seconds.
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
        """Scapy callback — runs for every captured packet."""
        if IP not in pkt:
            return

        # ── Build the raw record used by the flow aggregator ──────────────
        ip_layer = pkt[IP]
        ts = float(pkt.time)

        rec: dict = {
            "frame.time_epoch": str(ts),
            "ip.src":           ip_layer.src,
            "ip.dst":           ip_layer.dst,
            "ip.proto":         str(ip_layer.proto),
            "frame.len":        str(len(pkt)),
            "ip.ttl":           str(ip_layer.ttl),
        }
        sport = dport = 0
        if TCP in pkt:
            rec["tcp.srcport"] = str(pkt[TCP].sport)
            rec["tcp.dstport"] = str(pkt[TCP].dport)
            rec["tcp.flags"]   = str(int(pkt[TCP].flags))
            sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
            proto_name = "TCP"
        elif UDP in pkt:
            rec["udp.srcport"] = str(pkt[UDP].sport)
            rec["udp.dstport"] = str(pkt[UDP].dport)
            sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
            proto_name = "UDP"
        else:
            proto_name = "OTHER"

        self._pkt_buffer.append(rec)
        self._total_pkts += 1

        # ── Per-packet Kitsune update + per-src_ip buffer append ──────────
        if self._kitsune_enabled:
            try:
                src_mac = pkt[Ether].src if Ether in pkt else ip_layer.src
                raw_vec = self._kitsune.update(
                    timestamp = ts,
                    src_mac   = src_mac,
                    src_ip    = ip_layer.src,
                    dst_ip    = ip_layer.dst,
                    src_port  = sport,
                    dst_port  = dport,
                    pkt_len   = len(pkt),
                    protocol  = proto_name,
                )
                scaled = self._scaler.scale(raw_vec)
                self._kit_buffers[ip_layer.src].append(scaled)
            except Exception:
                # A single packet failing Kitsune shouldn't break capture.
                # Silently drop; if this becomes systematic, _kit_buffers
                # will stay empty and Stage-2 IoT will simply never fire.
                pass

        # ── Periodic flush ────────────────────────────────────────────────
        now = time.monotonic()
        if (len(self._pkt_buffer) >= PACKET_BUFFER or
                now - self._last_emit >= self.emit_interval):
            self._flush_buffer()
            self._last_emit = now

    def _flush_buffer(self) -> None:
        """Aggregate buffered packets into flows, attach Kitsune seqs, emit."""
        if not self._pkt_buffer:
            return

        flows = _aggregate_packets_to_flows(self._pkt_buffer)
        self._pkt_buffer.clear()

        total_bps = 0.0
        for flow in flows:
            # Attach Kitsune sequence if this src_ip's buffer is full.
            # The bridge consumes _kitsune_seq for IoT-classified flows.
            src_ip = flow.get("src_ip", "")
            buf    = self._kit_buffers.get(src_ip)
            if buf is not None and len(buf) == KITSUNE_SEQ_LEN:
                flow["_kitsune_seq"] = np.stack(list(buf))

            self.flow_ready.emit(flow)
            self._total_flows += 1
            total_bps += float(flow.get("flow_bytes_per_sec", 0))

        self.stats.emit({
            "flows_per_sec":  len(flows),
            "bandwidth_kbps": int(total_bps / 1024),
            "total_flows":    self._total_flows,
        })

    # ── Demo mode (unchanged) ─────────────────────────────────────────────────

    def _run_demo(self) -> None:
        """Emit synthetic flows for development without scapy/root."""
        while self._running:
            batch_size = random.randint(1, 3)
            for _ in range(batch_size):
                flow = dict(random.choice(_DEMO_FLOWS))
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


# ══════════════════════════════════════════════════════════════════════════════
# Packet → flow aggregation (Phase A enrichment, unchanged in Phase B)
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_packets_to_flows(packets: list[dict]) -> list[dict]:
    """
    Group raw packet dicts into bidirectional 5-tuple flows.
    Produces 38 of the 56 unified-schema features per flow.
    """
    import statistics

    raw: dict[tuple, dict] = {}

    for pkt in packets:
        src       = pkt.get("ip.src", "")
        dst       = pkt.get("ip.dst", "")
        sp        = int(pkt.get("tcp.srcport") or pkt.get("udp.srcport") or 0)
        dp        = int(pkt.get("tcp.dstport") or pkt.get("udp.dstport") or 0)
        proto_num = int(pkt.get("ip.proto", 0))
        key = tuple(sorted([(src, sp), (dst, dp)])) + (proto_num,)

        size  = int(pkt.get("frame.len", 0))
        ts    = float(pkt.get("frame.time_epoch", time.time()))
        flags = int(pkt.get("tcp.flags", 0))
        hdr_len = 54  # rough Eth+IP+TCP estimate

        if key not in raw:
            raw[key] = {
                "src_ip": src, "dst_ip": dst,
                "src_port": sp, "dst_port": dp,
                "protocol_num": proto_num,
                "fwd_sizes": [], "bwd_sizes": [],
                "fwd_times": [], "bwd_times": [],
                "all_times": [],
                "fwd_hdr_total": 0, "bwd_hdr_total": 0,
                "flag_FIN_count": 0, "flag_SYN_count": 0,
                "flag_RST_count": 0, "flag_PSH_count": 0,
                "flag_ACK_count": 0, "flag_URG_count": 0,
            }

        f = raw[key]
        if src == f["src_ip"]:
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

    def _stats(xs):
        if not xs: return 0.0, 0.0, 0.0, 0.0
        if len(xs) == 1: return float(xs[0]), float(xs[0]), float(xs[0]), 0.0
        return (float(min(xs)), float(max(xs)),
                float(statistics.mean(xs)), float(statistics.pstdev(xs)))

    def _iat(times):
        if len(times) < 2: return 0.0, 0.0, 0.0, 0.0
        ts = sorted(times)
        iats = [ts[i] - ts[i-1] for i in range(1, len(ts))]
        return (float(statistics.mean(iats)),
                float(statistics.pstdev(iats)) if len(iats) > 1 else 0.0,
                float(min(iats)), float(max(iats)))

    result = []
    for f in raw.values():
        n_fwd = len(f["fwd_sizes"])
        n_bwd = len(f["bwd_sizes"])
        bytes_fwd = sum(f["fwd_sizes"])
        bytes_bwd = sum(f["bwd_sizes"])
        all_times = f["all_times"]
        first_ts = min(all_times) if all_times else 0.0
        last_ts  = max(all_times) if all_times else 0.0
        duration = max(last_ts - first_ts, 1e-6)

        fwd_min, fwd_max, fwd_mean, fwd_std = _stats(f["fwd_sizes"])
        bwd_min, bwd_max, bwd_mean, bwd_std = _stats(f["bwd_sizes"])
        flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max = _iat(all_times)
        fwd_iat_mean,  fwd_iat_std,  fwd_iat_min,  fwd_iat_max  = _iat(f["fwd_times"])
        bwd_iat_mean,  bwd_iat_std,  bwd_iat_min,  bwd_iat_max  = _iat(f["bwd_times"])

        result.append({
            "src_ip": f["src_ip"], "dst_ip": f["dst_ip"],
            "src_port": f["src_port"], "dst_port": f["dst_port"],
            "protocol": f["protocol_num"],
            "flow_duration":      round(duration, 6),
            "total_fwd_packets":  n_fwd,
            "total_bwd_packets":  n_bwd,
            "total_fwd_bytes":    bytes_fwd,
            "total_bwd_bytes":    bytes_bwd,
            "fwd_pkt_len_min":  fwd_min, "fwd_pkt_len_max": fwd_max,
            "fwd_pkt_len_mean": fwd_mean, "fwd_pkt_len_std": fwd_std,
            "bwd_pkt_len_min":  bwd_min, "bwd_pkt_len_max": bwd_max,
            "bwd_pkt_len_mean": bwd_mean, "bwd_pkt_len_std": bwd_std,
            "flow_bytes_per_sec": round((bytes_fwd + bytes_bwd) / duration, 2),
            "flow_pkts_per_sec":  round((n_fwd + n_bwd) / duration, 2),
            "flow_iat_mean": flow_iat_mean, "flow_iat_std": flow_iat_std,
            "flow_iat_min":  flow_iat_min,  "flow_iat_max":  flow_iat_max,
            "fwd_iat_mean":  fwd_iat_mean,  "fwd_iat_std":   fwd_iat_std,
            "fwd_iat_min":   fwd_iat_min,   "fwd_iat_max":   fwd_iat_max,
            "bwd_iat_mean":  bwd_iat_mean,  "bwd_iat_std":   bwd_iat_std,
            "bwd_iat_min":   bwd_iat_min,   "bwd_iat_max":   bwd_iat_max,
            "fwd_header_length": f["fwd_hdr_total"],
            "bwd_header_length": f["bwd_hdr_total"],
            "flag_FIN": f["flag_FIN_count"],
            "flag_SYN": f["flag_SYN_count"],
            "flag_RST": f["flag_RST_count"],
            "flag_PSH": f["flag_PSH_count"],
            "flag_ACK": f["flag_ACK_count"],
            "flag_URG": f["flag_URG_count"],
        })

    return result
