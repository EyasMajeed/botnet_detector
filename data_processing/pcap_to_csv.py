"""
==========================================================================
PCAP → CSV Flow Feature Extractor
==========================================================================
AI-Based Botnet Detection Using Hybrid Deep Learning Models
CS498/499 - Group 07

Converts raw PCAP files into CSV files containing ~80 CICFlowMeter-style
flow features using tshark + Python.

Supports two datasets with automatic label merging:
  - IoT-23  : reads conn.log.labeled (Zeek/Bro format)
  - CTU-13  : reads .binetflow files (Argus bidirectional flow format)

This module:
  1. Uses tshark to extract per-packet fields from PCAPs efficiently
  2. Reconstructs bidirectional flows (5-tuple grouping)
  3. Computes CICFlowMeter-equivalent features per flow:
     - Flow duration & activity/idle times
     - Packet length statistics (min/max/mean/std) per direction
     - Inter-arrival time (IAT) statistics per direction
     - TCP flag counts (SYN, ACK, RST, FIN, PSH, URG, CWR, ECE)
     - Byte & packet rate features
     - Bulk transfer metrics
     - Subflow features
     - Header length features
  4. Merges labels from dataset-specific label files
  5. Outputs a single CSV ready for the preprocessing pipeline

Usage:
    # ── IoT-23 ──────────────────────────────────────────────────────────
    # Convert a single IoT-23 PCAP
    python pcap_to_csv.py --dataset iot23 --pcap ./capture.pcap --output ./flows.csv

    # Convert entire IoT-23 dataset directory
    python pcap_to_csv.py --dataset iot23 --data_dir ./iot23_full --output_dir ./iot23_csv

    # ── CTU-13 ──────────────────────────────────────────────────────────
    # Convert a single CTU-13 scenario PCAP (label dir = same folder as PCAP)
    python pcap_to_csv.py --dataset ctu13 --pcap ./scenario1/capture.pcap --output ./flows.csv

    # Convert entire CTU-13 dataset directory (all 13 scenarios)
    python pcap_to_csv.py --dataset ctu13 --data_dir ./CTU-13-Dataset --output_dir ./ctu13_csv

    # ── General ─────────────────────────────────────────────────────────
    # With row limit per PCAP (for testing large files)
    python pcap_to_csv.py --dataset ctu13 --data_dir ./CTU-13-Dataset --output_dir ./ctu13_csv --max_packets 500000

    # Generate a sample PCAP for testing (no dataset required)
    python pcap_to_csv.py --generate_sample --output_dir ./test_output

Authors: Group 07
==========================================================================
"""

import os
import sys
import csv
import glob
import struct
import subprocess
import argparse
import logging
import tempfile
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PCAP-to-CSV")


# =========================================================================
# 1. CONFIGURATION
# =========================================================================

# Flow timeout thresholds (matching CICFlowMeter defaults)
FLOW_TIMEOUT = 120.0          # seconds — flows idle > this are terminated
ACTIVITY_TIMEOUT = 5.0        # seconds — gap between active/idle periods

# tshark fields to extract per packet
TSHARK_FIELDS = [
    "frame.time_epoch",       # timestamp
    "ip.src",                 # source IP
    "ip.dst",                 # destination IP
    "tcp.srcport",            # TCP source port
    "tcp.dstport",            # TCP destination port
    "udp.srcport",            # UDP source port
    "udp.dstport",            # UDP destination port
    "ip.proto",               # protocol number (6=TCP, 17=UDP, 1=ICMP)
    "frame.len",              # total frame length
    "ip.len",                 # IP packet length
    "ip.ttl",                 # TTL
    "tcp.flags",              # TCP flags as hex
    "tcp.flags.syn",          # individual flags
    "tcp.flags.ack",
    "tcp.flags.reset",        # "reset" not "rst" in tshark
    "tcp.flags.fin",
    "tcp.flags.push",         # "push" not "psh" in tshark
    "tcp.flags.urg",
    "tcp.flags.cwr",
    "tcp.flags.ece",
    "tcp.hdr_len",            # TCP header length
    "ip.hdr_len",             # IP header length
    "tcp.window_size_value",  # TCP window size
    "ip.flags.df",            # Don't Fragment
    "tcp.len",                # TCP payload length
    "udp.length",             # UDP length
]

TSHARK_FIELD_STR = " -e ".join(TSHARK_FIELDS)

# Output CSV feature columns (CICFlowMeter-compatible naming)
OUTPUT_COLUMNS = [
    # ─── Identifiers ───
    "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    # ─── Timing ───
    "timestamp", "flow_duration",
    # ─── Packet counts ───
    "total_fwd_packets", "total_bwd_packets",
    # ─── Byte counts ───
    "total_fwd_bytes", "total_bwd_bytes",
    # ─── Packet length stats (forward) ───
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    # ─── Packet length stats (backward) ───
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    # ─── Overall packet length stats ───
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    # ─── Rate features ───
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    # ─── IAT stats (overall) ───
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    # ─── IAT stats (forward) ───
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    # ─── IAT stats (backward) ───
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    # ─── TCP flags ───
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG",
    "flag_CWR", "flag_ECE",
    "fwd_flag_PSH", "bwd_flag_PSH", "fwd_flag_URG", "bwd_flag_URG",
    # ─── Header lengths ───
    "fwd_header_length", "bwd_header_length",
    # ─── Packets per second per direction ───
    "fwd_pkts_per_sec", "bwd_pkts_per_sec",
    # ─── Packet size aggregates ───
    "min_pkt_size", "max_pkt_size",
    # ─── Bulk transfer features ───
    "fwd_avg_bytes_per_bulk", "fwd_avg_pkts_per_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_per_bulk", "bwd_avg_pkts_per_bulk", "bwd_avg_bulk_rate",
    # ─── Subflow features ───
    "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes",
    # ─── Window size ───
    "init_win_bytes_fwd", "init_win_bytes_bwd",
    # ─── Active/Idle times ───
    "active_mean", "active_std", "active_min", "active_max",
    "idle_mean", "idle_std", "idle_min", "idle_max",
    # ─── TTL ───
    "fwd_avg_ttl", "bwd_avg_ttl",
    # ─── Additional ───
    "down_up_ratio",
    "avg_pkt_size",
    "fwd_seg_size_avg", "bwd_seg_size_avg",
    # ─── Label ───
    "label", "detailed_label",
]


# =========================================================================
# 2. PACKET DATA STRUCTURE
# =========================================================================

@dataclass
class Packet:
    """Parsed packet from tshark output."""
    timestamp: float = 0.0
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0         # 6=TCP, 17=UDP, 1=ICMP
    frame_len: int = 0
    ip_len: int = 0
    ttl: int = 0
    tcp_flags: int = 0
    flag_syn: int = 0
    flag_ack: int = 0
    flag_rst: int = 0
    flag_fin: int = 0
    flag_psh: int = 0
    flag_urg: int = 0
    flag_cwr: int = 0
    flag_ece: int = 0
    tcp_hdr_len: int = 0
    ip_hdr_len: int = 0
    tcp_win_size: int = 0
    tcp_payload_len: int = 0
    udp_len: int = 0


@dataclass
class FlowRecord:
    """Accumulates packets belonging to one bidirectional flow."""
    flow_id: str = ""
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0
    start_time: float = 0.0
    last_time: float = 0.0

    # Forward = originator → responder; Backward = responder → originator
    fwd_packets: List[float] = field(default_factory=list)   # packet sizes
    bwd_packets: List[float] = field(default_factory=list)
    fwd_timestamps: List[float] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)
    all_timestamps: List[float] = field(default_factory=list)
    all_pkt_sizes: List[float] = field(default_factory=list)

    # TCP flags
    flag_counts: Dict[str, int] = field(default_factory=lambda: {
        "FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0,
        "URG": 0, "CWR": 0, "ECE": 0
    })
    fwd_flag_psh: int = 0
    bwd_flag_psh: int = 0
    fwd_flag_urg: int = 0
    bwd_flag_urg: int = 0

    # Header lengths
    fwd_header_lengths: List[int] = field(default_factory=list)
    bwd_header_lengths: List[int] = field(default_factory=list)

    # Window sizes
    init_win_fwd: int = -1
    init_win_bwd: int = -1

    # TTL
    fwd_ttls: List[int] = field(default_factory=list)
    bwd_ttls: List[int] = field(default_factory=list)


# =========================================================================
# 3. TSHARK PACKET EXTRACTION
# =========================================================================

def extract_packets_tshark(pcap_path: str, max_packets: int = None) -> List[Packet]:
    """
    Use tshark to extract per-packet fields from a PCAP file.
    Returns a list of Packet objects.
    """
    log.info(f"  Extracting packets with tshark from: {os.path.basename(pcap_path)}")

    # Build tshark command
    cmd = [
        "tshark", "-r", pcap_path,
        "-T", "fields",
        "-E", "separator=|",
        "-E", "header=n",
        "-E", "quote=n",
        "-E", "occurrence=f",
    ]
    for f in TSHARK_FIELDS:
        cmd.extend(["-e", f])

    if max_packets:
        cmd.extend(["-c", str(max_packets)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            log.error(f"  tshark error: {result.stderr[:500]}")
            return []
    except subprocess.TimeoutExpired:
        log.error("  tshark timed out (1 hour limit)")
        return []
    except FileNotFoundError:
        log.error("  tshark not found! Install with: sudo apt install tshark")
        return []

    packets = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        pkt = _parse_tshark_line(line)
        if pkt and pkt.src_ip and pkt.dst_ip:
            packets.append(pkt)

    log.info(f"  ✓ Extracted {len(packets):,} valid packets")
    return packets


def _parse_tshark_line(line: str) -> Optional[Packet]:
    """Parse a single tshark output line into a Packet object."""
    fields = line.split("|")
    if len(fields) < len(TSHARK_FIELDS):
        fields.extend([""] * (len(TSHARK_FIELDS) - len(fields)))

    try:
        pkt = Packet()
        pkt.timestamp = _safe_float(fields[0])
        pkt.src_ip = fields[1].strip()
        pkt.dst_ip = fields[2].strip()

        # Ports: try TCP first, then UDP
        tcp_sport = _safe_int(fields[3])
        tcp_dport = _safe_int(fields[4])
        udp_sport = _safe_int(fields[5])
        udp_dport = _safe_int(fields[6])

        pkt.src_port = tcp_sport if tcp_sport else udp_sport
        pkt.dst_port = tcp_dport if tcp_dport else udp_dport

        pkt.protocol = _safe_int(fields[7])
        pkt.frame_len = _safe_int(fields[8])
        pkt.ip_len = _safe_int(fields[9]) or pkt.frame_len
        pkt.ttl = _safe_int(fields[10])

        # TCP flags
        pkt.tcp_flags = _safe_hex(fields[11])
        pkt.flag_syn = _safe_int(fields[12])
        pkt.flag_ack = _safe_int(fields[13])
        pkt.flag_rst = _safe_int(fields[14])
        pkt.flag_fin = _safe_int(fields[15])
        pkt.flag_psh = _safe_int(fields[16])
        pkt.flag_urg = _safe_int(fields[17])
        pkt.flag_cwr = _safe_int(fields[18])
        pkt.flag_ece = _safe_int(fields[19])

        pkt.tcp_hdr_len = _safe_int(fields[20])
        pkt.ip_hdr_len = _safe_int(fields[21])
        pkt.tcp_win_size = _safe_int(fields[22])
        pkt.tcp_payload_len = _safe_int(fields[24])
        pkt.udp_len = _safe_int(fields[25])

        return pkt
    except Exception:
        return None


def _safe_float(s: str) -> float:
    try:
        return float(s.strip()) if s.strip() else 0.0
    except (ValueError, TypeError):
        return 0.0

def _safe_int(s: str) -> int:
    try:
        s = s.strip().lower() if s else ""
        if s in ("true", "1"):
            return 1
        if s in ("false", "0", ""):
            return 0
        return int(s)
    except (ValueError, TypeError):
        return 0

def _safe_hex(s: str) -> int:
    try:
        s = s.strip()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s) if s else 0
    except (ValueError, TypeError):
        return 0


# =========================================================================
# 4. FLOW RECONSTRUCTION
# =========================================================================

def reconstruct_flows(packets: List[Packet]) -> Dict[str, FlowRecord]:
    """
    Group packets into bidirectional flows using the 5-tuple:
        (src_ip, dst_ip, src_port, dst_port, protocol)

    Bidirectional: A→B and B→A belong to the same flow.
    Uses FLOW_TIMEOUT to split long-running connections.
    """
    log.info(f"  Reconstructing bidirectional flows (timeout={FLOW_TIMEOUT}s)...")

    flows: Dict[str, FlowRecord] = {}
    flow_counter = 0

    for pkt in packets:
        fwd_key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        bwd_key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)

        is_forward = True
        flow_key = None

        if str(fwd_key) in flows:
            flow = flows[str(fwd_key)]
            if pkt.timestamp - flow.last_time > FLOW_TIMEOUT:
                flow_key = None
            else:
                flow_key = str(fwd_key)
                is_forward = True
        elif str(bwd_key) in flows:
            flow = flows[str(bwd_key)]
            if pkt.timestamp - flow.last_time > FLOW_TIMEOUT:
                flow_key = None
            else:
                flow_key = str(bwd_key)
                is_forward = False

        if flow_key is None:
            flow_counter += 1
            flow_key = str(fwd_key)
            flow = FlowRecord(
                flow_id=f"flow_{flow_counter}",
                src_ip=pkt.src_ip,
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port,
                dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                start_time=pkt.timestamp,
            )
            flows[flow_key] = flow
            is_forward = True

        flow.last_time = pkt.timestamp
        flow.all_timestamps.append(pkt.timestamp)
        flow.all_pkt_sizes.append(pkt.ip_len)

        pkt_size = pkt.ip_len

        if is_forward:
            flow.fwd_packets.append(pkt_size)
            flow.fwd_timestamps.append(pkt.timestamp)
            flow.fwd_header_lengths.append((pkt.ip_hdr_len or 20) + (pkt.tcp_hdr_len or 0))
            flow.fwd_ttls.append(pkt.ttl)
            if pkt.flag_psh:
                flow.fwd_flag_psh += 1
            if pkt.flag_urg:
                flow.fwd_flag_urg += 1
            if flow.init_win_fwd == -1:
                flow.init_win_fwd = pkt.tcp_win_size
        else:
            flow.bwd_packets.append(pkt_size)
            flow.bwd_timestamps.append(pkt.timestamp)
            flow.bwd_header_lengths.append((pkt.ip_hdr_len or 20) + (pkt.tcp_hdr_len or 0))
            flow.bwd_ttls.append(pkt.ttl)
            if pkt.flag_psh:
                flow.bwd_flag_psh += 1
            if pkt.flag_urg:
                flow.bwd_flag_urg += 1
            if flow.init_win_bwd == -1:
                flow.init_win_bwd = pkt.tcp_win_size

        flow.flag_counts["SYN"] += pkt.flag_syn
        flow.flag_counts["ACK"] += pkt.flag_ack
        flow.flag_counts["RST"] += pkt.flag_rst
        flow.flag_counts["FIN"] += pkt.flag_fin
        flow.flag_counts["PSH"] += pkt.flag_psh
        flow.flag_counts["URG"] += pkt.flag_urg
        flow.flag_counts["CWR"] += pkt.flag_cwr
        flow.flag_counts["ECE"] += pkt.flag_ece

    log.info(f"  ✓ Reconstructed {len(flows):,} bidirectional flows")
    return flows


# =========================================================================
# 5. FEATURE COMPUTATION (CICFlowMeter-style)
# =========================================================================

def compute_flow_features(flow: FlowRecord) -> dict:
    """
    Compute ~80 CICFlowMeter-equivalent features for a single flow.
    """
    f = {}

    # ─── Identifiers ────────────────────────────────────────────────
    f["flow_id"] = flow.flow_id
    f["src_ip"] = flow.src_ip
    f["dst_ip"] = flow.dst_ip
    f["src_port"] = flow.src_port
    f["dst_port"] = flow.dst_port
    f["protocol"] = flow.protocol
    f["timestamp"] = flow.start_time

    # ─── Duration ───────────────────────────────────────────────────
    duration = flow.last_time - flow.start_time
    f["flow_duration"] = max(duration, 0.0)
    dur_safe = max(duration, 1e-6)

    # ─── Packet counts ──────────────────────────────────────────────
    n_fwd = len(flow.fwd_packets)
    n_bwd = len(flow.bwd_packets)
    n_total = n_fwd + n_bwd

    f["total_fwd_packets"] = n_fwd
    f["total_bwd_packets"] = n_bwd

    # ─── Byte counts ────────────────────────────────────────────────
    fwd_bytes = sum(flow.fwd_packets)
    bwd_bytes = sum(flow.bwd_packets)

    f["total_fwd_bytes"] = fwd_bytes
    f["total_bwd_bytes"] = bwd_bytes

    # ─── Packet length statistics (forward) ─────────────────────────
    f["fwd_pkt_len_min"] = _safe_min(flow.fwd_packets)
    f["fwd_pkt_len_max"] = _safe_max(flow.fwd_packets)
    f["fwd_pkt_len_mean"] = _safe_mean(flow.fwd_packets)
    f["fwd_pkt_len_std"] = _safe_std(flow.fwd_packets)

    # ─── Packet length statistics (backward) ────────────────────────
    f["bwd_pkt_len_min"] = _safe_min(flow.bwd_packets)
    f["bwd_pkt_len_max"] = _safe_max(flow.bwd_packets)
    f["bwd_pkt_len_mean"] = _safe_mean(flow.bwd_packets)
    f["bwd_pkt_len_std"] = _safe_std(flow.bwd_packets)

    # ─── Overall packet length statistics ───────────────────────────
    all_sizes = flow.fwd_packets + flow.bwd_packets
    f["pkt_len_min"] = _safe_min(all_sizes)
    f["pkt_len_max"] = _safe_max(all_sizes)
    f["pkt_len_mean"] = _safe_mean(all_sizes)
    f["pkt_len_std"] = _safe_std(all_sizes)
    f["pkt_len_var"] = _safe_var(all_sizes)

    # ─── Rate features ──────────────────────────────────────────────
    total_bytes = fwd_bytes + bwd_bytes
    f["flow_bytes_per_sec"] = total_bytes / dur_safe
    f["flow_pkts_per_sec"] = n_total / dur_safe

    # ─── IAT (Inter-Arrival Time) — overall ─────────────────────────
    all_iats = _compute_iats(sorted(flow.all_timestamps))
    f["flow_iat_mean"] = _safe_mean(all_iats)
    f["flow_iat_std"] = _safe_std(all_iats)
    f["flow_iat_min"] = max(_safe_min(all_iats), 0.0)   # clip negatives
    f["flow_iat_max"] = _safe_max(all_iats)

    # ─── IAT — forward ──────────────────────────────────────────────
    fwd_iats = _compute_iats(flow.fwd_timestamps)
    f["fwd_iat_total"] = sum(fwd_iats) if fwd_iats else 0
    f["fwd_iat_mean"] = _safe_mean(fwd_iats)
    f["fwd_iat_std"] = _safe_std(fwd_iats)
    f["fwd_iat_min"] = max(_safe_min(fwd_iats), 0.0)
    f["fwd_iat_max"] = _safe_max(fwd_iats)

    # ─── IAT — backward ────────────────────────────────────────────
    bwd_iats = _compute_iats(flow.bwd_timestamps)
    f["bwd_iat_total"] = sum(bwd_iats) if bwd_iats else 0
    f["bwd_iat_mean"] = _safe_mean(bwd_iats)
    f["bwd_iat_std"] = _safe_std(bwd_iats)
    f["bwd_iat_min"] = max(_safe_min(bwd_iats), 0.0)
    f["bwd_iat_max"] = _safe_max(bwd_iats)

    # ─── TCP flags ──────────────────────────────────────────────────
    f["flag_FIN"] = flow.flag_counts["FIN"]
    f["flag_SYN"] = flow.flag_counts["SYN"]
    f["flag_RST"] = flow.flag_counts["RST"]
    f["flag_PSH"] = flow.flag_counts["PSH"]
    f["flag_ACK"] = flow.flag_counts["ACK"]
    f["flag_URG"] = flow.flag_counts["URG"]
    f["flag_CWR"] = flow.flag_counts["CWR"]
    f["flag_ECE"] = flow.flag_counts["ECE"]

    f["fwd_flag_PSH"] = flow.fwd_flag_psh
    f["bwd_flag_PSH"] = flow.bwd_flag_psh
    f["fwd_flag_URG"] = flow.fwd_flag_urg
    f["bwd_flag_URG"] = flow.bwd_flag_urg

    # ─── Header lengths ─────────────────────────────────────────────
    f["fwd_header_length"] = sum(flow.fwd_header_lengths)
    f["bwd_header_length"] = sum(flow.bwd_header_lengths)

    # ─── Packets per second per direction ───────────────────────────
    f["fwd_pkts_per_sec"] = n_fwd / dur_safe
    f["bwd_pkts_per_sec"] = n_bwd / dur_safe

    # ─── Packet size extremes ───────────────────────────────────────
    f["min_pkt_size"] = _safe_min(all_sizes)
    f["max_pkt_size"] = _safe_max(all_sizes)

    # ─── Bulk transfer features ─────────────────────────────────────
    fwd_bulk = _compute_bulk_features(flow.fwd_packets, flow.fwd_timestamps)
    bwd_bulk = _compute_bulk_features(flow.bwd_packets, flow.bwd_timestamps)

    f["fwd_avg_bytes_per_bulk"] = fwd_bulk["avg_bytes"]
    f["fwd_avg_pkts_per_bulk"] = fwd_bulk["avg_pkts"]
    f["fwd_avg_bulk_rate"] = fwd_bulk["avg_rate"]
    f["bwd_avg_bytes_per_bulk"] = bwd_bulk["avg_bytes"]
    f["bwd_avg_pkts_per_bulk"] = bwd_bulk["avg_pkts"]
    f["bwd_avg_bulk_rate"] = bwd_bulk["avg_rate"]

    # ─── Subflow features ───────────────────────────────────────────
    n_subflows = max(1, _count_subflows(flow.all_timestamps))
    f["subflow_fwd_packets"] = n_fwd / n_subflows
    f["subflow_fwd_bytes"] = fwd_bytes / n_subflows
    f["subflow_bwd_packets"] = n_bwd / n_subflows
    f["subflow_bwd_bytes"] = bwd_bytes / n_subflows

    # ─── Initial window size ────────────────────────────────────────
    f["init_win_bytes_fwd"] = max(flow.init_win_fwd, 0)
    f["init_win_bytes_bwd"] = max(flow.init_win_bwd, 0)

    # ─── Active/Idle times ──────────────────────────────────────────
    active_times, idle_times = _compute_active_idle(flow.all_timestamps)
    f["active_mean"] = _safe_mean(active_times)
    f["active_std"] = _safe_std(active_times)
    f["active_min"] = _safe_min(active_times)
    f["active_max"] = _safe_max(active_times)
    f["idle_mean"] = _safe_mean(idle_times)
    f["idle_std"] = _safe_std(idle_times)
    f["idle_min"] = _safe_min(idle_times)
    f["idle_max"] = _safe_max(idle_times)

    # ─── TTL ────────────────────────────────────────────────────────
    f["fwd_avg_ttl"] = _safe_mean(flow.fwd_ttls)
    f["bwd_avg_ttl"] = _safe_mean(flow.bwd_ttls)

    # ─── Additional derived ─────────────────────────────────────────
    f["down_up_ratio"] = (n_bwd / n_fwd) if n_fwd > 0 else 0
    f["avg_pkt_size"] = (total_bytes / n_total) if n_total > 0 else 0
    f["fwd_seg_size_avg"] = (fwd_bytes / n_fwd) if n_fwd > 0 else 0
    f["bwd_seg_size_avg"] = (bwd_bytes / n_bwd) if n_bwd > 0 else 0

    # ─── Label placeholders ─────────────────────────────────────────
    f["label"] = ""
    f["detailed_label"] = ""

    return f


# ─── Helper functions ───────────────────────────────────────────────────

def _compute_iats(timestamps: List[float]) -> List[float]:
    """Compute inter-arrival times from sorted timestamps."""
    if len(timestamps) < 2:
        return []
    ts = sorted(timestamps)
    return [ts[i+1] - ts[i] for i in range(len(ts) - 1)]


def _compute_active_idle(timestamps: List[float]) -> Tuple[List[float], List[float]]:
    """Compute active and idle periods based on ACTIVITY_TIMEOUT."""
    if len(timestamps) < 2:
        return [0.0], [0.0]

    ts = sorted(timestamps)
    active_times = []
    idle_times = []
    active_start = ts[0]

    for i in range(1, len(ts)):
        gap = ts[i] - ts[i-1]
        if gap > ACTIVITY_TIMEOUT:
            active_duration = ts[i-1] - active_start
            active_times.append(active_duration)
            idle_times.append(gap)
            active_start = ts[i]

    active_times.append(ts[-1] - active_start)

    return active_times or [0.0], idle_times or [0.0]


def _count_subflows(timestamps: List[float]) -> int:
    """Count subflows based on activity timeout gaps."""
    if len(timestamps) < 2:
        return 1
    ts = sorted(timestamps)
    subflows = 1
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > ACTIVITY_TIMEOUT:
            subflows += 1
    return subflows


def _compute_bulk_features(packet_sizes: List[float],
                           timestamps: List[float]) -> dict:
    """
    Compute bulk transfer metrics (CICFlowMeter-style).
    A bulk transfer is 4+ consecutive packets with minimal IAT.
    """
    BULK_THRESHOLD = 4
    if len(packet_sizes) < BULK_THRESHOLD:
        return {"avg_bytes": 0, "avg_pkts": 0, "avg_rate": 0}

    iats = _compute_iats(timestamps)
    if not iats:
        return {"avg_bytes": 0, "avg_pkts": 0, "avg_rate": 0}

    median_iat = sorted(iats)[len(iats) // 2] if iats else 0
    threshold_iat = median_iat * 0.1

    bulk_bytes = []
    bulk_pkts = []
    bulk_durations = []
    current_bulk_bytes = packet_sizes[0]
    current_bulk_pkts = 1
    current_bulk_start = timestamps[0] if timestamps else 0

    for i, iat in enumerate(iats):
        if iat <= threshold_iat and threshold_iat > 0:
            current_bulk_bytes += packet_sizes[i + 1]
            current_bulk_pkts += 1
        else:
            if current_bulk_pkts >= BULK_THRESHOLD:
                bulk_bytes.append(current_bulk_bytes)
                bulk_pkts.append(current_bulk_pkts)
                dur = timestamps[i] - current_bulk_start
                bulk_durations.append(max(dur, 1e-6))
            current_bulk_bytes = packet_sizes[i + 1] if i + 1 < len(packet_sizes) else 0
            current_bulk_pkts = 1
            current_bulk_start = timestamps[i + 1] if i + 1 < len(timestamps) else 0

    if current_bulk_pkts >= BULK_THRESHOLD:
        bulk_bytes.append(current_bulk_bytes)
        bulk_pkts.append(current_bulk_pkts)
        if timestamps:
            bulk_durations.append(max(timestamps[-1] - current_bulk_start, 1e-6))

    n_bulks = len(bulk_bytes)
    if n_bulks == 0:
        return {"avg_bytes": 0, "avg_pkts": 0, "avg_rate": 0}

    return {
        "avg_bytes": sum(bulk_bytes) / n_bulks,
        "avg_pkts": sum(bulk_pkts) / n_bulks,
        "avg_rate": sum(b / d for b, d in zip(bulk_bytes, bulk_durations)) / n_bulks,
    }


def _safe_min(lst):
    return min(lst) if lst else 0
def _safe_max(lst):
    return max(lst) if lst else 0
def _safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0
def _safe_std(lst):
    return float(np.std(lst)) if lst else 0.0
def _safe_var(lst):
    return float(np.var(lst)) if lst else 0.0


# =========================================================================
# 6A. LABEL MERGING — IoT-23 (conn.log.labeled / Zeek format)
# =========================================================================

def load_labels_iot23(scenario_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    Load labels from IoT-23 conn.log.labeled file (Zeek/Bro tab-separated format).

    IoT-23 conn.log.labeled columns (tab-separated):
      0:ts  1:uid  2:id.orig_h  3:id.orig_p  4:id.resp_h  5:id.resp_p
      6:proto  7:service  8:duration  9:orig_bytes  10:resp_bytes
      11:conn_state  12:local_orig  13:local_resp  14:missed_bytes
      15:history  16:orig_pkts  17:orig_ip_bytes  18:resp_pkts
      19:resp_ip_bytes  20:tunnel_parents  21:label  22:detailed_label

    Matching strategies (in order):
      1. Exact: src_ip:src_port-dst_ip:dst_port-proto-timestamp_bucket
      2. Relaxed: src_ip-dst_ip-dst_port-proto
      3. Reverse: dst_ip-src_ip-src_port-proto
      4. Source-IP majority vote (infected device labeling)
    """
    label_files = glob.glob(os.path.join(scenario_dir, "**", "conn.log.labeled"), recursive=True)
    label_files += glob.glob(os.path.join(scenario_dir, "**", "*.log.labeled"), recursive=True)

    if not label_files:
        return {}

    log.info(f"  [IoT-23] Loading labels from: {label_files[0]}")

    # Protocol mapping: Zeek uses names, PCAP uses numbers
    proto_to_num = {"tcp": "6", "udp": "17", "icmp": "1"}

    labels = {}
    src_ip_labels = defaultdict(list)

    with open(label_files[0], "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
# FIX: handle merged last columns (space instead of tab)
            if len(parts) < 22 and len(parts) >= 20:
                last = parts[-1]
                extra = last.split()
                if len(extra) >= 2:
                    parts = parts[:-1] + extra
            if len(parts) >= 22:
                try:
                    ts = float(parts[0])
                    src_ip = parts[2]
                    dst_ip = parts[4]
                    src_port = parts[3]
                    dst_port = parts[5]
                    proto = parts[6].strip().lower()
                    proto_num = proto_to_num.get(proto, proto)
                    label = parts[21] if len(parts) > 21 else ""
                    detailed = parts[22] if len(parts) > 22 else ""

                    ts_bucket = int(ts)
                    key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto_num}-{ts_bucket}"
                    labels[key] = (label, detailed)

                    key2 = f"{src_ip}-{dst_ip}-{dst_port}-{proto_num}"
                    if key2 not in labels:
                        labels[key2] = (label, detailed)

                    key3 = f"{src_ip}-{dst_ip}-{dst_port}-{proto}"
                    if key3 not in labels:
                        labels[key3] = (label, detailed)

                    src_ip_labels[src_ip].append(label)

                except (ValueError, IndexError):
                    continue

    # Build majority-vote labels per source IP
    src_ip_majority = {}
    for ip, ip_labels in src_ip_labels.items():
        counts = Counter(ip_labels)
        malicious = {k: v for k, v in counts.items()
                     if k.lower() not in ("benign", "-", "")}
        if malicious:
            src_ip_majority[ip] = max(malicious, key=malicious.get)
        else:
            src_ip_majority[ip] = "Benign"

    labels["__src_ip_majority__"] = src_ip_majority

    log.info(f"  [IoT-23] ✓ Loaded {len(labels):,} label entries, "
             f"{len(src_ip_majority)} unique source IPs")
    return labels


# =========================================================================
# 6B. LABEL MERGING — CTU-13 (binetflow / Argus format)
# =========================================================================

def load_labels_ctu13(scenario_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    Load labels from CTU-13 .binetflow files (Argus bidirectional flow format).

    CTU-13 binetflow columns (comma-separated, with header):
      StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
      State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

    Label format examples:
      flow=Background-Established-cmpgw-CVUT
      flow=Normal-V42-UDP-CVUT-DNS-Server
      flow=From-Botnet-V42-TCP-Established
      flow=From-Botnet-V42-UDP-DNS
      flow=Background

    Binary label mapping:
      Contains "Botnet" → 1 (malicious)
      Contains "Normal" → 0 (benign)
      Contains "Background" → -1 (unlabeled — excluded from training)

    Matching strategies (in order):
      1. Exact: src_ip:src_port-dst_ip:dst_port-proto-timestamp_bucket
      2. Relaxed: src_ip-dst_ip-dst_port-proto
      3. Reverse direction: dst_ip:dst_port-src_ip:src_port-proto-ts_bucket
      4. Source-IP majority vote (infected device heuristic)

    Args:
        scenario_dir: Directory containing the .binetflow file
                      (same directory as the PCAP file for CTU-13)

    Returns:
        labels dict mapping flow keys → (binary_label, detailed_label)
        where binary_label is "1", "0", or "-1"
    """
    # CTU-13 binetflow files are in the same directory or in
    # 'detailed-bidirectional-flow-labels/' subdirectory
    binetflow_files = (
        glob.glob(os.path.join(scenario_dir, "*.binetflow")) +
        glob.glob(os.path.join(scenario_dir, "**", "*.binetflow"), recursive=True) +
        glob.glob(os.path.join(scenario_dir, "*.biargus")) +
        glob.glob(os.path.join(scenario_dir, "detailed-bidirectional-flow-labels", "*"))
    )
    binetflow_files = [f for f in binetflow_files if os.path.isfile(f)]

    if not binetflow_files:
        log.warning(f"  [CTU-13] No .binetflow file found in: {scenario_dir}")
        log.warning(f"  [CTU-13] Expected: <scenario_dir>/<name>.binetflow")
        return {}

    binetflow_path = binetflow_files[0]
    log.info(f"  [CTU-13] Loading labels from: {binetflow_path}")

    # CTU-13 protocol name → number mapping
    proto_to_num = {
        "tcp": "6", "udp": "17", "icmp": "1",
        "arp": "0", "igmp": "2", "esp": "50",
        "gre": "47", "ipv6-icmp": "58",
    }

    labels = {}
    src_ip_labels = defaultdict(list)
    total_rows = 0
    labeled_rows = 0

    with open(binetflow_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        # Detect column names (CTU-13 binetflow headers vary slightly)
        # Common variations: 'StartTime'/'starttime', 'SrcAddr'/'srcAddr', etc.
        fieldnames = reader.fieldnames or []
        col_map = _build_ctu13_col_map(fieldnames)

        if not col_map:
            log.error(f"  [CTU-13] Could not parse binetflow headers: {fieldnames[:10]}")
            return {}

        log.info(f"  [CTU-13] Detected columns: {list(col_map.values())}")

        for row in reader:
            total_rows += 1
            try:
                # Extract fields using flexible column mapping
                ts_str = row.get(col_map.get("StartTime", "StartTime"), "").strip()
                proto_raw = row.get(col_map.get("Proto", "Proto"), "").strip().lower()
                src_ip = row.get(col_map.get("SrcAddr", "SrcAddr"), "").strip()
                src_port = row.get(col_map.get("Sport", "Sport"), "0").strip()
                dst_ip = row.get(col_map.get("DstAddr", "DstAddr"), "").strip()
                dst_port = row.get(col_map.get("Dport", "Dport"), "0").strip()
                detailed_label = row.get(col_map.get("Label", "Label"), "").strip()

                if not src_ip or not dst_ip or not detailed_label:
                    continue

                # Convert port to plain integer string (CTU-13 uses hex for some ports)
                src_port = _ctu13_port_to_str(src_port)
                dst_port = _ctu13_port_to_str(dst_port)

                # Map protocol name → number
                proto_num = proto_to_num.get(proto_raw, "0")

                # Map detailed CTU-13 label → binary label
                binary_label = _ctu13_label_to_binary(detailed_label)

                # Parse timestamp
                ts = _parse_ctu13_timestamp(ts_str)
                ts_bucket = int(ts) if ts else 0

                # ── Strategy 1: Exact 5-tuple + timestamp bucket ──────────
                key1 = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto_num}-{ts_bucket}"
                labels[key1] = (binary_label, detailed_label)

                # ── Strategy 2: Relaxed (no timestamp) ────────────────────
                key2 = f"{src_ip}-{dst_ip}-{dst_port}-{proto_num}"
                if key2 not in labels:
                    labels[key2] = (binary_label, detailed_label)

                # ── Strategy 3: Also with proto name ──────────────────────
                key3 = f"{src_ip}-{dst_ip}-{dst_port}-{proto_raw}"
                if key3 not in labels:
                    labels[key3] = (binary_label, detailed_label)

                # ── Strategy 4: Reverse direction ─────────────────────────
                key4 = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto_num}-{ts_bucket}"
                if key4 not in labels:
                    labels[key4] = (binary_label, detailed_label)

                key5 = f"{dst_ip}-{src_ip}-{src_port}-{proto_num}"
                if key5 not in labels:
                    labels[key5] = (binary_label, detailed_label)

                # Track per-source-IP for majority vote
                src_ip_labels[src_ip].append(binary_label)
                labeled_rows += 1

            except (ValueError, KeyError, TypeError):
                continue

    # Build majority-vote labels per source IP
    # For CTU-13: infected devices generate mostly botnet traffic
    src_ip_majority = {}
    for ip, ip_labels in src_ip_labels.items():
        counts = Counter(ip_labels)
        # Prefer "1" (botnet) over "0" (benign) for infected IPs
        if counts.get("1", 0) > counts.get("0", 0):
            src_ip_majority[ip] = ("1", "flow=From-Botnet-majority-vote")
        else:
            src_ip_majority[ip] = ("0", "flow=Normal-majority-vote")

    labels["__src_ip_majority__"] = src_ip_majority

    log.info(f"  [CTU-13] ✓ Loaded {labeled_rows:,}/{total_rows:,} labeled rows, "
             f"{len(src_ip_majority)} unique source IPs")
    return labels


def _build_ctu13_col_map(fieldnames: List[str]) -> Dict[str, str]:
    """
    Build a mapping from canonical column names to actual column names.
    Handles case variations in CTU-13 binetflow files.
    """
    # Canonical name → list of possible actual names (case-insensitive)
    canonical_map = {
        "StartTime": ["StartTime", "starttime", "start_time", "Stime"],
        "Proto":     ["Proto", "proto", "Protocol", "protocol"],
        "SrcAddr":   ["SrcAddr", "srcaddr", "src_ip", "SrcIP"],
        "Sport":     ["Sport", "sport", "src_port", "SrcPort"],
        "DstAddr":   ["DstAddr", "dstaddr", "dst_ip", "DstIP"],
        "Dport":     ["Dport", "dport", "dst_port", "DstPort"],
        "Label":     ["Label", "label", "class", "Class"],
    }

    col_map = {}
    fieldnames_lower = {f.strip().lower(): f.strip() for f in (fieldnames or [])}

    for canonical, variants in canonical_map.items():
        for v in variants:
            if v.lower() in fieldnames_lower:
                col_map[canonical] = fieldnames_lower[v.lower()]
                break

    return col_map if len(col_map) >= 4 else {}


def _ctu13_port_to_str(port_str: str) -> str:
    """
    Convert CTU-13 port field to plain integer string.
    CTU-13 sometimes uses hex (0x0050) or service names (http).
    """
    port_str = port_str.strip().lower()
    if not port_str or port_str == "0":
        return "0"

    # Try hex
    if port_str.startswith("0x"):
        try:
            return str(int(port_str, 16))
        except ValueError:
            pass

    # Try plain integer
    try:
        return str(int(port_str))
    except ValueError:
        pass

    # Service name → port number mapping (common ones)
    service_ports = {
        "http": "80", "https": "443", "dns": "53", "ftp": "21",
        "ssh": "22", "smtp": "25", "telnet": "23", "pop3": "110",
        "imap": "143", "snmp": "161", "ntp": "123", "irc": "6667",
    }
    return service_ports.get(port_str, "0")


def _ctu13_label_to_binary(detailed_label: str) -> str:
    """
    Map CTU-13 detailed label string to binary label.

    CTU-13 label structure: flow=<Category>-<Details>
    Categories:
      - Background: unlabeled/noise → -1 (exclude from training)
      - Normal: legitimate traffic → 0
      - From-Botnet: malicious → 1
      - To-Botnet: malicious → 1

    Returns: "1" (botnet), "0" (benign), or "-1" (background/unknown)
    """
    label_lower = detailed_label.lower()

    if "botnet" in label_lower:
        return "1"
    elif "normal" in label_lower:
        return "0"
    elif "background" in label_lower:
        return "-1"   # will be filtered during preprocessing
    else:
        return "-1"   # unknown → exclude


def _parse_ctu13_timestamp(ts_str: str) -> float:
    """
    Parse CTU-13 timestamp into Unix epoch float.

    CTU-13 timestamp format: '2011-08-10 09:46:53.047277'
    or sometimes epoch float: '1312969613.047277'
    """
    ts_str = ts_str.strip()
    if not ts_str:
        return 0.0

    # Try plain float (Unix epoch)
    try:
        return float(ts_str)
    except ValueError:
        pass

    # Try datetime format
    from datetime import datetime
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    return 0.0


# =========================================================================
# 6C. UNIFIED LABEL MERGING (works for both datasets)
# =========================================================================

def merge_labels(flows_csv: str, labels: Dict, dataset: str = "iot23") -> None:
    """
    Merge labels into an existing flows CSV file using multi-strategy matching.
    Works for both IoT-23 and CTU-13 label formats.

    For CTU-13: rows with label="-1" (Background) are marked but NOT removed
    here — they can be filtered during preprocessing.
    """
    if not labels:
        log.warning("  No labels to merge.")
        return

    import pandas as pd
    df = pd.read_csv(flows_csv)
    matched = 0
    src_ip_majority = labels.pop("__src_ip_majority__", {})

    df["label"] = df["label"].astype(str).replace("nan", "")
    df["detailed_label"] = df["detailed_label"].astype(str).replace("nan", "")

    for idx, row in df.iterrows():
        src_ip = str(row.get("src_ip", ""))
        dst_ip = str(row.get("dst_ip", ""))
        src_port = str(int(row.get("src_port", 0)))
        dst_port = str(int(row.get("dst_port", 0)))
        proto = str(int(row.get("protocol", 0)))
        ts_bucket = int(row.get("timestamp", 0))

        found = False

        # Strategy 1: Exact 5-tuple + timestamp
        key1 = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}-{ts_bucket}"
        if key1 in labels:
            lbl = labels[key1]
            df.at[idx, "label"] = lbl[0]
            df.at[idx, "detailed_label"] = lbl[1]
            found = True

        # Strategy 2: Relaxed (no timestamp)
        if not found:
            key2 = f"{src_ip}-{dst_ip}-{dst_port}-{proto}"
            if key2 in labels:
                lbl = labels[key2]
                df.at[idx, "label"] = lbl[0]
                df.at[idx, "detailed_label"] = lbl[1]
                found = True

        # Strategy 3: Reverse direction
        if not found:
            key3 = f"{dst_ip}-{src_ip}-{src_port}-{proto}"
            if key3 in labels:
                lbl = labels[key3]
                df.at[idx, "label"] = lbl[0]
                df.at[idx, "detailed_label"] = lbl[1]
                found = True

        # Strategy 4: Source IP majority vote
        if not found and src_ip in src_ip_majority:
            maj = src_ip_majority[src_ip]
            if isinstance(maj, tuple):
                df.at[idx, "label"] = maj[0]
                df.at[idx, "detailed_label"] = maj[1]
            else:
                df.at[idx, "label"] = maj
                df.at[idx, "detailed_label"] = maj
            found = True

        if found:
            matched += 1

    # Restore majority dict
    labels["__src_ip_majority__"] = src_ip_majority

    df.to_csv(flows_csv, index=False)

    if dataset == "ctu13":
        # Report breakdown for CTU-13
        labeled = df[df["label"] != ""]
        botnet_count = (labeled["label"] == "1").sum()
        benign_count = (labeled["label"] == "0").sum()
        bg_count = (labeled["label"] == "-1").sum()
        log.info(f"  [CTU-13] ✓ Labels merged: {matched:,}/{len(df):,} flows matched")
        log.info(f"  [CTU-13]   Botnet (1): {botnet_count:,} | "
                 f"Benign (0): {benign_count:,} | "
                 f"Background (-1): {bg_count:,} (will be filtered in preprocessing)")
    else:
        log.info(f"  [IoT-23] ✓ Labels merged: {matched:,}/{len(df):,} flows matched")


# =========================================================================
# 7. MAIN CONVERSION FUNCTIONS
# =========================================================================

def convert_single_pcap(pcap_path: str, output_csv: str,
                        max_packets: int = None,
                        label_dir: str = None,
                        dataset: str = "iot23") -> str:
    """
    Convert a single PCAP file to a CSV with CICFlowMeter-style features.

    Args:
        pcap_path:   Path to the input PCAP file
        output_csv:  Path for the output CSV file
        max_packets: Max packets to process (None = all)
        label_dir:   Directory containing label files (None = no labeling)
        dataset:     "iot23" or "ctu13" — determines label loader
    """
    log.info(f"\n{'─'*60}")
    log.info(f"Processing: {pcap_path}")
    log.info(f"Output:     {output_csv}")
    log.info(f"Dataset:    {dataset.upper()}")

    # Step 1: Extract packets via tshark
    packets = extract_packets_tshark(pcap_path, max_packets)
    if not packets:
        log.warning(f"  ⚠ No packets extracted from {pcap_path}")
        return ""

    # Step 2: Reconstruct flows
    flows = reconstruct_flows(packets)

    # Step 3: Compute features
    log.info(f"  Computing CICFlowMeter-style features for {len(flows):,} flows...")
    rows = []
    for flow_key, flow in flows.items():
        features = compute_flow_features(flow)
        rows.append(features)

    # Step 4: Write CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(rows)

    cols_present = [c for c in OUTPUT_COLUMNS if c in df.columns]
    cols_extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    df = df[cols_present + cols_extra]
    df.to_csv(output_csv, index=False)

    log.info(f"  ✓ Written {len(df):,} flows → {output_csv}")

    # Step 5: Merge labels if a label directory is provided
    if label_dir:
        if dataset == "ctu13":
            labels = load_labels_ctu13(label_dir)
        else:
            labels = load_labels_iot23(label_dir)

        if labels:
            merge_labels(output_csv, labels, dataset=dataset)

    return output_csv


def convert_iot23_dataset(data_dir: str, output_dir: str,
                          max_packets: int = None) -> List[str]:
    """
    Convert all PCAP files in an IoT-23 dataset directory to CSVs.

    Expected IoT-23 directory structure:
        data_dir/
          CTU-IoT-Malware-Capture-34-1/
            2019-01-09-22-46-52-192.168.1.195.pcap
            bro/conn.log.labeled
          CTU-IoT-Malware-Capture-43-1/
            ...
    """
    log.info("=" * 70)
    log.info("  IoT-23 PCAP → CSV Converter")
    log.info("=" * 70)
    log.info(f"  Input:  {data_dir}")
    log.info(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pcap_files = _find_pcap_files(data_dir)
    if not pcap_files:
        log.error(f"No PCAP files found in {data_dir}")
        log.info("  Expected IoT-23 directory structure:")
        log.info("    data_dir/CTU-IoT-Malware-Capture-XX-1/*.pcap")
        return []

    log.info(f"  Found {len(pcap_files)} PCAP file(s)")

    output_csvs = []
    for i, pcap_path in enumerate(pcap_files, 1):
        log.info(f"\n[{i}/{len(pcap_files)}]")
        scenario = os.path.basename(os.path.dirname(pcap_path))
        pcap_name = Path(pcap_path).stem
        csv_name = f"{scenario}_{pcap_name}_flows.csv"
        csv_path = os.path.join(output_dir, csv_name)
        label_dir = os.path.dirname(pcap_path)

        result = convert_single_pcap(pcap_path, csv_path,
                                     max_packets=max_packets,
                                     label_dir=label_dir,
                                     dataset="iot23")
        if result:
            output_csvs.append(result)

    return _combine_csvs(output_csvs, output_dir, "iot23_all_flows.csv")


def convert_etf_dataset(data_dir: str, output_dir: str,
                        max_packets: int = None,
                        master_name: str = "etf_all_flows.csv") -> List[str]:
    """
    Convert all PCAP files in an ETF dataset folder to CSVs.

    ETF (Mendeley IoT Botnet Dataset) does NOT ship with conn.log.labeled
    or .binetflow label files. Labels are implicit in the folder name
    (e.g. malware/, benigniot/). This function therefore SKIPS label
    loading entirely — labels are assigned later in the preprocessor
    based on which folder each flow came from.

    Expected ETF directory structure:
        data_dir/
          *.pcap                 (or)
          subfolder/*.pcap       (recursive search supported)

    Args:
        data_dir:    ETF folder containing .pcap files (e.g. ETF/malware/)
        output_dir:  Where to write per-PCAP CSVs and the master CSV
        max_packets: Max packets per PCAP (None = all)
        master_name: Filename of the combined master CSV
                     (default: etf_all_flows.csv)
    """
    log.info("=" * 70)
    log.info("  ETF PCAP → CSV Converter (no label files expected)")
    log.info("=" * 70)
    log.info(f"  Input:  {data_dir}")
    log.info(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pcap_files = _find_pcap_files(data_dir)
    if not pcap_files:
        log.error(f"No PCAP files found in {data_dir}")
        return []

    log.info(f"  Found {len(pcap_files)} PCAP file(s)")
    log.info(f"  (Labels will be assigned later by the preprocessor)")

    output_csvs = []
    for i, pcap_path in enumerate(pcap_files, 1):
        log.info(f"\n[{i}/{len(pcap_files)}]")

        # Use parent folder name + pcap stem to build a unique CSV name
        parent = os.path.basename(os.path.dirname(pcap_path)) or "etf"
        pcap_name = Path(pcap_path).stem
        csv_name = f"etf_{parent}_{pcap_name}_flows.csv"
        csv_path = os.path.join(output_dir, csv_name)

        # CRITICAL: pass label_dir=None so no label loader runs.
        # ETF has no conn.log.labeled or binetflow files; labels come
        # from folder name later in the preprocessor.
        result = convert_single_pcap(pcap_path, csv_path,
                                     max_packets=max_packets,
                                     label_dir=None,
                                     dataset="iot23")
        if result:
            output_csvs.append(result)

    return _combine_csvs(output_csvs, output_dir, master_name)


def convert_ctu13_dataset(data_dir: str, output_dir: str,
                          max_packets: int = None) -> List[str]:
    """
    Convert all PCAP files in a CTU-13 dataset directory to CSVs.

    CTU-13 has two possible directory structures:

    Structure A — official download (one directory per scenario):
        data_dir/
          1/   (Scenario 1 — Neris)
            capture20110810.pcap
            capture20110810.binetflow
          2/   (Scenario 2 — Neris)
            ...
          13/
            ...

    Structure B — flat (all files in one directory):
        data_dir/
          capture20110810.pcap
          capture20110810.binetflow
          capture20110811.pcap
          capture20110811.binetflow

    The script handles both automatically.

    After conversion, Background-labeled flows (label=-1) are present
    in the CSV but should be filtered during preprocessing with:
        df = df[df['label'] != '-1']
    """
    log.info("=" * 70)
    log.info("  CTU-13 PCAP → CSV Converter")
    log.info("=" * 70)
    log.info(f"  Input:  {data_dir}")
    log.info(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pcap_files = _find_pcap_files(data_dir)
    if not pcap_files:
        log.error(f"No PCAP files found in {data_dir}")
        log.info("  Expected CTU-13 directory structure:")
        log.info("    data_dir/<scenario_num>/<scenario>.pcap")
        log.info("    data_dir/<scenario_num>/<scenario>.binetflow")
        return []

    log.info(f"  Found {len(pcap_files)} PCAP file(s)")

    output_csvs = []
    for i, pcap_path in enumerate(pcap_files, 1):
        log.info(f"\n[{i}/{len(pcap_files)}]")

        # Scenario name = parent directory name or PCAP stem
        scenario = os.path.basename(os.path.dirname(pcap_path))
        pcap_name = Path(pcap_path).stem
        csv_name = f"ctu13_{scenario}_{pcap_name}_flows.csv"
        csv_path = os.path.join(output_dir, csv_name)

        # Label dir = same directory as PCAP (binetflow is co-located)
        label_dir = os.path.dirname(pcap_path)

        result = convert_single_pcap(pcap_path, csv_path,
                                     max_packets=max_packets,
                                     label_dir=label_dir,
                                     dataset="ctu13")
        if result:
            output_csvs.append(result)

    combined = _combine_csvs(output_csvs, output_dir, "ctu13_all_flows.csv")

    # Print CTU-13-specific summary
    if combined:
        try:
            import pandas as pd
            master_path = os.path.join(output_dir, "ctu13_all_flows.csv")
            if os.path.exists(master_path):
                df = pd.read_csv(master_path)
                log.info(f"\n{'─'*60}")
                log.info("  CTU-13 Label Distribution (before filtering Background):")
                log.info(f"    Botnet  (label=1):  {(df['label']=='1').sum():>10,}")
                log.info(f"    Benign  (label=0):  {(df['label']=='0').sum():>10,}")
                log.info(f"    Bgrd    (label=-1): {(df['label']=='-1').sum():>10,}")
                log.info(f"    Unknown (label=''):  {(df['label']=='').sum():>10,}")
                log.info(f"    Total:              {len(df):>10,}")
                log.info(f"{'─'*60}")
                log.info("  ⚠  Filter Background rows before training:")
                log.info("     df = df[df['label'] != '-1']")
        except Exception:
            pass

    return combined


def _find_pcap_files(data_dir: str) -> List[str]:
    """Find all PCAP/PCAPNG/CAP files in a directory tree."""
    pcap_files = []
    for pattern in ["**/*.pcap", "**/*.pcapng", "**/*.cap"]:
        pcap_files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    return sorted(set(pcap_files))


def _combine_csvs(output_csvs: List[str], output_dir: str,
                  master_name: str) -> List[str]:
    """Combine multiple scenario CSVs into one master file."""
    if not output_csvs:
        return []

    log.info(f"\n{'─'*60}")
    log.info(f"Combining {len(output_csvs)} scenario CSV(s) into master file...")

    import pandas as pd
    dfs = []
    for csv_path in output_csvs:
        df = pd.read_csv(csv_path)
        df["scenario_id"] = Path(csv_path).stem
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)
    master_path = os.path.join(output_dir, master_name)
    master.to_csv(master_path, index=False)

    log.info(f"  ✓ Master CSV: {master_path} ({len(master):,} total flows)")
    output_csvs.append(master_path)

    log.info(f"\n{'='*70}")
    log.info("  CONVERSION COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"  PCAPs processed: {len(output_csvs) - 1}")
    log.info(f"  CSVs generated:  {len(output_csvs)}")
    log.info(f"  Output dir:      {output_dir}")
    log.info(f"{'='*70}")

    return output_csvs


# =========================================================================
# 8. GENERATE SAMPLE PCAP (for testing without a real dataset)
# =========================================================================

def generate_sample_pcap(output_path: str, n_packets: int = 5000) -> str:
    """
    Generate a small synthetic PCAP file for testing the converter.
    Uses dpkt to create realistic-looking TCP/UDP packets.
    """
    import dpkt

    log.info(f"Generating sample PCAP with {n_packets:,} packets...")

    writer = dpkt.pcap.Writer(open(output_path, "wb"))
    np.random.seed(42)

    base_ts = 1553810000.0  # March 2019

    iot_ips = [f"192.168.1.{i}" for i in range(100, 110)]
    target_ips = [f"10.0.0.{i}" for i in range(1, 50)]
    benign_ports = [80, 443, 53, 123, 1883, 8080]
    malicious_ports = [23, 2323, 5555, 37215, 48101]

    for i in range(n_packets):
        ts = base_ts + i * np.random.exponential(0.01)

        if np.random.random() < 0.4:
            src_ip = np.random.choice(iot_ips)
            dst_ip = np.random.choice(target_ips)
            dst_port = np.random.choice(benign_ports)
            payload_size = int(np.random.exponential(200))
            flags = dpkt.tcp.TH_ACK
        else:
            src_ip = np.random.choice(iot_ips[:3])
            dst_ip = f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}"
            dst_port = np.random.choice(malicious_ports)
            payload_size = int(np.random.exponential(50))
            flags = np.random.choice([dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK | dpkt.tcp.TH_SYN, dpkt.tcp.TH_RST])

        src_port = np.random.randint(1024, 65535)

        tcp = dpkt.tcp.TCP(
            sport=src_port,
            dport=dst_port,
            flags=flags,
            seq=np.random.randint(0, 2**32),
            data=b"\x00" * min(payload_size, 1400)
        )

        ip = dpkt.ip.IP(
            src=bytes(int(x) for x in src_ip.split(".")),
            dst=bytes(int(x) for x in dst_ip.split(".")),
            p=dpkt.ip.IP_PROTO_TCP,
            ttl=np.random.randint(32, 128),
            data=tcp
        )
        ip.len = len(ip)

        eth = dpkt.ethernet.Ethernet(
            data=ip,
            type=dpkt.ethernet.ETH_TYPE_IP
        )

        writer.writepkt(bytes(eth), ts=ts)

    writer.close()
    log.info(f"  ✓ Sample PCAP: {output_path} ({n_packets:,} packets)")
    return output_path


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCAP → CSV Converter for IoT-23 and CTU-13 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IoT-23 — single PCAP
  python pcap_to_csv.py --dataset iot23 --pcap ./capture.pcap --output ./flows.csv

  # IoT-23 — entire dataset directory
  python pcap_to_csv.py --dataset iot23 --data_dir ./iot23_full --output_dir ./iot23_csv

  # CTU-13 — single scenario PCAP
  python pcap_to_csv.py --dataset ctu13 --pcap ./1/capture20110810.pcap --output ./flows.csv

  # CTU-13 — entire dataset (all 13 scenarios)
  python pcap_to_csv.py --dataset ctu13 --data_dir ./CTU-13-Dataset --output_dir ./ctu13_csv

  # ETF (Mendeley) — IoT botnet captures (no label files; labels come from folder)
  python pcap_to_csv.py --dataset etf --data_dir ./ETF/malware    --output_dir ./pcap_csv/etf_malware
  python pcap_to_csv.py --dataset etf --data_dir ./ETF/benigniot  --output_dir ./pcap_csv/etf_benign

  # Test with a generated sample PCAP (no real data needed)
  python pcap_to_csv.py --generate_sample --output_dir ./test_output

  # Limit packets for quick testing
  python pcap_to_csv.py --dataset ctu13 --data_dir ./CTU-13 --output_dir ./out --max_packets 100000
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap", type=str,
                       help="Single PCAP file to convert")
    group.add_argument("--data_dir", type=str,
                       help="Dataset root directory (converts all PCAPs)")
    group.add_argument("--generate_sample", action="store_true",
                       help="Generate a sample PCAP and convert it (for testing)")

    parser.add_argument("--dataset", type=str, choices=["iot23", "ctu13", "etf"],
                        default="iot23",
                        help="Dataset type — determines label loader "
                             "(iot23/ctu13/etf, default: iot23). "
                             "ETF mode skips label loading entirely.")
    parser.add_argument("--output", type=str, default="./flows.csv",
                        help="Output CSV path (for single PCAP mode)")
    parser.add_argument("--output_dir", type=str, default="./output_csv",
                        help="Output directory (for dataset mode)")
    parser.add_argument("--max_packets", type=int, default=None,
                        help="Max packets to process per PCAP (for testing)")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Directory with label files (overrides default = PCAP dir)")

    args = parser.parse_args()

    if args.generate_sample:
        os.makedirs(args.output_dir, exist_ok=True)
        pcap_path = os.path.join(args.output_dir, "sample.pcap")
        generate_sample_pcap(pcap_path, n_packets=5000)
        convert_single_pcap(
            pcap_path,
            os.path.join(args.output_dir, "sample_flows.csv"),
            max_packets=args.max_packets,
            dataset=args.dataset,
        )

    elif args.pcap:
        label_dir = args.label_dir or os.path.dirname(args.pcap)
        convert_single_pcap(
            args.pcap, args.output,
            max_packets=args.max_packets,
            label_dir=label_dir,
            dataset=args.dataset,
        )

    else:  # --data_dir
        if args.dataset == "ctu13":
            convert_ctu13_dataset(args.data_dir, args.output_dir,
                                  max_packets=args.max_packets)
        elif args.dataset == "etf":
            # ETF mode: derive a sensible master filename from the input
            # folder so users running malware/ and benigniot/ separately
            # get distinct master CSVs (etf_malware_all_flows.csv vs
            # etf_benigniot_all_flows.csv) instead of overwriting each other.
            folder_name = os.path.basename(os.path.normpath(args.data_dir))
            master_name = f"etf_{folder_name}_all_flows.csv"
            convert_etf_dataset(args.data_dir, args.output_dir,
                                max_packets=args.max_packets,
                                master_name=master_name)
        else:
            convert_iot23_dataset(args.data_dir, args.output_dir,
                                  max_packets=args.max_packets)
