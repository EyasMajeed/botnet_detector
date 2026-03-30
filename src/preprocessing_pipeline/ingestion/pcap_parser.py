"""
pcap_parser.py — PCAP → Flow Records Converter
════════════════════════════════════════════════
Parses raw PCAP files into flow-level CSV using tshark (preferred) or scapy.
This module is used when datasets only provide PCAP files and you need
to extract flow features before running the dataset-specific preprocessor.

USAGE:
  python -m src.ingestion.pcap_parser --input path/to/file.pcap --output flows.csv

DEPENDENCIES:
  - tshark (Wireshark CLI) — preferred, must be on PATH
  - OR scapy (pip install scapy) — fallback, slower
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────
#  TSHARK-BASED EXTRACTION
# ─────────────────────────────────────────────
TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src", "ip.dst",
    "tcp.srcport", "udp.srcport",
    "tcp.dstport", "udp.dstport",
    "ip.proto",
    "frame.len",
    "tcp.flags",
    "ip.ttl",
    "dns.qry.name",
    "tls.handshake.version",
    "tls.handshake.ciphersuite",
]


def check_tshark() -> bool:
    """Check if tshark is available."""
    try:
        subprocess.run(["tshark", "--version"],
                       capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_with_tshark(pcap_path: str, output_csv: str):
    """Use tshark to extract packet fields, then aggregate into flows."""
    print(f"  Extracting packets with tshark from {pcap_path} ...")

    field_args = []
    for f in TSHARK_FIELDS:
        field_args.extend(["-e", f])

    cmd = [
        "tshark", "-r", pcap_path,
        "-T", "fields",
        *field_args,
        "-E", "separator=|",
        "-E", "header=y",
        "-E", "quote=n",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [ERROR] tshark failed: {result.stderr[:200]}")
        return False

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        print("  [WARN] No packets extracted")
        return False

    # Parse into packets
    header = lines[0].split("|")
    packets = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) == len(header):
            packets.append(dict(zip(header, parts)))

    print(f"  Extracted {len(packets):,} packets, aggregating into flows ...")

    # Aggregate into bidirectional flows
    flows = aggregate_packets_to_flows(packets)

    # Write CSV
    write_flows_csv(flows, output_csv)
    print(f"  Wrote {len(flows):,} flows to {output_csv}")
    return True


# ─────────────────────────────────────────────
#  SCAPY-BASED EXTRACTION (FALLBACK)
# ─────────────────────────────────────────────
def extract_with_scapy(pcap_path: str, output_csv: str):
    """Fallback: use scapy to read PCAP and aggregate into flows."""
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        print("  [ERROR] Neither tshark nor scapy available.")
        print("  Install tshark: sudo apt install tshark")
        print("  Or install scapy: pip install scapy")
        return False

    print(f"  Reading PCAP with scapy (this may take a while) ...")
    try:
        pkts = rdpcap(pcap_path)
    except Exception as e:
        print(f"  [ERROR] Could not read PCAP: {e}")
        return False

    print(f"  Read {len(pkts):,} packets, aggregating ...")

    packets = []
    for pkt in pkts:
        if IP in pkt:
            rec = {
                "frame.time_epoch": str(float(pkt.time)),
                "ip.src": pkt[IP].src,
                "ip.dst": pkt[IP].dst,
                "ip.proto": str(pkt[IP].proto),
                "frame.len": str(len(pkt)),
                "ip.ttl": str(pkt[IP].ttl),
            }
            if TCP in pkt:
                rec["tcp.srcport"] = str(pkt[TCP].sport)
                rec["tcp.dstport"] = str(pkt[TCP].dport)
                rec["tcp.flags"] = str(pkt[TCP].flags)
            elif UDP in pkt:
                rec["udp.srcport"] = str(pkt[UDP].sport)
                rec["udp.dstport"] = str(pkt[UDP].dport)
            packets.append(rec)

    flows = aggregate_packets_to_flows(packets)
    write_flows_csv(flows, output_csv)
    print(f"  Wrote {len(flows):,} flows to {output_csv}")
    return True


# ─────────────────────────────────────────────
#  FLOW AGGREGATION
# ─────────────────────────────────────────────
def aggregate_packets_to_flows(packets: list[dict]) -> list[dict]:
    """
    Group packets into bidirectional flows using the 5-tuple
    (src_ip, dst_ip, src_port, dst_port, proto).
    Compute flow-level statistics.
    """
    flow_map = defaultdict(list)

    for pkt in packets:
        src_ip = pkt.get("ip.src", "")
        dst_ip = pkt.get("ip.dst", "")
        src_port = pkt.get("tcp.srcport") or pkt.get("udp.srcport") or "0"
        dst_port = pkt.get("tcp.dstport") or pkt.get("udp.dstport") or "0"
        proto = pkt.get("ip.proto", "0")

        # Bidirectional key: sort endpoints
        key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)])) + (proto,)

        # Track direction: forward = matches first sorted endpoint
        is_fwd = (src_ip, src_port) <= (dst_ip, dst_port)

        pkt["_is_forward"] = is_fwd
        pkt["_src_port"] = src_port
        pkt["_dst_port"] = dst_port
        flow_map[key].append(pkt)

    flows = []
    for key, pkts in flow_map.items():
        ts_list = [float(p.get("frame.time_epoch", 0)) for p in pkts]
        sizes = [int(p.get("frame.len", 0)) for p in pkts]

        fwd_pkts = [p for p in pkts if p.get("_is_forward")]
        bwd_pkts = [p for p in pkts if not p.get("_is_forward")]

        fwd_sizes = [int(p.get("frame.len", 0)) for p in fwd_pkts]
        bwd_sizes = [int(p.get("frame.len", 0)) for p in bwd_pkts]

        duration = max(ts_list) - min(ts_list) if len(ts_list) > 1 else 0

        # IAT computation
        sorted_ts = sorted(ts_list)
        iats = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]

        # TCP flags
        flags = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
        for p in pkts:
            f = p.get("tcp.flags", "")
            if "F" in str(f).upper(): flags["FIN"] += 1
            if "S" in str(f).upper(): flags["SYN"] += 1
            if "R" in str(f).upper(): flags["RST"] += 1
            if "P" in str(f).upper(): flags["PSH"] += 1
            if "A" in str(f).upper(): flags["ACK"] += 1
            if "U" in str(f).upper(): flags["URG"] += 1

        flow = {
            "flow_duration": duration,
            "total_fwd_packets": len(fwd_pkts),
            "total_bwd_packets": len(bwd_pkts),
            "total_fwd_bytes": sum(fwd_sizes),
            "total_bwd_bytes": sum(bwd_sizes),
            "fwd_pkt_len_min": min(fwd_sizes) if fwd_sizes else 0,
            "fwd_pkt_len_max": max(fwd_sizes) if fwd_sizes else 0,
            "fwd_pkt_len_mean": np.mean(fwd_sizes) if fwd_sizes else 0,
            "fwd_pkt_len_std": np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0,
            "bwd_pkt_len_min": min(bwd_sizes) if bwd_sizes else 0,
            "bwd_pkt_len_max": max(bwd_sizes) if bwd_sizes else 0,
            "bwd_pkt_len_mean": np.mean(bwd_sizes) if bwd_sizes else 0,
            "bwd_pkt_len_std": np.std(bwd_sizes) if len(bwd_sizes) > 1 else 0,
            "flow_bytes_per_sec": sum(sizes) / duration if duration > 0 else 0,
            "flow_pkts_per_sec": len(pkts) / duration if duration > 0 else 0,
            "flow_iat_min": min(iats) if iats else 0,
            "flow_iat_max": max(iats) if iats else 0,
            "flow_iat_mean": np.mean(iats) if iats else 0,
            "flow_iat_std": np.std(iats) if len(iats) > 1 else 0,
            "protocol": pkts[0].get("ip.proto", "0"),
            "src_port": pkts[0].get("_src_port", "0"),
            "dst_port": pkts[0].get("_dst_port", "0"),
            "flag_FIN": flags["FIN"],
            "flag_SYN": flags["SYN"],
            "flag_RST": flags["RST"],
            "flag_PSH": flags["PSH"],
            "flag_ACK": flags["ACK"],
            "flag_URG": flags["URG"],
        }
        flows.append(flow)

    return flows


def write_flows_csv(flows: list[dict], output_path: str):
    """Write aggregated flows to CSV."""
    if not flows:
        return
    keys = flows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flows)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP files to flow-level CSV"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to PCAP file or directory of PCAPs")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        pcaps = sorted(input_path.glob("*.pcap")) + sorted(input_path.glob("*.pcapng"))
        if not pcaps:
            print(f"No PCAP files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(pcaps)} PCAP files")
        # Process each and concatenate
        all_flows = []
        has_tshark = check_tshark()
        for pcap in pcaps:
            tmp_csv = str(output_path) + ".tmp"
            if has_tshark:
                extract_with_tshark(str(pcap), tmp_csv)
            else:
                extract_with_scapy(str(pcap), tmp_csv)
            if os.path.exists(tmp_csv):
                import pandas as pd
                all_flows.append(pd.read_csv(tmp_csv))
                os.remove(tmp_csv)

        if all_flows:
            import pandas as pd
            combined = pd.concat(all_flows, ignore_index=True)
            combined.to_csv(str(output_path), index=False)
            print(f"\nTotal flows: {len(combined):,} → {output_path}")
    else:
        has_tshark = check_tshark()
        if has_tshark:
            extract_with_tshark(str(input_path), str(output_path))
        else:
            extract_with_scapy(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
