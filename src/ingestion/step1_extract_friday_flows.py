"""
════════════════════════════════════════════════════════════════════════
 Step 1 — Extract flows from CIC-IDS-2017 Friday PCAP
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT : data/raw/CIC-IDS-2017/Friday-WorkingHours.pcap
 OUTPUT: data/raw/CIC-IDS-2017/friday_flows.csv

 WHY THIS WORKS FOR CNN-LSTM:
   The Ares botnet runs from 13:02 to 14:02 (ADT).
   Infected machines (192.168.10.x) have BENIGN flows before 13:02
   and BOTNET flows during 13:02-14:02.  Per-device sequences built
   chronologically will therefore cross the benign→botnet boundary —
   which is exactly the temporal signal the LSTM needs.

 RUNTIME: ~20–40 min for the full 7GB PCAP on a typical machine.

 INSTALL:  pip install nfstream
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PCAP_PATH = ROOT / "data" / "raw" / "CIC-IDS-2017" / "Friday-WorkingHours.pcap"
OUT_DIR   = ROOT / "data" / "raw" / "CIC-IDS-2017"
OUT_PATH  = OUT_DIR / "friday_flows.csv"

# ── Botnet time window ─────────────────────────────────────────────────
# CIC-IDS-2017: Ares Botnet runs 13:02–14:02 ADT (= 16:02–17:02 UTC)
# UNB Fredericton, NB is Atlantic Daylight Time (UTC-3) in July 2017.
BOTNET_START_MS = 1499443320000   # 2017-07-07 16:02:00 UTC in milliseconds
BOTNET_END_MS   = 1499446920000   # 2017-07-07 17:02:00 UTC in milliseconds

# ── Internal victim subnet ────────────────────────────────────────────
# Ares C2 clients ran on 192.168.10.x machines inside the lab network.
# Port 8080 is the Ares C&C port — used as a secondary botnet indicator.
VICTIM_PREFIX  = "192.168."
ARES_C2_PORT   = 8080

# ── Feature columns we emit ───────────────────────────────────────────
# Must match the 49-feature unified schema in stage2_noniot_botnet.csv
NUMERIC_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK",
    "protocol", "src_port", "dst_port",
    "flow_active_time", "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "ttl_mean", "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
]
META_COLS = ["class_label", "device_type", "src_ip", "timestamp"]


def nfstream_to_row(flow) -> dict | None:
    """Convert one nfstream flow object to a dict matching our schema."""
    try:
        ts_ms   = float(flow.bidirectional_first_seen_ms)
        dur_s   = max(float(flow.bidirectional_duration_ms) / 1000.0, 1e-9)
        src_ip  = str(flow.src_ip)
        dst_port = int(flow.dst_port)
        src_port = int(flow.src_port)

        # ── Labelling logic ──────────────────────────────────────────
        # Primary: inside Ares time window AND from internal victim subnet
        in_window   = (BOTNET_START_MS <= ts_ms <= BOTNET_END_MS)
        from_victim = src_ip.startswith(VICTIM_PREFIX)

        # Secondary: any flow using the Ares C2 port (8080) is botnet
        ares_port   = (dst_port == ARES_C2_PORT or src_port == ARES_C2_PORT)

        label = "botnet" if ((in_window and from_victim) or ares_port) else "benign"

        fwd_pkts = float(flow.src2dst_packets)
        bwd_pkts = float(flow.dst2src_packets)
        fwd_bytes = float(flow.src2dst_bytes)
        bwd_bytes = float(flow.dst2src_bytes)
        tot_bytes = fwd_bytes + bwd_bytes
        tot_pkts  = fwd_pkts + bwd_pkts

        bps = tot_bytes / dur_s
        pps = tot_pkts  / dur_s

        # ms → seconds for IAT
        def ms2s(v, fallback=0.0):
            try: return float(v) / 1000.0
            except: return fallback

        flow_iat_mean = ms2s(flow.bidirectional_mean_piat_ms)
        pps_val = pps

        row = {
            # ── flow stats
            "flow_duration":       dur_s,
            "total_fwd_packets":   fwd_pkts,
            "total_bwd_packets":   bwd_pkts,
            "total_fwd_bytes":     fwd_bytes,
            "total_bwd_bytes":     bwd_bytes,
            # ── packet length stats
            "fwd_pkt_len_min":     float(flow.src2dst_min_ps),
            "fwd_pkt_len_max":     float(flow.src2dst_max_ps),
            "fwd_pkt_len_mean":    float(flow.src2dst_mean_ps),
            "fwd_pkt_len_std":     float(flow.src2dst_stddev_ps),
            "bwd_pkt_len_min":     float(flow.dst2src_min_ps),
            "bwd_pkt_len_max":     float(flow.dst2src_max_ps),
            "bwd_pkt_len_mean":    float(flow.dst2src_mean_ps),
            "bwd_pkt_len_std":     float(flow.dst2src_stddev_ps),
            # ── rates
            "flow_bytes_per_sec":  bps,
            "flow_pkts_per_sec":   pps,
            # ── IAT (ms → s)
            "flow_iat_mean":       flow_iat_mean,
            "flow_iat_std":        ms2s(flow.bidirectional_stddev_piat_ms),
            "flow_iat_min":        ms2s(flow.bidirectional_min_piat_ms),
            "flow_iat_max":        ms2s(flow.bidirectional_max_piat_ms),
            "fwd_iat_mean":        ms2s(flow.src2dst_mean_piat_ms),
            "fwd_iat_std":         ms2s(flow.src2dst_stddev_piat_ms),
            "fwd_iat_min":         ms2s(flow.src2dst_min_piat_ms),
            "fwd_iat_max":         ms2s(flow.src2dst_max_piat_ms),
            "bwd_iat_mean":        ms2s(flow.dst2src_mean_piat_ms),
            "bwd_iat_std":         ms2s(flow.dst2src_stddev_piat_ms),
            "bwd_iat_min":         ms2s(flow.dst2src_min_piat_ms),
            "bwd_iat_max":         ms2s(flow.dst2src_max_piat_ms),
            # ── header length (approx: min TCP+IPv4 header = 40 bytes)
            "fwd_header_length":   40.0 * fwd_pkts,
            "bwd_header_length":   40.0 * bwd_pkts,
            # ── TCP flags (src→dst direction)
            "flag_FIN":            float(flow.src2dst_fin_packets),
            "flag_SYN":            float(flow.src2dst_syn_packets),
            "flag_RST":            float(flow.src2dst_rst_packets),
            "flag_PSH":            float(flow.src2dst_psh_packets),
            "flag_ACK":            float(flow.src2dst_ack_packets),
            # ── 5-tuple
            "protocol":            float(flow.protocol),
            "src_port":            float(src_port),
            "dst_port":            float(dst_port),
            # ── time-window / behavioural
            "flow_active_time":    dur_s,
            "flow_idle_time":      0.0,
            "bytes_per_sec_window": bps,
            "pkts_per_sec_window":  pps_val,
            "periodicity_score":   1.0 / max(flow_iat_mean, 1e-6),
            "burst_rate":          pps_val,
            # ── packet-level (unavailable without DPI — zero-filled)
            "ttl_mean":            0.0,
            "dns_query_count":     0.0,
            "payload_bytes_mean":  0.0,
            "payload_bytes_std":   0.0,
            "payload_zero_ratio":  0.0,
            "payload_entropy":     0.0,
            # ── metadata
            "class_label":  label,
            "device_type":  "noniot",
            "src_ip":       src_ip,
            "timestamp":    ts_ms / 1000.0,   # epoch seconds
        }
        return row
    except Exception:
        return None


def main():
    if not PCAP_PATH.exists():
        print(f"[error] PCAP not found: {PCAP_PATH}")
        print("  Expected: data/raw/CIC-IDS-2017/Friday-WorkingHours.pcap")
        sys.exit(1)

    try:
        from nfstream import NFStreamer
    except ImportError:
        print("[error] nfstream not installed.  Run:  pip install nfstream")
        sys.exit(1)

    import csv
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALL_COLS = NUMERIC_FEATURES + META_COLS

    print("═" * 60)
    print("  Step 1 — CIC-IDS-2017 Friday Flow Extraction")
    print("═" * 60)
    print(f"  PCAP   : {PCAP_PATH}")
    print(f"  Output : {OUT_PATH}")
    print(f"  Botnet window : 16:02–17:02 UTC (13:02–14:02 ADT)")
    print("  ⏳  This takes 20–40 min for the full PCAP. Please wait …\n")

    n_total = n_bot = n_ben = n_skip = 0

    with open(OUT_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_COLS)
        writer.writeheader()

        streamer = NFStreamer(
            source=str(PCAP_PATH),
            statistical_analysis=True,
            idle_timeout=120,
            active_timeout=1800,
            accounting_mode=3,          # bidirectional mode
        )

        for flow in streamer:
            n_total += 1
            row = nfstream_to_row(flow)
            if row is None:
                n_skip += 1
                continue

            # Replace inf/nan
            for k, v in row.items():
                if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                    row[k] = 0.0

            if row["class_label"] == "botnet":
                n_bot += 1
            else:
                n_ben += 1

            writer.writerow(row)

            if n_total % 100_000 == 0:
                print(f"  Processed {n_total:,} flows  "
                      f"(botnet={n_bot:,}  benign={n_ben:,})")

    print(f"\n  Done!")
    print(f"  Total flows   : {n_total:,}")
    print(f"  Botnet flows  : {n_bot:,}  ({n_bot/max(n_total,1)*100:.1f}%)")
    print(f"  Benign flows  : {n_ben:,}  ({n_ben/max(n_total,1)*100:.1f}%)")
    print(f"  Skipped/error : {n_skip:,}")
    print(f"\n  ✓ Saved: {OUT_PATH}")

    if n_bot == 0:
        print("\n  ⚠️  WARNING: 0 botnet flows detected!")
        print("  The botnet window might be off by one hour.")
        print("  Check BOTNET_START_MS in this script and adjust by ±3600000 ms.")
    elif n_bot / max(n_ben, 1) > 0.5:
        print("\n  ⚠️  WARNING: unusually high botnet ratio.")
        print("  The ARES_C2_PORT=8080 filter may be over-labelling.")
        print("  Consider commenting out the ares_port check in label logic.")
    else:
        print("  ✅ Label distribution looks healthy. Proceed to step 2.")

    print("═" * 60)


if __name__ == "__main__":
    main()