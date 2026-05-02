"""
════════════════════════════════════════════════════════════════════════
 pcap_to_nbaiot_features.py  —  Extract N-BaIoT features from PCAPs
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 Reads IoT-23 PCAP files, runs each packet through the Kitsune
 feature extractor, and outputs a CSV with the same 115 columns
 as stage2_iot_botnet.csv — ready to combine with N-BaIoT data.

 INPUT  : data/raw/iot23/  (PCAP files, recursive)
 OUTPUT : data/processed/iot23_features.csv

 LABELLING (IoT-23 specific):
   CTU-IoT-Malware-Capture-*        → botnet  (infected devices)
   CTU-Honeypot-Capture-1-1         → benign  (Philips HUE lamp)
   CTU-Honeypot-Capture-5-1         → benign  (Amazon Echo)
   CTU-Honeypot-Capture-7-1         → benign  (Somfy door lock)
   CTU-IoT-Normal-Capture-*         → benign
   Other CTU-Honeypot-Capture-*     → botnet  (attacker honeypots)
   Unknown folder names             → skipped with warning

   IMPORTANT: In IoT-23, "Honeypot" means two different things:
     - Captures 1-1, 5-1, 7-1: real benign IoT devices (Philips Hue,
       Amazon Echo, Somfy). These are the 3 benign scenarios.
     - Other honeypot captures: attacker-side traffic (botnet).

   Use --label to force a label for a specific directory,
   e.g. when processing BoT-IoT or other datasets whose folder
   names do not follow IoT-23 naming conventions.

 NORMALISATION:
   Features are normalised using the SAME iot_scaler.json that was
   produced by preprocess_nbaiot.py — so both datasets share the
   exact same scale. This is critical: if you normalise separately,
   the two datasets will have different value ranges and the combined
   model will be confused.

 USAGE:
   python src/ingestion/pcap_to_nbaiot_features.py
   python src/ingestion/pcap_to_nbaiot_features.py --max-packets 50000
   python src/ingestion/pcap_to_nbaiot_features.py --pcap-dir data/raw/iot23
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'live'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from kitsune_extractor import KitsuneExtractor, FEATURE_NAMES
except ImportError:
    from src.live.kitsune_extractor import KitsuneExtractor, FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
PCAP_DIR    = ROOT / "data" / "raw" / "iot23"
OUTPUT_CSV  = ROOT / "data" / "processed" / "iot23_features.csv"
SCALER_PATH = ROOT / "models" / "stage2" / "iot_scaler.json"


# ════════════════════════════════════════════════════════════════════════
# LABEL INFERENCE
# ════════════════════════════════════════════════════════════════════════

# IoT-23 benign honeypot scenario numbers (Philips HUE, Amazon Echo, Somfy)
# Source: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/
# Scenario 21 = CTU-Honeypot-Capture-7-1  (Somfy Doorlock)   → BENIGN
# Scenario 22 = CTU-Honeypot-Capture-4-1  (Philips HUE)      → BENIGN
# Scenario 23 = CTU-Honeypot-Capture-5-1  (Amazon Echo)      → BENIGN
# All other CTU-Honeypot-Capture-* folders are attacker-side  → BOTNET
BENIGN_HONEYPOT_SCENARIOS = {"4-1", "5-1", "7-1"}


def infer_label(pcap_path: Path, label_override: str | None = None) -> str | None:
    """
    Infer benign/botnet from the capture folder name.

    IoT-23 honeypot naming is ambiguous — some honeypot captures are
    benign IoT devices, others are attacker-side botnet traffic.
    This function handles the distinction correctly.

    Parameters
    ----------
    pcap_path    : path to the PCAP file
    label_override : if set ("benign" or "botnet"), always return this
                     regardless of folder name — use for non-IoT-23 datasets
    """
    if label_override is not None:
        return label_override

    path_str = str(pcap_path).lower()

    # IoT-23 malware captures → always botnet
    if "malware" in path_str:
        return "botnet"

    # IoT-23 honeypot captures — check scenario number
    if "honeypot" in path_str:
        # Extract scenario suffix e.g. "CTU-Honeypot-Capture-7-1" → "7-1"
        import re
        match = re.search(r"honeypot-capture-(\d+-\d+)", path_str)
        if match:
            scenario = match.group(1)
            if scenario in BENIGN_HONEYPOT_SCENARIOS:
                return "benign"   # Philips HUE, Amazon Echo, Somfy
            else:
                return "botnet"   # attacker-side honeypot traffic
        return "botnet"  # unknown honeypot → assume botnet to be safe

    # Explicit benign/normal in name
    if "normal" in path_str or "benign" in path_str:
        return "benign"

    return None  # unknown — will be skipped


# ════════════════════════════════════════════════════════════════════════
# SINGLE PCAP EXTRACTOR
# ════════════════════════════════════════════════════════════════════════

def extract_pcap(pcap_path: Path,
                 label: str,
                 scaler: dict,
                 max_packets: int | None = None) -> pd.DataFrame | None:
    """
    Run all packets in pcap_path through Kitsune and return a DataFrame
    with 115 normalised features + metadata columns.
    """
    extractor  = KitsuneExtractor()
    rows: list[dict] = []
    n_packets  = 0
    device_name = pcap_path.parent.name  # e.g. CTU-IoT-Malware-Capture-1-1

    mins  = np.array(scaler["min"],  dtype=np.float32)
    maxs  = np.array(scaler["max"],  dtype=np.float32)
    scale = maxs - mins
    scale[scale == 0] = 1.0

    # Per-src_ip packet counter (used as seq_index)
    pkt_idx: dict[str, int] = defaultdict(int)

    try:
        reader = PcapReader(str(pcap_path))
    except Exception as e:
        print(f"    [ERROR] Cannot open: {e}")
        return None

    for pkt in reader:
        if max_packets and n_packets >= max_packets:
            break
        try:
            if IP not in pkt:
                continue
            ip   = pkt[IP]
            t    = float(pkt.time)
            src  = ip.src
            dst  = ip.dst
            size = len(pkt)
            src_mac = pkt[Ether].src if Ether in pkt else src

            if TCP in pkt:
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
                proto = "TCP"
            elif UDP in pkt:
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
                proto = "UDP"
            elif ICMP in pkt:
                sport, dport, proto = 0, 0, "ICMP"
            else:
                sport, dport, proto = 0, 0, "OTHER"
        except Exception:
            continue

        n_packets += 1

        raw = extractor.update(
            timestamp=t, src_mac=src_mac,
            src_ip=src, dst_ip=dst,
            src_port=sport, dst_port=dport,
            pkt_len=size, protocol=proto)

        # Normalise using the SAME scaler as N-BaIoT preprocessing
        scaled = np.clip((raw - mins) / scale, 0.0, 1.0)

        row = dict(zip(FEATURE_NAMES, scaled.tolist()))
        row["class_label"] = label
        row["attack_type"]  = label          # coarse — refine if needed
        row["device_name"]  = device_name
        row["src_ip"]       = src
        row["seq_index"]    = pkt_idx[src]
        pkt_idx[src]       += 1
        rows.append(row)

    reader.close()

    if not rows:
        return None

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main(pcap_dir: Path,
         output_csv: Path,
         scaler_path: Path,
         max_packets: int | None,
         label_override: str | None = None):

    print("=" * 62)
    print("  IoT-23 PCAP → N-BaIoT Features  —  Group 07")
    print(f"  Input  : {pcap_dir}")
    print(f"  Output : {output_csv}")
    print("=" * 62)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Run src/ingestion/preprocess_nbaiot.py first.")

    with open(scaler_path) as f:
        scaler = json.load(f)

    raw_max = max(scaler["max"])
    if raw_max < 2.0:
        raise ValueError(
            "Scaler appears to be from already-normalised data. "
            "Regenerate by running preprocess_nbaiot.py.")

    print(f"  Scaler loaded (raw max={raw_max:.1f}) ✓\n")

    # Find all PCAPs
    pcap_files = sorted(pcap_dir.rglob("*.pcap")) + \
                 sorted(pcap_dir.rglob("*.pcapng"))

    if not pcap_files:
        raise FileNotFoundError(f"No PCAP files found in {pcap_dir}")

    print(f"  Found {len(pcap_files)} PCAP file(s)\n")

    all_frames: list[pd.DataFrame] = []
    skipped = 0

    for pcap_path in pcap_files:
        label = infer_label(pcap_path, label_override=label_override)
        if label is None:
            print(f"  [SKIP] Cannot infer label: {pcap_path.name}")
            print(f"         Use --label botnet or --label benign to force.")
            skipped += 1
            continue

        size_mb = pcap_path.stat().st_size / 1e6
        print(f"  ── {pcap_path.parent.name}/{pcap_path.name} "
              f"({size_mb:.0f} MB)  label={label}")

        df = extract_pcap(pcap_path, label, scaler,
                          max_packets=max_packets)
        if df is None or len(df) == 0:
            print(f"    [SKIP] No usable packets")
            skipped += 1
            continue

        b   = (df["class_label"] == "benign").sum()
        bot = (df["class_label"] == "botnet").sum()
        print(f"    Extracted {len(df):,} rows  "
              f"(benign={b:,}  botnet={bot:,})")
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data extracted from any PCAP.")

    print(f"\n  Combining {len(all_frames)} files...")
    combined = pd.concat(all_frames, ignore_index=True)

    # Remove exact duplicates
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=FEATURE_NAMES, keep="first")
    dropped  = n_before - len(combined)

    vc = combined["class_label"].value_counts()
    print(f"\n  Final: {len(combined):,} rows "
          f"({dropped:,} duplicates removed)")
    for cls, cnt in vc.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(combined)*100:.1f}%)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    col_order = FEATURE_NAMES + [
        "class_label", "attack_type", "device_name", "src_ip", "seq_index"]
    combined[col_order].to_csv(output_csv, index=False)

    size_mb = output_csv.stat().st_size / 1e6
    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"  Rows     : {len(combined):,}")
    print(f"  Columns  : {len(col_order)}")
    print(f"  File     : {output_csv}  ({size_mb:.1f} MB)")
    print(f"  Skipped  : {skipped} PCAPs")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract N-BaIoT Kitsune features from IoT-23 PCAPs")
    ap.add_argument("--pcap-dir", default=str(PCAP_DIR),
                    help="Directory containing IoT-23 PCAP files.")
    ap.add_argument("--output",   default=str(OUTPUT_CSV),
                    help="Output CSV path.")
    ap.add_argument("--scaler",   default=str(SCALER_PATH),
                    help="Path to iot_scaler.json.")
    ap.add_argument("--max-packets", type=int, default=None,
                    help="Max packets per PCAP (default: all). "
                         "Use 50000 for quick testing.")
    ap.add_argument("--label", choices=["benign", "botnet"], default=None,
                    help="Force all PCAPs in --pcap-dir to this label. "
                         "Use for non-IoT-23 datasets (e.g. BoT-IoT) "
                         "whose folder names are not recognised.")
    args = ap.parse_args()

    main(Path(args.pcap_dir), Path(args.output),
         Path(args.scaler), args.max_packets,
         label_override=args.label)
