"""
════════════════════════════════════════════════════════════════════════
 combine_datasets.py  —  Merge N-BaIoT + IoT-23 for Combined Training
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 Merges stage2_iot_botnet.csv (N-BaIoT) with iot23_features.csv
 (IoT-23 extracted via Kitsune) into one combined training CSV.

 Since we have no IoT-23 normal captures, the combined dataset is:
   Benign : N-BaIoT benign rows only  (513K rows, 9 IoT devices)
   Botnet : N-BaIoT botnet rows       (1.97M rows, Mirai + Gafgyt flooding)
          + IoT-23 botnet rows        (extracted, Mirai scanning + C&C)

 The model trained on this dataset will detect:
   ✓ Mirai/Gafgyt UDP flooding    (learned from N-BaIoT)
   ✓ Mirai Telnet scanning        (learned from IoT-23)
   ✓ Mirai C&C communication      (learned from IoT-23)
   ✓ Gafgyt TCP scanning          (learned from N-BaIoT)

 INPUT  : data/processed/stage2_iot_botnet.csv   (N-BaIoT)
          data/processed/iot23_features.csv       (IoT-23)
 OUTPUT : data/processed/stage2_iot_combined.csv

 USAGE:
   python src/ingestion/combine_datasets.py
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'live'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from kitsune_extractor import FEATURE_NAMES
except ImportError:
    from src.live.kitsune_extractor import FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
NBAIOT_CSV   = ROOT / "data" / "processed" / "stage2_iot_botnet.csv"
IOT23_CSV    = ROOT / "data" / "processed" / "iot23_features.csv"
OUTPUT_CSV   = ROOT / "data" / "processed" / "stage2_iot_combined.csv"


def main(nbaiot_csv: Path, iot23_csv: Path, output_csv: Path):
    print("=" * 62)
    print("  Combine N-BaIoT + IoT-23 Datasets  —  Group 07")
    print("=" * 62)

    # ── Load N-BaIoT ─────────────────────────────────────────────────
    print(f"\n  Loading N-BaIoT: {nbaiot_csv}")
    if not nbaiot_csv.exists():
        raise FileNotFoundError(
            f"Not found: {nbaiot_csv}\n"
            "Run src/ingestion/preprocess_nbaiot.py first.")

    df_nbaiot = pd.read_csv(nbaiot_csv, low_memory=False)
    print(f"  {len(df_nbaiot):,} rows | {df_nbaiot.shape[1]} cols")
    vc = df_nbaiot["class_label"].value_counts()
    for cls, cnt in vc.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(df_nbaiot)*100:.1f}%)")

    # ── Load IoT-23 ───────────────────────────────────────────────────
    print(f"\n  Loading IoT-23 : {iot23_csv}")
    if not iot23_csv.exists():
        raise FileNotFoundError(
            f"Not found: {iot23_csv}\n"
            "Run src/ingestion/pcap_to_nbaiot_features.py first.")

    df_iot23 = pd.read_csv(iot23_csv, low_memory=False)
    print(f"  {len(df_iot23):,} rows | {df_iot23.shape[1]} cols")
    vc23 = df_iot23["class_label"].value_counts()
    for cls, cnt in vc23.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(df_iot23)*100:.1f}%)")

    # ── Align columns ─────────────────────────────────────────────────
    # Both must have the same 115 feature columns + 5 metadata columns
    meta_cols = ["class_label", "attack_type", "device_name",
                 "src_ip", "seq_index"]
    required  = FEATURE_NAMES + meta_cols

    for name, df in [("N-BaIoT", df_nbaiot), ("IoT-23", df_iot23)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [WARN] {name} missing columns: {missing[:5]}")
            for col in missing:
                if col in meta_cols:
                    df[col] = "unknown"
                else:
                    df[col] = 0.0

    # ── Strategy: keep N-BaIoT benign + both botnet sources ──────────
    print(f"\n  Combining...")
    print(f"  Strategy: N-BaIoT benign (all) + N-BaIoT botnet + IoT-23 botnet")

    # Tag source for tracking
    df_nbaiot = df_nbaiot.copy()
    df_iot23  = df_iot23.copy()
    df_nbaiot["source"] = "nbaiot"
    df_iot23["source"]  = "iot23"

    combined = pd.concat([df_nbaiot, df_iot23], ignore_index=True)

    # ── Remove cross-dataset exact duplicates ─────────────────────────
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=FEATURE_NAMES, keep="first")
    dropped  = n_before - len(combined)
    if dropped:
        print(f"  Removed {dropped:,} exact duplicate rows across datasets")

    # ── Final stats ───────────────────────────────────────────────────
    vc_final = combined["class_label"].value_counts()
    imb = vc_final.max() / max(vc_final.min(), 1)

    print(f"\n  Combined dataset: {len(combined):,} rows")
    print(f"  Class distribution:")
    for cls, cnt in vc_final.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(combined)*100:.1f}%)")
    print(f"  Imbalance ratio : {imb:.1f}:1")

    print(f"\n  Source breakdown:")
    for src, cnt in combined["source"].value_counts().items():
        print(f"    {src:>8s}: {cnt:>8,}  ({cnt/len(combined)*100:.1f}%)")

    print(f"\n  Device coverage:")
    for dev, cnt in combined["device_name"].value_counts().head(15).items():
        label = combined[combined["device_name"]==dev]["class_label"].mode()[0]
        print(f"    {dev:<45} {label:<8} {cnt:>8,}")

    # ── Save ──────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cols = FEATURE_NAMES + [c for c in
                                 meta_cols + ["source"]
                                 if c in combined.columns]
    combined[out_cols].to_csv(output_csv, index=False)

    size_mb = output_csv.stat().st_size / 1e6
    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"  Rows    : {len(combined):,}")
    print(f"  Columns : {len(out_cols)}")
    print(f"  File    : {output_csv}  ({size_mb:.1f} MB)")
    print(f"\n  NEXT STEP:")
    print(f"  Edit models/stage2/iot_detector.py and change:")
    print(f'    DATA_PATH = ROOT / "data" / "processed" / "stage2_iot_combined.csv"')
    print(f"  Then run: python models/stage2/iot_detector.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Combine N-BaIoT + IoT-23 into one training CSV")
    ap.add_argument("--nbaiot",  default=str(NBAIOT_CSV))
    ap.add_argument("--iot23",   default=str(IOT23_CSV))
    ap.add_argument("--output",  default=str(OUTPUT_CSV))
    args = ap.parse_args()

    main(Path(args.nbaiot), Path(args.iot23), Path(args.output))
