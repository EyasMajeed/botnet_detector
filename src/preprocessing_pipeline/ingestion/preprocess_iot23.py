"""
preprocess_iot23.py — IoT-23 Dataset Preprocessor
═══════════════════════════════════════════════════
Purpose : Stage-2 IoT CNN-LSTM botnet detector training data
Dataset : Aposemat IoT-23  (https://www.stratosphereips.org/datasets-iot23)
Format  : Zeek conn.log files (TSV) or pre-extracted CSVs

USAGE:
  1. Download the IoT-23 dataset from Stratosphere IPS:
       https://www.stratosphereips.org/datasets-iot23
     You can download individual scenarios (recommended) or the full set.

  2. Place conn.log.labeled files under:
       data/raw/iot23/
     Expected structure:
       data/raw/iot23/
         CTU-IoT-Malware-Capture-1-1/conn.log.labeled
         CTU-IoT-Malware-Capture-3-1/conn.log.labeled
         ...
     OR if using the lighter CSV version:
       data/raw/iot23/
         iot23_combined.csv

  3. Run:  python -m src.ingestion.preprocess_iot23
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    RAW_IOT23, PROCESSED_DIR, IOT23_COL_MAP,
    MAX_ROWS_PER_DATASET, RANDOM_SEED,
)
from src.features.feature_utils import (
    encode_protocol, flags_from_state,
    full_feature_pipeline,
)


# ─────────────────────────────────────────────
#  ZEEK CONN.LOG PARSER
# ─────────────────────────────────────────────
ZEEK_CONN_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]


def load_zeek_conn_log(filepath: Path) -> pd.DataFrame:
    """Parse a Zeek conn.log.labeled file."""
    rows = []
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    # Use the known column set; trim or pad each row
    n_cols = len(ZEEK_CONN_COLUMNS)
    cleaned = []
    for row in rows:
        if len(row) >= n_cols:
            cleaned.append(row[:n_cols])
        else:
            cleaned.append(row + ["-"] * (n_cols - len(row)))

    df = pd.DataFrame(cleaned, columns=ZEEK_CONN_COLUMNS)

    # Replace Zeek's '-' with NaN (suppress pandas FutureWarning)
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace("-", np.nan)

    return df


# ─────────────────────────────────────────────
#  STEP 1 — LOAD
# ─────────────────────────────────────────────
def load_iot23(data_dir: Path) -> pd.DataFrame:
    """Load all conn.log.labeled files or CSVs from IoT-23 directory."""
    frames = []

    # Zeek conn.log.labeled files (recursive search)
    conn_files = sorted(data_dir.rglob("conn.log.labeled"))

    if conn_files:
        print(f"  Found {len(conn_files)} conn.log.labeled files")
        for f in conn_files:
            print(f"    Loading {f.relative_to(data_dir)} ...", end=" ")
            df = load_zeek_conn_log(f)
            if len(df) > 0:
                df["_source_file"] = str(f.relative_to(data_dir))
                frames.append(df)
                print(f"({len(df):,} rows)")
            else:
                print("EMPTY")
    else:
        # Fall back to CSV files
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"[ERROR] No conn.log.labeled or .csv files in {data_dir}")
            print("  Download: https://www.stratosphereips.org/datasets-iot23")
            sys.exit(1)

        for f in csv_files:
            print(f"  Loading {f.name} ...", end=" ")
            try:
                df = pd.read_csv(f, low_memory=False)
                df["_source_file"] = f.name
                frames.append(df)
                print(f"({len(df):,} rows)")
            except Exception as e:
                print(f"SKIP ({e})")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total raw rows: {len(combined):,}")
    return combined


# ─────────────────────────────────────────────
#  STEP 2 — MAP COLUMNS
# ─────────────────────────────────────────────
def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename IoT-23 columns to unified schema."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for src, dst in IOT23_COL_MAP.items():
        if src in df.columns:
            rename[src] = dst
        elif src.lower() in col_lower:
            rename[col_lower[src.lower()]] = dst
    return df.rename(columns=rename)


# ─────────────────────────────────────────────
#  STEP 3 — DERIVED FEATURES
# ─────────────────────────────────────────────
def compute_iot23_derived(df: pd.DataFrame) -> pd.DataFrame:
    """IoT-23 specific derivations."""
    df = df.copy()

    # Numeric conversion for key fields
    for col in ["flow_duration", "total_fwd_bytes", "total_bwd_bytes",
                 "total_fwd_packets", "total_bwd_packets"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Protocol encoding
    if "protocol" in df.columns:
        df["protocol"] = encode_protocol(df["protocol"])

    # Flags from conn_state
    if "_state" in df.columns:
        flag_df = flags_from_state(df["_state"])
        for col in flag_df.columns:
            df[col] = flag_df[col]

    # Port cleanup
    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────
#  STEP 4 — LABEL NORMALIZATION
# ─────────────────────────────────────────────
def normalize_labels_iot23(df: pd.DataFrame) -> pd.DataFrame:
    """
    IoT-23 labels include:
      'Malicious', 'Benign', 'PartOfAHorizontalPortScan',
      'C&C', 'DDoS', 'Okiru', 'Torii', etc.
    Normalize to binary: 'benign' / 'botnet'
    """
    df = df.copy()

    if "_raw_label" not in df.columns:
        for alt in ["label", "Label"]:
            if alt in df.columns:
                df["_raw_label"] = df[alt]
                break

    if "_raw_label" not in df.columns:
        print("  [WARN] No label column — marking all 'unknown'")
        df["class_label"] = "unknown"
        df["device_type"] = "iot"
        return df

    raw = df["_raw_label"].astype(str).str.lower().str.strip()

    df["class_label"] = np.where(
        raw.str.contains("malicious|botnet|c&c|ddos|attack|portscan|"
                         "okiru|torii|mirai|hajime|kenjiro|muhstik|"
                         "hakai|linux|irc|spam|scan", regex=True),
        "botnet",
        np.where(
            raw.str.contains("benign|normal|legitimate", regex=True),
            "benign",
            "background"
        )
    )

    # Background → benign for binary classification
    df.loc[df["class_label"] == "background", "class_label"] = "benign"

    # IoT-23 is all IoT traffic
    df["device_type"] = "iot"

    return df


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("IoT-23 PREPROCESSING — Stage-2 IoT Training Data")
    print("=" * 60)

    print("\n[1/5] Loading raw data ...")
    df = load_iot23(RAW_IOT23)

    if MAX_ROWS_PER_DATASET and len(df) > MAX_ROWS_PER_DATASET:
        print(f"  Subsampling to {MAX_ROWS_PER_DATASET:,} rows ...")
        df = df.sample(MAX_ROWS_PER_DATASET, random_state=RANDOM_SEED)

    print("\n[2/5] Mapping columns ...")
    df = map_columns(df)

    print("\n[3/5] Computing derived features ...")
    df = compute_iot23_derived(df)

    print("\n[4/5] Normalizing labels ...")
    df = normalize_labels_iot23(df)

    df = df[df["class_label"].isin(["benign", "botnet"])].reset_index(drop=True)
    print(f"  Label distribution:\n{df['class_label'].value_counts().to_string()}")

    print("\n[5/5] Running feature engineering pipeline ...")
    df, scaler = full_feature_pipeline(df, normalize=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage2_iot_botnet.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Labels: {df['class_label'].value_counts().to_dict()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
