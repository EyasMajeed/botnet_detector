"""
preprocess_ctu13.py — CTU-13 Dataset Preprocessor
═══════════════════════════════════════════════════
Purpose : Stage-2 Non-IoT CNN-LSTM botnet detector training data
Dataset : CTU-13  (https://www.stratosphereips.org/datasets-ctu13)
Format  : Zeek conn.log / binetflow CSV files

USAGE:
  1. Download the CTU-13 dataset from Stratosphere IPS
  2. Place the scenario folders (or binetflow CSVs) under:
       data/raw/ctu13/
     Expected structure:
       data/raw/ctu13/
         capture20110810.binetflow          (scenario 1)
         capture20110811.binetflow          (scenario 2)
         ...
     OR if using the Kaggle CSV version:
       data/raw/ctu13/
         ctu13_dataset.csv

  3. Run:  python -m src.ingestion.preprocess_ctu13
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as module or script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    RAW_CTU13, PROCESSED_DIR, CTU13_COL_MAP,
    MAX_ROWS_PER_DATASET, RANDOM_SEED,
)
from src.features.feature_utils import (
    encode_protocol, flags_from_state,
    full_feature_pipeline,
)


# ─────────────────────────────────────────────
#  STEP 1 — LOAD RAW DATA
# ─────────────────────────────────────────────
def load_ctu13(data_dir: Path) -> pd.DataFrame:
    """Load all binetflow / CSV files from the CTU-13 directory."""
    frames = []

    # Try binetflow files first
    binetflow_files = sorted(data_dir.glob("*.binetflow"))
    csv_files = sorted(data_dir.glob("*.csv"))

    files = binetflow_files if binetflow_files else csv_files

    if not files:
        print(f"[ERROR] No .binetflow or .csv files found in {data_dir}")
        print("  Please download the CTU-13 dataset and place files there.")
        print("  Download: https://www.stratosphereips.org/datasets-ctu13")
        sys.exit(1)

    for f in files:
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
    """Rename dataset columns to unified schema names."""
    # Detect which column naming convention is used
    # The binetflow format has columns with different casing
    col_lower = {c.lower().strip(): c for c in df.columns}

    rename = {}
    for src, dst in CTU13_COL_MAP.items():
        src_l = src.lower().strip()
        if src in df.columns:
            rename[src] = dst
        elif src_l in col_lower:
            rename[col_lower[src_l]] = dst

    df = df.rename(columns=rename)
    return df


# ─────────────────────────────────────────────
#  STEP 3 — COMPUTE DERIVED COLUMNS
# ─────────────────────────────────────────────
def compute_ctu13_derived(df: pd.DataFrame) -> pd.DataFrame:
    """CTU-13 specific derivations."""
    df = df.copy()

    # Duration: sometimes stored as string with units
    if "flow_duration" in df.columns:
        df["flow_duration"] = pd.to_numeric(
            df["flow_duration"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        ).fillna(0)

    # Backward bytes = total - src
    if "_total_bytes" in df.columns and "total_bwd_bytes" not in df.columns:
        total_bytes = pd.to_numeric(df["_total_bytes"], errors="coerce").fillna(0)
        fwd_bytes = pd.to_numeric(df["total_fwd_bytes"], errors="coerce").fillna(0) if "total_fwd_bytes" in df.columns else 0
        df["total_bwd_bytes"] = (total_bytes - fwd_bytes).clip(lower=0)

    # Backward packets = total - src
    if "_total_pkts" in df.columns and "total_bwd_packets" not in df.columns:
        total_pkts = pd.to_numeric(df["_total_pkts"], errors="coerce").fillna(0)
        fwd_pkts = pd.to_numeric(df["total_fwd_packets"], errors="coerce").fillna(0) if "total_fwd_packets" in df.columns else 0
        df["total_bwd_packets"] = (total_pkts - fwd_pkts).clip(lower=0)

    # Protocol encoding
    if "protocol" in df.columns:
        df["protocol"] = encode_protocol(df["protocol"])

    # TCP flags from state field
    if "_state" in df.columns:
        flag_df = flags_from_state(df["_state"])
        for col in flag_df.columns:
            df[col] = flag_df[col]

    # Port cleaning
    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9]", "", regex=True),
                errors="coerce"
            ).fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────
#  STEP 4 — LABEL NORMALIZATION
# ─────────────────────────────────────────────
def normalize_labels_ctu13(df: pd.DataFrame) -> pd.DataFrame:
    """
    CTU-13 labels are strings like:
      'flow=From-Botnet', 'flow=From-Normal', 'flow=To-Normal',
      'Botnet', 'Normal', 'Background'
    Normalize to binary: 'benign' / 'botnet'
    """
    df = df.copy()

    if "_raw_label" not in df.columns:
        # Try common alternative column names
        for alt in ["Label", "label", "class"]:
            if alt in df.columns:
                df["_raw_label"] = df[alt]
                break

    if "_raw_label" not in df.columns:
        print("  [WARN] No label column found — marking all as 'unknown'")
        df["class_label"] = "unknown"
        df["device_type"] = "noniot"
        return df

    raw = df["_raw_label"].astype(str).str.lower().str.strip()

    df["class_label"] = np.where(
        raw.str.contains("botnet|malicious|attack", regex=True),
        "botnet",
        np.where(
            raw.str.contains("normal|benign|legitimate", regex=True),
            "benign",
            "background"  # background traffic — will be dropped or treated as benign
        )
    )

    # For Stage-2 Non-IoT: treat background as benign or drop
    df.loc[df["class_label"] == "background", "class_label"] = "benign"

    # CTU-13 is all non-IoT traffic
    df["device_type"] = "noniot"

    return df


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CTU-13 PREPROCESSING — Stage-2 Non-IoT Training Data")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading raw data ...")
    df = load_ctu13(RAW_CTU13)

    # Subsample if configured
    if MAX_ROWS_PER_DATASET and len(df) > MAX_ROWS_PER_DATASET:
        print(f"  Subsampling to {MAX_ROWS_PER_DATASET:,} rows ...")
        df = df.sample(MAX_ROWS_PER_DATASET, random_state=RANDOM_SEED)

    # Map columns
    print("\n[2/5] Mapping columns to unified schema ...")
    df = map_columns(df)

    # CTU-13 derived columns
    print("\n[3/5] Computing derived features ...")
    df = compute_ctu13_derived(df)

    # Labels
    print("\n[4/5] Normalizing labels ...")
    df = normalize_labels_ctu13(df)

    # Drop unknowns
    df = df[df["class_label"].isin(["benign", "botnet"])].reset_index(drop=True)
    print(f"  Label distribution:\n{df['class_label'].value_counts().to_string()}")

    # Full feature pipeline
    print("\n[5/5] Running feature engineering pipeline ...")
    df, scaler = full_feature_pipeline(df, normalize=True)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage2_noniot_botnet.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len([c for c in df.columns if c not in ['class_label', 'device_type', '_raw_label', '_detailed_label', '_attack_cat']])}")
    print(f"  Labels: {df['class_label'].value_counts().to_dict()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
