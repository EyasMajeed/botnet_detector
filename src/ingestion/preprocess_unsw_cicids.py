"""
preprocess_unsw_cicids.py — UNSW-NB15 + CICIDS2017 Preprocessor
═════════════════════════════════════════════════════════════════
Purpose : Stage-1 IoT vs Non-IoT traffic-type classifier training data

SUPPORTED INPUT FORMATS:
  Format A — Kaggle "Payload-Byte" dataset (yasiralifarrukh):
    Files: Payload_data_UNSW.csv, Payload_data_CICIDS2017.csv
    Shape: N × 1504 (1500 payload byte columns + metadata + label)
    All columns are integers. Last column is the label.

  Format B — Standard UNSW-NB15 CSV files:
    Files: UNSW-NB15_1.csv ... UNSW-NB15_4.csv
    Contains flow-level features like dur, spkts, dpkts, sbytes, dbytes, etc.

  Format C — CICFlowMeter output CSVs:
    Files: Monday-WorkingHours.pcap_Flow.csv, etc.
    Contains ~80 flow features from CICFlowMeter.

USAGE:
  1. Place your dataset files under:
       data/raw/unsw_cicids2017/
     For the Kaggle Payload-Byte dataset:
       data/raw/unsw_cicids2017/Payload_data_UNSW.csv
       data/raw/unsw_cicids2017/Payload_data_CICIDS2017.csv

  2. Run:  python -m src.ingestion.preprocess_unsw_cicids

DEVICE-TYPE LABELING STRATEGY:
  - UNSW-NB15 → labeled as mix of "iot" and "noniot" using port heuristic
  - CICIDS2017 → all labeled "noniot" (enterprise traffic)
  For best results, supplement with N-BaIoT data for IoT samples.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    RAW_UNSW_CIC, PROCESSED_DIR, UNSW_COL_MAP, CICFLOW_COL_MAP,
    MAX_ROWS_PER_DATASET, RANDOM_SEED,
)
from src.features.feature_utils import (
    encode_protocol, full_feature_pipeline,
)


# ─────────────────────────────────────────────
#  IoT HEURISTIC PARAMETERS
# ─────────────────────────────────────────────
IOT_PORTS = {1883, 8883, 5683, 5684, 1900, 23, 2323, 5353, 49152,
             8080, 8443, 554, 80, 443}
IOT_IPS = set()  # Add known IoT device IPs for better labeling


# ═════════════════════════════════════════════
#  FORMAT DETECTION
# ═════════════════════════════════════════════
def detect_format(filepath: Path) -> str:
    """
    Detect dataset format by reading the first few rows.
    Returns: 'payload_byte', 'unsw_csv', 'cicflow_csv', or 'unknown'
    """
    try:
        sample = pd.read_csv(filepath, nrows=5, low_memory=False)
    except Exception:
        return "unknown"

    cols = [str(c).strip() for c in sample.columns]
    n_cols = len(cols)

    # Payload-Byte format: ~1504 columns, mostly numeric column names (0-1499)
    if n_cols > 1000:
        numeric_col_count = sum(1 for c in cols if c.isdigit())
        if numeric_col_count > 900:
            return "payload_byte"

    # UNSW-NB15 standard format
    unsw_indicators = {"dur", "spkts", "dpkts", "sbytes", "dbytes", "proto"}
    col_lower = {c.lower() for c in cols}
    if len(unsw_indicators & col_lower) >= 4:
        return "unsw_csv"

    # CICFlowMeter format
    cic_indicators = {"flow duration", "total fwd packets", "flow bytes/s"}
    if len(cic_indicators & col_lower) >= 2:
        return "cicflow_csv"

    return "unknown"


# ═════════════════════════════════════════════
#  FORMAT A — KAGGLE PAYLOAD-BYTE DATASET
# ═════════════════════════════════════════════
def process_payload_byte(filepath: Path, source_name: str) -> pd.DataFrame:
    """
    Process the Payload-Byte format (N × 1504).

    Columns: 0..1499 = raw payload bytes, plus metadata columns and a label.
    Since these are raw byte representations (not flow features), we extract
    statistical features FROM the payload byte distributions to create
    flow-level-like features suitable for our unified schema.

    Features extracted from payload bytes:
      - Byte value statistics (mean, std, min, max, entropy)
      - Zero-byte ratio, printable ASCII ratio
      - Byte distribution features
      - These map to our flow-level feature schema as surrogates
    """
    print(f"    Detected: Payload-Byte format ({filepath.name})")
    print(f"    Loading (this may take a while for large files) ...")

    # Read in chunks for large files
    chunk_size = 50000
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        processed = extract_features_from_payload_bytes(chunk, source_name)
        chunks.append(processed)
        total_rows += len(chunk)
        print(f"      Processed {total_rows:,} rows ...", end="\r")

        # Optional: limit rows for memory
        if MAX_ROWS_PER_DATASET and total_rows >= MAX_ROWS_PER_DATASET:
            break

    print(f"      Processed {total_rows:,} rows total")

    return pd.concat(chunks, ignore_index=True)


def extract_features_from_payload_bytes(chunk: pd.DataFrame,
                                         source_name: str) -> pd.DataFrame:
    """
    Extract meaningful network features from raw payload byte columns.

    The payload bytes encode packet content as integers 0-255.
    We compute statistical features that capture the byte distribution
    characteristics — these are known to be effective for traffic classification
    as they reflect protocol structure, encryption, and payload patterns.
    """
    cols = list(chunk.columns)

    # Identify label column (usually last column, named 'Label' or similar)
    label_col = None
    for candidate in ["Label", "label", "CLASS", "class"]:
        if candidate in cols:
            label_col = candidate
            break
    # If no named label column, check if last column has small cardinality
    if label_col is None:
        last_col = cols[-1]
        if chunk[last_col].nunique() < 50:
            label_col = last_col

    # Identify byte columns (numeric columns that aren't the label)
    byte_cols = [c for c in cols if str(c).isdigit() or
                 (c != label_col and c not in ["Label", "label", "CLASS", "class"])]

    # Limit to actual payload columns (should be ~1500)
    # Filter to columns that are numeric
    byte_data = chunk[byte_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ── Extract statistical features from byte distributions ──
    result = pd.DataFrame(index=chunk.index)

    # Byte value statistics → map to flow-level packet length surrogates
    result["total_fwd_bytes"] = byte_data.sum(axis=1)
    result["total_bwd_bytes"] = 0  # single-direction in payload data

    # Packet length stats from byte values
    result["fwd_pkt_len_mean"] = byte_data.mean(axis=1)
    result["fwd_pkt_len_std"] = byte_data.std(axis=1)
    result["fwd_pkt_len_min"] = byte_data.min(axis=1)
    result["fwd_pkt_len_max"] = byte_data.max(axis=1)

    # Non-zero byte count → surrogate for actual payload size
    non_zero = (byte_data > 0).sum(axis=1)
    total_cols = len(byte_cols)
    result["total_fwd_packets"] = non_zero.clip(lower=1)
    result["total_bwd_packets"] = 0

    # Flow duration surrogate: based on byte diversity (higher diversity → longer flow)
    result["flow_duration"] = byte_data.nunique(axis=1).astype(float)

    # Byte entropy (Shannon entropy of byte distribution)
    # High entropy → encrypted/compressed; Low entropy → plaintext/repetitive
    def row_entropy(row):
        vals = row.values
        vals = vals[vals > 0]  # ignore zero padding
        if len(vals) == 0:
            return 0.0
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))

    result["flow_iat_mean"] = byte_data.apply(row_entropy, axis=1)
    result["flow_iat_std"] = result["flow_iat_mean"] * 0.1  # small variation surrogate

    # Zero-byte ratio → indicator of padding/protocol overhead
    zero_ratio = (byte_data == 0).sum(axis=1) / max(total_cols, 1)
    result["flow_idle_time"] = zero_ratio

    # Printable ASCII ratio (32-126) → indicator of plaintext vs binary
    ascii_count = ((byte_data >= 32) & (byte_data <= 126)).sum(axis=1)
    result["flow_active_time"] = ascii_count / max(total_cols, 1)

    # Byte rate surrogates
    dur = result["flow_duration"].replace(0, 1)
    result["flow_bytes_per_sec"] = result["total_fwd_bytes"] / dur
    result["flow_pkts_per_sec"] = result["total_fwd_packets"] / dur

    # IAT surrogates from byte differences
    byte_diff = byte_data.diff(axis=1).abs()
    result["fwd_iat_mean"] = byte_diff.mean(axis=1)
    result["fwd_iat_std"] = byte_diff.std(axis=1)
    result["fwd_iat_min"] = byte_diff.min(axis=1)
    result["fwd_iat_max"] = byte_diff.max(axis=1)

    # Backward IAT (zeros since unidirectional)
    for col in ["bwd_iat_min", "bwd_iat_max", "bwd_iat_mean", "bwd_iat_std"]:
        result[col] = 0

    # Backward packet length stats
    for col in ["bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std"]:
        result[col] = 0

    # Flow IAT
    result["flow_iat_min"] = result["fwd_iat_min"]
    result["flow_iat_max"] = result["fwd_iat_max"]

    # Header length surrogates
    result["fwd_header_length"] = 20  # typical TCP header
    result["bwd_header_length"] = 0

    # Protocol, ports: not available in payload-byte format → defaults
    result["protocol"] = 6  # assume TCP
    result["src_port"] = 0
    result["dst_port"] = 0

    # TCP flags: not available → 0
    for flag in ["flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG"]:
        result[flag] = 0

    # ── Labels ──
    if label_col is not None:
        raw_label = chunk[label_col].astype(str).str.strip().str.lower()

        # UNSW-NB15 labels: 0=normal, 1-9=attack categories
        # CICIDS2017 labels: 0=benign, others=attack
        if raw_label.str.isnumeric().all():
            numeric_label = pd.to_numeric(raw_label, errors="coerce").fillna(0)
            result["class_label"] = np.where(numeric_label == 0, "benign", "botnet")
        else:
            result["class_label"] = np.where(
                raw_label.isin(["0", "benign", "normal"]),
                "benign", "botnet"
            )
    else:
        result["class_label"] = "benign"

    # Device type
    if "cicids" in source_name.lower() or "cic" in source_name.lower():
        result["device_type"] = "noniot"
    elif "unsw" in source_name.lower():
        # UNSW has mixed traffic — use entropy-based heuristic
        # IoT devices tend to have more repetitive (lower entropy) payload patterns
        median_entropy = result["flow_iat_mean"].median()
        result["device_type"] = np.where(
            result["flow_iat_mean"] < median_entropy * 0.7, "iot", "noniot"
        )
    else:
        result["device_type"] = "noniot"

    result["_source_dataset"] = source_name

    return result


# ═════════════════════════════════════════════
#  FORMAT B — STANDARD UNSW-NB15 CSV
# ═════════════════════════════════════════════
def process_unsw_standard(filepath: Path) -> pd.DataFrame:
    """Process standard UNSW-NB15 CSV files with flow-level features."""
    print(f"    Detected: Standard UNSW-NB15 format ({filepath.name})")
    df = pd.read_csv(filepath, low_memory=False)

    col_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for src, dst in UNSW_COL_MAP.items():
        if src in df.columns:
            rename[src] = dst
        elif src.lower() in col_lower:
            rename[col_lower[src.lower()]] = dst
    df = df.rename(columns=rename)

    if "protocol" in df.columns:
        df["protocol"] = encode_protocol(df["protocol"])

    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "_raw_label" in df.columns:
        raw = pd.to_numeric(df["_raw_label"], errors="coerce")
        df["class_label"] = np.where(raw >= 1, "botnet", "benign")
    else:
        df["class_label"] = "benign"

    if IOT_IPS:
        src_ip = df.get("srcip", pd.Series("", index=df.index))
        dst_ip = df.get("dstip", pd.Series("", index=df.index))
        is_iot = src_ip.isin(IOT_IPS) | dst_ip.isin(IOT_IPS)
    else:
        sp = df.get("src_port", pd.Series(0, index=df.index))
        dp = df.get("dst_port", pd.Series(0, index=df.index))
        is_iot = sp.isin(IOT_PORTS) | dp.isin(IOT_PORTS)
    df["device_type"] = np.where(is_iot, "iot", "noniot")
    df["_source_dataset"] = "unsw_nb15"

    return df


# ═════════════════════════════════════════════
#  FORMAT C — CICFLOWMETER CSV
# ═════════════════════════════════════════════
def process_cicflow_csv(filepath: Path) -> pd.DataFrame:
    """Process CICFlowMeter output CSV files."""
    print(f"    Detected: CICFlowMeter format ({filepath.name})")
    try:
        df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    df.columns = df.columns.str.strip()

    rename = {}
    for src, dst in CICFLOW_COL_MAP.items():
        if src in df.columns:
            rename[src] = dst
        for c in df.columns:
            if c.strip().lower() == src.strip().lower():
                rename[c] = dst
                break
    df = df.rename(columns=rename)

    if "protocol" in df.columns:
        df["protocol"] = encode_protocol(df["protocol"])

    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "_raw_label" in df.columns:
        raw = df["_raw_label"].astype(str).str.strip().str.lower()
        df["class_label"] = np.where(raw.isin(["benign", "normal"]), "benign", "botnet")
    else:
        df["class_label"] = "benign"

    df["device_type"] = "noniot"
    df["_source_dataset"] = "cicids2017"
    return df


# ═════════════════════════════════════════════
#  MAIN LOADER — AUTO-DETECTS FORMAT
# ═════════════════════════════════════════════
def load_and_process_file(filepath: Path) -> pd.DataFrame:
    """Auto-detect format and process a single file."""
    fmt = detect_format(filepath)

    if fmt == "payload_byte":
        return process_payload_byte(filepath, filepath.stem)
    elif fmt == "unsw_csv":
        return process_unsw_standard(filepath)
    elif fmt == "cicflow_csv":
        return process_cicflow_csv(filepath)
    else:
        # Try as generic CSV
        print(f"    Unknown format for {filepath.name}, attempting generic load ...")
        try:
            df = pd.read_csv(filepath, low_memory=False, nrows=5)
            n_cols = len(df.columns)
            print(f"    Columns: {n_cols}, first few: {list(df.columns[:5])}")

            # If >1000 columns, likely payload-byte
            if n_cols > 1000:
                return process_payload_byte(filepath, filepath.stem)

            # Otherwise try standard
            df = pd.read_csv(filepath, low_memory=False)
            df["class_label"] = "benign"
            df["device_type"] = "noniot"
            df["_source_dataset"] = filepath.stem
            return df
        except Exception as e:
            print(f"    [ERROR] Cannot process {filepath.name}: {e}")
            return pd.DataFrame()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("UNSW-NB15 + CICIDS2017 PREPROCESSING — Stage-1 Training Data")
    print("=" * 60)

    if not RAW_UNSW_CIC.exists():
        RAW_UNSW_CIC.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    all_csvs = sorted(RAW_UNSW_CIC.glob("*.csv"))

    # Also check subdirectories
    for subdir in ["unsw", "cicids", "archive"]:
        sub = RAW_UNSW_CIC / subdir
        if sub.exists():
            all_csvs.extend(sorted(sub.glob("*.csv")))

    if not all_csvs:
        print(f"\n[ERROR] No CSV files found in {RAW_UNSW_CIC}")
        print("  Please place your dataset files there.")
        print("  For Kaggle Payload-Byte dataset:")
        print("    data/raw/unsw_cicids2017/Payload_data_UNSW.csv")
        print("    data/raw/unsw_cicids2017/Payload_data_CICIDS2017.csv")
        sys.exit(1)

    print(f"\n  Found {len(all_csvs)} CSV file(s):")
    for f in all_csvs:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")

    # Process each file
    frames = []
    for filepath in all_csvs:
        print(f"\n  Processing {filepath.name} ...")
        df = load_and_process_file(filepath)
        if not df.empty:
            frames.append(df)
            print(f"    → {len(df):,} rows")

    if not frames:
        print("\n[ERROR] No data successfully loaded.")
        sys.exit(1)

    # Merge
    print(f"\n  Merging {len(frames)} datasets ...")
    all_cols = set()
    for f in frames:
        all_cols.update(f.columns)
    for f in frames:
        for col in all_cols:
            if col not in f.columns:
                f[col] = 0

    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(df):,} rows")

    # Subsample if needed
    if MAX_ROWS_PER_DATASET and len(df) > MAX_ROWS_PER_DATASET:
        print(f"  Subsampling to {MAX_ROWS_PER_DATASET:,} rows ...")
        df = df.sample(MAX_ROWS_PER_DATASET, random_state=RANDOM_SEED)

    # Check distributions
    print(f"\n  device_type distribution:")
    print(f"    {df['device_type'].value_counts().to_dict()}")
    print(f"  class_label distribution:")
    print(f"    {df['class_label'].value_counts().to_dict()}")

    if df["device_type"].nunique() < 2:
        print("\n  [WARN] Only one device type found!")
        print("  For a balanced Stage-1 classifier you need both IoT and non-IoT.")
        print("  Consider supplementing with N-BaIoT data for IoT samples.\n")

    # Feature pipeline
    print("\n  Running feature engineering pipeline ...")
    df, scaler = full_feature_pipeline(df, normalize=True)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage1_iot_vs_noniot.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")
    if "device_type" in df.columns:
        print(f"  Device types: {df['device_type'].value_counts().to_dict()}")
    if "class_label" in df.columns:
        print(f"  Class labels:  {df['class_label'].value_counts().to_dict()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
