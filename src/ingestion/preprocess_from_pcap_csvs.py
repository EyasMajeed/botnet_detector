"""
══════════════════════════════════════════════════════════════════════════
 FIXED PREPROCESSING PIPELINE — preprocess_from_pcap_csvs.py
══════════════════════════════════════════════════════════════════════════
 Group 07 | CPCS499
 
 This script takes the RICH CSV files produced by pcap_to_csv.py and
 converts them into the 56-feature training-ready CSVs your models need.
 
 WHY THIS SCRIPT EXISTS:
   The original preprocessing pipeline had 3 bugs:
   
   1) IoT-23 conn.log parser failed because some files have the last 3
      columns (tunnel_parents, label, detailed-label) merged into one
      column due to inconsistent tab formatting. This caused the label
      column to be missing → all rows marked "unknown" → 0 rows after
      filtering → MinMaxScaler crash.
      
   2) CTU-13 extreme class imbalance (997K benign vs 2K botnet) because
      only the first binetflow file was loaded and most of that file is
      background/normal traffic.
      
   3) Packet-level features (TTL, header lengths) and TLS features were
      all zero-filled placeholders because the pipeline only used the
      conn.log metadata, not the actual PCAP data.
      
 THE FIX:
   Use pcap_to_csv.py FIRST to convert PCAPs → rich flow CSVs with real
   packet-level features. Then run THIS script to map those CSVs into
   the 56-feature unified schema.

 USAGE:
   Step 1: Convert PCAPs to CSVs (run this on your machine with tshark):
   
     # IoT-23
     python pcap_to_csv.py --dataset iot23 --data_dir ./data/raw/iot23 --output_dir ./data/pcap_csv/iot23
     
     # CTU-13
     python pcap_to_csv.py --dataset ctu13 --data_dir ./data/raw/ctu13 --output_dir ./data/pcap_csv/ctu13
   
   Step 2: Run this script:
     python preprocess_from_pcap_csvs.py

══════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Paths — adjust these to match your directory structure
BASE_DIR = Path(__file__).resolve().parent

# Input: CSVs produced by pcap_to_csv.py
PCAP_CSV_IOT23   = BASE_DIR / "data" / "pcap_csv" / "iot23" / "iot23_all_flows.csv"
PCAP_CSV_CTU13   = BASE_DIR / "data" / "pcap_csv" / "ctu13" / "ctu13_all_flows.csv"

# Input: Payload-byte CSVs for UNSW+CICIDS (these don't need PCAP conversion)
RAW_UNSW_CIC     = BASE_DIR / "data" / "raw" / "unsw_cicids2017"

# Output directory
PROCESSED_DIR    = BASE_DIR / "data" / "processed"

# If you also have the original conn.log.labeled files and want to use
# them as a FALLBACK for IoT-23 (in case pcap_to_csv.py wasn't run):
RAW_IOT23        = BASE_DIR / "data" / "raw" / "iot23"
RAW_CTU13        = BASE_DIR / "data" / "raw" / "ctu13"

# Processing parameters
MAX_ROWS = 1_000_000       # Max rows per dataset (None = no limit)
RANDOM_SEED = 42
TIME_WINDOW_SEC = 10.0

# ═══════════════════════════════════════════════════════════════════════
# UNIFIED 56-FEATURE SCHEMA
# ═══════════════════════════════════════════════════════════════════════

ALL_FEATURES = [
    # ── Flow-level (40 features) ──
    "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG",
    "protocol", "src_port", "dst_port",
    "flow_active_time", "flow_idle_time",
    # ── Time-window features (6) ──
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "window_flow_count", "window_unique_dsts",
    # ── Packet-level features (9) ──
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max",
    "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
    # ── TLS (1) ──
    "tls_features_available",
]

# Column mapping: pcap_to_csv.py output → unified schema
PCAP_CSV_COL_MAP = {
    "flow_duration":       "flow_duration",
    "total_fwd_packets":   "total_fwd_packets",
    "total_bwd_packets":   "total_bwd_packets",
    "total_fwd_bytes":     "total_fwd_bytes",
    "total_bwd_bytes":     "total_bwd_bytes",
    "fwd_pkt_len_min":     "fwd_pkt_len_min",
    "fwd_pkt_len_max":     "fwd_pkt_len_max",
    "fwd_pkt_len_mean":    "fwd_pkt_len_mean",
    "fwd_pkt_len_std":     "fwd_pkt_len_std",
    "bwd_pkt_len_min":     "bwd_pkt_len_min",
    "bwd_pkt_len_max":     "bwd_pkt_len_max",
    "bwd_pkt_len_mean":    "bwd_pkt_len_mean",
    "bwd_pkt_len_std":     "bwd_pkt_len_std",
    "flow_bytes_per_sec":  "flow_bytes_per_sec",
    "flow_pkts_per_sec":   "flow_pkts_per_sec",
    "flow_iat_mean":       "flow_iat_mean",
    "flow_iat_std":        "flow_iat_std",
    "flow_iat_min":        "flow_iat_min",
    "flow_iat_max":        "flow_iat_max",
    "fwd_iat_mean":        "fwd_iat_mean",
    "fwd_iat_std":         "fwd_iat_std",
    "fwd_iat_min":         "fwd_iat_min",
    "fwd_iat_max":         "fwd_iat_max",
    "bwd_iat_mean":        "bwd_iat_mean",
    "bwd_iat_std":         "bwd_iat_std",
    "bwd_iat_min":         "bwd_iat_min",
    "bwd_iat_max":         "bwd_iat_max",
    "fwd_header_length":   "fwd_header_length",
    "bwd_header_length":   "bwd_header_length",
    "flag_FIN":            "flag_FIN",
    "flag_SYN":            "flag_SYN",
    "flag_RST":            "flag_RST",
    "flag_PSH":            "flag_PSH",
    "flag_ACK":            "flag_ACK",
    "flag_URG":            "flag_URG",
    "protocol":            "protocol",
    "src_port":            "src_port",
    "dst_port":            "dst_port",
    "active_mean":         "flow_active_time",
    "idle_mean":           "flow_idle_time",
    # Packet-level from pcap_to_csv.py (REAL data, not placeholders!)
    "fwd_avg_ttl":         "ttl_mean",
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_time_window_features(df, window_sec=10.0):
    """Compute time-window features (periodicity, burst rate, etc.)."""
    # These are approximate when we don't have per-packet timestamps
    dur = df["flow_duration"].replace(0, np.nan)
    
    df["bytes_per_sec_window"] = (
        (df["total_fwd_bytes"] + df["total_bwd_bytes"]) / dur.clip(lower=1e-6)
    ).fillna(0)
    
    df["pkts_per_sec_window"] = (
        (df["total_fwd_packets"] + df["total_bwd_packets"]) / dur.clip(lower=1e-6)
    ).fillna(0)
    
    # Periodicity: ratio of IAT std to IAT mean (low = periodic = suspicious)
    iat_mean = df["flow_iat_mean"].replace(0, np.nan)
    df["periodicity_score"] = (
        1.0 - (df["flow_iat_std"] / iat_mean.clip(lower=1e-6)).clip(upper=1.0)
    ).fillna(0)
    
    # Burst rate: ratio of max IAT to mean IAT
    df["burst_rate"] = (
        df["flow_iat_max"] / iat_mean.clip(lower=1e-6)
    ).clip(upper=100).fillna(0)
    
    # Window-level features (approximated per-row since we don't have
    # cross-flow context; the real version would group by time windows)
    df["window_flow_count"] = 1
    df["window_unique_dsts"] = 1
    
    return df


def derive_packet_level_features(df):
    """
    Derive packet-level features from pcap_to_csv.py columns.
    These use REAL data instead of zero placeholders!
    """
    # TTL features — pcap_to_csv.py gives us fwd_avg_ttl and bwd_avg_ttl
    if "fwd_avg_ttl" in df.columns and "bwd_avg_ttl" in df.columns:
        fwd_ttl = pd.to_numeric(df["fwd_avg_ttl"], errors="coerce").fillna(0)
        bwd_ttl = pd.to_numeric(df["bwd_avg_ttl"], errors="coerce").fillna(0)
        df["ttl_mean"] = (fwd_ttl + bwd_ttl) / 2
        df["ttl_std"] = abs(fwd_ttl - bwd_ttl) / 2  # Approximation from averages
        df["ttl_min"] = np.minimum(fwd_ttl, bwd_ttl)
        df["ttl_max"] = np.maximum(fwd_ttl, bwd_ttl)
    elif "fwd_avg_ttl" in df.columns:
        ttl = pd.to_numeric(df["fwd_avg_ttl"], errors="coerce").fillna(0)
        df["ttl_mean"] = ttl
        df["ttl_std"] = 0
        df["ttl_min"] = ttl
        df["ttl_max"] = ttl
    
    # DNS query count — approximate from port usage
    if "dst_port" in df.columns:
        df["dns_query_count"] = (df["dst_port"] == 53).astype(int)
    else:
        df["dns_query_count"] = 0
    
    # Payload features — derive from packet length stats
    if "pkt_len_mean" in df.columns:
        df["payload_bytes_mean"] = pd.to_numeric(df["pkt_len_mean"], errors="coerce").fillna(0)
        df["payload_bytes_std"] = pd.to_numeric(df.get("pkt_len_std", 0), errors="coerce").fillna(0)
    elif "fwd_pkt_len_mean" in df.columns:
        df["payload_bytes_mean"] = pd.to_numeric(df["fwd_pkt_len_mean"], errors="coerce").fillna(0)
        df["payload_bytes_std"] = pd.to_numeric(df.get("fwd_pkt_len_std", 0), errors="coerce").fillna(0)
    
    # Payload zero ratio: approximate — small packets likely have high zero ratio
    if "avg_pkt_size" in df.columns and "min_pkt_size" in df.columns:
        avg = pd.to_numeric(df["avg_pkt_size"], errors="coerce").fillna(1).clip(lower=1)
        mn = pd.to_numeric(df["min_pkt_size"], errors="coerce").fillna(0)
        df["payload_zero_ratio"] = (mn / avg).clip(upper=1.0).fillna(0)
    else:
        df["payload_zero_ratio"] = 0
    
    # Payload entropy: not directly available, approximate from packet size variance
    if "pkt_len_var" in df.columns:
        var = pd.to_numeric(df["pkt_len_var"], errors="coerce").fillna(0)
        df["payload_entropy"] = np.log1p(var) / 20.0  # Normalized approximation
    else:
        df["payload_entropy"] = 0
    
    # TLS features available flag
    df["tls_features_available"] = 0  # Would need TLS-specific tshark fields
    
    return df


def align_to_schema(df):
    """Ensure DataFrame has exactly the 56 features in the right order."""
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df


def clean_and_normalize(df):
    """Clean NaN/Inf, cast to float32, and MinMax normalize."""
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    
    # Cast to numeric
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Replace inf
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], 0)
    
    # Cast to float32
    df[feat_cols] = df[feat_cols].astype(np.float32)
    
    # Check we have rows
    if len(df) == 0:
        print("  [ERROR] No rows to normalize! Check label filtering.")
        return df, None
    
    # MinMax normalize
    scaler = MinMaxScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    
    return df, scaler


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSOR 1: CTU-13 (from pcap_to_csv.py output)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_ctu13():
    """
    Process CTU-13 pcap_to_csv.py output → stage2_noniot_botnet.csv
    
    FIXES:
    - Uses PCAP-derived features (real TTL, real header lengths, real flags)
    - Properly handles Background labels (label=-1) by excluding them
    - Reports class balance and warns if severely imbalanced
    """
    print("=" * 60)
    print("CTU-13 PREPROCESSING (from PCAP CSVs)")
    print("=" * 60)
    
    # Try PCAP CSV first, fall back to binetflow
    if PCAP_CSV_CTU13.exists():
        print(f"\n  Loading PCAP-derived CSV: {PCAP_CSV_CTU13}")
        df = pd.read_csv(PCAP_CSV_CTU13, low_memory=False)
        print(f"  Loaded {len(df):,} flows")
        source = "pcap"
    else:
        print(f"\n  [WARN] PCAP CSV not found at: {PCAP_CSV_CTU13}")
        print(f"  Run pcap_to_csv.py first:")
        print(f"    python pcap_to_csv.py --dataset ctu13 --data_dir {RAW_CTU13} --output_dir {PCAP_CSV_CTU13.parent}")
        return False
    
    # ── Label normalization ──
    print("\n  Normalizing labels...")
    if "label" in df.columns:
        # CTU-13 labels from pcap_to_csv.py are numeric: 1=botnet, 0=benign, -1=background
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        
        # Filter out Background (-1) and unlabeled (NaN) rows
        n_before = len(df)
        df = df[df["label"].isin([0, 1])].copy()
        n_bg = n_before - len(df)
        print(f"  Removed {n_bg:,} Background/unlabeled rows ({n_bg/n_before*100:.1f}%)")
        
        df["class_label"] = df["label"].map({1: "botnet", 0: "benign"})
    else:
        print("  [ERROR] No 'label' column found in CTU-13 CSV!")
        return False
    
    df["device_type"] = "noniot"
    
    # Report class balance
    vc = df["class_label"].value_counts()
    print(f"\n  Class distribution:")
    for label, count in vc.items():
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    botnet_ratio = vc.get("botnet", 0) / len(df) if len(df) > 0 else 0
    if botnet_ratio < 0.01:
        print(f"\n  ⚠  WARNING: Only {botnet_ratio*100:.2f}% botnet samples!")
        print(f"     Consider loading more botnet-heavy scenarios (e.g., scenarios 1,2,9,10,11,13)")
        print(f"     Or use class weighting during training")
    
    # ── Subsample if needed ──
    if MAX_ROWS and len(df) > MAX_ROWS:
        # Stratified subsample to preserve class ratio
        df = df.groupby("class_label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(MAX_ROWS * len(x) / len(df))),
                               random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows (stratified)")
    
    # ── Map columns ──
    print("\n  Mapping columns to unified schema...")
    rename = {k: v for k, v in PCAP_CSV_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    
    # ── Derive packet-level features ──
    print("  Deriving packet-level features...")
    df = derive_packet_level_features(df)
    
    # ── Time-window features ──
    print("  Computing time-window features...")
    df = compute_time_window_features(df, TIME_WINDOW_SEC)
    
    # ── Align to schema ──
    df = align_to_schema(df)
    
    # ── Clean and normalize ──
    print("  Cleaning and normalizing...")
    df, scaler = clean_and_normalize(df)
    if df is None or len(df) == 0:
        return False
    
    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage2_noniot_botnet.csv"
    
    # Select only schema features + labels
    output_cols = ALL_FEATURES + ["class_label", "device_type"]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(out_path, index=False)
    
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Features: {len(ALL_FEATURES)}")
    print(f"    Labels: {df['class_label'].value_counts().to_dict()}")
    
    return True


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSOR 2: IoT-23 (from pcap_to_csv.py output)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_iot23():
    """
    Process IoT-23 pcap_to_csv.py output → stage2_iot_botnet.csv
    
    FIXES:
    - Completely bypasses the broken Zeek conn.log parser
    - Uses PCAP-derived features (real TTL, real flags, real header lengths)
    - Proper label handling from pcap_to_csv.py's label merger
    """
    print("\n" + "=" * 60)
    print("IoT-23 PREPROCESSING (from PCAP CSVs)")
    print("=" * 60)
    
    if PCAP_CSV_IOT23.exists():
        print(f"\n  Loading PCAP-derived CSV: {PCAP_CSV_IOT23}")
        df = pd.read_csv(PCAP_CSV_IOT23, low_memory=False)
        print(f"  Loaded {len(df):,} flows")
        source = "pcap"
    else:
        print(f"\n  [ERROR] PCAP CSV not found at: {PCAP_CSV_IOT23}")
        print(f"  Run pcap_to_csv.py first:")
        print(f"    python pcap_to_csv.py --dataset iot23 --data_dir {RAW_IOT23} --output_dir {PCAP_CSV_IOT23.parent}")
        
        # Try fallback: fixed conn.log parser
        print(f"\n  Attempting fallback: direct conn.log.labeled parsing...")
        df = fallback_load_iot23()
        if df is None or len(df) == 0:
            return False
        source = "connlog"
    
    # ── Label normalization ──
    print("\n  Normalizing labels...")
    if "label" in df.columns:
        raw = df["label"].astype(str).str.strip().str.lower()
        
        if source == "pcap":
            # Labels from pcap_to_csv.py IoT-23 label merger
            # IoT-23 labels: "Malicious", "Benign", specific malware names, etc.
            df["class_label"] = np.where(
                raw.str.contains("malicious|botnet|c&c|ddos|attack|portscan|"
                                 "okiru|torii|mirai|hajime|kenjiro|muhstik|"
                                 "hakai|linux|irc|spam|scan", regex=True, na=False),
                "botnet",
                np.where(
                    raw.str.contains("benign|normal|legitimate", regex=True, na=False),
                    "benign",
                    "unknown"
                )
            )
        elif source == "connlog":
            # Labels from fallback parser (_raw_label column)
            if "_raw_label" in df.columns:
                raw = df["_raw_label"].astype(str).str.strip().str.lower()
            df["class_label"] = np.where(
                raw.str.contains("malicious|botnet|c&c|ddos|attack|"
                                 "okiru|torii|mirai|hajime", regex=True, na=False),
                "botnet",
                np.where(
                    raw.str.contains("benign|normal", regex=True, na=False),
                    "benign",
                    "unknown"
                )
            )
    else:
        print("  [ERROR] No 'label' column!")
        return False
    
    # Filter to known labels only
    n_before = len(df)
    df = df[df["class_label"].isin(["benign", "botnet"])].copy()
    n_unknown = n_before - len(df)
    if n_unknown > 0:
        print(f"  Removed {n_unknown:,} unknown-label rows")
    
    if len(df) == 0:
        print("  [ERROR] No rows with valid labels after filtering!")
        print("  Check that pcap_to_csv.py successfully merged labels.")
        return False
    
    df["device_type"] = "iot"
    
    # Report class balance
    vc = df["class_label"].value_counts()
    print(f"\n  Class distribution:")
    for label, count in vc.items():
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # ── Subsample if needed ──
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.groupby("class_label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(MAX_ROWS * len(x) / len(df))),
                               random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows (stratified)")
    
    # ── Map columns ──
    print("\n  Mapping columns to unified schema...")
    rename = {k: v for k, v in PCAP_CSV_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    
    # ── Derive packet-level features ──
    print("  Deriving packet-level features...")
    df = derive_packet_level_features(df)
    
    # ── Time-window features ──
    print("  Computing time-window features...")
    df = compute_time_window_features(df, TIME_WINDOW_SEC)
    
    # ── Align, clean, normalize ──
    df = align_to_schema(df)
    print("  Cleaning and normalizing...")
    df, scaler = clean_and_normalize(df)
    if df is None or len(df) == 0:
        return False
    
    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage2_iot_botnet.csv"
    output_cols = [c for c in ALL_FEATURES + ["class_label", "device_type"] if c in df.columns]
    df[output_cols].to_csv(out_path, index=False)
    
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Features: {len(ALL_FEATURES)}")
    print(f"    Labels: {df['class_label'].value_counts().to_dict()}")
    
    return True


def fallback_load_iot23():
    """
    FIXED Zeek conn.log.labeled parser.
    
    The original parser failed because some IoT-23 files have the last
    3 columns merged. This version handles that by:
    1. Splitting on tabs
    2. If we get 21 columns instead of 23, splitting the last field on whitespace
    """
    if not RAW_IOT23.exists():
        print(f"  Fallback dir not found: {RAW_IOT23}")
        return None
    
    conn_files = sorted(RAW_IOT23.rglob("conn.log.labeled"))
    if not conn_files:
        print(f"  No conn.log.labeled files found in {RAW_IOT23}")
        return None
    
    EXPECTED_COLS = [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "proto", "service", "duration", "orig_bytes", "resp_bytes",
        "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
        "tunnel_parents", "label", "detailed-label",
    ]
    n_expected = len(EXPECTED_COLS)  # 23
    
    all_frames = []
    total_loaded = 0
    
    for f in conn_files:
        if MAX_ROWS and total_loaded >= MAX_ROWS:
            break
        
        rows = []
        with open(f, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                
                parts = line.split("\t")
                
                # ── THE FIX ──
                # If we got fewer columns than expected, the last few
                # may be space-separated instead of tab-separated.
                # Try splitting the last field on whitespace.
                if len(parts) < n_expected and len(parts) >= 20:
                    # The last field might contain "tunnel_parents label detailed-label"
                    last = parts[-1]
                    extra = last.split()
                    if len(extra) >= 3:
                        parts = parts[:-1] + extra[:3]
                    elif len(extra) == 2:
                        parts = parts[:-1] + extra + ["-"]
                
                # Pad or trim to expected length
                if len(parts) >= n_expected:
                    rows.append(parts[:n_expected])
                elif len(parts) >= 21:
                    rows.append(parts + ["-"] * (n_expected - len(parts)))
        
        if rows:
            chunk = pd.DataFrame(rows, columns=EXPECTED_COLS)
            chunk = chunk.replace("-", np.nan)
            chunk["_source_file"] = str(f.relative_to(RAW_IOT23))
            all_frames.append(chunk)
            total_loaded += len(chunk)
            print(f"    Loaded {f.relative_to(RAW_IOT23)}: {len(chunk):,} rows")
    
    if not all_frames:
        return None
    
    df = pd.concat(all_frames, ignore_index=True)
    print(f"  Total loaded: {len(df):,} rows")
    
    # Map columns
    col_map = {
        "duration": "flow_duration",
        "orig_bytes": "total_fwd_bytes",
        "resp_bytes": "total_bwd_bytes",
        "orig_pkts": "total_fwd_packets",
        "resp_pkts": "total_bwd_packets",
        "proto": "protocol",
        "id.orig_p": "src_port",
        "id.resp_p": "dst_port",
        "label": "_raw_label",
    }
    df = df.rename(columns=col_map)
    
    # Convert numerics
    for col in ["flow_duration", "total_fwd_bytes", "total_bwd_bytes",
                "total_fwd_packets", "total_bwd_packets", "src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Encode protocol
    proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].astype(str).str.lower().map(proto_map).fillna(0).astype(int)
    
    return df


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSOR 3: UNSW+CICIDS (unchanged — uses payload-byte CSVs)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_stage1_from_pcap_csvs():
    """
    Build Stage-1 training data by combining:
      - IoT-23 flows → device_type = "iot"
      - CTU-13 flows → device_type = "noniot"
    Both have real PCAP-derived features (TTL, flags, headers).
    """
    print("\n" + "=" * 60)
    print("STAGE-1 PREPROCESSING (from PCAP CSVs)")
    print("=" * 60)

    if not PCAP_CSV_IOT23.exists() or not PCAP_CSV_CTU13.exists():
        print("  [ERROR] Need both iot23_all_flows.csv and ctu13_all_flows.csv")
        return False

    # ── Load IoT-23 (IoT traffic) ──
    print("\n  Loading IoT-23 flows...")
    iot = pd.read_csv(PCAP_CSV_IOT23, low_memory=False)
    iot["device_type"] = "iot"
    print(f"    {len(iot):,} flows")

    # ── Load CTU-13 (Non-IoT traffic) ──
    print("  Loading CTU-13 flows...")
    noniot = pd.read_csv(PCAP_CSV_CTU13, low_memory=False)
    noniot["label"] = pd.to_numeric(noniot["label"], errors="coerce")
    noniot = noniot[noniot["label"].isin([0, 1])].copy()  # drop background
    noniot["device_type"] = "noniot"
    print(f"    {len(noniot):,} flows (after removing background)")

    # ── Balance the classes ──
    # Subsample IoT to reasonable size (match ~2:1 with non-IoT)
    n_noniot = len(noniot)
    n_iot_target = min(len(iot), n_noniot * 2)
    if len(iot) > n_iot_target:
        iot = iot.sample(n_iot_target, random_state=RANDOM_SEED)
        print(f"    Subsampled IoT to {n_iot_target:,} for balance")

    # ── Combine ──
    df = pd.concat([iot, noniot], ignore_index=True)
    print(f"\n  Combined: {len(df):,} flows")
    print(f"    IoT:    {(df['device_type']=='iot').sum():,}")
    print(f"    NonIoT: {(df['device_type']=='noniot').sum():,}")

    # ── Stage-1 label = device_type (not botnet/benign) ──
    df["class_label"] = df["device_type"]

    # ── Map columns ──
    rename = {k: v for k, v in PCAP_CSV_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # ── Derive features ──
    df = derive_packet_level_features(df)
    df = compute_time_window_features(df, TIME_WINDOW_SEC)
    df = align_to_schema(df)

    # ── Clean and normalize ──
    print("  Cleaning and normalizing...")
    df, scaler = clean_and_normalize(df)
    if df is None or len(df) == 0:
        return False

    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stage1_iot_vs_noniot.csv"
    output_cols = [c for c in ALL_FEATURES + ["class_label", "device_type"] if c in df.columns]
    df[output_cols].to_csv(out_path, index=False)

    print(f"\n  ✓ Saved: {out_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Device types: {df['device_type'].value_counts().to_dict()}")

    return True


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 58 + "╗")
    print("║  FIXED PREPROCESSING PIPELINE — Group 07                  ║")
    print("║  Uses pcap_to_csv.py output for real packet features      ║")
    print("╚" + "═" * 58 + "╝")
    
    results = {}
    
    # 1. CTU-13
    try:
        results["CTU-13"] = "✓ SUCCESS" if preprocess_ctu13() else "✗ FAILED"
    except Exception as e:
        results["CTU-13"] = f"✗ ERROR: {e}"
        import traceback; traceback.print_exc()
    
    # 2. IoT-23
    try:
        results["IoT-23"] = "✓ SUCCESS" if preprocess_iot23() else "✗ FAILED"
    except Exception as e:
        results["IoT-23"] = f"✗ ERROR: {e}"
        import traceback; traceback.print_exc()
    
    # 3. UNSW+CICIDS
 #  try:
  #     results["UNSW+CICIDS"] = "✓ SUCCESS" if preprocess_unsw_cicids() else "✗ FAILED (use previous output)"
 #  except Exception as e:
 #      results["UNSW+CICIDS"] = f"✗ ERROR: {e}"
 #      import traceback; traceback.print_exc()

    # 3. Stage-1
    try:
        results["Stage-1"] = "✓ SUCCESS" if preprocess_stage1_from_pcap_csvs() else "✗ FAILED"
    except Exception as e:
        results["Stage-1"] = f"✗ ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:20s} : {status}")
    
    # Check output files
    print("\nOutput files:")
    for fname in ["stage1_iot_vs_noniot.csv", "stage2_iot_botnet.csv", "stage2_noniot_botnet.csv"]:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {fname:35s} ({size_mb:.1f} MB)")
        else:
            print(f"  {fname:35s} (NOT FOUND)")
    
    print()


if __name__ == "__main__":
    main()
