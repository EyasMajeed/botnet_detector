"""
═══════════════════════════════════════════════════════════════════════
 CSE-CIC-IDS2018 IMPORTER (v6) — import_cicids2018.py
═══════════════════════════════════════════════════════════════════════
 Group 07 | CPCS499

 PURPOSE:
   Convert the raw CSE-CIC-IDS2018 CICFlowMeter CSV into the project's
   56-feature schema (matches S1_FEATURES in monitoring.py).

 INPUT:
   data/raw/cicids2018/Friday-02-03-2018.csv

 OUTPUT:
   data/processed/cicids2018_processed.csv
   56 features + class_label + device_type, raw values (no scaling)

 ALIGNMENT WITH monitoring.py:
   This v6 version produces output compatible with the v6 multi-source
   preprocessor and the v6 trainer. Key alignment decisions:

   1. 56-feature schema matching S1_FEATURES exactly
   2. Live-constant features forced to monitoring.py's hardcoded values:
        fwd_header_length=20.0, bwd_header_length=20.0,
        periodicity_score=0.0, burst_rate=0.0,
        window_flow_count=1.0, window_unique_dsts=1.0,
        payload_zero_ratio=0.0, payload_entropy=0.0
   3. tls_features_available computed from port set {443, 8443, 8883}
   4. dns_query_count from (dst_port==53 OR src_port==53)
   5. ttl_mean, ttl_std, ttl_min, ttl_max filled with 0 (CICFlowMeter
      doesn't expose TTL — that's a known CIC limitation, not our bug)
   6. NO scaling applied — trainer will fit StandardScaler on combined data

 USAGE:
   Windows: python import_cicids2018.py
   macOS  : python3 import_cicids2018.py
═══════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV   = BASE_DIR / "data" / "raw" / "cicids2018" / "Friday-02-03-2018.csv"
OUTPUT_CSV  = BASE_DIR / "data" / "processed" / "cicids2018_processed.csv"

FALLBACK_INPUTS = [
    BASE_DIR / "Friday-02-03-2018.csv",
    BASE_DIR / "data" / "Friday-02-03-2018.csv",
    Path(r"C:\Users\Administrator\Desktop\botnet_detector\data\raw\cicids2018\Friday-02-03-2018.csv"),
]


# ═══════════════════════════════════════════════════════════════════════
# 56-FEATURE SCHEMA (matches monitoring.py S1_FEATURES exactly)
# ═══════════════════════════════════════════════════════════════════════

S1_FEATURES = [
    "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes",   "total_bwd_bytes",
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
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "window_flow_count", "window_unique_dsts",
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max",
    "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
    "tls_features_available",
]
assert len(S1_FEATURES) == 56

# Live-constant features (forced to match monitoring.py)
LIVE_CONSTANT_FEATURES = {
    "fwd_header_length":  20.0,
    "bwd_header_length":  20.0,
    "periodicity_score":   0.0,
    "burst_rate":          0.0,
    "window_flow_count":   1.0,
    "window_unique_dsts":  1.0,
    "payload_zero_ratio":  0.0,
    "payload_entropy":     0.0,
}

TLS_PORTS = {443, 8443, 8883}

# Direct one-to-one column renames from CICFlowMeter to schema
DIRECT_RENAMES = {
    "Dst Port":          "dst_port",
    "Protocol":          "protocol",
    "Flow Duration":     "flow_duration",
    "Tot Fwd Pkts":      "total_fwd_packets",
    "Tot Bwd Pkts":      "total_bwd_packets",
    "TotLen Fwd Pkts":   "total_fwd_bytes",
    "TotLen Bwd Pkts":   "total_bwd_bytes",
    "Fwd Pkt Len Max":   "fwd_pkt_len_max",
    "Fwd Pkt Len Min":   "fwd_pkt_len_min",
    "Fwd Pkt Len Mean":  "fwd_pkt_len_mean",
    "Fwd Pkt Len Std":   "fwd_pkt_len_std",
    "Bwd Pkt Len Max":   "bwd_pkt_len_max",
    "Bwd Pkt Len Min":   "bwd_pkt_len_min",
    "Bwd Pkt Len Mean":  "bwd_pkt_len_mean",
    "Bwd Pkt Len Std":   "bwd_pkt_len_std",
    "Flow Byts/s":       "flow_bytes_per_sec",
    "Flow Pkts/s":       "flow_pkts_per_sec",
    "Flow IAT Mean":     "flow_iat_mean",
    "Flow IAT Std":      "flow_iat_std",
    "Flow IAT Max":      "flow_iat_max",
    "Flow IAT Min":      "flow_iat_min",
    "Fwd IAT Mean":      "fwd_iat_mean",
    "Fwd IAT Std":       "fwd_iat_std",
    "Fwd IAT Max":       "fwd_iat_max",
    "Fwd IAT Min":       "fwd_iat_min",
    "Bwd IAT Mean":      "bwd_iat_mean",
    "Bwd IAT Std":       "bwd_iat_std",
    "Bwd IAT Max":       "bwd_iat_max",
    "Bwd IAT Min":       "bwd_iat_min",
    "Active Mean":       "flow_active_time",
    "Idle Mean":         "flow_idle_time",
    "Pkt Len Mean":      "payload_bytes_mean",
    "Pkt Len Std":       "payload_bytes_std",
    "FIN Flag Cnt":      "flag_FIN",
    "SYN Flag Cnt":      "flag_SYN",
    "RST Flag Cnt":      "flag_RST",
    "PSH Flag Cnt":      "flag_PSH",
    "ACK Flag Cnt":      "flag_ACK",
    "URG Flag Cnt":      "flag_URG",
}


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def find_input():
    for candidate in [INPUT_CSV] + FALLBACK_INPUTS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"\n  Could not find CSE-CIC-IDS2018 CSV at any of:\n"
        f"    {INPUT_CSV}\n"
        + "\n".join(f"    {p}" for p in FALLBACK_INPUTS)
        + "\n  Edit INPUT_CSV at the top of this script."
    )


def safe_div(numerator, denominator):
    """Element-wise division returning 0 where denominator is 0/NaN."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], 0).fillna(0)


def derive_missing_features(df):
    """Compute the features that aren't direct CICFlowMeter matches."""

    # bytes_per_sec_window — recompute from totals (more reliable than CIC's)
    total_bytes = df["total_fwd_bytes"] + df["total_bwd_bytes"]
    df["bytes_per_sec_window"] = safe_div(total_bytes, df["flow_duration"])

    # pkts_per_sec_window
    total_pkts = df["total_fwd_packets"] + df["total_bwd_packets"]
    df["pkts_per_sec_window"] = safe_div(total_pkts, df["flow_duration"])

    # dns_query_count (matches monitoring.py: dst_port==53 OR src_port==53)
    src_port = df.get("src_port", pd.Series(np.zeros(len(df)), index=df.index))
    df["dns_query_count"] = ((df["dst_port"] == 53) | (src_port == 53)).astype(float)

    # tls_features_available (matches monitoring.py)
    df["tls_features_available"] = (
        df["dst_port"].isin(TLS_PORTS) | src_port.isin(TLS_PORTS)
    ).astype(float)

    # TTL features — CICFlowMeter doesn't export TTL. Fill with 0.
    # Note: this means CIC-IDS-2018 contributes no TTL signal. That's
    # acceptable — other sources (IoT-23, ETF, IEEE) provide TTL diversity.
    df["ttl_mean"] = 0.0
    df["ttl_std"]  = 0.0
    df["ttl_min"]  = 0.0
    df["ttl_max"]  = 0.0

    # src_port — CICFlowMeter omits it. Fill with 0.
    if "src_port" not in df.columns:
        df["src_port"] = 0.0

    return df


def force_live_constants(df):
    """Override features that live monitoring.py hardcodes."""
    for col, val in LIVE_CONSTANT_FEATURES.items():
        df[col] = val
    return df


def map_label(label_value):
    if pd.isna(label_value):
        return "unknown"
    s = str(label_value).strip().lower()
    if s in ("benign", "normal"):
        return "benign"
    return "botnet"


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  CSE-CIC-IDS2018 IMPORTER (v6) — Group 07                      ║")
    print("║  Maps CICFlowMeter → 56-feature schema (monitoring.py aligned) ║")
    print("╚" + "═" * 64 + "╝")

    try:
        input_path = find_input()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(f"\n  Loading: {input_path}")
    print(f"  (this may take a few minutes for the full dataset)")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Strip whitespace from CIC column names
    df.columns = [c.strip() for c in df.columns]

    # Verify required columns
    required = list(DIRECT_RENAMES.keys()) + ["Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  [ERROR] Required columns missing:")
        for c in missing:
            print(f"    ✗ {c}")
        print(f"\n  Available columns sample:")
        for c in df.columns[:20]:
            print(f"    {c}")
        sys.exit(1)

    # Convert critical columns to numeric, drop bad rows
    print(f"\n  Cleaning numeric columns...")
    n_before = len(df)
    for col in DIRECT_RENAMES.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    Dropped {n_dropped:,} rows with malformed values")

    df = df.replace([np.inf, -np.inf], 0)

    # Apply renames
    print(f"\n  Renaming {len(DIRECT_RENAMES)} columns to schema...")
    df = df.rename(columns=DIRECT_RENAMES)

    # Derive missing features
    print(f"  Deriving missing features...")
    df = derive_missing_features(df)

    # Force live-constant features
    print(f"  Forcing live-constant features (matches monitoring.py)...")
    df = force_live_constants(df)

    # Map labels
    print(f"  Mapping Label column...")
    df["class_label"] = df["Label"].apply(map_label)
    df["device_type"] = "noniot"

    n_before = len(df)
    df = df[df["class_label"].isin(["benign", "botnet"])].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    Dropped {n_dropped:,} unknown-label rows")

    # Schema alignment — fill any remaining missing features with 0
    for col in S1_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # Build output
    output_cols = list(S1_FEATURES) + ["class_label", "device_type"]
    df_out = df[output_cols].copy()

    # Cast features to float32
    for col in S1_FEATURES:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).astype(np.float32)

    # ── Summary ──
    print(f"\n  ── Final dataset summary ───────────────────────────────────")
    print(f"    Total rows  : {len(df_out):,}")
    print(f"    Total cols  : {df_out.shape[1]}")
    print(f"    Schema feats: 56/56 ✓")
    print(f"\n    Class distribution:")
    for cls, cnt in df_out["class_label"].value_counts().items():
        pct = cnt / len(df_out) * 100
        print(f"      {cls:>8s}: {cnt:>10,}  ({pct:5.1f}%)")

    # Verify forced constants
    print(f"\n    Live-constant feature verification:")
    for col, expected in LIVE_CONSTANT_FEATURES.items():
        actual = df_out[col].unique()
        ok = len(actual) == 1 and abs(float(actual[0]) - expected) < 1e-6
        flag = "✓" if ok else "✗"
        print(f"      {flag} {col:<22s} = {expected}")

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    file_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ Saved: {OUTPUT_CSV}")
    print(f"    Size: {file_size_mb:.1f} MB")
    print(f"\n  Next: python preprocess_from_pcap_csvs.py")


if __name__ == "__main__":
    main()
