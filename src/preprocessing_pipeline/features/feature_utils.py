"""
feature_utils.py — Shared feature engineering helpers.

Provides functions to:
  - Compute derived flow-level stats when raw columns are missing
  - Build time-window features
  - Compute packet-level surrogate features from conn-log data
  - Align any DataFrame to the unified feature schema
  - Normalize / scale the final feature matrix
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import (
    ALL_FEATURES, FLOW_FEATURES, TIME_WINDOW_FEATURES,
    PACKET_FEATURES, TLS_FEATURES, PROTO_MAP, LABEL_COLS,
)


# ─────────────────────────────────────────────────────
#  1.  PROTOCOL ENCODING
# ─────────────────────────────────────────────────────
def encode_protocol(series: pd.Series) -> pd.Series:
    """Map protocol strings/numbers to integer codes."""
    return series.astype(str).str.lower().str.strip().map(PROTO_MAP).fillna(0).astype(int)


# ─────────────────────────────────────────────────────
#  2.  DERIVED FLOW STATS
# ─────────────────────────────────────────────────────
def compute_derived_flow_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing flow-level columns with safe derivations."""
    df = df.copy()

    # Ensure core columns exist with 0 defaults
    for col in ["total_fwd_packets", "total_bwd_packets",
                 "total_fwd_bytes", "total_bwd_bytes", "flow_duration"]:
        if col not in df.columns:
            df[col] = 0

    # Replace non-numeric placeholders (skip labels and private cols)
    skip_cols = set(LABEL_COLS) | {c for c in df.columns if c.startswith("_")}
    for col in df.columns:
        if col in skip_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bytes per sec / packets per sec
    dur = df["flow_duration"].replace(0, np.nan)
    if "flow_bytes_per_sec" not in df.columns or df["flow_bytes_per_sec"].isna().all():
        total_bytes = df["total_fwd_bytes"].fillna(0) + df["total_bwd_bytes"].fillna(0)
        df["flow_bytes_per_sec"] = (total_bytes / dur).fillna(0)
    if "flow_pkts_per_sec" not in df.columns or df["flow_pkts_per_sec"].isna().all():
        total_pkts = df["total_fwd_packets"].fillna(0) + df["total_bwd_packets"].fillna(0)
        df["flow_pkts_per_sec"] = (total_pkts / dur).fillna(0)

    # Packet-length stats approximation (when only totals are available)
    for direction, pkt_col, byte_col in [
        ("fwd", "total_fwd_packets", "total_fwd_bytes"),
        ("bwd", "total_bwd_packets", "total_bwd_bytes"),
    ]:
        pkts = df[pkt_col].replace(0, np.nan)
        mean_len = (df[byte_col] / pkts).fillna(0)
        for stat, val in [("min", mean_len), ("max", mean_len),
                          ("mean", mean_len), ("std", 0)]:
            col_name = f"{direction}_pkt_len_{stat}"
            if col_name not in df.columns or df[col_name].isna().all():
                df[col_name] = val

    # IAT approximation (uniform assumption when only duration & pkt count known)
    for direction, pkt_col in [("fwd", "total_fwd_packets"),
                                ("bwd", "total_bwd_packets"),
                                ("flow", None)]:
        if pkt_col:
            n = df[pkt_col].replace(0, np.nan)
        else:
            n = (df["total_fwd_packets"] + df["total_bwd_packets"]).replace(0, np.nan)
        avg_iat = (df["flow_duration"] / n).fillna(0)
        for stat, val in [("min", avg_iat), ("max", avg_iat),
                          ("mean", avg_iat), ("std", 0)]:
            col_name = f"{direction}_iat_{stat}"
            if col_name not in df.columns or df[col_name].isna().all():
                df[col_name] = val

    # Header lengths approximation
    for col in ["fwd_header_length", "bwd_header_length"]:
        if col not in df.columns or df[col].isna().all():
            # TCP header = 20 bytes × packets; rough surrogate
            pkt_col = "total_fwd_packets" if "fwd" in col else "total_bwd_packets"
            df[col] = df[pkt_col].fillna(0) * 20

    # Ports
    for col in ["src_port", "dst_port"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # TCP flags — default 0 if missing
    for flag in ["flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG"]:
        if flag not in df.columns or df[flag].isna().all():
            df[flag] = 0

    # Active / idle time
    for col in ["flow_active_time", "flow_idle_time"]:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0

    return df


# ─────────────────────────────────────────────────────
#  3.  TCP FLAG EXTRACTION FROM STATE STRINGS
# ─────────────────────────────────────────────────────
def flags_from_state(state_series: pd.Series) -> pd.DataFrame:
    """
    Extract approximate TCP flag counts from Zeek conn_state strings.
    e.g. 'S0', 'SF', 'REJ', 'RSTO', etc.
    """
    flags = pd.DataFrame(0, index=state_series.index,
                         columns=["flag_FIN", "flag_SYN", "flag_RST",
                                  "flag_PSH", "flag_ACK", "flag_URG"])
    s = state_series.astype(str).str.upper()

    # SYN was attempted
    flags.loc[s.str.contains("S", na=False), "flag_SYN"] = 1
    # FIN (SF = normal close)
    flags.loc[s.str.contains("F", na=False), "flag_FIN"] = 1
    # RST
    flags.loc[s.str.contains("R", na=False), "flag_RST"] = 1
    # ACK implied in most established states
    flags.loc[s.isin(["SF", "S1", "S2", "S3", "RSTO", "RSTR", "OTH"]), "flag_ACK"] = 1

    return flags


# ─────────────────────────────────────────────────────
#  4.  TIME-WINDOW FEATURES
# ─────────────────────────────────────────────────────
def compute_time_window_features(df: pd.DataFrame,
                                  window_sec: float = 10.0) -> pd.DataFrame:
    """
    Compute time-window aggregate features.
    If no timestamp column is present, creates surrogate features
    from flow-level stats.
    """
    df = df.copy()
    n = len(df)

    # Surrogates when we don't have real timestamps
    df["bytes_per_second_window"] = df.get("flow_bytes_per_sec",
                                           pd.Series(0, index=df.index))
    df["packets_per_second_window"] = df.get("flow_pkts_per_sec",
                                              pd.Series(0, index=df.index))

    # connections_per_window: approximate via a rolling count if sorted by time
    df["connections_per_window"] = 1  # each row is one connection

    # distinct_dst_ips_window: not computable without IP column; use 1
    df["distinct_dst_ips_window"] = 1

    # periodicity_score: coefficient of variation of IAT
    iat_mean = df.get("flow_iat_mean", pd.Series(0, index=df.index))
    iat_std  = df.get("flow_iat_std", pd.Series(0, index=df.index))
    df["periodicity_score"] = np.where(
        iat_mean > 0, iat_std / iat_mean, 0
    )

    # burst_rate: ratio of packets to duration (already ≈ pkts/sec)
    df["burst_rate"] = df.get("flow_pkts_per_sec",
                               pd.Series(0, index=df.index))

    return df


# ─────────────────────────────────────────────────────
#  5.  PACKET / TLS PLACEHOLDER FEATURES
# ─────────────────────────────────────────────────────
def add_packet_and_tls_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Add zero-filled columns for packet-level and TLS features."""
    df = df.copy()
    for col in PACKET_FEATURES:
        if col not in df.columns:
            df[col] = 0
    for col in TLS_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df


# ─────────────────────────────────────────────────────
#  6.  SCHEMA ALIGNMENT
# ─────────────────────────────────────────────────────
def align_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder and pad DataFrame to match ALL_FEATURES exactly.
    Any missing column is filled with 0.
    Label columns are preserved if present.
    """
    df = df.copy()

    # Ensure every feature exists
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Keep labels if present
    keep = ALL_FEATURES[:]
    for lbl in LABEL_COLS + ["_raw_label", "_detailed_label", "_attack_cat"]:
        if lbl in df.columns:
            keep.append(lbl)

    return df[keep]


# ─────────────────────────────────────────────────────
#  7.  CLEANING
# ─────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows that are entirely NaN in feature columns
    - Replace inf with NaN then fill with 0
    - Cast numeric features to float32 for memory efficiency
    - Preserve string label columns untouched
    """
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]

    # Drop rows where ALL feature columns are NaN
    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)

    # Clean numeric features only — leave label columns alone
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[feat_cols] = df[feat_cols].astype(np.float32)

    return df


# ─────────────────────────────────────────────────────
#  8.  NORMALIZATION (MinMax to [0,1])
# ─────────────────────────────────────────────────────
def normalize_features(df: pd.DataFrame,
                        scaler: MinMaxScaler | None = None,
                        fit: bool = True) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMax scaling to all numeric feature columns.
    Returns (scaled_df, fitted_scaler).
    """
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    non_feat = [c for c in df.columns if c not in feat_cols]

    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
    else:
        df[feat_cols] = scaler.transform(df[feat_cols])

    return df, scaler


# ─────────────────────────────────────────────────────
#  9.  FULL PIPELINE HELPER
# ─────────────────────────────────────────────────────
def full_feature_pipeline(df: pd.DataFrame,
                           window_sec: float = 10.0,
                           normalize: bool = True) -> tuple[pd.DataFrame, MinMaxScaler | None]:
    """
    End-to-end: derived stats → time-window → placeholders → align → clean → normalize.
    """
    df = compute_derived_flow_stats(df)
    df = compute_time_window_features(df, window_sec)
    df = add_packet_and_tls_placeholders(df)
    df = align_to_schema(df)
    df = clean_dataframe(df)

    scaler = None
    if normalize:
        df, scaler = normalize_features(df)

    return df, scaler
