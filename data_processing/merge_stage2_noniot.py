"""
=============================================================================
Dataset Merger — Group 07 | CPCS499 Botnet Detection
=============================================================================
Purpose:
    Merge CTU-13 and CIC-IDS-2017 processed CSVs into the final
    data/processed/stage2_noniot_botnet.csv used by the Stage-2 Non-IoT
    CNN-LSTM trainer (noniot_detector_cnnlstm.py).

    This script:
        1. Loads both processed CSVs
        2. Aligns columns to a shared unified schema
        3. Verifies raw feature scale (values MUST be >> 1.0)
        4. Applies class-balance sampling (optional, preserves majority)
        5. Saves the merged CSV — DO NOT normalise here

Usage:
    Windows : python  data_processing/merge_stage2_noniot.py
    macOS   : python3 data_processing/merge_stage2_noniot.py

Options:
    --ctu13    path to ctu13_processed.csv  (default: data/processed/ctu13_processed.csv)
    --cicids   path to cicids2017_processed.csv
    --output   path to stage2_noniot_botnet.csv
    --balance  max ratio of benign:botnet rows (default: 5.0, i.e. up to 5x benign)
    --seed     random seed for reproducibility (default: 42)

Dependencies:
    Windows : pip  install pandas numpy tqdm
    macOS   : pip3 install pandas numpy tqdm
=============================================================================
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified feature schema — ALL non-metadata, non-label columns.
# Every dataset must contribute (or fill 0 for) these features.
# The SCALER in noniot_detector_cnnlstm.py is fitted on exactly these cols.
# ---------------------------------------------------------------------------
UNIFIED_FEATURES = [
    # Flow-level statistical features
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    # TCP flag counts
    "fwd_syn_flag_cnt",
    "fwd_ack_flag_cnt",
    "fwd_rst_flag_cnt",
    "fwd_psh_flag_cnt",
    # Derived rate features
    "bytes_per_second",
    "packets_per_second",
    # Protocol / port
    "protocol_num",
    "src_port",
    "dst_port",
    "tcp_state",
    # TLS indicator
    "tls_features_available",
]

# Metadata columns kept in CSV but DROPPED before training
METADATA_COLS = ["src_ip", "timestamp", "device_type", "class_label"]


def load_and_align(path: str, source_tag: str) -> pd.DataFrame | None:
    """Load a processed dataset CSV and align to UNIFIED_FEATURES schema."""
    if not os.path.exists(path):
        log.warning(f"{source_tag}: file not found at {path} — skipping.")
        return None

    log.info(f"Loading {source_tag}: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    out = pd.DataFrame()

    # ── metadata ──────────────────────────────────────────────────────────
    out["src_ip"]      = df.get("src_ip",      pd.Series(["0.0.0.0"] * len(df))).values
    out["timestamp"]   = df.get("timestamp",   pd.Series([None]       * len(df))).values
    out["device_type"] = df.get("device_type", pd.Series(["noniot"]   * len(df))).values
    out["class_label"] = pd.to_numeric(df.get("class_label", pd.Series([0]*len(df))),
                                       errors="coerce").fillna(0).astype(int).values

    # ── unified features — prefer exact name, else fill 0 ─────────────────
    for feat in UNIFIED_FEATURES:
        if feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0).values
        else:
            # Try case-insensitive partial match
            matches = [c for c in df.columns
                       if c.strip().lower().replace(" ", "_") == feat]
            if matches:
                out[feat] = pd.to_numeric(df[matches[0]], errors="coerce").fillna(0).values
            else:
                out[feat] = 0.0

    # ── append any EXTRA numeric columns from the source ──────────────────
    # (CIC-IDS-2017 has ~78 features; many are informative)
    already = set(UNIFIED_FEATURES) | set(METADATA_COLS)
    for col in df.columns:
        stripped = col.strip().lower().replace(" ", "_").replace("/", "_per_")
        if stripped in already:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            out[stripped] = pd.to_numeric(df[col], errors="coerce").fillna(0).values

    # ── replace inf ────────────────────────────────────────────────────────
    out.replace([np.inf, -np.inf], 0, inplace=True)

    # Tag source for diagnostics
    out["_source"] = source_tag

    log.info(f"  → {len(out):,} rows aligned | "
             f"botnet={(out.class_label==1).sum():,} "
             f"({out.class_label.mean()*100:.1f}%)")
    return out


def balance_classes(df: pd.DataFrame, max_ratio: float, seed: int) -> pd.DataFrame:
    """
    Downsample majority class so that benign:botnet <= max_ratio.
    We NEVER upsample — that inflates training data artificially.
    """
    n_bot = (df.class_label == 1).sum()
    n_ben = (df.class_label == 0).sum()

    if n_bot == 0:
        log.error("No botnet rows in merged dataset — check label encoding.")
        return df

    max_benign = int(n_bot * max_ratio)
    if n_ben > max_benign:
        log.info(f"Downsampling benign: {n_ben:,} → {max_benign:,} "
                 f"(ratio {max_ratio:.1f}:1)")
        ben_df = df[df.class_label == 0].sample(max_benign, random_state=seed)
        bot_df = df[df.class_label == 1]
        df = pd.concat([ben_df, bot_df]).sample(frac=1, random_state=seed)
    return df


def merge(ctu13_path: str, cicids_path: str,
          output_path: str, max_ratio: float, seed: int) -> None:

    parts = []

    ctu13_df  = load_and_align(ctu13_path,  "CTU-13")
    cicids_df = load_and_align(cicids_path, "CIC-IDS-2017")

    if ctu13_df is not None:
        parts.append(ctu13_df)
    if cicids_df is not None:
        parts.append(cicids_df)

    if not parts:
        log.error(
            "No datasets loaded. Run process_ctu13.py and process_cicids2017.py first.\n"
            "Expected files:\n"
            f"  {ctu13_path}\n"
            f"  {cicids_path}"
        )
        return

    merged = pd.concat(parts, ignore_index=True)
    log.info(f"\nMerged: {len(merged):,} rows total | "
             f"benign={(merged.class_label==0).sum():,} | "
             f"botnet={(merged.class_label==1).sum():,}")

    # ── CRITICAL raw scale verification ────────────────────────────────────
    feat_cols = [c for c in merged.columns
                 if c not in METADATA_COLS + ["_source"]
                 and merged[c].dtype in [np.float64, np.float32,
                                         np.int64,   np.int32]]
    max_vals = merged[feat_cols].max()

    log.info("\n=== RAW SCALE CHECK (MUST show values >> 1.0) ===")
    top5 = max_vals.sort_values(ascending=False).head(5)
    log.info(top5.to_string())

    if max_vals.max() <= 1.0:
        log.error(
            "\n" + "="*60 + "\n"
            "STOP: All features <= 1.0 — data is ALREADY NORMALISED.\n"
            "The scaler will be fitted on normalised data → scale_max ~ 1.\n"
            "This is the root cause bug identified in the diagnosis.\n"
            "Fix: use original raw CTU-13 / CIC-IDS-2017 CSVs.\n"
            + "="*60
        )
        return

    log.info(f"Raw scale OK. max_feature_value = {max_vals.max():.2f}")

    # ── class balance ──────────────────────────────────────────────────────
    merged = balance_classes(merged, max_ratio, seed)

    # ── remove diagnostic tag ──────────────────────────────────────────────
    merged.drop(columns=["_source"], errors="ignore", inplace=True)

    # ── save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"SAVED: {output_path}")
    log.info(f"Total rows   : {len(merged):,}")
    log.info(f"Features     : {len(feat_cols)}")
    log.info(f"Benign       : {(merged.class_label==0).sum():,}")
    log.info(f"Botnet       : {(merged.class_label==1).sum():,}")
    log.info(f"Max raw value: {max_vals.max():.2f}")
    log.info("Next step: run models/stage2/noniot_detector_cnnlstm.py")
    log.info(f"{'='*60}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge CTU-13 + CIC-IDS-2017 into stage2_noniot_botnet.csv"
    )
    parser.add_argument("--ctu13",   default="data/processed/ctu13_processed.csv")
    parser.add_argument("--cicids",  default="data/processed/cicids2017_processed.csv")
    parser.add_argument("--output",  default="data/processed/stage2_noniot_botnet.csv")
    parser.add_argument("--balance", type=float, default=5.0,
                        help="Max benign:botnet ratio (default 5.0)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    merge(args.ctu13, args.cicids, args.output, args.balance, args.seed)
