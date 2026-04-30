"""
=============================================================================
Stage-2 Outlier Fix — Group 07 | CPCS499 Botnet Detection
=============================================================================
Problem identified from verify_pipeline.py output:

    flow_bytes_per_sec   = 5,863,000,000,000  (5.86 TRILLION)
    bytes_per_sec_window = 5,863,000,000,000
    flow_pkts_per_sec    = 26,000,000,000

These are divide-by-zero artefacts from near-zero duration flows:
    bytes/duration ≈ 450,000 / 0.000077µs → ∞ → stored as max float

Why this destroys training:
    StandardScaler computes: scale_ = std(column)
    If one value is 5.86T and all others are < 1M, std ≈ 5.86T
    After transform: all normal flows → ≈ 0.0
    The model sees no variation in rate features → cannot learn

Fix:
    Clip each feature to its 99.9th percentile (computed per-feature on the
    training data). This removes < 0.1% of rows' effect while preserving
    the full range of legitimate network traffic.

    We do NOT use a fixed cap (e.g., 1e9) because:
    - different features have different natural ranges
    - percentile clipping is data-driven and reproducible

Output:
    data/processed/stage2_noniot_botnet.csv  (overwritten in-place)
    data/processed/stage2_outlier_caps.json  (caps saved for inference pipeline)

Usage:
    macOS   : python3 data_processing/fix_outliers_stage2.py
    Windows : python  data_processing/fix_outliers_stage2.py

Dependencies:
    pip3 install pandas numpy tqdm
=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CSV_PATH  = "data/processed/stage2_noniot_botnet.csv"
CAPS_PATH = "data/processed/stage2_outlier_caps.json"

PERCENTILE     = 99.9   # clip above this percentile
CHUNK_SIZE     = 200_000

# Columns to SKIP when clipping (binary/categorical, or metadata)
SKIP_CLIP = {
    "class_label", "device_type", "src_ip", "timestamp",
    "protocol_num", "tcp_state", "src_port", "dst_port",
    "tls_features_available",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK",
    "fwd_syn_flag_cnt", "fwd_ack_flag_cnt", "fwd_rst_flag_cnt",
    "fwd_psh_flag_cnt",
}


def compute_caps(df: pd.DataFrame, feat_cols: list[str]) -> dict[str, float]:
    """Compute 99.9th percentile for each feature column."""
    caps = {}
    log.info(f"Computing {PERCENTILE}th percentile caps on {len(df):,} rows...")
    for col in tqdm(feat_cols, desc="Computing caps"):
        val = np.percentile(df[col].values, PERCENTILE)
        caps[col] = float(val)
    return caps


def apply_caps(df: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    """Clip each feature column to its cap value."""
    n_clipped_total = 0
    for col, cap in caps.items():
        if col not in df.columns:
            continue
        mask = df[col] > cap
        n_clipped = mask.sum()
        if n_clipped > 0:
            df.loc[mask, col] = cap
            n_clipped_total += n_clipped
    log.info(f"Total values clipped: {n_clipped_total:,}")
    return df


def main():
    if not os.path.exists(CSV_PATH):
        log.error(f"File not found: {CSV_PATH}")
        log.error("Run merge_stage2_noniot.py first.")
        return

    log.info(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Identify numeric feature columns to clip
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in SKIP_CLIP
    ]
    log.info(f"Feature columns to check for outliers: {len(feat_cols)}")

    # ── Show worst offenders BEFORE clipping ──────────────────────────────
    max_vals = df[feat_cols].max()
    top10 = max_vals.sort_values(ascending=False).head(10)
    log.info("\nTop-10 max values BEFORE clipping:")
    log.info(top10.to_string())

    # ── Identify columns that actually need clipping ───────────────────────
    # A column needs clipping if its max >> its 99th percentile
    # (indicates extreme outliers, not a natural range)
    p99  = df[feat_cols].quantile(0.99)
    p999 = df[feat_cols].quantile(0.999)
    ratio = max_vals / p999.replace(0, np.nan)

    suspicious = ratio[ratio > 10].index.tolist()
    log.info(f"\nColumns with max > 10x their 99.9th percentile "
             f"(extreme outliers): {len(suspicious)}")
    for col in suspicious:
        log.info(f"  {col:<40} max={max_vals[col]:.3e}  "
                 f"p99.9={p999[col]:.3e}  ratio={ratio[col]:.1f}x")

    # ── Compute caps ───────────────────────────────────────────────────────
    caps = compute_caps(df, feat_cols)

    # Save caps for use in inference pipeline (live traffic normalization)
    os.makedirs(os.path.dirname(CAPS_PATH) or ".", exist_ok=True)
    json.dump({
        "percentile": PERCENTILE,
        "note": (f"99.9th percentile caps computed on stage2_noniot_botnet.csv "
                 f"({len(df):,} rows). Apply before StandardScaler in inference."),
        "caps": caps
    }, open(CAPS_PATH, "w"), indent=2)
    log.info(f"Caps saved: {CAPS_PATH}")

    # ── Apply clipping ─────────────────────────────────────────────────────
    df = apply_caps(df, caps)

    # ── Show max values AFTER clipping ────────────────────────────────────
    max_after = df[feat_cols].max()
    top5_after = max_after.sort_values(ascending=False).head(5)
    log.info("\nTop-5 max values AFTER clipping:")
    log.info(top5_after.to_string())

    # ── Scale sanity check ─────────────────────────────────────────────────
    global_max = max_after.max()
    if global_max < 100:
        log.warning(
            f"Max value after clipping = {global_max:.2f} — unusually low.\n"
            "Verify that clipping did not remove all signal from rate features."
        )
    else:
        log.info(f"\nScale check PASS: max after clipping = {global_max:.2f}")

    # ── Replace inf / nan introduced by any division ───────────────────────
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # ── Overwrite the CSV ──────────────────────────────────────────────────
    log.info(f"\nSaving corrected CSV → {CSV_PATH}")
    df.to_csv(CSV_PATH, index=False)

    log.info(f"\n{'='*60}")
    log.info("Outlier fix COMPLETE.")
    log.info(f"Rows     : {len(df):,}")
    log.info(f"Benign   : {(df.class_label==0).sum():,}")
    log.info(f"Botnet   : {(df.class_label==1).sum():,}")
    log.info(f"Max value: {global_max:.2f}")
    log.info("\nNext steps:")
    log.info("  1. Delete old model artifacts:")
    log.info("       rm models/stage2/noniot_scaler.json")
    log.info("       rm models/stage2/noniot_cnn_lstm.pt")
    log.info("  2. Retrain:")
    log.info("       python3 models/stage2/noniot_detector_cnnlstm.py")
    log.info("  3. Verify:")
    log.info("       python3 data_processing/verify_pipeline.py")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
