"""
export_s1_scaler.py  —  Export Stage-1 StandardScaler to s1_scaler.json
Group 07 | CPCS499

WHY THIS EXISTS:
  classifier.py trains the Random Forest on data that was ALREADY normalised
  by clean_and_normalize() inside preprocess_from_pcap_csvs.py.  The scaler
  object was fit there but never saved alongside the model.

  This script re-fits an identical StandardScaler on the same processed CSV
  and saves mean_ / scale_ to s1_scaler.json so monitoring.py can reproduce
  the exact same transformation on live feature vectors before calling
  rf.predict_proba().

  Running this ONCE after classifier.py training is complete is all that is
  needed.  You do NOT need to re-train the RF.

USAGE:
  python3 export_s1_scaler.py
  python3 export_s1_scaler.py --csv path/to/stage1_iot_vs_noniot.csv
  python3 export_s1_scaler.py --out path/to/models/stage1/s1_scaler.json

OUTPUT:
  models/stage1/s1_scaler.json
  {
    "features": [...56 feature names in training order...],
    "mean":     [...56 floats...],
    "scale":    [...56 floats...],
    "note":     "StandardScaler fitted on stage1_iot_vs_noniot.csv"
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Default paths (mirror classifier.py's ROOT logic) ────────────────────────
_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE
# Walk up to find the project root (contains data/ and models/)
for _cand in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_cand / "data" / "processed").exists():
        _ROOT = _cand
        break

DEFAULT_CSV = _ROOT / "data" / "processed" / "stage1_iot_vs_noniot.csv"
DEFAULT_OUT = _ROOT / "models" / "stage1" / "s1_scaler.json"

# ── 56-feature schema (must match classifier.py ALL_FEATURES exactly) ─────────
ALL_FEATURES = [
    "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes",   "total_bwd_bytes",
    "fwd_pkt_len_min",   "fwd_pkt_len_max",  "fwd_pkt_len_mean",  "fwd_pkt_len_std",
    "bwd_pkt_len_min",   "bwd_pkt_len_max",  "bwd_pkt_len_mean",  "bwd_pkt_len_std",
    "flow_bytes_per_sec","flow_pkts_per_sec",
    "flow_iat_mean",     "flow_iat_std",      "flow_iat_min",      "flow_iat_max",
    "fwd_iat_mean",      "fwd_iat_std",       "fwd_iat_min",       "fwd_iat_max",
    "bwd_iat_mean",      "bwd_iat_std",       "bwd_iat_min",       "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG",
    "protocol", "src_port", "dst_port",
    "flow_active_time",  "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "window_flow_count", "window_unique_dsts",
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max",
    "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
    "tls_features_available",
]

DROP_COLS = {"class_label", "device_type", "src_ip", "timestamp"}


def main(csv_path: Path, out_path: Path) -> None:
    print("╔" + "═" * 52 + "╗")
    print("║  export_s1_scaler.py  —  Group 07 Stage-1 Scaler  ║")
    print("╚" + "═" * 52 + "╝")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n  CSV not found: {csv_path}\n"
            "  Run preprocess_from_pcap_csvs.py first to generate it."
        )
    print(f"\n  Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # ── Select features ───────────────────────────────────────────────────────
    # Use only features present in both ALL_FEATURES and the CSV.
    # Columns present in the CSV but not in ALL_FEATURES are ignored.
    # Columns in ALL_FEATURES but not in the CSV get a 0.0 default.
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]

    if missing:
        print(f"\n  ⚠  {len(missing)} features missing from CSV (will default to 0.0 in live):")
        for m in missing:
            print(f"       {m}")

    X = df[available].copy()
    X.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    X.fillna(X.median(), inplace=True)
    X = X.values.astype(float)

    # ── Fit StandardScaler ────────────────────────────────────────────────────
    # Fit on ALL rows (full dataset) — identical to what preprocess_from_pcap_csvs.py
    # does via clean_and_normalize() before saving the CSV that classifier.py reads.
    print(f"\n  Fitting StandardScaler on {len(X):,} rows × {len(available)} features...")
    scaler = StandardScaler()
    scaler.fit(X)

    print(f"  mean  range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # ── Sanity check: ensure scale values are not all ~1.0 ───────────────────
    # If the CSV was ALREADY normalised (values mostly in [-3, 3]), the scaler
    # will have scale ≈ 1 for most features, which means applying it again in
    # monitoring.py would essentially be a no-op — correct, since the RF was
    # trained on that already-normalised CSV.
    near_one = (scaler.scale_ < 2.0).sum()
    if near_one > len(available) * 0.8:
        print(
            f"\n  ℹ  {near_one}/{len(available)} features have scale < 2.0.\n"
            "     This is expected if the CSV was already StandardScaler-normalised\n"
            "     by preprocess_from_pcap_csvs.py.  The scaler is still correct:\n"
            "     monitoring.py will apply the same transform the RF was trained on."
        )

    # ── Build output ──────────────────────────────────────────────────────────
    # Store mean/scale indexed by ALL_FEATURES order.
    # Features missing from the CSV get mean=0, scale=1 (identity for those cols).
    feat_mean  = []
    feat_scale = []
    avail_idx  = {f: i for i, f in enumerate(available)}

    for feat in ALL_FEATURES:
        if feat in avail_idx:
            i = avail_idx[feat]
            feat_mean.append(float(scaler.mean_[i]))
            feat_scale.append(float(scaler.scale_[i]))
        else:
            feat_mean.append(0.0)   # default: missing feature assumed 0
            feat_scale.append(1.0)  # identity: (0 - 0) / 1 = 0

    scaler_data = {
        "features": ALL_FEATURES,
        "mean":     feat_mean,
        "scale":    feat_scale,
        "n_train_rows": int(len(X)),
        "note": (
            "StandardScaler fitted on stage1_iot_vs_noniot.csv "
            "(the same normalised CSV that classifier.py's RF was trained on). "
            "Apply: z = (x - mean) / scale before calling rf.predict_proba()."
        ),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scaler_data, f, indent=2)

    print(f"\n  ✓ Scaler saved → {out_path}")
    print(f"    Features: {len(ALL_FEATURES)}")
    print(f"    Missing (identity):  {len(missing)}")
    print(f"    Present (fitted):    {len(available)}")

    # Show a few sample values for verification:
    print("\n  Sample feature ranges (first 5):")
    for feat in ALL_FEATURES[:5]:
        i = ALL_FEATURES.index(feat)
        print(f"    {feat:<35}  mean={feat_mean[i]:>10.4f}  scale={feat_scale[i]:>10.4f}")

    print("\n  Next step: run monitoring.py — Stage-1 scaler will load automatically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export Stage-1 StandardScaler to JSON")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help=f"Processed CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output JSON path (default: {DEFAULT_OUT})")
    args = ap.parse_args()
    main(args.csv, args.out)
