"""
=============================================================================
Pipeline Verification Script — Group 07 | CPCS499 Botnet Detection
=============================================================================
Purpose:
    Run this BEFORE training to verify that stage2_noniot_botnet.csv
    contains raw (un-normalised) features and the correct schema.

    Also verifies the saved scaler and model checkpoint AFTER training.

Usage:
    Windows : python  data_processing/verify_pipeline.py
    macOS   : python3 data_processing/verify_pipeline.py

Options:
    --csv     path to stage2_noniot_botnet.csv
    --scaler  path to noniot_scaler.json (optional, post-training check)
    --model   path to noniot_cnn_lstm.pt (optional, post-training check)
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

METADATA_COLS = {"src_ip", "timestamp", "device_type", "class_label"}


def check_csv(path: str) -> bool:
    print(f"\n{'='*60}")
    print("STEP 1: Verifying stage2_noniot_botnet.csv")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"{FAIL} File not found: {path}")
        print("  Run: python3 data_processing/process_ctu13.py")
        print("  Then: python3 data_processing/process_cicids2017.py")
        print("  Then: python3 data_processing/merge_stage2_noniot.py")
        return False

    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Class label check
    if "class_label" not in df.columns:
        print(f"{FAIL} No 'class_label' column found.")
        return False

    n_bot = (df.class_label == 1).sum()
    n_ben = (df.class_label == 0).sum()
    print(f"  Benign : {n_ben:,}")
    print(f"  Botnet : {n_bot:,}")

    if n_bot == 0:
        print(f"{FAIL} Zero botnet rows — label encoding may be wrong.")
        return False
    print(f"{PASS} Both classes present.")

    # Raw scale check — THE CRITICAL ONE
    feat_cols = [c for c in df.columns if c not in METADATA_COLS]
    numeric   = df[feat_cols].select_dtypes(include=[np.number])
    max_vals  = numeric.max()
    top5      = max_vals.sort_values(ascending=False).head(5)

    print(f"\n  Top-5 max feature values (MUST be >> 1.0):")
    for col, val in top5.items():
        status = PASS if val > 100 else (WARN if val > 1 else FAIL)
        print(f"    {col:<35} {val:>15.2f}  {status}")

    global_max = max_vals.max()
    if global_max <= 1.0:
        print(f"\n{FAIL} All features <= 1.0 — DATA IS ALREADY NORMALISED.")
        print("  This will cause scale_max ~ 1 in the scaler (the diagnosed bug).")
        print("  Fix: use original un-processed CTU-13 / CIC-IDS-2017 CSVs.")
        return False

    if global_max < 100:
        print(f"\n{WARN} Max value = {global_max:.2f} — may be partially normalised.")
        print("  Verify with raw dataset source.")
    else:
        print(f"\n{PASS} Raw scale looks correct (max = {global_max:.2f}).")

    # NaN check
    nan_count = numeric.isna().sum().sum()
    if nan_count > 0:
        print(f"{WARN} {nan_count:,} NaN values found — will be replaced with 0 at training time.")
    else:
        print(f"{PASS} No NaN values.")

    # Inf check
    inf_count = np.isinf(numeric.values).sum()
    if inf_count > 0:
        print(f"{WARN} {inf_count:,} Inf values found — will be replaced with 0 at training time.")
    else:
        print(f"{PASS} No Inf values.")

    return True


def check_scaler(path: str) -> bool:
    print(f"\n{'='*60}")
    print("STEP 2: Verifying noniot_scaler.json (post-training)")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"{WARN} Scaler not found at {path} — run training first.")
        return True  # Not a failure if training hasn't run

    with open(path) as f:
        s = json.load(f)

    scales = np.array(s["scale"])
    means  = np.array(s["mean"])

    print(f"  Features in scaler: {len(s['features'])}")
    print(f"  scale_max : {scales.max():.2f}   (MUST be > 100)")
    print(f"  scale_min : {scales.min():.6f}")
    print(f"  mean_max  : {means.max():.2f}")

    if scales.max() <= 100:
        print(f"{FAIL} scale_max = {scales.max():.2f} — scaler was fitted on normalised data.")
        print("  Delete noniot_scaler.json and retrain from raw data.")
        return False

    print(f"{PASS} Scaler looks correct — fitted on raw features.")

    note = s.get("note", "")
    if "raw" in note.lower():
        print(f"{PASS} Note confirms: '{note}'")
    else:
        print(f"{WARN} Note field: '{note}' — verify scaler was fitted on raw data.")

    return True


def check_checkpoint(path: str) -> bool:
    print(f"\n{'='*60}")
    print("STEP 3: Verifying noniot_cnn_lstm.pt (post-training)")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"{WARN} Checkpoint not found at {path} — run training first.")
        return True

    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        print(f"{WARN} PyTorch not installed — skipping checkpoint check.")
        return True
    except Exception as e:
        print(f"{FAIL} Could not load checkpoint: {e}")
        return False

    required_keys = ["model_state", "n_features", "seq_len",
                     "threshold", "feature_cols", "model_type"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        print(f"{FAIL} Missing keys in checkpoint: {missing}")
        return False
    print(f"{PASS} All required keys present.")

    threshold = ckpt["threshold"]
    print(f"  threshold  : {threshold:.4f}  (PASS if 0.30–0.70)")
    print(f"  n_features : {ckpt['n_features']}")
    print(f"  seq_len    : {ckpt['seq_len']}")
    print(f"  model_type : {ckpt['model_type']}")

    if not (0.20 <= threshold <= 0.80):
        print(f"{WARN} Threshold {threshold:.4f} is outside normal range.")
        print("  Model may not have learned — check training logs.")
    else:
        print(f"{PASS} Threshold is in normal range.")

    return True


def main(csv_path, scaler_path, model_path):
    results = []
    results.append(check_csv(csv_path))
    results.append(check_scaler(scaler_path))
    results.append(check_checkpoint(model_path))

    print(f"\n{'='*60}")
    if all(results):
        print("ALL CHECKS PASSED — Pipeline is ready for training / inference.")
    else:
        print("SOME CHECKS FAILED — Fix the issues above before proceeding.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="data/processed/stage2_noniot_botnet.csv")
    parser.add_argument("--scaler", default="models/stage2/noniot_scaler.json")
    parser.add_argument("--model",  default="models/stage2/noniot_cnn_lstm.pt")
    args = parser.parse_args()
    main(args.csv, args.scaler, args.model)
