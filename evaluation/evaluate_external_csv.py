"""
═══════════════════════════════════════════════════════════════════════════
 evaluate_external_csv.py
 Group 07 | CPCS499  —  Hybrid AI-Based Botnet Detection
═══════════════════════════════════════════════════════════════════════════

 Purpose
 ───────
 Run end-to-end evaluation of the trained Stage-1 (RF) and Stage-2 Non-IoT
 (CNN-LSTM) models on an EXTERNAL CICFlowMeter-format CSV — e.g. any of
 the published CSE-CIC-IDS-2018 daily files such as Friday-02-03-2018.csv.

 Why this script exists
 ──────────────────────
 Your existing pipelines (preprocess_from_pcap_csvs.py, classifier.py,
 noniot_detector_cnnlstm.py) are designed for TRAINING from your prepared
 raw datasets. There was no offline evaluator that:
   1. Maps CICFlowMeter column names → your unified 56-feature schema
   2. Runs Stage-1 then Stage-2 in a single pass over a labeled CSV
   3. Reports Precision / Recall / F1 / ROC-AUC + threshold sweep
   4. Treats Recall as the primary metric (per project rules)

 This script imports your existing Stage1Classifier and Stage2Detector
 classes — it does NOT duplicate model logic. If those classes change,
 this script picks up the changes automatically.

 Save location
 ─────────────
   <project_root>/evaluation/evaluate_external_csv.py

 Usage
 ─────
 Windows:
   python evaluation/evaluate_external_csv.py --csv path/to/Friday-02-03-2018.csv
 macOS:
   python3 evaluation/evaluate_external_csv.py --csv path/to/Friday-02-03-2018.csv

 Recommended first run (subsample + lower threshold, given the known scaler issue):
   python3 evaluation/evaluate_external_csv.py \
       --csv path/to/Friday-02-03-2018.csv \
       --max_rows 200000 \
       --threshold 0.30
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

# ─── Project imports (assumes script lives at <project_root>/evaluation/) ───
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.stage1.classifier import Stage1Classifier, ALL_FEATURES
from models.stage2.noniot_detector_cnnlstm import Stage2Detector, DEVICE


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING:  CICFlowMeter (CSE-CIC-IDS-2018)  →  unified 56-feature
# ═══════════════════════════════════════════════════════════════════════════
# Approximately 30 / 56 features map cleanly. The remaining 26 (TTL stats,
# payload entropy, DNS counts, TLS flag, time-window aggregates, etc.) are
# not present in CICFlowMeter output and are zero-filled — same behavior as
# production CSV ingestion.
CICFLOW_TO_UNIFIED = {
    "Flow Duration":      "flow_duration",
    "Tot Fwd Pkts":       "total_fwd_packets",
    "Tot Bwd Pkts":       "total_bwd_packets",
    "TotLen Fwd Pkts":    "total_fwd_bytes",
    "TotLen Bwd Pkts":    "total_bwd_bytes",
    "Fwd Pkt Len Min":    "fwd_pkt_len_min",
    "Fwd Pkt Len Max":    "fwd_pkt_len_max",
    "Fwd Pkt Len Mean":   "fwd_pkt_len_mean",
    "Fwd Pkt Len Std":    "fwd_pkt_len_std",
    "Bwd Pkt Len Min":    "bwd_pkt_len_min",
    "Bwd Pkt Len Max":    "bwd_pkt_len_max",
    "Bwd Pkt Len Mean":   "bwd_pkt_len_mean",
    "Bwd Pkt Len Std":    "bwd_pkt_len_std",
    "Flow Byts/s":        "flow_bytes_per_sec",
    "Flow Pkts/s":        "flow_pkts_per_sec",
    "Flow IAT Mean":      "flow_iat_mean",
    "Flow IAT Std":       "flow_iat_std",
    "Flow IAT Min":       "flow_iat_min",
    "Flow IAT Max":       "flow_iat_max",
    "Fwd IAT Mean":       "fwd_iat_mean",
    "Fwd IAT Std":        "fwd_iat_std",
    "Fwd IAT Min":        "fwd_iat_min",
    "Fwd IAT Max":        "fwd_iat_max",
    "Bwd IAT Mean":       "bwd_iat_mean",
    "Bwd IAT Std":        "bwd_iat_std",
    "Bwd IAT Min":        "bwd_iat_min",
    "Bwd IAT Max":        "bwd_iat_max",
    "Fwd Header Len":     "fwd_header_length",
    "Bwd Header Len":     "bwd_header_length",
    "FIN Flag Cnt":       "flag_FIN",
    "SYN Flag Cnt":       "flag_SYN",
    "RST Flag Cnt":       "flag_RST",
    "PSH Flag Cnt":       "flag_PSH",
    "ACK Flag Cnt":       "flag_ACK",
    "URG Flag Cnt":       "flag_URG",
    "Protocol":           "protocol",
    "Dst Port":           "dst_port",
    "Active Mean":        "flow_active_time",   # closest available proxy
    "Idle Mean":          "flow_idle_time",     # closest available proxy
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _print_header(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _build_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter columns to unified schema and zero-fill missing."""
    df = df.rename(columns=CICFLOW_TO_UNIFIED)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.replace([np.inf, -np.inf], 0).astype(np.float32)
    return df


def _tile_to_sequences(X: np.ndarray, seq_len: int) -> torch.Tensor:
    """
    Tile each row to a (seq_len, n_features) sequence by replication.

    This is REQUIRED when no src_ip column is available (as in the
    published CSE-CIC-IDS-2018 daily CSVs). The downside is that the
    LSTM cannot extract real temporal patterns — it sees a constant
    sequence — so behaviour collapses toward a CNN classifier.

    For proper temporal evaluation re-extract flows from the PCAPs
    using src/ingestion/pcap_to_csv.py, which preserves src_ip.
    """
    B, F = X.shape
    seqs = np.broadcast_to(X[:, None, :], (B, seq_len, F))
    return torch.tensor(seqs.copy(), dtype=torch.float32)


def _predict_in_batches(model, seqs: torch.Tensor, batch: int = 512) -> np.ndarray:
    """Run sigmoid(model(seqs)) in batches to avoid OOM."""
    probs = np.zeros(len(seqs), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk  = seqs[i:i + batch].to(DEVICE)
            logits = model(chunk).cpu().numpy().squeeze(-1)
            probs[i:i + batch] = 1.0 / (1.0 + np.exp(-logits))
    return probs


def _save_confusion_plot(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Botnet"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Botnet"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Stage-1 + Stage-2 Non-IoT on a CICFlowMeter CSV."
    )
    ap.add_argument("--csv", required=True, type=str,
                    help="Path to CICFlowMeter-format CSV (e.g. CIC-IDS-2018 day file).")
    ap.add_argument("--out_dir", default=str(ROOT / "evaluation" / "results"),
                    help="Output directory for predictions / plots / summary.")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Optional cap on rows read (recommended for first run).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the saved Stage-2 threshold (e.g. 0.30 for higher recall).")
    ap.add_argument("--label_col", default="Label",
                    help="Name of the ground-truth column in the CSV.")
    ap.add_argument("--positive_label", default="Bot",
                    help="String value that marks the positive (botnet) class.")
    ap.add_argument("--stage1_model", default=str(ROOT / "models" / "stage1" / "rf_model.pkl"))
    ap.add_argument("--stage2_model", default=str(ROOT / "models" / "stage2" / "noniot_cnn_lstm.pt"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem

    # ─── 1. Load CSV ────────────────────────────────────────────────────
    _print_header(f"STEP 1 — Loading {args.csv}")
    t0 = time.perf_counter()
    df = pd.read_csv(args.csv, nrows=args.max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Rows loaded : {len(df):,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Read time   : {time.perf_counter() - t0:.1f}s")

    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. "
            f"Available columns: {list(df.columns)[:8]}..."
        )

    # ─── 2. Ground truth ────────────────────────────────────────────────
    y_true = (df[args.label_col].astype(str).str.strip() == args.positive_label).astype(int).values
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    print(f"\n  Ground truth — Botnet: {n_pos:,}   Benign: {n_neg:,}   "
          f"(pos rate {n_pos / len(y_true):.1%})")

    if n_pos == 0:
        print("  [WARN] No positive (botnet) examples — recall will be undefined.")

    # ─── 3. Map CICFlowMeter → unified 56-feature schema ────────────────
    _print_header("STEP 2 — Mapping columns to unified 56-feature schema")
    df_unified = _build_unified_features(df)
    n_mapped   = sum(1 for v in CICFLOW_TO_UNIFIED.values() if v in ALL_FEATURES)
    print(f"  Mapped {n_mapped} CICFlowMeter columns directly.")
    print(f"  Remaining {56 - n_mapped} unified features zero-filled "
          f"(TTL/payload/TLS/time-window — not in CICFlowMeter output).")

    # ─── 4. Stage-1 ─────────────────────────────────────────────────────
    _print_header("STEP 3 — Stage-1: IoT vs Non-IoT")
    s1 = Stage1Classifier.load(args.stage1_model)
    device_types, s1_confs = s1.predict(df_unified)
    if isinstance(device_types, str):
        device_types, s1_confs = [device_types], [s1_confs]
    device_types = np.asarray(device_types)
    s1_confs     = np.asarray(s1_confs, dtype=np.float32)

    n_iot    = int((device_types == "iot").sum())
    n_noniot = int((device_types == "noniot").sum())
    print(f"  Predicted IoT    : {n_iot:,}   ({n_iot / len(device_types):.1%})")
    print(f"  Predicted Non-IoT: {n_noniot:,}   ({n_noniot / len(device_types):.1%})")
    if n_iot / max(len(device_types), 1) > 0.05:
        print("  [WARN] Substantial 'iot' predictions on a Non-IoT dataset.")
        print("         This indicates Stage-1 calibration drift — flag for the report.")

    # ─── 5. Stage-2 Non-IoT ─────────────────────────────────────────────
    _print_header("STEP 4 — Stage-2 Non-IoT CNN-LSTM")
    s2 = Stage2Detector.load(Path(args.stage2_model))
    if args.threshold is not None:
        print(f"  Overriding threshold: {s2.threshold:.3f} → {args.threshold:.3f}")
        s2.threshold = float(args.threshold)
    print(f"  seq_len    : {s2.seq_len}")
    print(f"  n_features : {s2.n_features}")
    print(f"  threshold  : {s2.threshold:.3f}")

    # Align + scale (uses Stage2Detector's own logic & saved scaler)
    X_scaled = s2._align_and_scale(df_unified)
    print(f"  Scaled shape: {X_scaled.shape}")

    # Tile to (B, seq_len, n_features). See _tile_to_sequences caveat.
    seqs = _tile_to_sequences(X_scaled, s2.seq_len)
    print("  Tiling each flow into a constant length-{} sequence.".format(s2.seq_len))
    print("  NOTE: temporal advantage of LSTM is reduced — see script docstring.")

    t1     = time.perf_counter()
    probs  = _predict_in_batches(s2.model, seqs, batch=512)
    y_pred = (probs >= s2.threshold).astype(int)
    print(f"  Inference time: {time.perf_counter() - t1:.1f}s")

    # ─── 6. Metrics ─────────────────────────────────────────────────────
    _print_header("STEP 5 — Metrics (Stage-2 on full CSV)")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Botnet"], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")

    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"  Confusion matrix (rows = actual, cols = predicted):")
    print(f"               Benign     Botnet")
    print(f"   Benign  {tn:>10,}  {fp:>10,}")
    print(f"   Botnet  {fn:>10,}  {tp:>10,}")
    print(f"\n  Precision           : {p:.4f}")
    print(f"  Recall              : {r:.4f}    ← PROJECT-PRIORITY METRIC")
    print(f"  F1                  : {f1:.4f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  False Negative Rate : {fnr:.4f}    (missed botnets)")
    print(f"  False Positive Rate : {fpr:.4f}    (false alarms)")

    _save_confusion_plot(
        cm, f"Stage-2 Non-IoT — {Path(args.csv).name}",
        out_dir / f"cm_{stem}.png",
    )

    # ─── 7. Threshold sweep (recall-focused) ────────────────────────────
    _print_header("STEP 6 — Threshold sweep")
    print(f"  {'threshold':>10}  {'precision':>10}  {'recall':>10}  "
          f"{'f1':>10}  {'TP':>10}  {'FN':>10}  {'FP':>10}")
    sweep_rows = []
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        yp = (probs >= thr).astype(int)
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", pos_label=1, zero_division=0
        )
        tp_t = int(((yp == 1) & (y_true == 1)).sum())
        fn_t = int(((yp == 0) & (y_true == 1)).sum())
        fp_t = int(((yp == 1) & (y_true == 0)).sum())
        print(f"  {thr:>10.2f}  {pp:>10.4f}  {rr:>10.4f}  {ff:>10.4f}  "
              f"{tp_t:>10,}  {fn_t:>10,}  {fp_t:>10,}")
        sweep_rows.append({
            "threshold": thr, "precision": pp, "recall": rr, "f1": ff,
            "TP": tp_t, "FN": fn_t, "FP": fp_t,
        })

    # Find lowest threshold meeting recall >= 0.85 target
    recall_target = 0.85
    qualifying = [r_ for r_ in sweep_rows if r_["recall"] >= recall_target]
    if qualifying:
        best = max(qualifying, key=lambda x: x["precision"])
        print(f"\n  Best threshold for recall ≥ {recall_target}: "
              f"{best['threshold']:.2f}  "
              f"(precision={best['precision']:.4f}, recall={best['recall']:.4f})")
    else:
        print(f"\n  [WARN] No threshold in sweep achieves recall ≥ {recall_target}.")

    pd.DataFrame(sweep_rows).to_csv(
        out_dir / f"threshold_sweep_{stem}.csv", index=False)

    # ─── 8. Per-row predictions ─────────────────────────────────────────
    out_csv = out_dir / f"predictions_{stem}.csv"
    pd.DataFrame({
        "row":             np.arange(len(df)),
        "true_label":      np.where(y_true == 1, "Botnet", "Benign"),
        "stage1_device":   device_types,
        "stage1_conf":     s1_confs,
        "stage2_prob":     probs,
        "predicted_label": np.where(y_pred == 1, "Botnet", "Benign"),
        "correct":         (y_pred == y_true),
    }).to_csv(out_csv, index=False)

    # ─── 9. Summary JSON ────────────────────────────────────────────────
    summary = {
        "csv":                 str(args.csv),
        "rows_evaluated":      int(len(df)),
        "ground_truth":        {"botnet": n_pos, "benign": n_neg},
        "stage1": {
            "model":         str(args.stage1_model),
            "predicted_iot": n_iot,
            "predicted_noniot": n_noniot,
        },
        "stage2": {
            "model":              str(args.stage2_model),
            "threshold":          float(s2.threshold),
            "seq_len":            int(s2.seq_len),
            "n_features":         int(s2.n_features),
            "sequence_mode":      "tiled (no src_ip in CSV)",
            "precision":          float(p),
            "recall":             float(r),
            "f1":                 float(f1),
            "roc_auc":            auc,
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
        },
    }
    with open(out_dir / f"summary_{stem}.json", "w") as f_:
        json.dump(summary, f_, indent=2)

    _print_header("Done")
    print(f"  Per-row predictions : {out_csv}")
    print(f"  Threshold sweep     : {out_dir / f'threshold_sweep_{stem}.csv'}")
    print(f"  Confusion matrix    : {out_dir / f'cm_{stem}.png'}")
    print(f"  Summary JSON        : {out_dir / f'summary_{stem}.json'}")
    print()


if __name__ == "__main__":
    main()