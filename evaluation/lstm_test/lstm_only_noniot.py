"""
================================================================================
 LSTM-only Ablation - Stage-2 Non-IoT Detector
 Group 07 | CPCS499 | AI-Based Botnet Detection
================================================================================

 PURPOSE
 -------
 Train and evaluate an LSTM-only variant of the Stage-2 Non-IoT detector
 (no CNN block) using the EXACT same data, splits, scaler, and
 hyperparameters as the hybrid CNN-LSTM in
 models/stage2/noniot_detector_cnnlstm.py.

 This isolates the empirical contribution of the convolutional spatial
 extractor. Any difference in test-set Recall / Precision / F1 between
 this run and the hybrid run is attributable to the CNN component alone.

 ARCHITECTURE  (LSTM directly on raw feature sequences)
 ------------------------------------------------------
   Input  (B, seq_len=20, n_features)
     -> LSTM(input=n_features, hidden=128, layers=2, dropout=0.3)
                              ^---- input dim is n_features, not 256,
                                   because there is no CNN compression
     -> h_n[-1]                                       # (B, 128)
     -> Linear(128, 64) + ReLU + Dropout(0.4)
     -> Linear(64, 1)  (logit)
     -> divide by TEMPERATURE  (matches hybrid logit scaling)

 Why this is the right LSTM-only baseline:
   We feed the raw normalised flow features straight into the LSTM, without
   any local-pattern extractor. If F1 drops vs. the hybrid, the convolutional
   spine was extracting useful local relationships that the LSTM could not
   easily learn on its own.

 INPUT
 -----
   data/processed/stage2_noniot_botnet.csv
   (raw, un-normalised - produced by data_processing/merge_stage2_noniot.py)

 OUTPUT (under evaluation/lstm_test/results/noniot/)
 ---------------------------------------------------
   metrics.json          Accuracy, precision, recall, F1, AUC, threshold,
                         param count, model_type tag
   training_curves.png   Loss + val accuracy + val recall over epochs
   confusion_matrix.png  Test-set confusion matrix
   roc_curve.png         Test-set ROC curve
   model.pt              Trained checkpoint
   scaler.json           StandardScaler (saved separately so the
                         production noniot_scaler.json is NOT clobbered)

 USAGE
 -----
   Windows : python evaluation\\lstm_test\\lstm_only_noniot.py
   macOS   : python3 evaluation/lstm_test/lstm_only_noniot.py

 DEPENDENCIES
 ------------
   Windows : pip  install pandas numpy scikit-learn torch matplotlib
   macOS   : pip3 install pandas numpy scikit-learn torch matplotlib

 COMMON ERRORS
 -------------
   - "FileNotFoundError: stage2_noniot_botnet.csv"
       -> Run data_processing/merge_stage2_noniot.py first.
   - "AssertionError: SCALER SANITY CHECK FAILED"
       -> Your CSV is already normalised. Regenerate it from raw flows.
   - "RuntimeError: No sequences built"
       -> Internal-IP filter dropped too many rows, OR fewer than SEQ_LEN
          flows per src_ip group. Check data quality.
   - Slow training on CPU
       -> LSTM-only has ~25K-50K fewer params than the hybrid but the
          recurrent dependency stops cuDNN fusion gains. Use GPU if available.
================================================================================
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make the project root importable.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.stage2 import noniot_detector_cnnlstm as nm  # noqa: E402


# ---------------------------------------------------------------------------
# Output paths (separate from production model outputs)
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "results" / "noniot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = OUT_DIR / "model.pt"
SCALER_PATH  = OUT_DIR / "scaler.json"
METRICS_PATH = OUT_DIR / "metrics.json"
CURVE_PATH   = OUT_DIR / "training_curves.png"
CM_PATH      = OUT_DIR / "confusion_matrix.png"
ROC_PATH     = OUT_DIR / "roc_curve.png"

nm.SCALER_PATH = SCALER_PATH

DEVICE        = nm.DEVICE
SEQ_LEN       = nm.SEQ_LEN
TARGET_RECALL = nm.TARGET_RECALL
RANDOM_SEED   = nm.RANDOM_SEED
TEMPERATURE   = nm.TEMPERATURE


# ---------------------------------------------------------------------------
# LSTM-ONLY model  (CNN removed, LSTM consumes raw feature sequences)
# ---------------------------------------------------------------------------
class LstmOnlyDetector(nn.Module):
    """
    LSTM with the same hidden size (128), layer count (2), and dropout (0.3)
    as the hybrid, but with input dim = n_features (instead of 256 from the
    CNN). Same classifier head as the hybrid.
    """

    def __init__(self, n_features: int, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) - already in batch_first layout
        _, (h_n, _) = self.lstm(x)
        logit = self.head(h_n[-1])
        return logit / self.temperature


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_curves(train_loss, val_loss, val_acc, val_recall):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss, label="Train Loss")
    axes[0].plot(val_loss, label="Val Loss")
    axes[0].set_title("LSTM-only Non-IoT - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(val_acc, label="Val Accuracy")
    axes[1].plot(val_recall, label="Val Recall")
    axes[1].axhline(TARGET_RECALL, ls="--", c="red",
                    label=f"Recall target {TARGET_RECALL}")
    axes[1].set_title("LSTM-only Non-IoT - Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=120)
    plt.close()
    print(f"  Curves -> {CURVE_PATH}")


def plot_cm(labels, preds, le):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("LSTM-only Non-IoT - Confusion Matrix\n"
                 "(TN=benign  FP=false alarm  FN=missed botnet  TP=botnet)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    plt.close()
    print(f"  CM     -> {CM_PATH}")


def plot_roc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM-only Non-IoT - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=120)
    plt.close()
    print(f"  ROC    -> {ROC_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("  LSTM-only Ablation - Stage-2 Non-IoT")
    print(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 60)

    # ---- Step 1: load + scaler + sequences (hybrid pipeline, redirected) ----
    df, feat_cols, le = nm.load_raw_data()
    n_feat = len(feat_cols)

    X_raw, scaler = nm.fit_and_save_scaler(df, feat_cols)
    X_tr, y_tr, X_val, y_val, X_te, y_te = nm.build_sequences(
        df, X_raw, scaler, feat_cols
    )
    tr_ldr, val_ldr, te_ldr, pw = nm.make_loaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te
    )

    # ---- Step 2: build LSTM-only model ----
    model = LstmOnlyDetector(n_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  LSTM-only parameters: {n_params:,}")

    # ---- Step 3: train using the hybrid's training loop ----
    model, tl, vl, va, vr = nm.train_model(model, tr_ldr, val_ldr, pw)

    # ---- Step 4: threshold + test-set evaluation ----
    threshold = nm.find_threshold(model, val_ldr)
    metrics, probs, labels = nm.evaluate(model, te_ldr, threshold, le)
    metrics["model_type"] = "lstm_only"
    metrics["branch"] = "noniot"
    metrics["n_parameters"] = int(n_params)
    metrics["n_features"] = int(n_feat)

    # ---- Step 5: plots + save ----
    plot_curves(tl, vl, va, vr)
    preds = (probs >= threshold).astype(int)
    plot_cm(labels, preds, le)
    plot_roc(labels, probs)

    torch.save({
        "model_state": model.state_dict(),
        "n_features": n_feat,
        "seq_len": SEQ_LEN,
        "threshold": float(threshold),
        "feature_cols": feat_cols,
        "label_encoder": le,
        "model_type": "lstm_only",
    }, MODEL_PATH)
    print(f"  Model -> {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics -> {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("  LSTM-only DONE")
    print(f"  AUC={metrics['auc_roc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  "
          f"Precision={metrics['precision']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()