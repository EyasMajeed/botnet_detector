"""
================================================================================
 LSTM-only Ablation - Stage-2 IoT Detector
 Group 07 | CPCS499 | AI-Based Botnet Detection
================================================================================

 PURPOSE
 -------
 Train and evaluate an LSTM-only variant of the Stage-2 IoT detector
 (no CNN block) using the EXACT same data, splits, and hyperparameters
 as the hybrid CNN-LSTM in models/stage2/iot_detector.py.

 ARCHITECTURE  (LSTM directly on raw feature sequences)
 ------------------------------------------------------
   Input  (B, seq_len=20, n_features)
     -> LSTM(input=n_features, hidden=128, layers=2, dropout=0.3)
     -> h_n[-1]                                       # (B, 128)
     -> Dropout(0.4) + Linear(128, 64) + ReLU + Linear(64, 1)

 INPUT
 -----
   data/processed/stage2_iot_combined.csv

 OUTPUT (under evaluation/lstm_test/results/iot/)
 ------------------------------------------------
   metrics.json          Accuracy, precision, recall, F1, AUC, threshold
   training_curves.png   Loss + val accuracy + val recall over epochs
   confusion_matrix.png  Test-set confusion matrix
   roc_curve.png         Test-set ROC curve
   model.pt              Trained checkpoint

 USAGE
 -----
   Windows : python evaluation\\lstm_test\\lstm_only_iot.py
   macOS   : python3 evaluation/lstm_test/lstm_only_iot.py

 DEPENDENCIES
 ------------
   Windows : pip  install pandas numpy scikit-learn torch matplotlib
   macOS   : pip3 install pandas numpy scikit-learn torch matplotlib

 COMMON ERRORS
 -------------
   - "FileNotFoundError: stage2_iot_combined.csv"
       -> Run src/ingestion/preprocess_nbaiot.py and combine_datasets.py first.
   - Slow training on CPU
       -> LSTM-only on 115 features is heavier than the hybrid's
          (CNN-compressed-to-256). Use GPU if available.
================================================================================
"""
from __future__ import annotations

import gc
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
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.stage2 import iot_detector as im  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results" / "iot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = OUT_DIR / "model.pt"
METRICS_PATH = OUT_DIR / "metrics.json"
CURVE_PATH   = OUT_DIR / "training_curves.png"
CM_PATH      = OUT_DIR / "confusion_matrix.png"
ROC_PATH     = OUT_DIR / "roc_curve.png"

DEVICE      = im.DEVICE
SEQ_LEN     = im.SEQ_LEN
RANDOM_SEED = im.RANDOM_SEED


# ---------------------------------------------------------------------------
# LSTM-ONLY model for IoT branch
# ---------------------------------------------------------------------------
class LstmOnlyDetectorIoT(nn.Module):
    """LSTM only - no convolution. Sees raw flow features over time directly."""

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features) - already in batch_first layout
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_curves(tl, vl, va, vr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tl, label="Train", color="#2196F3")
    axes[0].plot(vl, label="Val", color="#F44336")
    axes[0].set_title("LSTM-only IoT - Loss")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].plot([a * 100 for a in va], label="Accuracy", color="#4CAF50")
    axes[1].plot([r * 100 for r in vr], label="Recall", color="#FF9800", ls="--")
    axes[1].axhline(90, color="gray", ls=":", lw=.8, label="90% target")
    axes[1].set_title("LSTM-only IoT - Accuracy & Recall (val)")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"  Curves -> {CURVE_PATH}")


def plot_cm(labels, preds, le, threshold):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"LSTM-only IoT - Confusion Matrix (t={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    print(f"  CM     -> {CM_PATH}")


def plot_roc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=1.5, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LSTM-only IoT - ROC Curve")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=150)
    plt.close(fig)
    print(f"  ROC    -> {ROC_PATH}")


def evaluate_collect(model, test_loader, le, threshold):
    model.eval()
    probs_all, true_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs_all.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().tolist())
            true_all.extend(yb.cpu().tolist())

    probs = np.array(probs_all)
    true = np.array(true_all, dtype=int)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy":      float(accuracy_score(true, preds)),
        "precision":     float(precision_score(true, preds, zero_division=0)),
        "recall":        float(recall_score(true, preds, zero_division=0)),
        "recall_benign": float(recall_score(true, preds, pos_label=0, zero_division=0)),
        "f1":            float(f1_score(true, preds, zero_division=0)),
        "auc_roc":       float(roc_auc_score(true, probs)),
        "threshold":     float(threshold),
    }

    print(f"\n{'-' * 62}")
    print(f"  LSTM-only IoT - Test Results  (t={threshold:.4f})")
    print(f"{'-' * 62}")
    print(f"  Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  Precision     : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall(botnet): {metrics['recall'] * 100:.2f}%  <- priority")
    print(f"  Recall(benign): {metrics['recall_benign'] * 100:.2f}%")
    print(f"  F1-Score      : {metrics['f1'] * 100:.2f}%")
    print(f"\n{classification_report(true, preds, target_names=le.classes_)}")

    return metrics, probs, true


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  LSTM-only Ablation - Stage-2 IoT")
    print(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 60)

    df, feature_cols, le, idx_tr, idx_va, idx_te = im.load_data()
    n_features = len(feature_cols)

    train_ds = im.load_and_build("train", feature_cols, le, idx_tr, df)
    val_ds   = im.load_and_build("val",   feature_cols, le, idx_va, df)
    test_ds  = im.load_and_build("test",  feature_cols, le, idx_te, df)
    del df
    gc.collect()

    print("\n  Creating dataloaders...")
    train_loader = im.make_loader(train_ds, shuffle=True,  max_sequences=200_000)
    val_loader   = im.make_loader(val_ds,   shuffle=False, max_sequences=50_000)
    test_loader  = im.make_loader(test_ds,  shuffle=False)

    model = LstmOnlyDetectorIoT(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  LSTM-only parameters: {n_params:,}")

    model, best_thresh, tl, vl, va, vr = im.train_model(
        model, train_loader, val_loader, train_ds)
    print(f"\n  Best val threshold: {best_thresh:.4f}")

    plot_curves(tl, vl, va, vr)
    metrics, probs, labels = evaluate_collect(
        model, test_loader, le, threshold=best_thresh)
    metrics["model_type"]   = "lstm_only"
    metrics["branch"]       = "iot"
    metrics["n_parameters"] = int(n_params)
    metrics["n_features"]   = int(n_features)
    preds = (probs >= best_thresh).astype(int)
    plot_cm(labels, preds, le, best_thresh)
    plot_roc(labels, probs)

    torch.save({
        "model_state": model.state_dict(),
        "n_features": n_features,
        "seq_len": SEQ_LEN,
        "threshold": float(best_thresh),
        "feature_cols": feature_cols,
        "label_encoder": le,
        "model_type": "lstm_only",
    }, MODEL_PATH)
    print(f"  Model -> {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics -> {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("  LSTM-only IoT DONE")
    print(f"  AUC={metrics['auc_roc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  "
          f"Precision={metrics['precision']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()