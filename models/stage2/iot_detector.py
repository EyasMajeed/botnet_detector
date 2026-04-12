"""
════════════════════════════════════════════════════════════════════════
 Stage-2 IoT Botnet Detector  —  CNN-LSTM
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════

 INPUT  : data/processed/stage2_iot_botnet.csv
           (produced by preprocess_from_pcap_csvs.py → preprocess_iot23)

 OUTPUT : models/stage2/iot_cnn_lstm.pt     ← trained PyTorch model
          models/stage2/iot_metadata.json   ← seq_len, features, threshold
          models/stage2/results/iot_*       ← metrics, plots

 ARCHITECTURE:
   Flow records → sliding-window sequences (seq_len=10)
   → Conv1D (spatial patterns across features)
   → MaxPool1D
   → Conv1D (deeper patterns)
   → LSTM   (temporal / sequential patterns across flows)
   → Dropout
   → Linear → Sigmoid  (benign / botnet)

 INTERFACE (used by inference_bridge.py):
   detector = Stage2Detector.load("models/stage2/iot_cnn_lstm.pt")
   label, confidence = detector.predict(feature_df)

 SUCCESS CRITERION: ≥ 90% accuracy AND high recall (priority: recall)
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

ROOT        = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "data" / "processed" / "stage2_iot_botnet.csv"
MODEL_DIR   = ROOT / "models" / "stage2"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_PATH  = MODEL_DIR / "iot_cnn_lstm.pt"
META_PATH   = MODEL_DIR / "iot_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

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
    "flag_FIN","flag_SYN","flag_RST","flag_PSH","flag_ACK","flag_URG",
    "protocol","src_port","dst_port",
    "flow_active_time",  "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score",    "burst_rate",
    "window_flow_count",    "window_unique_dsts",
    "ttl_mean","ttl_std","ttl_min","ttl_max",
    "dns_query_count",
    "payload_bytes_mean","payload_bytes_std",
    "payload_zero_ratio","payload_entropy",
    "tls_features_available",
]

LABEL_COL   = "class_label"     # values: "benign" | "botnet"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ── Sequence / model hyperparameters ────────────────────────────────
SEQ_LEN      = 10      # flows per sequence fed into CNN-LSTM
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 1e-3
PATIENCE     = 5       # early stopping patience

# ── Decision threshold (lower → higher recall, fewer missed botnets) ─
THRESHOLD    = 0.40    # per project priority: recall over precision

DEVICE = torch.device(  "cuda" if torch.cuda.is_available() else 
                        "mps"  if torch.backends.mps.is_available() else
                        "cpu")

# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class FlowSequenceDataset(Dataset):
    """
    Converts flat flow records into overlapping sequences of length SEQ_LEN.
    Each sample = (SEQ_LEN × n_features) tensor → label of the LAST flow.

    Using the last flow's label preserves temporal ordering and ensures
    the model learns to predict based on the sequence history.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        # Valid start indices: need seq_len flows ahead
        self.indices = list(range(seq_len - 1, len(X)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end   = self.indices[idx] + 1
        start = end - self.seq_len
        seq   = self.X[start:end]          # (seq_len, n_features)
        label = self.y[self.indices[idx]]  # scalar
        return seq, label


# ═══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class CnnLstmDetector(nn.Module):
    """
    Hybrid CNN-LSTM for botnet detection.

    Input shape : (batch, seq_len, n_features)
    Pipeline    :
      Permute → Conv1D (spatial) → MaxPool → Conv1D (deeper)
      → Permute back → LSTM (temporal) → Dropout → Linear → Sigmoid
    """

    def __init__(self, n_features: int, seq_len: int):
        super().__init__()

        # ── CNN block ────────────────────────────────────────────────
        # Permute to (batch, n_features, seq_len) for Conv1d
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # ── LSTM block ───────────────────────────────────────────────
        # After CNN: shape is (batch, 128, seq_len') → permute to (batch, seq_len', 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False,
        )

        # ── Classifier head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)          # → (batch, n_features, seq_len)
        x = self.conv1(x)               # → (batch, 64, seq_len')
        x = self.conv2(x)               # → (batch, 128, seq_len')
        x = x.permute(0, 2, 1)          # → (batch, seq_len', 128)
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch, 64)
        x = h_n[-1]                     # last layer hidden state → (batch, 64)
        x = self.classifier(x)          # → (batch, 1)
        return x.squeeze(1)             # → (batch,)


# ═══════════════════════════════════════════════════════════════════════
# STAGE-2 DETECTOR WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class Stage2Detector:
    """
    Wrapper used by inference_bridge.py.
    Interface:
        detector = Stage2Detector.load("models/stage2/iot_cnn_lstm.pt")
        label, confidence = detector.predict(feature_df)
    """

    def __init__(self, model: CnnLstmDetector, seq_len: int,
                 n_features: int, threshold: float, label_encoder: LabelEncoder):
        self.model         = model.to(DEVICE)
        self.seq_len       = seq_len
        self.n_features    = n_features
        self.threshold     = threshold
        self.label_encoder = label_encoder
        self.model.eval()

    # ── Inference ────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Parameters
        ----------
        df : single-row (or multi-row) DataFrame with 56 features.
             If fewer than seq_len rows, pads with zeros at the front.

        Returns
        -------
        label      : str   — "benign" or "botnet"
        confidence : float — probability of botnet (0–1)
        """
        X = self._align(df)

        # Pad if needed
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), self.n_features),
                           dtype=np.float32)
            X = np.vstack([pad, X])

        # Use the last seq_len rows
        seq = torch.tensor(X[-self.seq_len:], dtype=torch.float32)
        seq = seq.unsqueeze(0).to(DEVICE)   # (1, seq_len, n_features)

        with torch.no_grad():
            prob = self.model(seq).item()

        label = "botnet" if prob >= self.threshold else "benign"
        return label, float(prob)

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, model_path: str | Path, meta_path: str | Path) -> None:
        model_path = Path(model_path)
        meta_path  = Path(meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state":   self.model.state_dict(),
            "n_features":    self.n_features,
            "seq_len":       self.seq_len,
            "threshold":     self.threshold,
            "label_encoder": self.label_encoder,
        }, model_path)

        meta = {
            "seq_len":    self.seq_len,
            "n_features": self.n_features,
            "threshold":  self.threshold,
            "model_type": "iot_cnn_lstm",
            "features":   ALL_FEATURES,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ Model  → {model_path}")
        print(f"  ✓ Meta   → {meta_path}")

    @classmethod
    def load(cls, model_path: str | Path) -> "Stage2Detector":
        ckpt = torch.load(model_path, map_location=DEVICE)
        n_features = ckpt["n_features"]
        seq_len    = ckpt["seq_len"]
        model      = CnnLstmDetector(n_features, seq_len)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, seq_len, n_features,
                   ckpt["threshold"], ckpt["label_encoder"])

    # ── Internal ─────────────────────────────────────────────────────
    @staticmethod
    def _align(df: pd.DataFrame) -> np.ndarray:
        for col in ALL_FEATURES:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0
        return df[ALL_FEATURES].values.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    print(f"\n  Loading data from:\n    {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n  [ERROR] File not found: {DATA_PATH}\n"
            "  Run preprocess_from_pcap_csvs.py first."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows  |  columns: {df.shape[1]}")

    df = df[df[LABEL_COL].isin(["benign", "botnet"])].copy()
    print(f"  After label filter: {len(df):,} rows")

    vc = df[LABEL_COL].value_counts()
    print("\n  Class distribution:")
    for cls, cnt in vc.items():
        pct = cnt / len(df) * 100
        print(f"    {cls:>8s}: {cnt:>8,}  ({pct:.1f}%)")

    # Fill any missing features
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    X = df[ALL_FEATURES].values.astype(np.float32)

    le = LabelEncoder()
    # Ensure botnet = 1, benign = 0
    le.fit(["benign", "botnet"])
    y = le.transform(df[LABEL_COL]).astype(np.float32)
    print("\n  Label encoding: benign=0, botnet=1")
    return X, y, le


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def make_loader(X, y, seq_len, shuffle=True, use_weighted_sampler=False):
    ds = FlowSequenceDataset(X, y, seq_len)

    if use_weighted_sampler and shuffle:
        # Oversample botnet sequences to combat class imbalance
        labels = [int(ds.y[i].item()) for i in ds.indices]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_model(model, train_loader, val_loader):
    # Compute pos_weight to handle any residual imbalance
    all_labels = torch.cat([y for _, y in train_loader])
    n_neg = (all_labels == 0).sum().item()
    n_pos = (all_labels == 1).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                               dtype=torch.float32).to(DEVICE)

    criterion = nn.BCELoss()          # Sigmoid already applied in model
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    best_val_recall = 0.0
    best_state      = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    val_accuracies, val_recalls = [], []

    print(f"\n  Training on {DEVICE}  |  epochs={EPOCHS}  |  "
          f"batch={BATCH_SIZE}  |  seq_len={SEQ_LEN}")
    print(f"  pos_weight (botnet): {pos_weight.item():.2f}")
    print(f"  Decision threshold : {THRESHOLD}")
    print()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        train_loss = running_loss / len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        all_probs, all_true = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                probs = model(X_batch)
                loss  = criterion(probs, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        val_loss  = val_loss_sum / len(val_loader.dataset)
        all_probs = np.array(all_probs)
        all_true  = np.array(all_true, dtype=int)
        preds_bin = (all_probs >= THRESHOLD).astype(int)

        val_acc    = accuracy_score(all_true, preds_bin)
        val_recall = recall_score(all_true, preds_bin, zero_division=0)
        val_f1     = f1_score(all_true, preds_bin, zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_recalls.append(val_recall)

        scheduler.step(val_recall)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={val_acc*100:.2f}%  recall={val_recall*100:.2f}%  "
              f"f1={val_f1*100:.2f}%")

        # ── Early stopping (monitored on recall) ──────────────────────
        if val_recall > best_val_recall:
            best_val_recall   = val_recall
            best_state        = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(best recall: {best_val_recall*100:.2f}%)")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, train_losses, val_losses, val_accuracies, val_recalls


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, le):
    model.eval()
    all_probs, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch.to(DEVICE))
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true, dtype=int)
    preds_bin = (all_probs >= THRESHOLD).astype(int)

    acc  = accuracy_score(all_true, preds_bin)
    prec = precision_score(all_true, preds_bin, zero_division=0)
    rec  = recall_score(all_true, preds_bin, zero_division=0)
    f1   = f1_score(all_true, preds_bin, zero_division=0)
    auc  = roc_auc_score(all_true, all_probs)

    print(f"\n{'─'*55}")
    print("  Stage-2 IoT CNN-LSTM — Test Results")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc*100:.2f}%  (target ≥ 90%)")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%  ← priority metric")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Threshold : {THRESHOLD}")
    print(f"\n{classification_report(all_true, preds_bin, target_names=le.classes_)}")

    if acc >= 0.90:
        print("  ✅  SUCCESS — ≥90% accuracy criterion MET")
    else:
        print(f"  ⚠   Accuracy {acc*100:.2f}% is below the 90% target")

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "auc_roc":   round(auc,  4),
        "threshold": THRESHOLD,
    }


def plot_training_curves(train_losses, val_losses, val_accuracies, val_recalls):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(train_losses, label="Train Loss", color="#2196F3")
    axes[0].plot(val_losses,   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("IoT CNN-LSTM — Training & Validation Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy + Recall
    axes[1].plot([a*100 for a in val_accuracies],
                 label="Val Accuracy", color="#4CAF50")
    axes[1].plot([r*100 for r in val_recalls],
                 label="Val Recall",   color="#FF9800", linestyle="--")
    axes[1].axhline(90, color="gray", linestyle=":", linewidth=0.8,
                    label="90% target")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%")
    axes[1].set_title("IoT CNN-LSTM — Accuracy & Recall")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "iot_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 Training curves → {out}")


def plot_confusion_matrix(model, test_loader, le):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch.to(DEVICE))
            preds = (probs.cpu().numpy() >= THRESHOLD).astype(int)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy().astype(int))

    cm   = confusion_matrix(all_true, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Reds")
    ax.set_title("IoT CNN-LSTM — Confusion Matrix")
    fig.tight_layout()
    out = RESULTS_DIR / "iot_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix → {out}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("╔" + "═"*53 + "╗")
    print("║  STAGE-2 IoT DETECTOR TRAINING — Group 07           ║")
    print("║  CNN-LSTM  |  Benign vs Botnet (IoT traffic)         ║")
    print("╚" + "═"*53 + "╝")

    # 1. Load data
    X, y, le = load_data()
    n_features = X.shape[1]

    # 2. Train / val / test split  (60 / 20 / 20)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_SEED, stratify=y_tv)

    print(f"\n  Split → Train: {len(X_train):,}  |  "
          f"Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # 3. DataLoaders
    train_loader = make_loader(X_train, y_train, SEQ_LEN,
                               shuffle=True, use_weighted_sampler=True)
    val_loader   = make_loader(X_val,   y_val,   SEQ_LEN, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  SEQ_LEN, shuffle=False)

    # 4. Build model
    model = CnnLstmDetector(n_features=n_features, seq_len=SEQ_LEN).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,}")

    # 5. Train
    model, t_loss, v_loss, v_acc, v_rec = train_model(
        model, train_loader, val_loader)

    # 6. Evaluate on test set
    metrics = evaluate(model, test_loader, le)

    # 7. Save plots
    plot_training_curves(t_loss, v_loss, v_acc, v_rec)
    plot_confusion_matrix(model, test_loader, le)

    # 8. Save model + metadata
    detector = Stage2Detector(model, SEQ_LEN, n_features, THRESHOLD, le)
    detector.save(MODEL_PATH, META_PATH)

    # 9. Save metrics JSON
    metrics_path = RESULTS_DIR / "iot_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄 Metrics report → {metrics_path}")

    print("\n╔" + "═"*53 + "╗")
    print("║  TRAINING COMPLETE                                   ║")
    print(f"║  Model   → {MODEL_PATH}            ║")
    print(f"║  Results → {RESULTS_DIR}              ║")
    print("╚" + "═"*53 + "╝\n")


if __name__ == "__main__":
    main()
