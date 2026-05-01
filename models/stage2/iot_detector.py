"""
════════════════════════════════════════════════════════════════════════
 Stage-2 IoT Botnet Detector  —  CNN-LSTM  (memory-efficient)
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT  : data/processed/stage2_iot_botnet.csv
 OUTPUT : models/stage2/iot_cnn_lstm.pt
          models/stage2/iot_metadata.json
          models/stage2/results/iot_*

 MEMORY DESIGN:
   The naive approach (build all ~1.9M sequences as a full array) needs
   ~12.8 GiB RAM and causes MemoryError on most machines.

   This version never materialises the full sequence array. Instead:
     1. Each device's flow rows are loaded as a compact numpy array
        (int16 indices + labels, not float32 feature copies)
     2. A custom Dataset stores only PER-DEVICE INDEX ARRAYS and
        produces sequences on-the-fly in __getitem__ by slicing
        the pre-loaded per-device feature matrix
     3. The per-device feature matrices are capped to MAX_ROWS_PER_DEVICE
        to bound peak RAM per device
     4. Train/val/test split is done at the FLOW level (stratified 60/20/20)
        BEFORE sequences are built — only the assigned rows are kept
        per split, so no split holds more than its share of the data

   Peak RAM usage: ~1-2 GB instead of 12+ GB.
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "data" / "processed" / "stage2_iot_combined.csv"
MODEL_DIR   = ROOT / "models" / "stage2"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_PATH  = MODEL_DIR / "iot_cnn_lstm.pt"
META_PATH   = MODEL_DIR / "iot_metadata.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
SEQ_LEN            = 20
BATCH_SIZE         = 512
EPOCHS             = 30
LR                 = 3e-4
PATIENCE           = 6
TARGET_RECALL      = 0.95
MIN_PRECISION_GATE = 0.70
RANDOM_SEED        = 42

# Per-device row cap: limits RAM per device's feature matrix.
# Each device row = 115 float32 = 460 bytes.
# 50_000 rows × 460 bytes × 9 devices ≈ 207 MB total for features.
MAX_ROWS_PER_DEVICE = 50_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT DATASET
# ════════════════════════════════════════════════════════════════════════

class PerDeviceSeqDataset(Dataset):
    """
    Stores per-device feature matrices and produces sequences on-the-fly.

    Memory layout:
      self.device_X : dict[device_name → float32 array (n_rows, n_feat)]
      self.index    : list of (device_name, start_row) tuples
                      one entry per valid sequence starting position

    __getitem__ slices device_X[device][start:start+SEQ_LEN] when called,
    so only ONE sequence (SEQ_LEN × n_feat × 4 bytes) is in memory at
    a time per worker — not the entire sequence pool.
    """

    def __init__(self, device_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
                 seq_len: int):
        """
        Parameters
        ----------
        device_arrays : {device_name: (X_float32, y_int64)}
                        rows already in temporal order (sorted by seq_index)
        seq_len       : sliding window length
        """
        self.seq_len   = seq_len
        self.device_X  : dict[str, np.ndarray] = {}
        self.device_y  : dict[str, np.ndarray] = {}
        self.index     : list[tuple[str, int]] = []   # (device, start_row)

        n_benign = 0; n_botnet = 0; n_devices = 0

        for device, (X, y) in device_arrays.items():
            if len(X) < seq_len:
                continue
            self.device_X[device] = X
            self.device_y[device] = y
            n_devices += 1
            for start in range(len(X) - seq_len + 1):
                self.index.append((device, start))
                label = y[start + seq_len - 1]
                if label == 0: n_benign  += 1
                else:          n_botnet  += 1

        if not self.index:
            raise ValueError(f"No sequences. SEQ_LEN={seq_len} too large?")

        self._n_benign = n_benign
        self._n_botnet = n_botnet
        print(f"  Dataset: {len(self.index):,} sequences from {n_devices} devices  "
              f"(benign={n_benign:,}  botnet={n_botnet:,})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        device, start = self.index[idx]
        X_seq = self.device_X[device][start : start + self.seq_len]   # (seq_len, feat)
        label = float(self.device_y[device][start + self.seq_len - 1])
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(label)


# ════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════

class CnnLstmDetector(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(1)


# ════════════════════════════════════════════════════════════════════════
# STAGE-2 DETECTOR WRAPPER
# ════════════════════════════════════════════════════════════════════════

class Stage2Detector:
    def __init__(self, model, seq_len, n_features,
                 threshold, feature_cols, label_encoder):
        self.model        = model.to(DEVICE); self.model.eval()
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.threshold    = threshold
        self.feature_cols = feature_cols
        self.label_encoder = label_encoder

    def predict(self, df: pd.DataFrame):
        X = self._align(df)
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), self.n_features), dtype=np.float32)
            X = np.vstack([pad, X])
        seq = torch.tensor(X[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(seq)).item()
        return ("botnet" if prob >= self.threshold else "benign"), float(prob)

    def predict_sequence(self, seq_array: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a pre-computed (seq_len, n_features) Kitsune sequence.

        Used by the live monitoring pipeline: live_capture.py maintains a
        per-src_ip rolling buffer of scaled Kitsune vectors and calls this
        method once the buffer has seq_len packets.

        Bypasses _align() because Kitsune features have their own column order
        (FEATURE_NAMES in src/live/kitsune_extractor.py), already matched at
        training time. Use predict(df) instead for DataFrames with named columns.
        """
        if seq_array.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {seq_array.shape}")
        if seq_array.shape[0] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len} rows, got {seq_array.shape[0]}"
            )
        if seq_array.shape[1] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features} cols, got {seq_array.shape[1]}"
            )

        seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(seq)).item()
        return ("botnet" if prob >= self.threshold else "benign"), float(prob)

    def save(self, model_path, meta_path):
        torch.save({"model_state": self.model.state_dict(),
                    "n_features": self.n_features, "seq_len": self.seq_len,
                    "threshold": self.threshold, "feature_cols": self.feature_cols,
                    "label_encoder": self.label_encoder,
                    "model_type": "iot_cnn_lstm"}, Path(model_path))
        with open(meta_path, "w") as f:
            json.dump({"n_features": self.n_features, "seq_len": self.seq_len,
                       "threshold": self.threshold, "model_type": "iot_cnn_lstm",
                       "features": self.feature_cols}, f, indent=2)
        print(f"  Model saved -> {model_path}")
        print(f"  Meta  saved -> {meta_path}")

    @classmethod
    def load(cls, model_path):
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        m    = CnnLstmDetector(ckpt["n_features"])
        m.load_state_dict(ckpt["model_state"])
        return cls(m, ckpt["seq_len"], ckpt["n_features"], ckpt["threshold"],
                   ckpt["feature_cols"], ckpt["label_encoder"])

    def _align(self, df):
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns: df[col] = 0.0
        return df[self.feature_cols].values.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING + SPLIT + DATASET CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════

def load_and_build(split: str,
                   feature_cols: list[str],
                   le: LabelEncoder,
                   flow_indices: np.ndarray,
                   df_full: pd.DataFrame) -> PerDeviceSeqDataset:
    """
    Given a subset of row indices (flow_indices into df_full),
    build a PerDeviceSeqDataset for that split.
    Caps each device to MAX_ROWS_PER_DEVICE rows (stratified by label).
    """
    df_split = df_full.iloc[flow_indices].copy()
    device_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for device, grp in df_split.groupby("src_ip"):
        grp = grp.sort_values("seq_index")

        # Stratified cap to keep RAM bounded
        if len(grp) > MAX_ROWS_PER_DEVICE:
            b   = grp[grp["class_label"] == "benign"]
            bot = grp[grp["class_label"] == "botnet"]
            ratio = len(b) / len(grp)
            n_b   = max(SEQ_LEN, int(MAX_ROWS_PER_DEVICE * ratio))
            n_bot = max(SEQ_LEN, MAX_ROWS_PER_DEVICE - n_b)
            n_b   = min(n_b,   len(b))
            n_bot = min(n_bot, len(bot))
            grp = pd.concat([
                b.head(n_b),        # keep temporal order (head = earliest)
                bot.head(n_bot)
            ]).sort_values("seq_index")

        X = grp[feature_cols].values.astype(np.float32)
        y = le.transform(grp["class_label"]).astype(np.int64)
        device_arrays[device] = (X, y)

    print(f"\n  [{split}] building sequences...")
    return PerDeviceSeqDataset(device_arrays, SEQ_LEN)


def load_data():
    print(f"\n  Loading: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Not found: {DATA_PATH}\n"
            "Run src/ingestion/preprocess_nbaiot.py first.")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(df):,} rows | {df.shape[1]} cols")

    META = {"class_label", "attack_type", "device_name", "src_ip", "seq_index", "source"}
    feature_cols = [c for c in df.columns if c not in META]
    print(f"  Feature columns: {len(feature_cols)}")

    vc = df["class_label"].value_counts()
    print(f"\n  Class distribution:")
    for cls, cnt in vc.items():
        print(f"    {cls:>8s}: {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")

    le = LabelEncoder(); le.fit(["benign", "botnet"])

    # Stratified 60/20/20 split at FLOW level (on indices, no data copied yet)
    y_all = le.transform(df["class_label"]).astype(np.int64)
    idx   = np.arange(len(df))

    idx_tv, idx_te = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_SEED, stratify=y_all)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=0.25, random_state=RANDOM_SEED,
        stratify=y_all[idx_tv])

    print(f"\n  Flow-level stratified split:")
    for name, ii in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        b = (y_all[ii] == 0).sum(); bot = (y_all[ii] == 1).sum()
        print(f"    {name:5s}: {len(ii):>9,} flows  "
              f"benign={b:,} ({b/len(ii)*100:.1f}%)  "
              f"botnet={bot:,} ({bot/len(ii)*100:.1f}%)")

    return df, feature_cols, le, idx_tr, idx_va, idx_te


# ════════════════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ════════════════════════════════════════════════════════════════════════

def make_loader(ds: PerDeviceSeqDataset,
                shuffle: bool = True,
                max_sequences: int | None = None) -> DataLoader:
    """
    Wrap dataset in DataLoader.
    When max_sequences is set, use a random SubsetSampler to cap epoch size.
    """
    if max_sequences is not None and len(ds) > max_sequences:
        # Stratified subsample: match class ratio
        labels = np.array([ds.device_y[d][s + SEQ_LEN - 1]
                           for d, s in ds.index])
        idx0 = np.where(labels == 0)[0]; idx1 = np.where(labels == 1)[0]
        ratio = len(idx0) / len(labels)
        n0 = min(int(max_sequences * ratio), len(idx0))
        n1 = min(max_sequences - n0, len(idx1))
        np.random.seed(RANDOM_SEED)
        chosen = np.concatenate([np.random.choice(idx0, n0, replace=False),
                                  np.random.choice(idx1, n1, replace=False)])
        np.random.shuffle(chosen)
        sampler = torch.utils.data.SubsetRandomSampler(chosen.tolist())
        b = (labels[chosen] == 0).sum(); bot = (labels[chosen] == 1).sum()
        print(f"    Capped to {len(chosen):,}  (benign={b:,}  botnet={bot:,})")
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, train_ds: PerDeviceSeqDataset):
    n_neg = train_ds._n_benign
    n_pos = train_ds._n_botnet
    raw_pw = n_neg / max(n_pos, 1)
    pw_val = float(np.clip(raw_pw, 0.1, 10.0))
    pos_weight = torch.tensor([pw_val], dtype=torch.float32).to(DEVICE)

    print(f"\n  Train seqs: benign={n_neg:,}  botnet={n_pos:,}  "
          f"pos_weight={pw_val:.3f}")
    print(f"  Device:{DEVICE} | epochs:{EPOCHS} | batch:{BATCH_SIZE} | "
          f"seq_len:{SEQ_LEN} | lr:{LR}")
    print(f"  BCEWithLogitsLoss | gate={MIN_PRECISION_GATE}\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_f1, best_thresh, best_state = 0.0, 0.5, None
    no_improve = 0
    tl_h, vl_h, va_h, vr_h = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); run_loss = 0.0; n_seen = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item() * len(yb); n_seen += len(yb)
        t_loss = run_loss / max(n_seen, 1)

        model.eval()
        vl_sum, probs_all, true_all = 0.0, [], []; n_val = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                vl_sum += criterion(logits, yb).item() * len(yb)
                probs_all.extend(torch.sigmoid(logits).cpu().tolist())
                true_all.extend(yb.cpu().tolist())
                n_val += len(yb)

        v_loss    = vl_sum / max(n_val, 1)
        probs_arr = np.array(probs_all)
        true_arr  = np.array(true_all, dtype=int)
        v_auc     = roc_auc_score(true_arr, probs_arr)

        # Find threshold that maximises F1 on val — scanning a grid
        # avoids the near-zero collapse from ROC-based recall targeting.
        best_t, best_t_f1 = 0.5, 0.0
        for t_cand in np.linspace(0.05, 0.95, 91):
            p_cand = (probs_arr >= t_cand).astype(int)
            f_cand = f1_score(true_arr, p_cand, zero_division=0)
            if f_cand > best_t_f1:
                best_t_f1 = f_cand
                best_t    = float(t_cand)
        val_thresh = best_t

        preds   = (probs_arr >= val_thresh).astype(int)
        v_acc   = accuracy_score(true_arr, preds)
        v_prec  = precision_score(true_arr, preds, zero_division=0)
        v_rec   = recall_score(true_arr, preds, zero_division=0)
        v_f1    = f1_score(true_arr, preds, zero_division=0)
        ben_rec = recall_score(true_arr, preds, pos_label=0, zero_division=0)

        tl_h.append(t_loss); vl_h.append(v_loss)
        va_h.append(v_acc);  vr_h.append(v_rec)
        scheduler.step(v_rec)

        gate_ok = v_prec >= MIN_PRECISION_GATE
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train={t_loss:.4f}  val={v_loss:.4f}  auc={v_auc:.4f}  "
              f"acc={v_acc*100:.1f}%  prec={v_prec*100:.1f}%  "
              f"rec(bot)={v_rec*100:.1f}%  rec(ben)={ben_rec*100:.1f}%  "
              f"f1={v_f1*100:.1f}%  t={val_thresh:.3f}"
              + ("" if gate_ok else "  [gate FAIL]"))

        if gate_ok and v_f1 > best_f1:
            best_f1     = v_f1
            best_thresh = val_thresh
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stop epoch {epoch} | "
                      f"best F1 {best_f1*100:.2f}% @ t={best_thresh:.3f}")
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_thresh, tl_h, vl_h, va_h, vr_h


# ════════════════════════════════════════════════════════════════════════
# EVALUATION & PLOTS
# ════════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, le, threshold: float):
    model.eval(); probs_all, true_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs_all.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().tolist())
            true_all.extend(yb.cpu().tolist())

    probs = np.array(probs_all); true = np.array(true_all, dtype=int)
    preds = (probs >= threshold).astype(int)
    acc     = accuracy_score(true, preds)
    prec    = precision_score(true, preds, zero_division=0)
    rec     = recall_score(true, preds, zero_division=0)
    f1      = f1_score(true, preds, zero_division=0)
    auc     = roc_auc_score(true, probs)
    ben_rec = recall_score(true, preds, pos_label=0, zero_division=0)

    print(f"\n{'─'*62}")
    print(f"  Stage-2 IoT CNN-LSTM — Test Results  (t={threshold:.4f})")
    print(f"{'─'*62}")
    print(f"  Accuracy      : {acc*100:.2f}%  (target >= 90%)")
    print(f"  AUC-ROC       : {auc:.4f}")
    print(f"  Precision     : {prec*100:.2f}%")
    print(f"  Recall(botnet): {rec*100:.2f}%  <- priority")
    print(f"  Recall(benign): {ben_rec*100:.2f}%")
    print(f"  F1-Score      : {f1*100:.2f}%")
    print(f"\n{classification_report(true, preds, target_names=le.classes_)}")
    print("  SUCCESS -- >=90% accuracy" if acc >= 0.90
          else f"  WARNING: {acc*100:.2f}% below 90% target")

    return dict(accuracy=round(acc,4), precision=round(prec,4),
                recall_botnet=round(rec,4), recall_benign=round(ben_rec,4),
                f1_score=round(f1,4), auc_roc=round(auc,4),
                threshold=round(threshold,4))


def plot_curves(tl, vl, va, vr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tl, label="Train", color="#2196F3")
    axes[0].plot(vl, label="Val",   color="#F44336")
    axes[0].set_title("IoT CNN-LSTM — Loss"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot([a*100 for a in va], label="Accuracy", color="#4CAF50")
    axes[1].plot([r*100 for r in vr], label="Recall",   color="#FF9800", ls="--")
    axes[1].axhline(90, color="gray", ls=":", lw=.8, label="90% target")
    axes[1].set_title("IoT CNN-LSTM — Accuracy & Recall (val)")
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout(); out = RESULTS_DIR / "iot_training_curves.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(f"  Curves -> {out}")


def plot_cm(model, test_loader, le, threshold: float):
    model.eval(); preds_all, true_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            p = (torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
                 >= threshold).astype(int)
            preds_all.extend(p.tolist()); true_all.extend(yb.tolist())
    cm = confusion_matrix(true_all, preds_all)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"IoT CNN-LSTM — Confusion Matrix (t={threshold:.3f})")
    fig.tight_layout(); out = RESULTS_DIR / "iot_confusion_matrix.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(f"  CM    -> {out}")


def plot_roc(model, test_loader):
    model.eval(); probs_all, true_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs_all.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().tolist())
            true_all.extend(yb.tolist())
    probs = np.array(probs_all); true = np.array(true_all, dtype=int)
    fpr, tpr, _ = roc_curve(true, probs)
    auc = roc_auc_score(true, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=1.5, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"k--",lw=.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("IoT CNN-LSTM — ROC Curve")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); out = RESULTS_DIR / "iot_roc_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(f"  ROC   -> {out}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    print("=" * 60)
    print("  STAGE-2 IoT CNN-LSTM TRAINING -- Group 07")
    print("  N-BaIoT | Memory-efficient | Stratified split")
    print("=" * 60)

    # 1. Load CSV and get split indices (no large arrays yet)
    df, feature_cols, le, idx_tr, idx_va, idx_te = load_data()
    n_features = len(feature_cols)

    # 2. Build per-split datasets (on-the-fly sequence generation)
    train_ds = load_and_build("train", feature_cols, le, idx_tr, df)
    val_ds   = load_and_build("val",   feature_cols, le, idx_va, df)
    test_ds  = load_and_build("test",  feature_cols, le, idx_te, df)

    # Free the full DataFrame — datasets hold only their device slices
    del df
    import gc; gc.collect()

    # 3. DataLoaders  (train/val capped; test full)
    print("\n  Creating dataloaders...")
    train_loader = make_loader(train_ds, shuffle=True,  max_sequences=200_000)
    val_loader   = make_loader(val_ds,   shuffle=False, max_sequences=50_000)
    test_loader  = make_loader(test_ds,  shuffle=False)

    # 4. Model
    model = CnnLstmDetector(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters: {n_params:,}")

    # 5. Train
    model, best_thresh, tl, vl, va, vr = train_model(
        model, train_loader, val_loader, train_ds)
    print(f"\n  Best val threshold: {best_thresh:.4f}")

    # 6. Evaluate & plots
    metrics = evaluate(model, test_loader, le, threshold=best_thresh)
    plot_curves(tl, vl, va, vr)
    plot_cm(model, test_loader, le, threshold=best_thresh)
    plot_roc(model, test_loader)

    # 7. Save
    detector = Stage2Detector(model, SEQ_LEN, n_features,
                               best_thresh, feature_cols, le)
    detector.save(MODEL_PATH, META_PATH)
    out = RESULTS_DIR / "iot_metrics.json"
    with open(out, "w") as f: json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {out}")

    print("\n" + "=" * 60)
    print("  DONE  |  models/stage2/iot_cnn_lstm.pt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
