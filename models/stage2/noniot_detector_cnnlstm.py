"""
════════════════════════════════════════════════════════════════════════
 Stage-2 Non-IoT CNN-LSTM  —  v3  (Internal-device Filter)
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT : data/processed/stage2_noniot_botnet.csv
 OUTPUT: models/stage2/noniot_cnn_lstm.pt
         models/stage2/noniot_metadata.json
         models/stage2/results/noniot_*

 ROOT CAUSE OF PREVIOUS FAILURES:
   CIC-IDS-2017 Friday PCAP contains flows from ~500+ unique source IPs.
   The vast majority are EXTERNAL IPs with only 1-5 flows each — they
   can never form a 20-flow sequence. With 72 "devices" but 501K flows,
   the sequence builder only produced 7,274 sequences because most
   source IPs had fewer than SEQ_LEN=20 flows.

 FIX — Internal-device filter:
   Before building sequences, keep ONLY flows from the internal lab
   network (src_ip starts with '192.168.'). These are the actual
   victim machines:
     - 192.168.10.x : victim machines (infected with Ares botnet 13-14h)
     - 192.168.x.x  : other internal machines (benign all day)
   Each of these has thousands of flows → hundreds of sequences.
   External IPs are noise for temporal sequence modeling.

 RESULT:
   ~20 internal devices × ~5000 sequences each = ~100K sequences
   Botnet fraction stays ~6.6% → ~6,600 botnet sequences for training
   Every infected machine shows the benign→botnet→benign transition.
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "data" / "processed" / "stage2_noniot_botnet.csv"
MODEL_DIR   = ROOT / "models" / "stage2"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_PATH  = MODEL_DIR / "noniot_cnn_lstm.pt"
META_PATH   = MODEL_DIR / "noniot_metadata.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
SEQ_LEN            = 20
STRIDE             = 1
MAX_SEQ_PER_DEVICE = 5_000   # cap per device (prevents one server dominating)
INTERNAL_PREFIX    = ("192.168.", "172.31.", "147.32.")  # CIC-2017, CIC-2018 (AWS), CTU-13
BATCH_SIZE         = 512
EPOCHS             = 80
LR                 = 1e-4
PATIENCE           = 10
TARGET_RECALL      = 0.85
MIN_PRECISION_GATE = 0.12    # realistic floor for 6.6% botnet base rate
RANDOM_SEED        = 42

TEMPERATURE        = 2.0    # scales logits to spread probability outputs
DROP_COLS = {"class_label", "device_type", "src_ip", "timestamp", "ttl_mean"}
LABEL_COL = "class_label"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(1) / TEMPERATURE


# ══════════════════════════════════════════════════════════════════════
# SEQUENCE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def build_sequences(df: pd.DataFrame, feature_cols: list,
                    le: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    n_mixed = 0
    n_devices = 0

    for src_ip, grp in df.groupby("src_ip"):
        grp   = grp.sort_values("timestamp")
        X     = grp[feature_cols].values.astype(np.float32)
        y_raw = le.transform(grp[LABEL_COL]).astype(np.float32)
        n     = len(X)
        if n < SEQ_LEN:
            continue

        n_devices += 1
        starts = list(range(0, n - SEQ_LEN + 1, STRIDE))
        if len(starts) > MAX_SEQ_PER_DEVICE:
            chosen = np.round(
                np.linspace(0, len(starts)-1, MAX_SEQ_PER_DEVICE)
            ).astype(int)
            starts = [starts[i] for i in chosen]

        dev_labels = set()
        for s in starts:
            lbl = y_raw[s + SEQ_LEN - 1]
            all_X.append(X[s : s + SEQ_LEN])
            all_y.append(lbl)
            dev_labels.add(int(lbl))

        if len(dev_labels) > 1:
            n_mixed += 1

    if not all_X:
        raise RuntimeError("No sequences built — no internal device had ≥ SEQ_LEN flows")

    X_arr = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    n_bot = int(y_arr.sum())
    n_ben = int((y_arr == 0).sum())
    print(f"  Built {len(X_arr):,} sequences from {n_devices} internal devices")
    print(f"  Botnet={n_bot:,}  Benign={n_ben:,}  "
          f"Botnet%={n_bot/len(y_arr)*100:.1f}%")
    print(f"  Devices with label transitions: {n_mixed}  "
          f"← provides temporal benign→botnet signal")
    return X_arr, y_arr


# ══════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════

def load_and_build(feature_cols: list, le: LabelEncoder):
    print(f"  Loading {DATA_PATH.name} …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(df):,} rows | {df.shape[1]} cols")

    df = df[df[LABEL_COL].isin(["benign", "botnet"])].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)

    vc = df[LABEL_COL].value_counts()
    print(f"\n  Full dataset: {vc.to_dict()}")

    # ── KEY FIX: filter to internal lab devices only ──────────────────
    before = len(df)
    df = df[df["src_ip"].str.startswith(INTERNAL_PREFIX)].copy()
    print(f"\n  After internal-only filter (src_ip in known internal subnets):")
    print(f"    {len(df):,} flows  (removed {before-len(df):,} external-IP flows)")
    print(f"    Unique internal devices: {df['src_ip'].nunique()}")

    vc2 = df[LABEL_COL].value_counts()
    for cls, cnt in vc2.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

    if df.empty:
        raise RuntimeError(
            f"No flows matched internal IP prefixes {INTERNAL_PREFIX}. "
            "Check that step1 and step2 preserved src_ip correctly.")

    print(f"\n  Building sequences (SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE}, "
          f"cap={MAX_SEQ_PER_DEVICE:,}/device) …")
    X, y = build_sequences(df, feature_cols, le)

    # ── Stratified sequence-level split: 60 / 20 / 20 ──────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED)

    print(f"\n  Stratified 60/20/20 split:")
    print(f"    train: {len(y_tr):>7,}  "
          f"(bot={int(y_tr.sum()):,}  ben={int((y_tr==0).sum()):,})")
    print(f"    val  : {len(y_val):>7,}  "
          f"(bot={int(y_val.sum()):,}  ben={int((y_val==0).sum()):,})")
    print(f"    test : {len(y_te):>7,}  "
          f"(bot={int(y_te.sum()):,}  ben={int((y_te==0).sum()):,})")
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te):
    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    pw    = float(np.clip(n_neg / max(n_pos, 1), 1.0, 50.0))
    print(f"\n  pos_weight = {pw:.3f}  (n_neg={n_neg:,}  n_pos={n_pos:,})")

    w       = np.where(y_tr == 1, 1./max(n_pos,1), 1./max(n_neg,1))
    sampler = WeightedRandomSampler(w, num_samples=len(y_tr), replacement=True)

    pin = (DEVICE.type == "cuda")
    tr_ldr  = DataLoader(SeqDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE,
                         sampler=sampler, num_workers=0, pin_memory=pin)
    val_ldr = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE*2,
                         shuffle=False, num_workers=0)
    te_ldr  = DataLoader(SeqDataset(X_te,  y_te),  batch_size=BATCH_SIZE*2,
                         shuffle=False, num_workers=0)
    return tr_ldr, val_ldr, te_ldr, pw


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_model(model, tr_ldr, val_ldr, pw):
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_f1, best_state, wait = 0.0, None, 0
    tl_h, vl_h, va_h, vr_h = [], [], [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in tr_ldr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(yb)

        model.eval()
        probs, labs, vl = [], [], 0.0
        with torch.no_grad():
            for Xb, yb in val_ldr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                lg = model(Xb)
                vl += crit(lg, yb).item() * len(yb)
                probs.extend(torch.sigmoid(lg).cpu().numpy())
                labs.extend(yb.cpu().numpy())

        probs = np.array(probs); labs = np.array(labs)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labs, preds)
        rec = recall_score(labs, preds, zero_division=0)
        pre = precision_score(labs, preds, zero_division=0)
        f1  = f1_score(labs, preds, zero_division=0)
        sch.step()

        tl_h.append(ep_loss / max(len(tr_ldr.dataset), 1))
        vl_h.append(vl / max(len(val_ldr.dataset), 1))
        va_h.append(acc); vr_h.append(rec)

        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:03d}  tr={tl_h[-1]:.4f}  vl={vl_h[-1]:.4f}"
                  f"  acc={acc:.4f}  rec={rec:.4f}  pre={pre:.4f}  f1={f1:.4f}")

        if f1 > best_f1 and pre >= MIN_PRECISION_GATE:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best val F1 = {best_f1:.4f}")
    return model, tl_h, vl_h, va_h, vr_h


def find_threshold(model, val_ldr):
    """
    Priority for botnet detection:
      1. Maximize recall (missing a botnet is worse than a false alarm)
      2. Subject to precision >= MIN_PRECISION_GATE (above base rate)
      3. Among solutions satisfying both, pick highest F1

    Scan from low→high threshold so we find the highest recall first,
    then tighten until precision gate is met.
    """
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for Xb, yb in val_ldr:
            probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy())
            labs.extend(yb.numpy())
    probs = np.array(probs); labs = np.array(labs)

    # Collect all (threshold, recall, precision, f1) tuples
    candidates = []
    for t in np.arange(0.01, 0.99, 0.01):
        p   = (probs >= t).astype(int)
        rec = recall_score(labs, p, zero_division=0)
        pre = precision_score(labs, p, zero_division=0)
        f1  = f1_score(labs, p, zero_division=0)
        candidates.append((t, rec, pre, f1))

    # Strategy 1: recall >= TARGET_RECALL and precision >= gate → best F1
    s1 = [(t, r, p, f) for t, r, p, f in candidates
          if r >= TARGET_RECALL and p >= MIN_PRECISION_GATE]
    if s1:
        best = max(s1, key=lambda x: x[3])
        print(f"  Threshold = {best[0]:.2f}  "
              f"(recall={best[1]:.3f}  prec={best[2]:.3f}  F1={best[3]:.4f})")
        return float(best[0])

    # Strategy 2: relax recall target, keep precision gate → highest recall
    s2 = [(t, r, p, f) for t, r, p, f in candidates if p >= MIN_PRECISION_GATE]
    if s2:
        best = max(s2, key=lambda x: (x[1], x[3]))   # sort by recall then F1
        print(f"  [warn] recall target not fully met — best achievable recall")
        print(f"  Threshold = {best[0]:.2f}  "
              f"(recall={best[1]:.3f}  prec={best[2]:.3f}  F1={best[3]:.4f})")
        return float(best[0])

    # Strategy 3: drop precision gate entirely → highest F1
    best = max(candidates, key=lambda x: x[3])
    print(f"  [warn] precision gate not met at any threshold")
    print(f"  Threshold = {best[0]:.2f}  "
          f"(recall={best[1]:.3f}  prec={best[2]:.3f}  F1={best[3]:.4f})")
    return float(best[0])


def evaluate(model, te_ldr, threshold, le):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for Xb, yb in te_ldr:
            probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy())
            labs.extend(yb.numpy())
    probs = np.array(probs); labs = np.array(labs)
    preds = (probs >= threshold).astype(int)

    print("\n" + "="*55 + "\n  TEST RESULTS\n" + "="*55)
    print(classification_report(labs, preds, target_names=le.classes_))
    return {
        "accuracy":  float(accuracy_score(labs, preds)),
        "precision": float(precision_score(labs, preds, zero_division=0)),
        "recall":    float(recall_score(labs, preds, zero_division=0)),
        "f1":        float(f1_score(labs, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labs, probs)),
        "threshold": threshold,
    }, probs, labs


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_curves(tl, vl, va, vr):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(tl, label="Train"); ax[0].plot(vl, label="Val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(va); ax[1].set_title("Val Accuracy")
    ax[2].plot(vr); ax[2].set_title("Val Recall")
    for a in ax: a.set_xlabel("Epoch")
    plt.tight_layout()
    out = RESULTS_DIR / "noniot_training_curves.png"
    plt.savefig(out, dpi=120); plt.close(); print(f"  Curves → {out}")

def plot_cm(labs, preds, le):
    cm   = confusion_matrix(labs, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Non-IoT CNN-LSTM — Confusion Matrix")
    plt.tight_layout()
    out = RESULTS_DIR / "noniot_confusion_matrix.png"
    plt.savefig(out, dpi=120); plt.close(); print(f"  CM     → {out}")

def plot_roc(labs, probs):
    fpr, tpr, _ = roc_curve(labs, probs)
    auc = roc_auc_score(labs, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Non-IoT CNN-LSTM — ROC"); plt.legend(); plt.tight_layout()
    out = RESULTS_DIR / "noniot_roc.png"
    plt.savefig(out, dpi=120); plt.close(); print(f"  ROC    → {out}")


# ══════════════════════════════════════════════════════════════════════
# DETECTOR WRAPPER  (for inference_bridge.py)
# ══════════════════════════════════════════════════════════════════════

class Stage2Detector:
    def __init__(self, model, seq_len, n_features, threshold, le, feat_cols):
        self.model = model.to(DEVICE); self.model.eval()
        self.seq_len = seq_len; self.n_features = n_features
        self.threshold = threshold; self.label_encoder = le
        self.feature_cols = feat_cols

    def _align(self, df):
        arr = np.zeros((len(df), self.n_features), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in df.columns:
                arr[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.).values
        return arr

    def predict(self, df):
        X = self._align(df)
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), self.n_features), np.float32)
            X   = np.vstack([pad, X])
        seq = torch.tensor(X[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(seq)).item()
        return ("botnet" if prob >= self.threshold else "benign"), float(prob)

    def predict_sequence(self, seq_array: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a pre-scaled (seq_len, n_features) flow sequence.

        Used by the live monitoring pipeline: live_capture.py maintains a
        per-src_ip rolling buffer of completed flow dicts and, once the buffer
        has seq_len entries, scales them with noniot_scaler.json and passes
        the resulting array here.

        Bypasses _align() because the caller has already mapped its source
        feature dicts onto self.feature_cols. Use predict(df) instead for
        DataFrames with named columns that need alignment.
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

    def save(self, mp, mtp):
        torch.save({"model_state": self.model.state_dict(),
                    "seq_len": self.seq_len, "n_features": self.n_features,
                    "threshold": self.threshold, "label_encoder": self.label_encoder,
                    "feature_cols": self.feature_cols, "model_type": "noniot_cnn_lstm"}, mp)
        json.dump({"seq_len": self.seq_len, "n_features": self.n_features,
                   "threshold": self.threshold, "feature_cols": self.feature_cols,
                   "model_type": "noniot_cnn_lstm"}, open(mtp, "w"), indent=2)
        print(f"  Model → {mp}\n  Meta  → {mtp}")

    @classmethod
    def load(cls, mp):
        ck = torch.load(mp, map_location=DEVICE, weights_only=False)
        m  = CnnLstmDetector(ck["n_features"]).to(DEVICE)
        m.load_state_dict(ck["model_state"])
        return cls(m, ck["seq_len"], ck["n_features"], ck["threshold"],
                   ck["label_encoder"], ck["feature_cols"])


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    print("\n" + "═"*55)
    print("  Stage-2 Non-IoT CNN-LSTM  v3  (Internal Filter)")
    print(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}   STRIDE={STRIDE}")
    print("═"*55)

    le = LabelEncoder(); le.fit(["benign", "botnet"])

    sample    = pd.read_csv(DATA_PATH, nrows=1)
    feat_cols = [c for c in sample.columns if c not in DROP_COLS]
    n_feat    = len(feat_cols)
    print(f"\n  Features: {n_feat}")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_and_build(feat_cols, le)
    tr_ldr, val_ldr, te_ldr, pw = make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te)

    model    = CnnLstmDetector(n_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}\n")

    model, tl, vl, va, vr = train_model(model, tr_ldr, val_ldr, pw)
    plot_curves(tl, vl, va, vr)

    thr = find_threshold(model, val_ldr)
    m, probs, labs = evaluate(model, te_ldr, thr, le)
    plot_cm(labs, (probs >= thr).astype(int), le)
    plot_roc(labs, probs)

    Stage2Detector(model, SEQ_LEN, n_feat, thr, le, feat_cols).save(MODEL_PATH, META_PATH)
    json.dump(m, open(RESULTS_DIR / "noniot_metrics.json", "w"), indent=2)

    print(f"\n  AUC={m['auc_roc']:.4f}  Recall={m['recall']:.4f}"
          f"  Precision={m['precision']:.4f}  F1={m['f1']:.4f}")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()
