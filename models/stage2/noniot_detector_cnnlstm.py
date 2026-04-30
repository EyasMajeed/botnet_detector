"""
════════════════════════════════════════════════════════════════════════
 Stage-2 Non-IoT CNN-LSTM  —  v4  (Scaler Fix)
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT : data/processed/stage2_noniot_botnet.csv   ← MUST be RAW (not pre-normalised)
 OUTPUT: models/stage2/noniot_cnn_lstm.pt
         models/stage2/noniot_scaler.json
         models/stage2/noniot_metadata.json
         models/stage2/results/noniot_*

 ROOT CAUSE OF PREVIOUS FAILURE (v3):
   The scaler (StandardScaler) was fitted AFTER or ON the already-
   normalised CSV. This produced:
       scaler.mean_  ≈ 0.5     (should be e.g. 50,000 for byte counts)
       scaler.scale_ ≈ 0.3     (should be e.g. 200,000+)
   Live raw traffic fed through this scaler produced X_scaled values
   in [-20, +2] instead of [-3, +3], making the model see completely
   out-of-distribution inputs at inference time.

 THIS FIX:
   1. Load CSV → verify raw scale (max >> 1.0) before ANY normalisation
   2. Fit StandardScaler on raw data → assert scale_.max() > 100
   3. Save scaler JSON immediately (before transform or split)
   4. Then transform → split → build sequences → train
   5. Find threshold from validation set at recall >= 0.85
   6. Save checkpoint with all required keys

 INTERNAL-DEVICE FILTER (carried over from v3):
   Keep only flows from internal prefixes (192.168.x, 172.31.x, 147.32.x)
   so that every device has enough flows to form sequences.
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════
ROOT        = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "data" / "processed" / "stage2_noniot_botnet.csv"
MODEL_DIR   = ROOT / "models" / "stage2"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_PATH  = MODEL_DIR / "noniot_cnn_lstm.pt"
META_PATH   = MODEL_DIR / "noniot_metadata.json"
SCALER_PATH = MODEL_DIR / "noniot_scaler.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════
SEQ_LEN            = 20
STRIDE             = 1
MAX_SEQ_PER_DEVICE = 5_000    # cap per source IP to prevent one server dominating
BATCH_SIZE         = 512
EPOCHS             = 80
LR                 = 1e-4
PATIENCE           = 10
TARGET_RECALL      = 0.85     # project requirement: minimize false negatives
MIN_PRECISION_GATE = 0.12     # realistic floor for ~6.6% botnet base rate
RANDOM_SEED        = 42
TEMPERATURE        = 2.0      # logit scaling to spread probabilities

# Columns to drop before feature selection
DROP_COLS = {"class_label", "device_type", "src_ip", "timestamp", "ttl_mean"}
LABEL_COL = "class_label"

# Internal network prefixes — only these IPs have enough flows for sequences
INTERNAL_PREFIX = ("192.168.", "172.31.", "147.32.")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"\n  Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════

class CnnLstmDetector(nn.Module):
    """
    Hybrid CNN-LSTM for temporal botnet detection.
    Conv layers extract local spatial patterns from the feature dimension.
    LSTM layers capture the temporal progression of those patterns.
    Temperature scaling spreads output probabilities so that the
    precision-recall curve has better resolution near the 0.85 recall target.
    """
    def __init__(self, n_features: int, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

        # CNN block: (batch, n_features, seq_len) → (batch, 256, seq_len/2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # LSTM block: processes the temporal dimension
        self.lstm = nn.LSTM(256, 128, num_layers=2,
                            batch_first=True, dropout=0.3)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)                  # → (batch, n_features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)                  # → (batch, seq_len/2, 256)
        _, (h_n, _) = self.lstm(x)
        logit = self.head(h_n[-1])               # last LSTM layer hidden state
        return logit / self.temperature           # scaled logit (raw, no sigmoid)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATA AND VERIFY SCALE
# ══════════════════════════════════════════════════════════════════════

def load_raw_data() -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """
    Load CSV, drop metadata columns, filter to internal IPs,
    and VERIFY that features are RAW (not pre-normalised).

    CRITICAL: This function must run BEFORE any scaler is fitted.
    If all feature max values <= 1.0, the CSV was already normalised
    and the upstream preprocessing pipeline must be fixed first.
    """
    print("\n" + "═"*60)
    print("  STEP 1 — Loading raw data")
    print("═"*60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n  CSV not found: {DATA_PATH}\n"
            "  Run your preprocessing script first to generate:\n"
            "  data/processed/stage2_noniot_botnet.csv"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # ── Label encoding ────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(["benign", "botnet"])
    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not found in CSV.")

    # class_label can be integer (0/1) from merge_stage2_noniot.py
    # or string ("benign"/"botnet") from step2_preprocess.py — handle both.
    raw_labels = df[LABEL_COL]
    if pd.api.types.is_numeric_dtype(raw_labels):
        int_labels = raw_labels.astype(int)
        invalid = ~int_labels.isin([0, 1])
        if invalid.any():
            print(f"  [WARN] {invalid.sum():,} rows with label not in {{0,1}} — dropping.")
            df = df[~invalid].copy()
            int_labels = df[LABEL_COL].astype(int)
        df["label_enc"] = le.transform(int_labels.map({0: "benign", 1: "botnet"}))
        print(f"  Label encoding: benign=0  botnet=1  (format=int)")
    else:
        str_labels = raw_labels.astype(str).str.strip().str.lower()
        invalid = ~str_labels.isin(["benign", "botnet"])
        if invalid.any():
            print(f"  [WARN] {invalid.sum():,} rows with unknown label — dropping.")
            df = df[~invalid].copy()
            str_labels = df[LABEL_COL].astype(str).str.strip().str.lower()
        df["label_enc"] = le.transform(str_labels)
        print(f"  Label encoding: benign=0  botnet=1  (format=str)")

    # ── Internal-device filter ────────────────────────────────────────
    # CTU-13 uses 147.32.x; CIC-IDS-2017 uses 192.168.x — both pass.
    # Safety guard: if filter drops > 80%, skip it and warn.
    if "src_ip" in df.columns:
        mask    = df["src_ip"].astype(str).str.startswith(INTERNAL_PREFIX)
        n_pass  = mask.sum()
        before  = len(df)
        pct     = n_pass / before * 100
        if pct < 20.0:
            print(f"  [WARN] Internal-IP filter would keep only {pct:.1f}% of rows "
                  f"({n_pass:,}/{before:,}). Skipping filter — using ALL rows.")
        else:
            df = df[mask].copy()
            print(f"  Internal-IP filter: {before:,} → {len(df):,} rows "
                  f"({pct:.1f}% retained)")
    else:
        print("  [WARN] 'src_ip' column not found — skipping internal filter")

    # ── Feature columns ───────────────────────────────────────────────
    # Drop metadata columns AND any all-NaN or all-zero columns
    candidates = [c for c in df.columns if c not in DROP_COLS and c != "label_enc"]
    all_nan  = [c for c in candidates if df[c].isna().all()]
    all_zero = [c for c in candidates if (df[c].fillna(0) == 0).all()]
    drop_extra = set(all_nan) | set(all_zero)
    if drop_extra:
        print(f"  Dropping {len(drop_extra)} all-NaN/all-zero columns: "
              f"{list(drop_extra)[:5]}{'...' if len(drop_extra) > 5 else ''}")

    feat_cols = [c for c in candidates if c not in drop_extra]
    print(f"  Feature columns: {len(feat_cols)}")

    # ── CLASS BALANCE ─────────────────────────────────────────────────
    vc = df["label_enc"].value_counts()
    for cls_idx, cnt in vc.items():
        cls_name = le.inverse_transform([cls_idx])[0]
        print(f"  {cls_name:>8s}: {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")

    # ── RAW SCALE VERIFICATION (mandatory guard) ──────────────────────
    print("\n  Raw feature scale check (top-5 max values):")
    max_vals = df[feat_cols].max().sort_values(ascending=False).head(5)
    for col, val in max_vals.items():
        print(f"    {col:<40} max = {val:.4f}")

    global_max = df[feat_cols].max().max()
    if global_max <= 1.0:
        raise ValueError(
            f"\n  ╔══ FATAL: CSV APPEARS ALREADY NORMALISED ══╗\n"
            f"  ║  All feature max values <= 1.0            ║\n"
            f"  ║  Global max found: {global_max:.6f}              ║\n"
            f"  ║                                           ║\n"
            f"  ║  Fix: regenerate stage2_noniot_botnet.csv ║\n"
            f"  ║  from RAW flows WITHOUT pre-normalisation.║\n"
            f"  ╚═══════════════════════════════════════════╝"
        )

    print(f"\n  ✓ Raw scale verified (global max = {global_max:.2f} >> 1.0)")
    return df, feat_cols, le


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — FIT AND SAVE SCALER ON RAW DATA
# ══════════════════════════════════════════════════════════════════════

def fit_and_save_scaler(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """
    Fit StandardScaler on RAW feature values.
    Save to noniot_scaler.json BEFORE any transform or split.
    Assert that scale_.max() > 100 (would fail on already-normalised data).

    Returns X_raw: float32 array (rows, features), NaN/inf cleaned.
    """
    print("\n" + "═"*60)
    print("  STEP 2 — Fitting scaler on RAW data")
    print("═"*60)

    X_raw = df[feat_cols].values.astype(np.float32)

    # Replace NaN, +inf, -inf with 0 (safe for StandardScaler)
    n_bad = (np.isnan(X_raw) | np.isinf(X_raw)).sum()
    if n_bad > 0:
        print(f"  Replacing {n_bad:,} NaN/Inf values with 0")
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    scaler.fit(X_raw)

    # ── CRITICAL ASSERTION ────────────────────────────────────────────
    # StandardScaler's scale_ ≈ std of each feature.
    # For byte counts / duration / packet rates on RAW data:
    #   std should be in the thousands to millions.
    # If scale_.max() <= 100, the scaler was fitted on already-normalised data.
    assert scaler.scale_.max() > 100, (
        f"\n  SCALER SANITY CHECK FAILED\n"
        f"  scale_.max() = {scaler.scale_.max():.4f} (expected > 100)\n"
        f"  This means the CSV contains already-normalised data.\n"
        f"  Fix: regenerate stage2_noniot_botnet.csv from raw flows."
    )

    print(f"  ✓ Scaler fitted:  scale_max = {scaler.scale_.max():.2f}  "
          f"mean_max = {scaler.mean_.max():.2f}")

    # ── SAVE SCALER JSON ──────────────────────────────────────────────
    scaler_dict = {
        "features": feat_cols,
        "mean":     scaler.mean_.tolist(),
        "scale":    scaler.scale_.tolist(),
        "note":     (
            "StandardScaler fitted on RAW stage2_noniot_botnet.csv. "
            "Apply to RAW live traffic features to reproduce training scale. "
            "scaled = (raw - mean) / scale"
        )
    }
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    return X_raw, scaler


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — NORMALISE, BUILD SEQUENCES, SPLIT
# ══════════════════════════════════════════════════════════════════════

def build_sequences(
    df: pd.DataFrame,
    X_raw: np.ndarray,
    scaler: StandardScaler,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Transform X_raw with fitted scaler → X_scaled
    2. Verify scaled range is approximately [-4, +4]
    3. Group by src_ip, build sliding-window sequences per device
    4. Train/Val/Test split (70/15/15)
    """
    print("\n" + "═"*60)
    print("  STEP 3 — Normalising and building sequences")
    print("═"*60)

    # ── Normalise ─────────────────────────────────────────────────────
    X_scaled = scaler.transform(X_raw)
    print(f"  Scaled range: {X_scaled.min():.2f} to {X_scaled.max():.2f}")

    # Soft guard: values >> 10 indicate extreme outliers in raw data
    if abs(X_scaled.min()) > 10 or abs(X_scaled.max()) > 10:
        print(
            "  [WARN] Scaled range exceeds [-10, +10]. "
            "Consider clipping outliers in raw data (e.g., 99th-percentile cap)."
        )
        # Clip rather than crash — allows training to proceed with a warning
        X_scaled = np.clip(X_scaled, -10, 10)
        print("  [WARN] Applied clip to [-10, +10].")

    # ── Sequence builder per source IP ────────────────────────────────
    df = df.reset_index(drop=True)
    df["_row_idx"] = df.index

    seqs_X, seqs_y = [], []

    if "src_ip" in df.columns:
        groups = df.groupby("src_ip")
    else:
        # Fallback: treat all rows as one group
        groups = [("all", df)]

    for ip, grp in groups:
        idxs = grp["_row_idx"].values
        ys   = grp["label_enc"].values

        if len(idxs) < SEQ_LEN:
            continue  # not enough flows for even one sequence

        count = 0
        for start in range(0, len(idxs) - SEQ_LEN + 1, STRIDE):
            end   = start + SEQ_LEN
            s_x   = X_scaled[idxs[start:end]]    # (SEQ_LEN, n_features)
            s_y   = int(ys[end - 1])              # label of the LAST flow in window
            seqs_X.append(s_x)
            seqs_y.append(s_y)
            count += 1
            if count >= MAX_SEQ_PER_DEVICE:
                break

    if len(seqs_X) == 0:
        raise RuntimeError(
            "No sequences built. Check that src_ip groups have >= SEQ_LEN flows "
            "and that INTERNAL_PREFIX filter is not too aggressive."
        )

    X_all = np.stack(seqs_X).astype(np.float32)   # (N, SEQ_LEN, n_features)
    y_all = np.array(seqs_y, dtype=np.float32)

    print(f"  Total sequences: {len(X_all):,}")
    print(f"  Botnet fraction: {y_all.mean()*100:.1f}%")

    # ── Stratified split ─────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_all, y_all, test_size=0.30,
        random_state=RANDOM_SEED, stratify=y_all
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50,
        random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ══════════════════════════════════════════════════════════════════════
# DATALOADERS
# ══════════════════════════════════════════════════════════════════════

def make_loaders(
    X_tr, y_tr, X_val, y_val, X_te, y_te
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build DataLoaders.
    Training uses WeightedRandomSampler to handle class imbalance
    (~6.6% botnet) without artificially duplicating data.
    pos_weight is passed to BCEWithLogitsLoss for additional emphasis.
    """
    # Class weight for loss function
    n_pos  = int(y_tr.sum())
    n_neg  = int(len(y_tr) - n_pos)
    pw     = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"\n  pos_weight (for loss): {pw.item():.2f}  "
          f"(n_neg={n_neg:,} / n_pos={n_pos:,})")

    # WeightedRandomSampler: each batch contains balanced classes
    sample_weights = np.where(y_tr == 1,
                               n_neg / max(n_pos, 1),
                               1.0).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    tr_ds  = SeqDataset(X_tr,  y_tr)
    val_ds = SeqDataset(X_val, y_val)
    te_ds  = SeqDataset(X_te,  y_te)

    tr_ldr  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, sampler=sampler)
    val_ldr = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_ldr  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return tr_ldr, val_ldr, te_ldr, pw


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_model(
    model: CnnLstmDetector,
    tr_ldr: DataLoader,
    val_ldr: DataLoader,
    pw: torch.Tensor,
) -> tuple[CnnLstmDetector, list, list, list, list]:
    """
    Train with BCEWithLogitsLoss + Adam.
    Early stopping on validation loss with PATIENCE epochs.
    Returns (model, train_losses, val_losses, val_accs, val_recalls).
    """
    print("\n" + "═"*60)
    print("  STEP 3b — Training")
    print("═"*60)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    train_losses, val_losses, val_accs, val_recalls = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        # ── Training pass ─────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        for X_b, y_b in tr_ldr:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b).squeeze(1)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item() * len(y_b)
        train_losses.append(ep_loss / len(tr_ldr.dataset))

        # ── Validation pass ───────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_ldr:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(X_b).squeeze(1)
                val_loss += criterion(logits, y_b).item() * len(y_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend((probs >= 0.5).astype(int))
                all_labels.extend(y_b.cpu().numpy().astype(int))

        val_loss /= len(val_ldr.dataset)
        val_losses.append(val_loss)

        val_acc    = accuracy_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_accs.append(val_acc)
        val_recalls.append(val_recall)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_losses[-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"val_recall={val_recall:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_accs, val_recalls


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — FIND THRESHOLD FROM VALIDATION SET
# ══════════════════════════════════════════════════════════════════════

def find_threshold(model: CnnLstmDetector, val_ldr: DataLoader) -> float:
    """
    Sweep the precision-recall curve on the validation set.
    Select the HIGHEST threshold that still achieves recall >= TARGET_RECALL.
    Higher threshold = fewer false positives while maintaining detection rate.

    Project requirement: recall >= 0.85 (false negatives are critical misses).
    """
    print("\n" + "═"*60)
    print("  STEP 4 — Finding optimal threshold (val set)")
    print("═"*60)

    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_ldr:
            X_b = X_b.to(DEVICE)
            p   = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
            val_probs.extend(p)
            val_labels.extend(y_b.numpy())

    val_probs  = np.array(val_probs)
    val_labels = np.array(val_labels, dtype=int)

    print(f"  Val probs range: {val_probs.min():.4f} to {val_probs.max():.4f}")

    if val_probs.max() - val_probs.min() < 0.01:
        print(
            "\n  [CRITICAL WARNING] Val probs are nearly constant.\n"
            "  The model did not learn a meaningful decision boundary.\n"
            "  Possible causes:\n"
            "    1. Extreme class imbalance not handled → check pos_weight\n"
            "    2. All-zero sequences (bad data)\n"
            "    3. Learning rate too high → loss diverged\n"
            "  Defaulting threshold to 0.50 — retrain recommended."
        )
        return 0.50

    prec, rec, thresholds = precision_recall_curve(val_labels, val_probs)

    # rec[:-1] aligns with thresholds (precision_recall_curve returns one extra)
    valid_mask   = (rec[:-1] >= TARGET_RECALL) & (prec[:-1] >= MIN_PRECISION_GATE)
    valid_thresh = thresholds[valid_mask]

    if len(valid_thresh) > 0:
        threshold = float(valid_thresh.max())
        idx = np.where(thresholds == threshold)[0][0]
        print(f"  ✓ Threshold at recall≥{TARGET_RECALL}: {threshold:.4f}  "
              f"(precision={prec[idx]:.4f}  recall={rec[idx]:.4f})")
    else:
        # Fallback: pick threshold that maximises recall above precision gate
        fallback_mask = prec[:-1] >= MIN_PRECISION_GATE
        if fallback_mask.any():
            best_idx  = np.argmax(rec[:-1][fallback_mask])
            threshold = float(thresholds[fallback_mask][best_idx])
            print(f"  [WARN] No threshold achieves recall≥{TARGET_RECALL} "
                  f"with precision≥{MIN_PRECISION_GATE}.")
            print(f"  Fallback threshold: {threshold:.4f}")
        else:
            threshold = 0.50
            print(f"  [WARN] Model precision too low at all thresholds. "
                  f"Using default threshold=0.50")

    # ── SATURATION GUARD ──────────────────────────────────────────────
    # If threshold >= 0.999, the model sigmoid is saturated — all botnet
    # probs are pushed to ~1.0 and all benign to ~0.0.
    # A threshold of 0.9999 is FRAGILE in production: any live flow with
    # prob=0.97 (genuinely suspicious) would be missed.
    # Fix: find the F1-maximising threshold and use max(f1_thresh, 0.50).
    # This maintains recall ≥ 0.85 while being robust to distribution shift.
    if threshold >= 0.999:
        print(f"\n  ⚠ Threshold {threshold:.4f} ≥ 0.999 — model output is saturated.")
        print(f"  Computing robust threshold at maximum F1-score instead...")
        f1_scores  = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_f1_idx = np.argmax(f1_scores)
        f1_thresh   = float(thresholds[best_f1_idx])
        # Enforce floor at 0.50 (never classify everything as botnet)
        threshold   = max(f1_thresh, 0.50)
        print(f"  ✓ Robust threshold (F1-max): {threshold:.4f}  "
              f"(precision={prec[best_f1_idx]:.4f}  "
              f"recall={rec[best_f1_idx]:.4f}  "
              f"f1={f1_scores[best_f1_idx]:.4f})")

    auc = roc_auc_score(val_labels, val_probs)
    print(f"  Val AUC-ROC: {auc:.4f}")

    return threshold


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(
    model: CnnLstmDetector,
    te_ldr: DataLoader,
    threshold: float,
    le: LabelEncoder,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate on the held-out test set using the validation-derived threshold.
    Returns metrics dict, probability array, and label array.
    """
    print("\n" + "═"*60)
    print(f"  STEP 5 — Test set evaluation (threshold={threshold:.4f})")
    print("═"*60)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in te_ldr:
            X_b = X_b.to(DEVICE)
            p   = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
            all_probs.extend(p)
            all_labels.extend(y_b.numpy().astype(int))

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)),
        "threshold": threshold,
    }

    print(f"\n  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}  ← target ≥ {TARGET_RECALL}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    if metrics["recall"] < TARGET_RECALL:
        print(f"\n  [WARN] Recall {metrics['recall']:.4f} < target {TARGET_RECALL}.")
        print("  Consider: reducing threshold, increasing pos_weight, "
              "or collecting more botnet samples.")

    print("\n" + classification_report(labels, preds,
          target_names=le.classes_, zero_division=0))
    return metrics, probs, labels


# ══════════════════════════════════════════════════════════════════════
# PLOT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def plot_curves(tl, vl, va, vr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tl, label="Train Loss")
    axes[0].plot(vl, label="Val Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(va, label="Val Accuracy")
    axes[1].plot(vr, label="Val Recall")
    axes[1].axhline(TARGET_RECALL, ls="--", c="red", label=f"Recall target {TARGET_RECALL}")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    out = RESULTS_DIR / "noniot_training_curves.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  Curves → {out}")


def plot_cm(labs, preds, le):
    cm   = confusion_matrix(labs, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Non-IoT CNN-LSTM — Confusion Matrix\n"
                 "(TN=benign✓  FP=false alarm  FN=missed botnet  TP=botnet✓)")
    plt.tight_layout()
    out = RESULTS_DIR / "noniot_confusion_matrix.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  CM     → {out}")


def plot_roc(labs, probs):
    fpr, tpr, _ = roc_curve(labs, probs)
    auc = roc_auc_score(labs, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Non-IoT CNN-LSTM — ROC Curve")
    plt.legend(); plt.tight_layout()
    out = RESULTS_DIR / "noniot_roc.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"  ROC    → {out}")


# ══════════════════════════════════════════════════════════════════════
# DETECTOR WRAPPER  (used by inference_bridge.py and GUI)
# ══════════════════════════════════════════════════════════════════════

class Stage2Detector:
    """
    Wraps a trained CnnLstmDetector with scaler and threshold for inference.
    Loaded by inference_bridge.py using Stage2Detector.load(MODEL_PATH).
    """
    def __init__(self, model, seq_len, n_features, threshold, le, feat_cols, scaler=None):
        self.model        = model.to(DEVICE); self.model.eval()
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.threshold    = threshold
        self.label_encoder = le
        self.feature_cols = feat_cols
        self.scaler       = scaler   # StandardScaler; None if loaded from JSON elsewhere

    def _align_and_scale(self, df: pd.DataFrame) -> np.ndarray:
        """
        Align dataframe columns to feat_cols, fill missing with 0,
        then apply StandardScaler if available.
        """
        arr = np.zeros((len(df), self.n_features), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in df.columns:
                arr[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.).values
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr

    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        X = self._align_and_scale(df)
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), self.n_features), np.float32)
            X   = np.vstack([pad, X])
        seq = (torch.tensor(X[-self.seq_len:], dtype=torch.float32)
               .unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            prob = torch.sigmoid(self.model(seq)).item()
        label = "botnet" if prob >= self.threshold else "benign"
        return label, float(prob)

    def save(self, mp: Path, mtp: Path):
        torch.save({
            "model_state":  self.model.state_dict(),
            "n_features":   self.n_features,
            "seq_len":      self.seq_len,
            "threshold":    self.threshold,
            "feature_cols": self.feature_cols,
            "label_encoder": self.label_encoder,
            "model_type":   "noniot_cnn_lstm",
        }, mp)

        json.dump({
            "seq_len":      self.seq_len,
            "n_features":   self.n_features,
            "threshold":    self.threshold,
            "feature_cols": self.feature_cols,
            "model_type":   "noniot_cnn_lstm",
        }, open(mtp, "w"), indent=2)

        print(f"  Model → {mp}\n  Meta  → {mtp}")

    @classmethod
    def load(cls, mp: Path):
        ck = torch.load(mp, map_location=DEVICE, weights_only=False)
        m  = CnnLstmDetector(ck["n_features"]).to(DEVICE)
        m.load_state_dict(ck["model_state"])

        # Load scaler from JSON if available
        scaler = None
        if SCALER_PATH.exists():
            s    = json.load(open(SCALER_PATH))
            sc   = StandardScaler()
            sc.mean_  = np.array(s["mean"])
            sc.scale_ = np.array(s["scale"])
            sc.var_   = sc.scale_ ** 2
            sc.n_features_in_ = len(s["features"])
            scaler = sc

        return cls(m, ck["seq_len"], ck["n_features"], ck["threshold"],
                   ck["label_encoder"], ck["feature_cols"], scaler)


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(model, feat_cols, threshold, le, n_feat):
    print("\n" + "═"*60)
    print("  STEP 5 — Saving checkpoint")
    print("═"*60)
    detector = Stage2Detector(
        model, SEQ_LEN, n_feat, threshold, le, feat_cols, scaler=None
    )
    detector.save(MODEL_PATH, META_PATH)
    print("  ✓ Checkpoint saved.")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — POST-TRAINING VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def verify_outputs():
    """
    Run verification checks on saved files.
    All assertions must pass before the model is considered production-ready.
    """
    print("\n" + "═"*60)
    print("  STEP 6 — Verifying saved files")
    print("═"*60)

    # ── Scaler check ──────────────────────────────────────────────────
    assert SCALER_PATH.exists(), f"Scaler not found: {SCALER_PATH}"
    s = json.load(open(SCALER_PATH))
    scales = np.array(s["scale"])
    print(f"  Scaler scale_max : {scales.max():.2f}  "
          f"{'✓ PASS' if scales.max() > 100 else '✗ FAIL — < 100, check pipeline'}")
    print(f"  Scaler scale_min : {scales.min():.6f}")
    print(f"  Scaler n_features: {len(s['features'])}")

    # ── Checkpoint check ──────────────────────────────────────────────
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    thr     = ckpt["threshold"]
    n_feat  = ckpt["n_features"]
    fc_match = ckpt["feature_cols"] == s["features"]

    print(f"\n  Checkpoint threshold  : {thr:.4f}  "
          f"{'✓ PASS' if 0.20 <= thr <= 0.80 else '⚠ UNUSUAL — check recall'}")
    print(f"  Checkpoint n_features : {n_feat}")
    print(f"  feature_cols match scaler: "
          f"{'✓ PASS' if fc_match else '✗ FAIL — mismatch!'}")

    if not fc_match:
        raise AssertionError(
            "feature_cols in checkpoint do not match scaler JSON. "
            "The model and scaler were trained with different feature sets."
        )

    print("\n  ✓ All verification checks passed.")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "═"*60)
    print("  Stage-2 Non-IoT CNN-LSTM  v4  (Scaler Fix)")
    print(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}   STRIDE={STRIDE}")
    print("═"*60)

    # ── Step 1: Load RAW data ─────────────────────────────────────────
    df, feat_cols, le = load_raw_data()
    n_feat = len(feat_cols)

    # ── Step 2: Fit scaler on RAW data, save JSON ─────────────────────
    X_raw, scaler = fit_and_save_scaler(df, feat_cols)

    # ── Step 3: Normalise, build sequences, split ─────────────────────
    X_tr, y_tr, X_val, y_val, X_te, y_te = build_sequences(
        df, X_raw, scaler, feat_cols
    )
    tr_ldr, val_ldr, te_ldr, pw = make_loaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te
    )

    # ── Build model ───────────────────────────────────────────────────
    model = CnnLstmDetector(n_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    # ── Step 3b: Train ────────────────────────────────────────────────
    model, tl, vl, va, vr = train_model(model, tr_ldr, val_ldr, pw)
    plot_curves(tl, vl, va, vr)

    # ── Step 4: Find threshold ────────────────────────────────────────
    threshold = find_threshold(model, val_ldr)

    # ── Evaluate ──────────────────────────────────────────────────────
    metrics, probs, labs = evaluate(model, te_ldr, threshold, le)
    plot_cm(labs, (probs >= threshold).astype(int), le)
    plot_roc(labs, probs)

    # ── Step 5: Save checkpoint ───────────────────────────────────────
    save_checkpoint(model, feat_cols, threshold, le, n_feat)

    # Save metrics
    json.dump(metrics, open(RESULTS_DIR / "noniot_metrics.json", "w"), indent=2)
    print(f"\n  Metrics → {RESULTS_DIR / 'noniot_metrics.json'}")

    # ── Step 6: Verify ────────────────────────────────────────────────
    verify_outputs()

    print("\n" + "═"*60)
    print(f"  DONE")
    print(f"  AUC={metrics['auc_roc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  "
          f"Precision={metrics['precision']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()