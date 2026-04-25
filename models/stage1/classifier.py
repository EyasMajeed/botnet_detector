"""
════════════════════════════════════════════════════════════════════════
 Stage-1 Classifier  —  IoT vs Non-IoT
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════

 INPUT  : data/processed/stage1_iot_vs_noniot.csv
           (produced by preprocess_from_pcap_csvs.py)

 OUTPUT : models/stage1/rf_model.pkl   ← Random Forest (primary)
          models/stage1/xgb_model.pkl  ← XGBoost      (baseline)
          models/stage1/results/       ← metrics, plots, report

 INTERFACE (used by inference_bridge.py):
    clf = Stage1Classifier.load("models/stage1/rf_model.pkl")
    device_type, confidence = clf.predict(feature_df)

 SUCCESS CRITERION: ≥ 95% accuracy on held-out test set
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

ROOT         = Path(__file__).resolve().parents[2]   # project root
DATA_PATH    = ROOT / "data" / "processed" / "stage1_iot_vs_noniot.csv"
MODEL_DIR    = ROOT / "models" / "stage1"
RESULTS_DIR  = MODEL_DIR / "results"
RF_PATH      = MODEL_DIR / "rf_model.pkl"
XGB_PATH     = MODEL_DIR / "xgb_model.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# UNIFIED 56-FEATURE SCHEMA  (must match preprocess_from_pcap_csvs.py)
# ═══════════════════════════════════════════════════════════════════════

ALL_FEATURES = [
    # ── Flow-level (40) ──────────────────────────────────────────────
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
    # ── Time-window (6) ──────────────────────────────────────────────
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score",    "burst_rate",
    "window_flow_count",    "window_unique_dsts",
    # ── Packet-level (9) ─────────────────────────────────────────────
    "ttl_mean","ttl_std","ttl_min","ttl_max",
    "dns_query_count",
    "payload_bytes_mean","payload_bytes_std",
    "payload_zero_ratio","payload_entropy",
    # ── TLS (1) ──────────────────────────────────────────────────────
    "tls_features_available",
]

LABEL_COL   = "class_label"          # values: "iot" | "noniot"
RANDOM_SEED = 42
TEST_SIZE   = 0.20                    # 80 / 20 split

# ═══════════════════════════════════════════════════════════════════════
# STAGE-1 CLASSIFIER WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class Stage1Classifier:
    """
    Thin wrapper around a scikit-learn classifier.
    Exposes the interface expected by inference_bridge.py:
        device_type, confidence = clf.predict(feature_df)
    """

    def __init__(self, model, label_encoder: LabelEncoder):
        self.model         = model
        self.label_encoder = label_encoder

    # ── Inference ────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Single-row (or multi-row) DataFrame with the 56 unified features.

        Returns
        -------
        device_type : str   — "iot" or "noniot"
        confidence  : float — probability of the predicted class (0–1)
        """
        X = self._align(df)
        proba      = self.model.predict_proba(X)          # shape (n, 2)
        pred_enc   = np.argmax(proba, axis=1)
        confidence = proba[np.arange(len(pred_enc)), pred_enc]
        device_types = self.label_encoder.inverse_transform(pred_enc)

        if len(device_types) == 1:
            return str(device_types[0]), float(confidence[0])
        return list(device_types), list(confidence.astype(float))

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)
        print(f"  ✓ Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Stage1Classifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(obj["model"], obj["label_encoder"])

    # ── Internal ─────────────────────────────────────────────────────
    @staticmethod
    def _align(df: pd.DataFrame) -> np.ndarray:
        """Ensure exactly the 56 features in the correct order."""
        for col in ALL_FEATURES:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0
        return df[ALL_FEATURES].values.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load CSV, return X array, y array (encoded), and the LabelEncoder."""
    print(f"\n  Loading data from:\n    {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n  [ERROR] File not found: {DATA_PATH}\n"
            "  Run preprocess_from_pcap_csvs.py first to generate it."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows  |  columns: {df.shape[1]}")

    # ── Validate label column ────────────────────────────────────────
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in CSV.")
    df = df[df[LABEL_COL].isin(["iot", "noniot"])].copy()
    print(f"  After label filter: {len(df):,} rows")

    # ── Class distribution ───────────────────────────────────────────
    vc = df[LABEL_COL].value_counts()
    print("\n  Class distribution:")
    for cls, cnt in vc.items():
        print(f"    {cls:>8s}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

    # ── Features ─────────────────────────────────────────────────────
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"\n  [WARN] {len(missing)} features missing → filled with 0:")
        print(f"    {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[ALL_FEATURES].values.astype(np.float32)

    # ── Encode labels ─────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])
    print(f"\n  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X, y, le


def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray,
              le: LabelEncoder) -> dict:
    """Compute and print all evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"\n{'─'*55}")
    print(f"  {name} — Test Results")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc*100:.2f}%  (target ≥ 95%)")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    if acc >= 0.95:
        print("  ✅  SUCCESS — ≥95% accuracy criterion MET")
    else:
        print(f"  ⚠   Accuracy {acc*100:.2f}% is below the 95% target")

    return {
        "model":     name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "auc_roc":   round(auc,  4),
    }


def _plot_confusion_matrix(name: str, model, X_test, y_test, le, filename: str):
    """Save a confusion-matrix PNG to RESULTS_DIR."""
    cm   = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix → {out}")


def _plot_feature_importance(model, top_n: int = 20):
    """Save a top-N feature importance bar chart for Random Forest."""
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:top_n]
    names       = [ALL_FEATURES[i] for i in idx]
    vals        = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], vals[::-1], color="#2196F3")
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances (Stage-1)")
    fig.tight_layout()
    out = RESULTS_DIR / "rf_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 Feature importance → {out}")


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("\n" + "═"*55)
    print("  Training: Random Forest (primary model)")
    print("═"*55)

    rf = RandomForestClassifier(
        n_estimators=200,       # enough trees for stability
        max_depth=None,         # grow full trees (data is clean/normalized)
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",    # standard for classification
        class_weight="balanced",# handles minor class imbalance
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)

    # 5-fold CV on training set
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5,
                                scoring="accuracy", n_jobs=-1)
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")

    metrics = _evaluate("Random Forest", rf, X_test, y_test, le)
    _plot_confusion_matrix("Random Forest", rf, X_test, y_test, le,
                           "rf_confusion_matrix.png")
    _plot_feature_importance(rf, top_n=20)

    clf = Stage1Classifier(rf, le)
    clf.save(RF_PATH)
    return clf, metrics


def train_xgboost(X_train, X_test, y_train, y_test, le):
    if not HAS_XGB:
        return None, None

    print("\n" + "═"*55)
    print("  Training: XGBoost (baseline comparison)")
    print("═"*55)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    cv_scores = cross_val_score(xgb, X_train, y_train, cv=5,
                                scoring="accuracy", n_jobs=-1)
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")

    metrics = _evaluate("XGBoost", xgb, X_test, y_test, le)
    _plot_confusion_matrix("XGBoost", xgb, X_test, y_test, le,
                           "xgb_confusion_matrix.png")

    clf = Stage1Classifier(xgb, le)
    clf.save(XGB_PATH)
    return clf, metrics


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════

def _save_comparison_report(all_metrics: list[dict]):
    """Print a side-by-side table and save JSON report."""
    print("\n" + "═"*55)
    print("  MODEL COMPARISON")
    print("═"*55)
    header = f"  {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    print(header)
    print("  " + "─" * 53)
    for m in all_metrics:
        print(f"  {m['model']:<20} "
              f"{m['accuracy']*100:>6.2f}% "
              f"{m['precision']*100:>6.2f}% "
              f"{m['recall']*100:>6.2f}% "
              f"{m['f1_score']*100:>6.2f}% "
              f"{m['auc_roc']:>7.4f}")

    out = RESULTS_DIR / "comparison_report.json"
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  📄 Report saved → {out}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═"*53 + "╗")
    print("║  STAGE-1 CLASSIFIER TRAINING — Group 07             ║")
    print("║  IoT vs Non-IoT  |  Random Forest + XGBoost         ║")
    print("╚" + "═"*53 + "╝")

    # 1. Load data
    X, y, le = _load_data()

    # 2. Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n  Split → Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    all_metrics = []

    # 3. Random Forest (primary)
    _, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, le)
    all_metrics.append(rf_metrics)

    # 4. XGBoost (baseline)
    _, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, le)
    if xgb_metrics:
        all_metrics.append(xgb_metrics)

    # 5. Comparison table + JSON report
    _save_comparison_report(all_metrics)

    print("\n╔" + "═"*53 + "╗")
    print("║  TRAINING COMPLETE                                   ║")
    print(f"║  RF model  → {RF_PATH}            ║")
    print(f"║  Results   → {RESULTS_DIR}                ║")
    print("╚" + "═"*53 + "╝\n")


if __name__ == "__main__":
    main()
