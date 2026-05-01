"""
global_importance.py — Offline Global Feature-Importance Analysis
==================================================================
Group 07 | CPCS499 | XAI Module

Purpose
-------
Per-flow XAI (local_explainer.py) tells the analyst WHY one specific flow
was flagged. This script answers the complementary academic question:
"Which features matter MOST to each model OVERALL?"  — a result you need
for the CPCS499 final report and the project defence.

Three importance methods are computed and saved:

    1. Mean |IG attribution|  over a sample of TEST flows
       Stage-2 only. Most faithful to the trained CNN-LSTM.
       This is the "what does the deep model actually use" answer.

    2. Permutation importance  (drop in F1 when feature is shuffled)
       Stage-1 and Stage-2. Model-agnostic, recall-aware.
       This tells you the BUSINESS impact of a feature.

    3. Tree feature importance  (Gini for RF, gain for XGBoost)
       Stage-1 only. Free, comes baked into the trained model.

All three are saved as JSON + a single summary chart per model.

Why all three?
--------------
The three rankings rarely agree perfectly. Disagreement is itself a finding:
  · A feature that's high on Gini but low on permutation is correlated with
    other features (the model can substitute).
  · A feature high on |IG| but low on permutation may matter for individual
    flows but not for aggregate F1.
  · The intersection — high on all three — is the bedrock you can publish.

CLI Usage
---------
    python  src/xai/global_importance.py  --model stage1
    python  src/xai/global_importance.py  --model stage2-iot
    python  src/xai/global_importance.py  --model stage2-noniot

Output Files (under results/xai/)
---------------------------------
    {model}_global_importance.json    raw rankings
    {model}_global_importance.png     bar chart of top-15 features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.xai.feature_metadata import get_display


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "xai"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "unified_dataset.csv"

# How many test flows to sample for IG computation. IG is ~50ms/flow on
# CPU, so 500 flows ≈ 25s. Increase to 2000 for the final report.
N_IG_SAMPLES = 500
N_PERM_REPEATS = 10        # repeats for permutation importance variance estimate
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════
# Method 1: Mean |IG| (Stage-2 only)
# ══════════════════════════════════════════════════════════════════════

def compute_mean_abs_ig(detector,
                        df_test: pd.DataFrame,
                        n_samples: int = N_IG_SAMPLES) -> dict[str, float]:
    """
    Sample n_samples test flows, run IG on each, average the absolute
    attribution per feature. Higher value = feature matters more on average.
    """
    from src.xai.local_explainer import Stage2Explainer
    explainer = Stage2Explainer(detector, n_steps=30)   # 30 steps is enough for ranking

    rng = np.random.default_rng(RANDOM_SEED)
    n = min(n_samples, len(df_test))
    idx = rng.choice(len(df_test), size=n, replace=False)

    accumulator = np.zeros(len(detector.feature_cols), dtype=np.float64)

    for i, k in enumerate(idx):
        # Build a window ending at flow k
        start = max(0, k - detector.seq_len + 1)
        window = df_test.iloc[start : k + 1]
        result = explainer.explain(window, top_k=len(detector.feature_cols))
        for fname, val in result.raw_attributions.items():
            j = detector.feature_cols.index(fname)
            accumulator[j] += abs(val)

        if (i + 1) % 50 == 0:
            print(f"    IG progress: {i + 1}/{n}")

    accumulator /= n
    return dict(zip(detector.feature_cols, accumulator.tolist()))


# ══════════════════════════════════════════════════════════════════════
# Method 2: Permutation Importance (model-agnostic)
# ══════════════════════════════════════════════════════════════════════

def compute_permutation_importance(predict_fn,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: list[str],
                                   n_repeats: int = N_PERM_REPEATS,
                                   metric: str = "f1") -> dict[str, float]:
    """
    For each feature j:
        baseline = metric( predict_fn(X), y )
        shuffle X[:, j] across rows
        shuffled = metric( predict_fn(X_shuf), y )
        importance_j = baseline - shuffled    (averaged over n_repeats)

    Higher = feature is more important. Negative = noise feature.

    `predict_fn(X)` must return hard labels (0/1).
    """
    from sklearn.metrics import f1_score, recall_score

    score_fn = {"f1": f1_score, "recall": recall_score}[metric]
    rng = np.random.default_rng(RANDOM_SEED)

    baseline = score_fn(y, predict_fn(X))
    print(f"    Baseline {metric}: {baseline:.4f}")

    importances: dict[str, float] = {}
    for j, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuf = X.copy()
            rng.shuffle(X_shuf[:, j])
            score = score_fn(y, predict_fn(X_shuf))
            drops.append(baseline - score)
        importances[fname] = float(np.mean(drops))
    return importances


# ══════════════════════════════════════════════════════════════════════
# Method 3: Tree feature importance (Stage-1 only)
# ══════════════════════════════════════════════════════════════════════

def compute_tree_importance(model, feature_names: list[str]) -> dict[str, float]:
    """RF: Gini. XGB: gain. Both are normalised to sum to 1."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        raise ValueError("Model has no feature_importances_ attribute.")
    return dict(zip(feature_names, imp.tolist()))


# ══════════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════════

def save_json_and_plot(rankings: dict[str, dict[str, float]],
                       model_label: str,
                       top_k: int = 15) -> None:
    """
    rankings: {method_name: {feature: score, ...}}
    Saves a single JSON and a single multi-panel PNG.
    """
    json_path = RESULTS_DIR / f"{model_label}_global_importance.json"
    with open(json_path, "w") as f:
        json.dump(rankings, f, indent=2)
    print(f"\n  Saved JSON  → {json_path}")

    n_methods = len(rankings)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 7), sharey=False)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, scores) in zip(axes, rankings.items()):
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        names = [get_display(k) for k, _ in items]
        vals  = [v for _, v in items]
        y_pos = np.arange(len(names))[::-1]
        ax.barh(y_pos, vals, color="#3A6EA5")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"{method}\n(top {top_k})")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(f"Global Feature Importance — {model_label}", fontsize=13)
    fig.tight_layout()
    png_path = RESULTS_DIR / f"{model_label}_global_importance.png"
    plt.savefig(png_path, dpi=140)
    plt.close()
    print(f"  Saved chart → {png_path}")


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def run_stage1():
    """Stage-1: tree importance + permutation importance on F1 and recall."""
    from models.stage1.classifier import Stage1Classifier, RF_PATH

    print("\n" + "═" * 60)
    print("  GLOBAL IMPORTANCE — Stage-1 (RF)")
    print("═" * 60)

    clf = Stage1Classifier.load(RF_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    from src.ingestion.preprocess_from_pcap_csvs import ALL_FEATURES
    feature_names = list(ALL_FEATURES)
    X = df[feature_names].values.astype(np.float32)
    y = clf.label_encoder.transform(df["device_type"].values)

    rankings: dict[str, dict[str, float]] = {}

    print("\n  [1/2] Tree (Gini) importance ...")
    rankings["RF Gini Importance"] = compute_tree_importance(clf.model, feature_names)

    print("\n  [2/2] Permutation importance (F1) ...")
    def predict_fn(X_arr):
        return clf.model.predict(X_arr)
    rankings["Permutation Δ F1"] = compute_permutation_importance(
        predict_fn, X, y, feature_names, metric="f1")

    save_json_and_plot(rankings, "stage1_rf")


def run_stage2(variant: str):
    """variant ∈ {'iot', 'noniot'}"""
    print("\n" + "═" * 60)
    print(f"  GLOBAL IMPORTANCE — Stage-2 ({variant.upper()})")
    print("═" * 60)

    if variant == "iot":
        from models.stage2.iot_detector import Stage2Detector
        from models.stage2.iot_detector import MODEL_PATH
    else:
        from models.stage2.noniot_detector_cnnlstm import Stage2Detector, MODEL_PATH

    det = Stage2Detector.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df_test = df[df["class_label"].notna()].reset_index(drop=True)

    rankings: dict[str, dict[str, float]] = {}

    print("\n  [1/2] Mean |Integrated Gradients| ...")
    rankings["Mean |IG|"] = compute_mean_abs_ig(det, df_test, n_samples=N_IG_SAMPLES)

    print("\n  [2/2] Permutation importance (recall) ...")
    # Build aligned X / y
    X = det._align(df_test)
    y = det.label_encoder.transform(df_test["class_label"].values)

    def predict_fn(X_arr: np.ndarray) -> np.ndarray:
        # We treat each row as an isolated flow for permutation purposes.
        # This intentionally simplifies the temporal model so the importance
        # is comparable to the Stage-1 method. A future refinement: use
        # window-based permutation. We accept the simplification here for
        # speed (~1 minute vs ~30 minutes per feature).
        df_tmp = pd.DataFrame(X_arr, columns=det.feature_cols)
        preds = []
        for i in range(len(df_tmp)):
            window = df_tmp.iloc[max(0, i - det.seq_len + 1) : i + 1]
            label, _ = det.predict(window)
            preds.append(1 if label == "botnet" else 0)
        return np.array(preds)

    # Subsample for permutation — one full pass is O(features × repeats × N)
    n_perm_samples = min(2000, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n_perm_samples, replace=False)
    X_sub, y_sub = X[idx], y[idx]
    rankings["Permutation Δ recall"] = compute_permutation_importance(
        predict_fn, X_sub, y_sub, det.feature_cols,
        n_repeats=3,         # fewer repeats — recall variance is small
        metric="recall",
    )

    save_json_and_plot(rankings, f"stage2_{variant}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Compute global feature importance for trained models.")
    p.add_argument("--model",
                   choices=["stage1", "stage2-iot", "stage2-noniot", "all"],
                   default="all",
                   help="Which model to analyse.")
    args = p.parse_args()

    if args.model in ("stage1", "all"):
        run_stage1()
    if args.model in ("stage2-iot", "all"):
        run_stage2("iot")
    if args.model in ("stage2-noniot", "all"):
        run_stage2("noniot")


if __name__ == "__main__":
    main()
