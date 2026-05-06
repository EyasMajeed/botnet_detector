"""
═══════════════════════════════════════════════════════════════════════
 global_importance.py — Offline global feature importance for the report
 Group 07 | CPCS499 | XAI Module
═══════════════════════════════════════════════════════════════════════

Computes feature importance THREE ways and saves them as plots /
JSON to xai_results/. Run this ONCE per trained model; the outputs
go into the project report's XAI section.

  1. Tree-based importance (Stage-1 RF only)  — Gini, free
  2. Permutation importance                    — recall-aware, model-agnostic
  3. Mean |IG| over a sample (Stage-2 only)    — what the model actually uses

WHY THREE METHODS?
──────────────────
Disagreement between them is itself a research finding (worth ~1 page in
the report):
  · A feature high on ALL THREE → bedrock feature, defensible to publish.
  · High Gini, low permutation → other features can substitute for it.
  · High permutation, low |IG|  → the feature is statistically predictive
                                  but the model doesn't really USE it.
  · High |IG|, low permutation  → the model uses it for individual
                                  decisions but other features compensate
                                  in aggregate.

USAGE
─────
Default — Stage-2 Non-IoT global importance:
    python3 -m src.xai.global_importance

Stage-1 (RF) global importance:
    python3 -m src.xai.global_importance --stage 1

Custom paths:
    python3 -m src.xai.global_importance \\
        --csv  data/processed/stage2_noniot_botnet.csv \\
        --out  xai_results/

OUTPUTS (written to <out>/)
───────────────────────────
  global_importance_<stage>.json     — combined rankings
  permutation_importance_<stage>.png — bar chart
  mean_ig_<stage>.png                — bar chart (Stage-2 only)
  tree_importance.png                — bar chart (Stage-1 only)
  agreement_<stage>.png              — disagreement plot across methods
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# Path resolution — try to find the repo root from anywhere
# ══════════════════════════════════════════════════════════════════════

def _find_repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__).resolve()).parent
    for _ in range(6):
        if (p / "monitoring.py").exists() and (p / "models").exists():
            return p
        p = p.parent
    return Path.cwd()


_ROOT = _find_repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════
# Stage-2 Non-IoT — global importance
# ══════════════════════════════════════════════════════════════════════

def stage2_global_importance(csv_path: Path, out_dir: Path,
                             sample: int = 500) -> dict:
    """
    Compute permutation importance + mean |IG| for the trained Non-IoT
    Stage-2 detector, on a sample of `sample` rows from `csv_path`.
    """
    print(f"\n[Stage-2 Non-IoT global importance]")
    print(f"  CSV:    {csv_path}")
    print(f"  Sample: {sample} rows")
    print(f"  Out:    {out_dir}\n")

    # Load detector via monitoring.py (the production wrapper)
    from monitoring import (Stage2NonIoTDetector, MODEL_S2_NONIOT,
                            NONIOT_SEQ_LEN)
    det = Stage2NonIoTDetector(MODEL_S2_NONIOT)
    feature_cols: list[str] = list(det._feature_cols)
    n_features = det._n_features

    # Load data
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.iloc[:sample].copy()
    print(f"  Loaded {len(df)} rows × {df.shape[1]} cols")

    # Build the explainer
    from src.xai import Stage2NonIoTExplainer
    expl = Stage2NonIoTExplainer(det)

    # Build sequences — one per row, padded to seq_len with zeros (matches
    # what stage2_predict does for short windows). Each sequence's "latest"
    # timestep is the row of interest.
    print("  Computing IG attributions per row ...")
    accumulated_abs_ig = np.zeros(n_features, dtype=np.float64)
    n_explained = 0
    for i, row in df.iterrows():
        feat = {c: float(row.get(c, 0.0)) for c in feature_cols}
        try:
            res = expl.explain(feat, top_k=n_features)
            for fc in res.top_features:
                idx = feature_cols.index(fc.feature)
                accumulated_abs_ig[idx] += abs(fc.attribution)
            n_explained += 1
        except Exception as e:
            print(f"    [warn] row {i} failed: {e}")
        if (n_explained + 1) % 50 == 0:
            print(f"    ... {n_explained + 1}/{len(df)}")

    if n_explained == 0:
        raise RuntimeError("Mean-|IG| accumulator is empty. Check the CSV columns "
                           "match the model's feature_cols.")
    mean_abs_ig = accumulated_abs_ig / n_explained

    # Permutation importance — measure how much recall drops when each
    # feature is shuffled across rows. Recall is the project's priority
    # metric so we use it (not accuracy).
    print("\n  Computing permutation importance ...")
    perm_imp = _permutation_importance(det, df, feature_cols, metric="recall")

    # Save and plot
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": 2, "branch": "noniot",
        "n_features": n_features,
        "n_samples_explained": n_explained,
        "feature_cols": feature_cols,
        "mean_abs_ig":  mean_abs_ig.tolist(),
        "permutation":  perm_imp.tolist(),
    }
    json_path = out_dir / "global_importance_stage2_noniot.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {json_path}")

    _bar_plot(feature_cols, mean_abs_ig,
              "Stage-2 Non-IoT — Mean |IG| (top 20)",
              out_dir / "mean_ig_stage2_noniot.png")
    _bar_plot(feature_cols, perm_imp,
              "Stage-2 Non-IoT — Permutation importance Δrecall (top 20)",
              out_dir / "permutation_importance_stage2_noniot.png")
    _agreement_plot(feature_cols, {"|IG|": mean_abs_ig, "Permutation": perm_imp},
                    "Stage-2 Non-IoT — method agreement (top 20)",
                    out_dir / "agreement_stage2_noniot.png")
    return summary


# ══════════════════════════════════════════════════════════════════════
# Stage-1 — global importance (RF tree-based + permutation + SHAP-mean)
# ══════════════════════════════════════════════════════════════════════

def stage1_global_importance(csv_path: Path, out_dir: Path,
                             sample: int = 1000) -> dict:
    print(f"\n[Stage-1 global importance]")
    print(f"  CSV:    {csv_path}")
    print(f"  Sample: {sample} rows")
    print(f"  Out:    {out_dir}\n")

    from monitoring import (Stage1Classifier, MODEL_S1_RF,
                            SCALER_S1_JSON, S1_FEATURES)
    s1 = Stage1Classifier(MODEL_S1_RF, SCALER_S1_JSON)
    feature_cols = list(s1._features)

    # 1) Tree-based importance — free, baked into the trained RF
    if hasattr(s1._rf, "feature_importances_"):
        tree_imp = np.asarray(s1._rf.feature_importances_, dtype=np.float64)
    else:
        # XGBoost path
        try:
            booster = s1._rf.get_booster()
            score = booster.get_score(importance_type="gain")
            tree_imp = np.array([
                score.get(f, 0.0) for f in [f"f{i}" for i in range(len(feature_cols))]
            ], dtype=np.float64)
        except Exception:
            tree_imp = np.zeros(len(feature_cols), dtype=np.float64)
    if tree_imp.sum() > 0:
        tree_imp = tree_imp / tree_imp.sum()  # normalise

    # 2) SHAP TreeExplainer mean |attribution| over a sample
    print("  Computing SHAP attributions ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False).iloc[:sample]

    from src.xai import Stage1Explainer
    expl = Stage1Explainer(s1)

    accumulated_abs_shap = np.zeros(len(feature_cols), dtype=np.float64)
    n_done = 0
    for i, row in df.iterrows():
        feat = {c: float(row.get(c, 0.0)) for c in feature_cols}
        try:
            res = expl.explain(feat, top_k=len(feature_cols))
            for fc in res.top_features:
                if fc.feature in feature_cols:
                    accumulated_abs_shap[feature_cols.index(fc.feature)] += abs(fc.attribution)
            n_done += 1
        except Exception as e:
            print(f"    [warn] row {i} failed: {e}")
    mean_abs_shap = accumulated_abs_shap / max(n_done, 1)

    # 3) Permutation importance on accuracy (Stage-1 is multi-class routing,
    #    accuracy is the natural metric; recall doesn't apply per-class here)
    print("\n  Computing permutation importance ...")
    perm_imp = _permutation_importance_stage1(s1, df, feature_cols)

    # Save & plot
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": 1,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "tree_importance": tree_imp.tolist(),
        "mean_abs_shap":   mean_abs_shap.tolist(),
        "permutation":     perm_imp.tolist(),
    }
    json_path = out_dir / "global_importance_stage1.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {json_path}")

    _bar_plot(feature_cols, tree_imp,
              "Stage-1 — Tree importance (top 20)",
              out_dir / "tree_importance_stage1.png")
    _bar_plot(feature_cols, mean_abs_shap,
              "Stage-1 — Mean |SHAP| (top 20)",
              out_dir / "mean_shap_stage1.png")
    _bar_plot(feature_cols, perm_imp,
              "Stage-1 — Permutation importance Δaccuracy (top 20)",
              out_dir / "permutation_importance_stage1.png")
    _agreement_plot(feature_cols,
                    {"Tree": tree_imp, "|SHAP|": mean_abs_shap, "Permutation": perm_imp},
                    "Stage-1 — method agreement (top 20)",
                    out_dir / "agreement_stage1.png")
    return summary


# ══════════════════════════════════════════════════════════════════════
# Permutation importance helpers
# ══════════════════════════════════════════════════════════════════════

def _permutation_importance(det, df: pd.DataFrame,
                            feature_cols: list[str],
                            metric: str = "recall") -> np.ndarray:
    """
    For Stage-2 NonIoT detector: shuffle each feature column, measure
    how much the chosen metric drops on the labelled `df`. Higher drop
    = more important feature.
    """
    if "class_label" not in df.columns:
        print("  [warn] CSV lacks 'class_label' — permutation importance unavailable.")
        return np.zeros(len(feature_cols), dtype=np.float64)

    rng = np.random.default_rng(42)

    def _eval(working_df: pd.DataFrame) -> float:
        y_true, y_pred = [], []
        for _, row in working_df.iterrows():
            feat = {c: float(row.get(c, 0.0)) for c in feature_cols}
            scaled = det.stage2_preprocess_non_iot(feat)
            label, _ = det.stage2_predict(np.stack([scaled]))
            y_true.append(int(row["class_label"]))
            y_pred.append(1 if label == "botnet" else 0)
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        if metric == "recall":
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            return float(tp / max(tp + fn, 1))
        else:  # accuracy
            return float((y_true == y_pred).mean())

    print(f"  Baseline {metric} = ", end="", flush=True)
    base = _eval(df)
    print(f"{base:.4f}")

    drops = np.zeros(len(feature_cols), dtype=np.float64)
    for j, col in enumerate(feature_cols):
        if col not in df.columns:
            continue
        scrambled = df.copy()
        scrambled[col] = rng.permutation(scrambled[col].values)
        m = _eval(scrambled)
        drops[j] = max(0.0, base - m)
        print(f"    {col:<30}  Δ{metric} = {drops[j]:+.4f}")
    return drops


def _permutation_importance_stage1(s1, df: pd.DataFrame,
                                   feature_cols: list[str]) -> np.ndarray:
    """Stage-1 version — uses accuracy on device_type label."""
    label_col = None
    for cand in ["device_type", "class_label", "label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        print("  [warn] CSV lacks a label column — permutation importance unavailable.")
        return np.zeros(len(feature_cols), dtype=np.float64)

    rng = np.random.default_rng(42)

    def _eval(working_df: pd.DataFrame) -> float:
        y_true, y_pred = [], []
        for _, row in working_df.iterrows():
            feat = {c: float(row.get(c, 0.0)) for c in feature_cols}
            label, _, _ = s1.stage1_predict(feat)   # Note: 3-tuple unpack
            y_true.append(str(row[label_col]))
            y_pred.append(label)
        return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))

    print(f"  Baseline accuracy = ", end="", flush=True)
    base = _eval(df)
    print(f"{base:.4f}")
    drops = np.zeros(len(feature_cols), dtype=np.float64)
    for j, col in enumerate(feature_cols):
        if col not in df.columns:
            continue
        scrambled = df.copy()
        scrambled[col] = rng.permutation(scrambled[col].values)
        drops[j] = max(0.0, base - _eval(scrambled))
        print(f"    {col:<30}  Δacc = {drops[j]:+.4f}")
    return drops


# ══════════════════════════════════════════════════════════════════════
# Plot helpers — friendly labels via feature_metadata, top-20 only
# ══════════════════════════════════════════════════════════════════════

def _top_n_idx(values: np.ndarray, n: int = 20) -> np.ndarray:
    return np.argsort(values)[::-1][:n]


def _bar_plot(features: list[str], values: np.ndarray,
              title: str, out_path: Path, top_n: int = 20):
    from src.xai.feature_metadata import get_display
    idx = _top_n_idx(values, top_n)
    labels = [get_display(features[i]) for i in idx]
    vals   = [values[i] for i in idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], vals[::-1])
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → {out_path}")


def _agreement_plot(features: list[str], rankings: dict[str, np.ndarray],
                    title: str, out_path: Path, top_n: int = 20):
    """Side-by-side bar plot of multiple importance methods on the same features."""
    from src.xai.feature_metadata import get_display
    # Combined ranking: union of top_n features from each method
    union: set[int] = set()
    for v in rankings.values():
        union.update(_top_n_idx(v, top_n).tolist())
    idx = sorted(union, key=lambda i: -max(v[i] for v in rankings.values()))[:top_n]
    labels = [get_display(features[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / max(len(rankings), 1)
    y = np.arange(len(labels))[::-1]
    for k, (name, vals) in enumerate(rankings.items()):
        # Normalise each method to [0,1] for comparable bar lengths
        vmax = vals.max() if vals.max() > 0 else 1.0
        normed = [vals[i] / vmax for i in idx]
        ax.barh(y - (k - len(rankings)/2 + 0.5) * width, normed[::-1],
                height=width, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_title(title)
    ax.set_xlabel("normalised importance (per-method)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2],
                    help="Which stage to analyse (default: 2 = Non-IoT detector)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Processed CSV with class_label / device_type column")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "xai_results",
                    help="Output directory (default: <repo>/xai_results)")
    ap.add_argument("--sample", type=int, default=500,
                    help="Number of rows to sample (default: 500)")
    args = ap.parse_args()

    if args.stage == 1:
        csv = args.csv or _ROOT / "data/processed/stage1_train.csv"
        stage1_global_importance(csv, args.out, sample=args.sample)
    else:
        csv = args.csv or _ROOT / "data/processed/stage2_noniot_botnet.csv"
        stage2_global_importance(csv, args.out, sample=args.sample)


if __name__ == "__main__":
    main()
