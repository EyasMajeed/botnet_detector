"""
================================================================================
 evaluate_ablations_cicids.py
 Group 07 | CPCS499 -- AI-Based Botnet Detection
================================================================================

 PURPOSE
 -------
 Evaluate ALL THREE Stage-2 Non-IoT architectures (hybrid CNN-LSTM,
 CNN-only, LSTM-only) on the SAME external CICFlowMeter-format CSV --
 e.g. Friday-02-03-2018.csv from CSE-CIC-IDS-2018 -- so the ablation
 study can be reported under out-of-distribution conditions.

 This script reuses helpers from the existing
 evaluation/cicids2018/evaluate_external_csv.py (column mapping, tiling,
 batched inference, plotting) and adds:

   1. An architecture dispatcher that loads the correct nn.Module class
      based on the checkpoint's "model_type" field.
   2. Auto-location of each model's matching scaler.json (so the hybrid
      uses the production scaler and each ablation uses its own).
   3. A uniform threshold sweep across all three models for a fair
      OOD comparison.
   4. A side-by-side comparison summary (JSON + Markdown + bar chart).

 IMPORTANT INTERPRETATION CAVEAT (read before reporting any number)
 ------------------------------------------------------------------
 CICFlowMeter daily CSVs do NOT carry src_ip per flow, so this script
 falls back to TILING each flow into a constant length-20 sequence.
 Constant input collapses the LSTM into a fixed-point computation, so
 the hybrid's temporal advantage is invisible in this regime. Treat the
 results as a feature-space generalisation check, not as a validation
 of the LSTM contribution. For the latter, re-extract Friday-02-03 from
 PCAP using src/ingestion/pcap_to_csv.py to recover src_ip groups, then
 re-run with --use_src_ip.

 SAVE LOCATION
 -------------
   <project_root>/evaluation/cicids2018/evaluate_ablations_cicids.py

 USAGE
 -----
   Windows:
     python evaluation\\cicids2018\\evaluate_ablations_cicids.py ^
         --csv data\\raw\\cicids2018\\Friday-02-03-2018.csv ^
         --max_rows 200000

   macOS:
     python3 evaluation/cicids2018/evaluate_ablations_cicids.py \\
         --csv data/raw/cicids2018/Friday-02-03-2018.csv \\
         --max_rows 200000

 EXPECTED OUTPUT
 ---------------
   evaluation/cicids2018/results/ablations/
     ablation_summary_<csv_stem>.json     Per-arch metrics + sweeps
     ablation_summary_<csv_stem>.md       Markdown table for the report
     ablation_bars_<csv_stem>.png         4-panel bar chart
     cm_<arch>_<csv_stem>.png             One CM per architecture
     sweep_<arch>_<csv_stem>.csv          One sweep per architecture

 DEPENDENCIES
 ------------
   Windows: pip  install pandas numpy scikit-learn torch matplotlib
   macOS:   pip3 install pandas numpy scikit-learn torch matplotlib

 COMMON ERRORS
 -------------
   - "FileNotFoundError" on a model.pt
       -> Train the corresponding ablation first. Skip with --skip cnn,lstm
          to evaluate only the hybrid.
   - "size mismatch for ..."
       -> Checkpoint was trained with a different feature count.
          Re-train against the current data/processed/stage2_noniot_botnet.csv.
   - "No 'model_type' in checkpoint"
       -> Old hybrid checkpoint without the model_type tag. The script
          treats it as 'cnn_lstm' by default. To be explicit, re-train.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project imports. Done BEFORE we touch any model class to ensure
# the existing evaluator's helpers are available.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse helpers from the existing CICIDS2018 evaluator (no duplication).
# These are pure functions with no side effects on import.
from evaluation.cicids2018.evaluate_external_csv import (  # noqa: E402
    CICFLOW_TO_UNIFIED,
    _build_unified_features,
    _tile_to_sequences,
    _predict_in_batches,
    _save_confusion_plot,
    _print_header,
)
from evaluation.cicids2018.stratified_eval import stratify as _stratify_fn  # noqa: E402
from models.stage1.classifier import Stage1Classifier, ALL_FEATURES  # noqa: E402


def _safe_stratify(df: pd.DataFrame) -> np.ndarray:
    """
    Return a per-row label of 'active' or 'empty' using the exact same
    definition as evaluation/cicids2018/stratified_eval.py:

      empty  = (TotLen Fwd Pkts == 0) AND (Tot Bwd Pkts == 0)
      active = otherwise

    If the required CIC columns are missing (e.g. someone passes a
    pre-mapped CSV), every row is labelled 'active' so downstream code
    still produces meaningful results -- but a warning is printed so the
    user knows the active-flow filter degenerated to a no-op.
    """
    needed = ("TotLen Fwd Pkts", "Tot Bwd Pkts")
    if not all(c in df.columns for c in needed):
        print(f"  [WARN] CIC columns {needed} not found in CSV. "
              "Active/empty stratification disabled — all rows treated as 'active'.")
        return np.full(len(df), "active", dtype=object)
    return _stratify_fn(df)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ===========================================================================
# ARCHITECTURE DEFINITIONS (must match the training scripts byte-for-byte)
# ===========================================================================
# We re-define them HERE rather than importing from the ablation training
# scripts because those scripts monkey-patch nm.SCALER_PATH at module import
# time, which pollutes the production scaler path resolution in this process.
# ---------------------------------------------------------------------------

TEMPERATURE = 2.0  # matches noniot_detector_cnnlstm.TEMPERATURE

# IMPORTANT — these mirrors MUST match the training scripts' module nesting
# exactly, otherwise state_dict keys won't line up. The production hybrid uses
# nn.Sequential blocks named 'conv1', 'conv2', 'head', producing keys like
# 'conv1.0.weight' (the Conv1d), 'conv1.1.weight' (the BN), 'head.0.*' (first
# Linear), 'head.3.*' (second Linear, since head[1]=ReLU, head[2]=Dropout
# carry no params). Mirroring that nesting here is what makes load_state_dict
# work without renaming. Forward-pass output is divided by TEMPERATURE because
# the training scripts did the same before BCEWithLogitsLoss — skipping it
# would shift sigmoid outputs and silently miscalibrate every threshold.


class _HybridCnnLstm(nn.Module):
    """Mirror of models.stage2.noniot_detector_cnnlstm.CnnLstmDetector."""
    def __init__(self, n_features: int):
        super().__init__()
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
        self.lstm = nn.LSTM(256, 128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(128, 64),     # head.0
            nn.ReLU(),              # head.1
            nn.Dropout(0.4),        # head.2
            nn.Linear(64, 1),       # head.3
        )

    def forward(self, x):                       # (B, S, F)
        x = x.permute(0, 2, 1)                  # (B, F, S)
        x = self.conv1(x)                       # (B, 128, S/2)
        x = self.conv2(x)                       # (B, 256, S/2)
        x = x.permute(0, 2, 1)                  # (B, S/2, 256)
        _, (h_n, _) = self.lstm(x)
        logit = self.head(h_n[-1])              # (B, 1)
        return logit / TEMPERATURE


class _CnnOnly(nn.Module):
    """Mirror of evaluation/cnn_test/cnn_only_noniot.CnnOnlyDetector."""
    def __init__(self, n_features: int):
        super().__init__()
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
        # AdaptiveAvgPool1d has no params -> not in state_dict, name irrelevant.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 64),     # head.0
            nn.ReLU(),              # head.1
            nn.Dropout(0.4),        # head.2
            nn.Linear(64, 1),       # head.3
        )

    def forward(self, x):                       # (B, S, F)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x).squeeze(-1)            # (B, 256)
        logit = self.head(x)                    # (B, 1)
        return logit / TEMPERATURE


class _LstmOnly(nn.Module):
    """Mirror of evaluation/lstm_test/lstm_only_noniot.LstmOnlyDetector."""
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=128, num_layers=2,
            batch_first=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),     # head.0
            nn.ReLU(),              # head.1
            nn.Dropout(0.4),        # head.2
            nn.Linear(64, 1),       # head.3
        )

    def forward(self, x):                       # (B, S, F)
        _, (h_n, _) = self.lstm(x)
        logit = self.head(h_n[-1])              # (B, 1)
        return logit / TEMPERATURE


_MODEL_TYPE_TO_CLASS = {
    "noniot_cnn_lstm": _HybridCnnLstm,
    "cnn_lstm":        _HybridCnnLstm,   # tolerate older tag
    "cnn_only":        _CnnOnly,
    "lstm_only":       _LstmOnly,
}


# ===========================================================================
# CHECKPOINT + SCALER LOADING
# ===========================================================================

def _load_scaler(scaler_path: Path) -> StandardScaler | None:
    """Load a StandardScaler from a JSON file written by the training scripts."""
    if not scaler_path.exists():
        return None
    s = json.load(open(scaler_path))
    sc = StandardScaler()
    sc.mean_  = np.array(s["mean"])
    sc.scale_ = np.array(s["scale"])
    sc.var_   = sc.scale_ ** 2
    sc.n_features_in_ = len(s["features"])
    return sc


def _load_arch_checkpoint(model_path: Path) -> dict[str, Any]:
    """
    Load a checkpoint and dispatch to the correct nn.Module class based on
    its model_type field. Returns a dict with the loaded model, threshold,
    n_features, seq_len, feature_cols, and the matching scaler.

    Auto-locates the scaler:
      hybrid    -> models/stage2/noniot_scaler.json (production)
      cnn_only  -> <model_dir>/scaler.json
      lstm_only -> <model_dir>/scaler.json
    """
    ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_type = ck.get("model_type", "cnn_lstm")  # fallback for old checkpoints
    cls = _MODEL_TYPE_TO_CLASS.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type '{model_type}' in {model_path}. "
            f"Expected one of {sorted(_MODEL_TYPE_TO_CLASS)}"
        )

    n_features = int(ck["n_features"])
    model = cls(n_features).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Locate the matching scaler.
    if model_type in ("noniot_cnn_lstm", "cnn_lstm"):
        scaler_path = ROOT / "models" / "stage2" / "noniot_scaler.json"
    else:
        scaler_path = model_path.parent / "scaler.json"
    scaler = _load_scaler(scaler_path)
    if scaler is None:
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path} for model_type={model_type}. "
            "Re-run the corresponding training script."
        )

    return {
        "model":        model,
        "model_type":   model_type,
        "n_features":   n_features,
        "seq_len":      int(ck["seq_len"]),
        "threshold":    float(ck["threshold"]),
        "feature_cols": list(ck["feature_cols"]),
        "scaler":       scaler,
    }


def _align_and_scale(df: pd.DataFrame,
                     feature_cols: list[str],
                     scaler: StandardScaler) -> np.ndarray:
    """Same logic as Stage2Detector._align_and_scale, but standalone."""
    arr = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            arr[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.).values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return scaler.transform(arr).astype(np.float32)


# ===========================================================================
# INFERENCE + METRICS (split, so we can compute strata cheaply)
# ===========================================================================

def _run_inference_one(arch_label: str,
                      model_path: Path,
                      df_unified: pd.DataFrame) -> tuple[np.ndarray | None, dict | None]:
    """
    Load the checkpoint and run inference on every row. Returns
    (probs, checkpoint_meta) on success, or (None, error_dict) on failure.
    Inference is the expensive step — we only do it once per architecture
    regardless of how many strata are requested.
    """
    _print_header(f"Inference: {arch_label}  ({model_path})")
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None, {"status": "missing", "model_path": str(model_path)}

    try:
        ck = _load_arch_checkpoint(model_path)
    except Exception as e:
        print(f"  [ERROR] {arch_label}: {e}")
        return None, {"status": "error", "model_path": str(model_path), "error": str(e)}

    print(f"  model_type   : {ck['model_type']}")
    print(f"  n_features   : {ck['n_features']}")
    print(f"  seq_len      : {ck['seq_len']}")
    print(f"  train_thresh : {ck['threshold']:.3f}")

    X = _align_and_scale(df_unified, ck["feature_cols"], ck["scaler"])
    seqs = _tile_to_sequences(X, ck["seq_len"])

    t1 = time.perf_counter()
    probs = _predict_in_batches(ck["model"], seqs, batch=512)
    print(f"  inference_s  : {time.perf_counter() - t1:.1f}s")
    return probs, ck


def _metrics_for_subset(probs: np.ndarray,
                        y_true: np.ndarray,
                        mask: np.ndarray,
                        train_threshold: float,
                        threshold_sweep: list[float],
                        cm_path: Path,
                        title_prefix: str) -> dict[str, Any]:
    """
    Compute precision/recall/F1/AUC on a boolean-mask subset of `probs`,
    plus a uniform threshold sweep. Saves a confusion-matrix PNG. Used
    once per (architecture, stratum) pair.
    """
    sub_probs  = probs[mask]
    sub_y_true = y_true[mask]
    n_sub = int(mask.sum())
    n_pos = int(sub_y_true.sum())
    n_neg = n_sub - n_pos

    if n_sub == 0:
        return {"status": "empty", "n": 0}
    if n_pos == 0 or n_neg == 0:
        # Subset is degenerate (one class only). Report counts but skip
        # most metrics; AUC and binary precision/recall are undefined.
        return {
            "status": "degenerate",
            "n": n_sub, "botnet_flows": n_pos, "benign_flows": n_neg,
            "note": "subset has only one class; binary metrics undefined",
        }

    # At training-time threshold.
    yp_train = (sub_probs >= train_threshold).astype(int)
    p_t, r_t, f_t, _ = precision_recall_fscore_support(
        sub_y_true, yp_train, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = float(roc_auc_score(sub_y_true, sub_probs))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(sub_y_true, yp_train, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    _save_confusion_plot(cm, title=f"{title_prefix}  thr={train_threshold:.2f}",
                         path=cm_path)

    # Uniform threshold sweep (fair comparison across architectures).
    sweep_rows = []
    for thr in threshold_sweep:
        yp = (sub_probs >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            sub_y_true, yp, average="binary", pos_label=1, zero_division=0
        )
        sweep_rows.append({
            "threshold": thr, "precision": p, "recall": r, "f1": f,
            "TP": int(((yp == 1) & (sub_y_true == 1)).sum()),
            "FN": int(((yp == 0) & (sub_y_true == 1)).sum()),
            "FP": int(((yp == 1) & (sub_y_true == 0)).sum()),
        })
    sweep_df = pd.DataFrame(sweep_rows)

    qual = sweep_df[sweep_df["recall"] >= 0.85]
    if len(qual):
        best = qual.sort_values("precision", ascending=False).iloc[0]
        best_basis = "recall>=0.85, max precision"
    else:
        best = sweep_df.sort_values("f1", ascending=False).iloc[0]
        best_basis = "max F1 (recall target unattainable in sweep)"

    print(f"    n={n_sub:,}  bot={n_pos:,}  ben={n_neg:,}")
    print(f"    [thr={train_threshold:.2f}]  P={p_t:.4f}  R={r_t:.4f}  "
          f"F1={f_t:.4f}  AUC={auc:.4f}")
    print(f"    best in sweep: thr={best['threshold']:.2f}  ({best_basis})  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}")

    return {
        "status":          "ok",
        "n":               n_sub,
        "botnet_flows":    n_pos,
        "benign_flows":    n_neg,
        "mean_prob_true_botnet": float(sub_probs[sub_y_true == 1].mean()),
        "mean_prob_true_benign": float(sub_probs[sub_y_true == 0].mean()),
        "at_train_threshold": {
            "threshold": float(train_threshold),
            "precision": float(p_t), "recall": float(r_t), "f1": float(f_t),
            "auc_roc":   auc,
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        },
        "best_in_sweep": {
            "basis":     best_basis,
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall":    float(best["recall"]),
            "f1":        float(best["f1"]),
        },
        "sweep": sweep_rows,
    }


# ===========================================================================
# COMPARISON ARTEFACTS
# ===========================================================================

def _save_comparison_bars(arch_results: list[dict[str, Any]],
                          stratum: str,
                          out_path: Path,
                          metric_source: str = "best_in_sweep") -> None:
    """4-panel bar chart for one stratum: P/R/F1/AUC across architectures."""
    rows = [(r["arch"], r["strata"].get(stratum)) for r in arch_results
            if r.get("status") == "ok" and r.get("strata", {}).get(stratum, {}).get("status") == "ok"]
    if not rows:
        print(f"  [SKIP] No successful evaluations for stratum '{stratum}'.")
        return
    archs    = [r[0] for r in rows]
    metrics_panels = ["precision", "recall", "f1", "auc_roc"]

    def _val(s: dict, metric: str) -> float:
        if metric == "auc_roc":
            return s["at_train_threshold"]["auc_roc"]
        return s[metric_source].get(metric, float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, m in zip(axes.flat, metrics_panels):
        vals = [_val(r[1], m) for r in rows]
        bars = ax.bar(archs, vals, color=["#4C72B0", "#DD8452", "#55A467"][:len(archs)])
        ax.set_title(f"{m}  (basis: {metric_source})", fontsize=10)
        ax.set_ylim(0, 1.05)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    n_subset = rows[0][1]["n"]
    n_pos    = rows[0][1]["botnet_flows"]
    fig.suptitle(
        f"CICIDS2018 ablation -- stratum: {stratum.upper()}  "
        f"(n={n_subset:,}, botnet={n_pos:,})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _save_markdown_summary(arch_results: list[dict[str, Any]],
                           strata: list[str],
                           csv_stem: str,
                           rows_evaluated: int,
                           gt_pos: int,
                           gt_neg: int,
                           out_path: Path) -> None:
    """Markdown summary with one table per stratum (best-in-sweep + train-thr)."""
    lines = [
        f"# CICIDS2018 ablation -- `{csv_stem}`",
        "",
        f"- Rows evaluated: **{rows_evaluated:,}**",
        f"- Ground truth: **{gt_pos:,} botnet** / **{gt_neg:,} benign**",
        "",
        "> **Caveat.** Constant-tile sequences (no `src_ip`) collapse the "
        "LSTM into a fixed-point computation. These numbers compare the "
        "models' *feature-space* generalisation on each stratum, not "
        "their temporal modelling. Re-extract the day from PCAP for a "
        "temporal comparison.",
        "",
        "> **Why stratify?** CICFlowMeter splits one logical TCP "
        "conversation into multiple flow records on FIN/RST/idle-timeout, "
        "producing empty residue records that inherit the host-level "
        "`Bot` label without containing any C2 behaviour. The model "
        "correctly assigns these flows ~0 probability — but they drag "
        "the unstratified recall down. The **ACTIVE** subset is the "
        "meaningful comparison; **EMPTY** is included as a sanity "
        "check (all three architectures should score ~0 there).",
        "",
    ]

    for stratum in strata:
        # Subset header and counts (pulled from the first OK arch's record).
        first_ok = next(
            (r["strata"].get(stratum) for r in arch_results
             if r.get("status") == "ok" and r.get("strata", {}).get(stratum, {}).get("status") == "ok"),
            None,
        )
        if first_ok is None:
            lines += [f"## stratum: {stratum.upper()}", "", "*(no successful evaluations)*", ""]
            continue

        n_sub = first_ok["n"]; n_pos_s = first_ok["botnet_flows"]; n_neg_s = first_ok["benign_flows"]
        lines += [
            f"## stratum: {stratum.upper()}  (n={n_sub:,}, botnet={n_pos_s:,}, benign={n_neg_s:,})",
            "",
            "**At each model's own training-time threshold**",
            "",
            "| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in arch_results:
            s = r.get("strata", {}).get(stratum)
            if not s or s.get("status") != "ok":
                lines.append(f"| {r['arch']} | — | *unavailable* | — | — | — | — | — | — | — |")
                continue
            a = s["at_train_threshold"]
            lines.append(
                f"| {r['arch']} | {a['threshold']:.2f} | {a['precision']:.4f} | "
                f"{a['recall']:.4f} | {a['f1']:.4f} | {a['auc_roc']:.4f} | "
                f"{a['tp']:,} | {a['fn']:,} | {a['fp']:,} | {a['tn']:,} |"
            )
        lines += [
            "",
            "**At each model's best threshold in the uniform sweep**",
            "",
            "| Architecture | basis | thr | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in arch_results:
            s = r.get("strata", {}).get(stratum)
            if not s or s.get("status") != "ok":
                lines.append(f"| {r['arch']} | — | — | *unavailable* | — | — |")
                continue
            b = s["best_in_sweep"]
            lines.append(
                f"| {r['arch']} | {b['basis']} | {b['threshold']:.2f} | "
                f"{b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} |"
            )
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **Recall is the project's primary metric** (minimise false negatives).",
        "- The *training-time threshold* row shows operational behaviour: each "
        "model was tuned to hit recall>=0.85 on its own validation set; this "
        "row reveals whether that calibration generalises out-of-distribution.",
        "- The *uniform sweep* row is the fair architecture comparison: same "
        "threshold grid, same data, same scaler-per-model logic. Differences "
        "are attributable to architecture only.",
        "- The **ACTIVE** stratum corresponds to the 114,614-flow active "
        "subset cited in the CS499 thesis, where the hybrid scored "
        "P=0.984 R=0.977. Compare ablation rows to that baseline.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run hybrid + CNN-only + LSTM-only on a CICFlowMeter CSV "
                    "and produce a comparison summary."
    )
    ap.add_argument("--csv", required=True, type=str,
                    help="Path to CICFlowMeter-format CSV (e.g. CICIDS2018 day file).")
    ap.add_argument("--out_dir",
                    default=str(ROOT / "evaluation" / "cicids2018" /
                                "results" / "ablations"),
                    help="Output directory for plots + summaries.")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Optional cap on rows read.")
    ap.add_argument("--hybrid_model",
                    default=str(ROOT / "models" / "stage2" / "noniot_cnn_lstm.pt"))
    ap.add_argument("--cnn_only_model",
                    default=str(ROOT / "evaluation" / "cnn_test" /
                                "results" / "noniot" / "model.pt"))
    ap.add_argument("--lstm_only_model",
                    default=str(ROOT / "evaluation" / "lstm_test" /
                                "results" / "noniot" / "model.pt"))
    ap.add_argument("--label_col", default="Label",
                    help="Ground-truth label column name in the CSV.")
    ap.add_argument("--skip", default="",
                    help="Comma-separated archs to skip: hybrid, cnn-only, lstm-only.")
    ap.add_argument("--strata", default="all,active,empty",
                    help=("Comma-separated subsets to evaluate. "
                          "Choices: all, active, empty. "
                          "'active' matches the 114,614-flow subset cited in "
                          "the CS499 thesis. Default = all,active,empty."))
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = csv_path.stem
    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    requested_strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    valid_strata = {"all", "active", "empty"}
    bad = [s for s in requested_strata if s not in valid_strata]
    if bad:
        sys.exit(f"  [FATAL] Unknown strata: {bad}. Valid choices: {sorted(valid_strata)}")

    # --- Load CSV ---
    _print_header(f"Loading {csv_path}")
    if not csv_path.exists():
        sys.exit(f"  [FATAL] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, nrows=args.max_rows)
    df.columns = [c.strip() for c in df.columns]   # CIC files often have leading-space cols
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    # --- Resolve label column ---
    if args.label_col not in df.columns:
        cand = [c for c in df.columns if c.strip().lower() == args.label_col.lower()]
        if not cand:
            sys.exit(f"  [FATAL] Label column '{args.label_col}' not in CSV. "
                     f"Available: {list(df.columns)[:10]}...")
        args.label_col = cand[0]
    raw_labels = df[args.label_col].astype(str).str.strip().str.lower()
    y_true = (raw_labels != "benign").astype(int).values
    n_pos = int(y_true.sum()); n_neg = int(len(y_true) - n_pos)
    print(f"  Ground truth: botnet={n_pos:,}  benign={n_neg:,}")

    # --- Stratify (active vs empty) using the same definition as
    #     evaluation/cicids2018/stratified_eval.py ---
    _print_header("Stratifying flows")
    populations = _safe_stratify(df)
    n_active = int((populations == "active").sum())
    n_empty  = int((populations == "empty").sum())
    print(f"  Active: {n_active:,}   Empty: {n_empty:,}")
    if n_empty > 0:
        # Mirror the breakdown that stratified_eval.py prints for sanity.
        active_bot = int(((populations == "active") & (y_true == 1)).sum())
        empty_bot  = int(((populations == "empty")  & (y_true == 1)).sum())
        print(f"  Among true Botnet flows : active={active_bot:,}  empty={empty_bot:,}")
        print(f"  Among true Benign flows : active={n_active - active_bot:,}  "
              f"empty={n_empty - empty_bot:,}")

    # --- Map columns to unified schema ---
    _print_header("Mapping columns to unified 56-feature schema")
    df_unified = _build_unified_features(df)
    n_mapped = sum(1 for v in CICFLOW_TO_UNIFIED.values() if v in ALL_FEATURES)
    print(f"  Mapped {n_mapped} CICFlowMeter columns directly.")
    print(f"  Remaining {56 - n_mapped} unified features zero-filled.")

    # --- Architectures to evaluate ---
    targets = [
        ("hybrid",    Path(args.hybrid_model)),
        ("cnn-only",  Path(args.cnn_only_model)),
        ("lstm-only", Path(args.lstm_only_model)),
    ]
    targets = [(name, p) for name, p in targets if name not in skip_set]

    # Uniform threshold sweep — same grid for every architecture.
    threshold_sweep = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # --- Inference (once per architecture) + per-stratum metrics ---
    arch_results = []
    for name, model_path in targets:
        probs, ck_or_err = _run_inference_one(name, model_path, df_unified)
        if probs is None:
            arch_results.append({"arch": name, **ck_or_err})
            continue

        ck = ck_or_err
        arch_record = {
            "arch":       name,
            "status":     "ok",
            "model_path": str(model_path),
            "model_type": ck["model_type"],
            "n_features": ck["n_features"],
            "seq_len":    ck["seq_len"],
            "strata":     {},
        }

        # Save per-row predictions once (full population — strata are mask views).
        # This file is regeneratable so we keep it small enough to be useful for
        # downstream inspection without committing to git.
        pd.DataFrame({
            "row":           np.arange(len(probs)),
            "true_label":    np.where(y_true == 1, "Botnet", "Benign"),
            "stage2_prob":   probs,
            "population":    populations,
        }).to_csv(out_dir / f"predictions_{name}_{csv_stem}.csv", index=False)

        # Compute metrics on every requested stratum.
        for stratum in requested_strata:
            print(f"\n  --- {name}  ×  stratum={stratum.upper()} ---")
            if stratum == "all":
                mask = np.ones(len(populations), dtype=bool)
            elif stratum == "active":
                mask = populations == "active"
            elif stratum == "empty":
                mask = populations == "empty"
            else:                                                # pragma: no cover
                continue

            sweep_path = out_dir / f"sweep_{name}_{stratum}_{csv_stem}.csv"
            cm_path    = out_dir / f"cm_{name}_{stratum}_{csv_stem}.png"

            metrics = _metrics_for_subset(
                probs=probs,
                y_true=y_true,
                mask=mask,
                train_threshold=ck["threshold"],
                threshold_sweep=threshold_sweep,
                cm_path=cm_path,
                title_prefix=f"{name}  [{stratum}]",
            )
            arch_record["strata"][stratum] = metrics

            if metrics.get("status") == "ok":
                pd.DataFrame(metrics["sweep"]).to_csv(sweep_path, index=False)

        arch_results.append(arch_record)

    # --- Save aggregate artefacts ---
    _print_header("Writing aggregate artefacts")
    summary = {
        "csv":             str(csv_path),
        "rows_evaluated":  int(len(df)),
        "ground_truth":    {"botnet": n_pos, "benign": n_neg},
        "stratification":  {"active": n_active, "empty": n_empty,
                            "definition": ("empty = TotLen Fwd Pkts == 0 "
                                           "AND Tot Bwd Pkts == 0")},
        "strata_evaluated": requested_strata,
        "threshold_sweep": threshold_sweep,
        "tile_caveat":     ("Constant-tile sequences (no src_ip) collapse the "
                            "LSTM into a fixed-point computation. Treat results "
                            "as feature-space generalisation only."),
        "results":         arch_results,
    }
    summary_json = out_dir / f"ablation_summary_{csv_stem}.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON summary -> {summary_json}")

    md_path = out_dir / f"ablation_summary_{csv_stem}.md"
    _save_markdown_summary(arch_results, requested_strata, csv_stem,
                           len(df), n_pos, n_neg, md_path)
    print(f"  Markdown     -> {md_path}")

    # One bar chart per stratum.
    for stratum in requested_strata:
        bars_path = out_dir / f"ablation_bars_{stratum}_{csv_stem}.png"
        _save_comparison_bars(arch_results, stratum, bars_path,
                              metric_source="best_in_sweep")
        if bars_path.exists():
            print(f"  Bars [{stratum}] -> {bars_path}")

    print("\n" + "=" * 60)
    print("  DONE  |  CICIDS2018 ablation comparison")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()