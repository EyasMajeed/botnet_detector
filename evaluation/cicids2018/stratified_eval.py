"""
═══════════════════════════════════════════════════════════════════════════
 stratified_eval.py
 Group 07 | CPCS499  —  Hybrid AI-Based Botnet Detection
═══════════════════════════════════════════════════════════════════════════

 Purpose
 ───────
 Test the hypothesis raised by diagnose_results.py: that the recall
 plateau on Friday-02-03-2018.csv is caused by a population of
 'header-only / payload-less' flows that the CIC-IDS-2018 labeling
 marks as 'Bot' (because they originate from an infected host) but
 that contain none of the C2-conversation features the model was
 trained to recognize.

 The script splits flows into two sub-populations and reports
 Precision / Recall / F1 / confusion matrix on each:

   ACTIVE  : Tot Bwd Pkts > 0  OR  TotLen Fwd Pkts > 0
   EMPTY   : Tot Bwd Pkts == 0 AND TotLen Fwd Pkts == 0

 Expected outcome if the hypothesis is correct
 ─────────────────────────────────────────────
   ACTIVE  : Recall jumps to >> 0.85, Precision stays high
   EMPTY   : Recall stays near 0, model outputs constant low probability

 If both sub-populations show low recall, the hypothesis is wrong and
 the failure mode is different — investigate further.

 Save location
 ─────────────
   <project_root>/evaluation/stratified_eval.py

 Usage
 ─────
 macOS:
   python3 evaluation/stratified_eval.py \\
       --predictions evaluation/results/predictions_Friday-02-03-2018.csv \\
       --csv data/raw/cicids2018/Friday-02-03-2018.csv \\
       --max_rows 200000

 Windows:
   python evaluation\\stratified_eval.py ^
       --predictions evaluation\\results\\predictions_Friday-02-03-2018.csv ^
       --csv data\\raw\\cicids2018\\Friday-02-03-2018.csv ^
       --max_rows 200000
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


def _print_header(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def stratify(df: pd.DataFrame) -> np.ndarray:
    """
    Label each flow as 'active' or 'empty'.

    'empty' = both directions carried zero payload bytes AND zero
              backward packets — these flows have no C2 content to
              detect. They are typically TCP control-packet residue
              (RST, FIN-only, half-open) emitted when CICFlowMeter
              splits one logical connection into multiple flow records.
    """
    fwd_bytes = pd.to_numeric(df["TotLen Fwd Pkts"], errors="coerce").fillna(0)
    bwd_pkts  = pd.to_numeric(df["Tot Bwd Pkts"],   errors="coerce").fillna(0)
    is_empty  = (fwd_bytes == 0) & (bwd_pkts == 0)
    return np.where(is_empty, "empty", "active")


def report_subset(name: str, sub: pd.DataFrame) -> dict:
    """Compute and print metrics for a flow subset. Returns a dict of metrics."""
    n = len(sub)
    if n == 0:
        print(f"\n  [{name}]  empty subset")
        return {"name": name, "n": 0}

    y_true = (sub["true_label"]      == "Botnet").astype(int).values
    y_pred = (sub["predicted_label"] == "Botnet").astype(int).values
    probs  = pd.to_numeric(sub["stage2_prob"], errors="coerce").fillna(0).values

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    if n_pos == 0:
        print(f"\n  [{name}]  n={n:,}  no positive (Bot) examples — skipping")
        return {"name": name, "n": n, "botnet_flows": 0, "benign_flows": n_neg}

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probs)) if n_neg > 0 else float("nan")
    except ValueError:
        auc = float("nan")

    print(f"\n  [{name}]  n={n:,}")
    print(f"    Botnet flows         : {n_pos:,}")
    print(f"    Benign flows         : {n_neg:,}")
    print(f"    Precision            : {p:.4f}")
    print(f"    Recall               : {r:.4f}")
    print(f"    F1                   : {f1:.4f}")
    print(f"    ROC-AUC              : {auc:.4f}")
    print(f"    TP / FN / FP / TN    : {tp:,} / {fn:,} / {fp:,} / {tn:,}")
    print(f"    Mean Stage-2 prob (true Bot)    : "
          f"{probs[y_true == 1].mean():.4f}")
    if n_neg > 0:
        print(f"    Mean Stage-2 prob (true Benign) : "
              f"{probs[y_true == 0].mean():.4f}")

    return {
        "name": name, "n": n,
        "botnet_flows": n_pos, "benign_flows": n_neg,
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": auc,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "mean_prob_true_botnet": float(probs[y_true == 1].mean()),
        "mean_prob_true_benign": float(probs[y_true == 0].mean()) if n_neg > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="predictions_*.csv from evaluate_external_csv.py")
    ap.add_argument("--csv", required=True,
                    help="Original CICFlowMeter CSV (same one used during evaluation).")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="MUST match the value used during evaluation.")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    preds_path = Path(args.predictions)
    csv_path   = Path(args.csv)
    out_dir    = Path(args.out_dir) if args.out_dir else preds_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    # ─── Load and join ──────────────────────────────────────────────────
    _print_header("Loading predictions and original CSV")
    preds = pd.read_csv(preds_path, low_memory=False)
    df    = pd.read_csv(csv_path, nrows=args.max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Predictions rows: {len(preds):,}")
    print(f"  CSV rows        : {len(df):,}")

    if len(preds) != len(df):
        raise ValueError(
            f"Row count mismatch: predictions={len(preds):,} vs csv={len(df):,}. "
            f"Re-run with the same --max_rows."
        )

    joined = pd.concat([df.reset_index(drop=True),
                        preds.reset_index(drop=True)], axis=1)

    # ─── Stratify ───────────────────────────────────────────────────────
    joined["population"] = stratify(joined)

    _print_header("Population distribution")
    print(f"  By population:")
    print(joined["population"].value_counts().to_string())

    print(f"\n  Among true Botnet flows:")
    print(joined.loc[joined["true_label"] == "Botnet",
                     "population"].value_counts().to_string())

    print(f"\n  Among true Benign flows:")
    print(joined.loc[joined["true_label"] == "Benign",
                     "population"].value_counts().to_string())

    # ─── Stratified metrics ─────────────────────────────────────────────
    _print_header("Metrics: ALL flows  (baseline, matches earlier evaluation)")
    res_all    = report_subset("ALL", joined)

    _print_header("Metrics: ACTIVE flows  (have payload OR backward packets)")
    res_active = report_subset("ACTIVE", joined[joined["population"] == "active"])

    _print_header("Metrics: EMPTY flows  (zero payload AND zero backward packets)")
    res_empty  = report_subset("EMPTY",  joined[joined["population"] == "empty"])

    # ─── Verdict ────────────────────────────────────────────────────────
    _print_header("Verdict")
    if res_active.get("recall", 0) >= 0.85:
        print("  ✓ HYPOTHESIS CONFIRMED.")
        print(f"    On ACTIVE conversation flows the model achieves "
              f"recall = {res_active['recall']:.4f} "
              f"(precision = {res_active['precision']:.4f}).")
        print( "    The recall plateau in the unstratified evaluation is an")
        print( "    artifact of the EMPTY-flow population, which the model")
        print( "    correctly classifies as not-C2 (constant probability ≈")
        print(f"    {res_empty.get('mean_prob_true_botnet', 0):.4f}).")
    else:
        print("  ✗ HYPOTHESIS NOT CONFIRMED.")
        print( "    Active-flow recall is also < 0.85. The model is missing")
        print( "    real C2 traffic, not just empty residue. Further")
        print( "    investigation needed (per-port behavior, fine-tuning).")

    # ─── Save summary ───────────────────────────────────────────────────
    out_json = out_dir / f"stratified_summary_{stem}.json"
    with open(out_json, "w") as fp:
        json.dump({
            "all":    res_all,
            "active": res_active,
            "empty":  res_empty,
        }, fp, indent=2)
    print(f"\n  Saved → {out_json}\n")


if __name__ == "__main__":
    main()