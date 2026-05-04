"""
================================================================================
 Ablation Comparison Aggregator
 Group 07 | CPCS499 | AI-Based Botnet Detection
================================================================================

 PURPOSE
 -------
 Aggregate the metric JSONs produced by:
   - models/stage2/results/iot_metrics.json      (hybrid IoT)
   - models/stage2/results/noniot_metrics.json   (hybrid Non-IoT)
   - evaluation/cnn_test/results/iot/metrics.json
   - evaluation/cnn_test/results/noniot/metrics.json
   - evaluation/lstm_test/results/iot/metrics.json
   - evaluation/lstm_test/results/noniot/metrics.json

 Produces:
   - comparison_table.csv     One row per (branch, model_type) combination
   - comparison_bars.png      Recall / Precision / F1 / AUC bar chart
   - comparison_summary.md    Auto-generated markdown table for ABLATION_REPORT.md

 USAGE
 -----
   Windows : python evaluation\\ablation_comparison\\aggregate_metrics.py
   macOS   : python3 evaluation/ablation_comparison/aggregate_metrics.py

 DEPENDENCIES
 ------------
   Windows : pip  install pandas matplotlib
   macOS   : pip3 install pandas matplotlib

 EXPECTED OUTPUT
 ---------------
   evaluation/ablation_comparison/results/
     comparison_table.csv
     comparison_bars.png
     comparison_summary.md

 COMMON ERRORS
 -------------
   - "FileNotFoundError" for any individual metrics JSON
       -> The corresponding training script has not been run yet.
          The aggregator will skip missing files and warn, NOT crash.
   - "KeyError: 'recall'"
       -> An older metrics JSON is missing fields. Re-run the training script.

 NOTES
 -----
 - The script does NOT re-train any model. It only aggregates files written
   by the six training runs.
 - "Hybrid" rows are read from the production model paths so this works
   without copying the hybrid JSON anywhere.
================================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
HERE      = Path(__file__).resolve().parent
OUT_DIR   = HERE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_PATH   = OUT_DIR / "comparison_table.csv"
BARS_PATH    = OUT_DIR / "comparison_bars.png"
SUMMARY_PATH = OUT_DIR / "comparison_summary.md"

# (label, branch, model_type, metrics_json_path)
SOURCES = [
    ("Hybrid CNN-LSTM", "iot",    "hybrid",
     ROOT / "models" / "stage2" / "results" / "iot_metrics.json"),
    ("Hybrid CNN-LSTM", "noniot", "hybrid",
     ROOT / "models" / "stage2" / "results" / "noniot_metrics.json"),
    ("CNN-only",        "iot",    "cnn_only",
     ROOT / "evaluation" / "cnn_test" / "results" / "iot" / "metrics.json"),
    ("CNN-only",        "noniot", "cnn_only",
     ROOT / "evaluation" / "cnn_test" / "results" / "noniot" / "metrics.json"),
    ("LSTM-only",       "iot",    "lstm_only",
     ROOT / "evaluation" / "lstm_test" / "results" / "iot" / "metrics.json"),
    ("LSTM-only",       "noniot", "lstm_only",
     ROOT / "evaluation" / "lstm_test" / "results" / "noniot" / "metrics.json"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_load(path: Path) -> dict | None:
    """Return JSON contents or None if the file is missing / unreadable."""
    if not path.exists():
        print(f"  [WARN] missing metrics: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] could not read {path}: {exc}")
        return None


def collect_rows() -> list[dict]:
    """Build a list of dict rows, one per available metric file."""
    rows = []
    for label, branch, model_type, path in SOURCES:
        m = safe_load(path)
        if m is None:
            rows.append({
                "model": label,
                "branch": branch,
                "model_type": model_type,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "auc_roc": None,
                "threshold": None,
                "n_parameters": None,
                "source": str(path.relative_to(ROOT)),
                "available": False,
            })
            continue

        rows.append({
            "model": label,
            "branch": branch,
            "model_type": model_type,
            "accuracy":     m.get("accuracy"),
            "precision":    m.get("precision"),
            "recall":       m.get("recall") or m.get("recall_botnet"),
            "f1":           m.get("f1") or m.get("f1_score"),
            "auc_roc":      m.get("auc_roc"),
            "threshold":    m.get("threshold"),
            "n_parameters": m.get("n_parameters"),
            "source":       str(path.relative_to(ROOT)),
            "available":    True,
        })
    return rows


def save_table(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_PATH, index=False)
    print(f"  Comparison table -> {TABLE_PATH}")


def plot_bars(df: pd.DataFrame) -> None:
    """One sub-plot per metric; bars grouped by model_type, coloured by branch."""
    metric_cols = ["recall", "precision", "f1", "auc_roc"]
    metric_titles = {
        "recall":    "Recall (priority - minimise false negatives)",
        "precision": "Precision",
        "f1":        "F1-score",
        "auc_roc":   "AUC-ROC",
    }
    available_df = df[df["available"]].copy()
    if available_df.empty:
        print("  [WARN] no metrics available - skipping bar plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    model_order  = ["hybrid", "cnn_only", "lstm_only"]
    branch_order = ["iot", "noniot"]
    width  = 0.35
    colors = {"iot": "#2196F3", "noniot": "#FF9800"}

    for ax, metric in zip(axes, metric_cols):
        x_positions = list(range(len(model_order)))
        for i, branch in enumerate(branch_order):
            sub = available_df[available_df["branch"] == branch]
            values = []
            for mt in model_order:
                row = sub[sub["model_type"] == mt]
                values.append(float(row[metric].iloc[0])
                              if not row.empty and row[metric].iloc[0] is not None
                              else 0.0)
            offset = (i - 0.5) * width
            bars = ax.bar([p + offset for p in x_positions], values,
                          width, label=branch.upper(), color=colors[branch])
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.005,
                        f"{v:.3f}" if v else "-",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([m.replace("_", "-") for m in model_order])
        ax.set_title(metric_titles[metric])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right")

    fig.suptitle("Ablation Study: Hybrid CNN-LSTM vs CNN-only vs LSTM-only",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(BARS_PATH, dpi=140)
    plt.close(fig)
    print(f"  Bar chart -> {BARS_PATH}")


def write_markdown_summary(df: pd.DataFrame) -> None:
    """Write a markdown table that can be pasted into ABLATION_REPORT.md."""
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc",
                   "n_parameters"]
    headers = ["Branch", "Model"] + [c.replace("_", " ").title() for c in metric_cols]

    lines = []
    lines.append("# Auto-generated comparison summary")
    lines.append("")
    lines.append(f"Generated by `aggregate_metrics.py` from "
                 f"{(df['available'] == True).sum()} of {len(df)} metric files.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in df.iterrows():
        cells = [str(row["branch"]), str(row["model"])]
        for c in metric_cols:
            v = row[c]
            if v is None or pd.isna(v):
                cells.append("n/a")
            elif c == "n_parameters":
                cells.append(f"{int(v):,}")
            else:
                cells.append(f"{float(v):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Re-run the missing training scripts to populate the n/a rows.")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"  Summary markdown -> {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Ablation Comparison Aggregator")
    print("=" * 60)
    rows = collect_rows()
    df = pd.DataFrame(rows)
    save_table(df)
    plot_bars(df)
    write_markdown_summary(df)
    print("\n  DONE\n")


if __name__ == "__main__":
    main()