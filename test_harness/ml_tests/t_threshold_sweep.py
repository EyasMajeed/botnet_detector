"""
M-T1 Threshold sweep on synthetic mixed CSV.

Generates a balanced mixed CSV (benign + botnet, multi-IP), runs
inference, then sweeps thresholds and reports recall/precision/F1 at
each. Saves a CSV to artifacts for later analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.ml_tests._inference import predict_csv
from test_harness.ml_tests._metrics import threshold_sweep
from test_harness.utils.paths import for_test


TEST_ID = "M-T1"


def run(n_iot: int = 200, n_noniot: int = 400, n_botnet: int = 200,
        seed: int = 0) -> dict:
    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(dirs["artifacts"] / "mixed.csv",
                             n_iot=n_iot, n_noniot=n_noniot,
                             n_botnet=n_botnet, seed=seed,
                             unique_src_ips=20)
    res = predict_csv(str(csv_path))
    if res.get("skipped"):
        return {"test_id": TEST_ID, "name": "Threshold sweep (synthetic)",
                "severity": "MEDIUM", "verdict": "SKIPPED",
                "expected": "Sweep produces ROC-like curve",
                "actual":   res.get("skip_reason"),
                "csv": str(csv_path)}
    if res.get("error"):
        return {"test_id": TEST_ID, "name": "Threshold sweep (synthetic)",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "inference succeeds",
                "actual":   f"inference error: {res['error']}",
                "csv": str(csv_path)}

    # Read labels in the same row order the predictor saw.
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    y_true = [1 if r.get("class_label") == "botnet" else 0 for r in rows]
    preds  = res["results"]
    # The bridge returns one dict per row in order. Anything that came
    # back as 'unknown' (IoT routed) we treat as 'no Stage-2 score'.
    y_score: list[float] = []
    y_true_aligned: list[int] = []
    for label, r in zip(preds, rows):
        if label.get("label") == "unknown":
            continue
        y_true_aligned.append(1 if r.get("class_label") == "botnet" else 0)
        y_score.append(float(label.get("confidence", 0.0)))
    if not y_score:
        return {"test_id": TEST_ID, "name": "Threshold sweep (synthetic)",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "at least some non-IoT scores produced",
                "actual":   "all rows came back as 'unknown'",
                "csv": str(csv_path), "raw": {"n_results": len(preds)}}

    sweep = threshold_sweep(y_true_aligned, y_score,
                            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    sweep_csv = dirs["artifacts"] / "threshold_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    # Pick the F1-best row.
    best = max(sweep, key=lambda r: r["f1"])
    if best["recall"] >= 0.80 and best["precision"] >= 0.80:
        verdict, actual = "PASS", (
            f"best F1 at τ={best['threshold']}: P={best['precision']}, "
            f"R={best['recall']}, F1={best['f1']}, AUC={threshold_sweep_auc(sweep)}")
    else:
        verdict, actual = "FAIL", (
            f"best F1 at τ={best['threshold']} only "
            f"P={best['precision']}, R={best['recall']}; below 0.80")
    return {"test_id": TEST_ID, "name": "Threshold sweep (synthetic)",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": "F1 ≥ 0.80 at some threshold; precision and recall both ≥ 0.80",
            "actual":   actual,
            "csv": str(csv_path),
            "raw": {"sweep": sweep, "n_used": len(y_score),
                    "n_unknown": len(preds) - len(y_score)}}


def threshold_sweep_auc(sweep: list[dict]) -> float:
    """Approximate AUC from a sweep table — handy for the summary."""
    pts = sorted(((1 - r["precision"], r["recall"]) for r in sweep))
    return round(sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                     for a, b in zip(pts, pts[1:])), 4)
