"""
M-CAL Confidence calibration (reliability) on synthetic mixed input.

Splits scores into 10 bins and reports observed positive rate vs the
mean predicted probability per bin. Saves the table to artifacts and
returns a single summary scalar — Expected Calibration Error (ECE).
"""

from __future__ import annotations

import csv

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.ml_tests._inference import predict_csv
from test_harness.utils.paths import for_test


TEST_ID = "M-CAL"


def run(n_each: int = 400, n_bins: int = 10, seed: int = 0,
        ece_threshold: float = 0.10) -> dict:
    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(
        dirs["artifacts"] / "calib.csv",
        n_iot=0, n_noniot=n_each, n_botnet=n_each, seed=seed,
        unique_src_ips=20,
    )
    res = predict_csv(str(csv_path))
    if res.get("skipped"):
        return _skip("Confidence calibration", res.get("skip_reason"), str(csv_path))
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    truth = [1 if r["class_label"] == "botnet" else 0 for r in rows]
    pairs = []
    for label, t in zip(res["results"], truth):
        if label.get("label") == "unknown":
            continue
        pairs.append((float(label.get("confidence", 0.0)), t))
    if not pairs:
        return _skip("Confidence calibration", "all unknown", str(csv_path))

    # Bin and compute ECE.
    bins = [[] for _ in range(n_bins)]
    for s, t in pairs:
        idx = min(int(s * n_bins), n_bins - 1)
        bins[idx].append((s, t))
    total = len(pairs)
    table = []
    ece = 0.0
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not b:
            table.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                          "avg_conf": None, "obs_rate": None})
            continue
        avg_conf = sum(s for s, _ in b) / len(b)
        obs_rate = sum(t for _, t in b) / len(b)
        table.append({"bin_lo": lo, "bin_hi": hi, "count": len(b),
                      "avg_conf": round(avg_conf, 4),
                      "obs_rate": round(obs_rate, 4),
                      "abs_gap":  round(abs(avg_conf - obs_rate), 4)})
        ece += (len(b) / total) * abs(avg_conf - obs_rate)

    out_csv = dirs["artifacts"] / "reliability.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)

    if ece <= ece_threshold:
        verdict, actual = "PASS", f"ECE = {ece:.4f}"
    else:
        verdict, actual = "FAIL", (
            f"ECE = {ece:.4f} exceeds {ece_threshold}; "
            "model confidence is not well-calibrated.")
    return {"test_id": TEST_ID, "name": "Confidence calibration",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"ECE ≤ {ece_threshold}",
            "actual":   actual, "csv": str(csv_path),
            "raw": {"ece": round(ece, 4), "n_bins": n_bins,
                    "n_used": total, "table": table}}


def _skip(name: str, reason: str, csv_path: str) -> dict:
    return {"test_id": TEST_ID, "name": name,
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "ECE within threshold",
            "actual":   reason, "csv": csv_path}
