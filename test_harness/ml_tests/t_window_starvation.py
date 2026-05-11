"""
M-WS Per-src_ip Stage-2 window starvation.

The Non-IoT detector keeps a per-src_ip deque (maxlen 20). When every
flow comes from a unique src_ip, the LSTM receives 1 real + 19 zero-padded
vectors — a distribution it never saw at training time.

We compare detection accuracy on:
    A. balanced — many flows, 5 unique src_ips → full sequences
    B. starved  — many flows, every flow a unique src_ip → 1+19 padding

A meaningful drop A → B exposes the brittleness.
"""

from __future__ import annotations

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.ml_tests._inference import predict_csv
from test_harness.ml_tests._metrics import metrics
from test_harness.utils.paths import for_test


TEST_ID = "M-WS"


def _eval(csv_path: str) -> dict:
    res = predict_csv(csv_path)
    if res.get("skipped"):
        return {"skipped": True, "reason": res.get("skip_reason")}
    rows = list(open(csv_path, encoding="utf-8").readlines())  # +1 for header
    import csv as _csv
    truth = [1 if r.get("class_label") == "botnet" else 0
             for r in _csv.DictReader(open(csv_path, encoding="utf-8"))]
    preds = res["results"]
    y_true_aligned: list[int] = []
    y_pred: list[int] = []
    for label, t in zip(preds, truth):
        if label.get("label") == "unknown":
            continue
        y_true_aligned.append(t)
        y_pred.append(1 if label.get("label") == "botnet" else 0)
    if not y_pred:
        return {"skipped": True, "reason": "all rows 'unknown'"}
    return metrics(y_true_aligned, y_pred)


def run(n_botnet: int = 200, n_benign: int = 200) -> dict:
    dirs = for_test(TEST_ID)
    bal = synthetic_csv(dirs["artifacts"] / "balanced.csv",
                        n_iot=0, n_noniot=n_benign, n_botnet=n_botnet,
                        unique_src_ips=5)
    star = synthetic_csv(dirs["artifacts"] / "starved.csv",
                         n_iot=0, n_noniot=n_benign, n_botnet=n_botnet,
                         unique_src_ips=10_000)   # effectively unique-per-flow

    a = _eval(str(bal))
    b = _eval(str(star))
    if a.get("skipped"):
        return {"test_id": TEST_ID, "name": "Window starvation",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "F1 drop ≤ 5 percentage points from A to B",
                "actual":   a.get("reason"), "csv": str(bal)}
    if b.get("skipped"):
        return {"test_id": TEST_ID, "name": "Window starvation",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "F1 drop ≤ 5 pp from A to B",
                "actual":   b.get("reason"), "csv": str(star)}

    drop_f1     = round(a["f1"]     - b["f1"],     4)
    drop_recall = round(a["recall"] - b["recall"], 4)

    if drop_f1 <= 0.05:
        verdict, actual = "PASS", (
            f"F1: {a['f1']} → {b['f1']} (Δ={drop_f1}); "
            f"recall: {a['recall']} → {b['recall']} (Δ={drop_recall})")
    else:
        verdict, actual = "FAIL", (
            f"F1 drops {drop_f1*100:.1f} pp under unique-IP starvation: "
            f"{a['f1']} → {b['f1']}; recall drops {drop_recall*100:.1f} pp "
            f"({a['recall']} → {b['recall']}). The LSTM has not seen "
            "1-real + 19-padded sequences at training time.")
    return {"test_id": TEST_ID, "name": "Window starvation",
            "severity": "HIGH", "verdict": verdict,
            "expected": "F1 drop ≤ 0.05 between balanced and starved",
            "actual":   actual,
            "csv":      str(bal),
            "raw": {"balanced": a, "starved": b,
                    "drop_f1": drop_f1, "drop_recall": drop_recall}}
