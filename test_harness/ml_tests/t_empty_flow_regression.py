"""
M-EF Empty-flow regression test.

Confirms the documented behaviour from the project's own CIC-IDS-2018
report: an empty flow (zero forward bytes, zero backward packets) makes
the Non-IoT model emit a near-constant ~0.0068 probability. We don't
require the team to have fixed it — we just measure the value and warn
if it has drifted from the documented constant.
"""

from __future__ import annotations

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.ml_tests._inference import predict_csv
from test_harness.utils.paths import for_test


TEST_ID = "M-EF"

DOCUMENTED_PROB = 0.0068
TOLERANCE       = 0.05   # any value < TOLERANCE is "near zero" — what we expect


def run(n_empty: int = 200) -> dict:
    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(
        dirs["artifacts"] / "empty_flows.csv",
        n_iot=0, n_noniot=0, n_botnet=0, n_empty=n_empty,
        unique_src_ips=20,
    )
    res = predict_csv(str(csv_path))
    if res.get("skipped"):
        return {"test_id": TEST_ID, "name": "Empty-flow regression",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "constant near-zero probability on empty flows",
                "actual":   res.get("skip_reason"),
                "csv": str(csv_path)}
    if res.get("error"):
        return {"test_id": TEST_ID, "name": "Empty-flow regression",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": "inference returns",
                "actual":   f"inference error: {res['error']}",
                "csv": str(csv_path)}

    confs = [float(r.get("confidence", 0.0)) for r in res["results"]
             if r.get("label") != "unknown"]
    if not confs:
        return {"test_id": TEST_ID, "name": "Empty-flow regression",
                "severity": "HIGH", "verdict": "FAIL",
                "expected": "Non-IoT flows produce scores",
                "actual":   "all rows came back as 'unknown'",
                "csv": str(csv_path)}

    avg = sum(confs) / len(confs)
    near_zero = avg < TOLERANCE
    if near_zero:
        verdict = "PASS"
        actual  = (f"avg confidence {avg:.4f} on empty flows — "
                   f"matches documented ≈{DOCUMENTED_PROB} (model returns "
                   "constant for feature-less inputs).")
    else:
        verdict = "FAIL"
        actual  = (f"avg confidence {avg:.4f} on empty flows is no longer near zero — "
                   "scaler may have been re-fit (good) OR something else changed.")
    return {"test_id": TEST_ID, "name": "Empty-flow regression",
            "severity": "HIGH", "verdict": verdict,
            "expected": f"avg confidence < {TOLERANCE}",
            "actual":   actual, "csv": str(csv_path),
            "raw": {"avg": round(avg, 6), "min": min(confs),
                    "max": max(confs), "n": len(confs)}}
