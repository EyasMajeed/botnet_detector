"""P-07 Backwards timestamps — must not produce NaN/Inf or negative IATs."""

from __future__ import annotations

import math

from test_harness.generators.pcap_gen import timestamp_reversal_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "P-07"


def run(n_packets: int = 500) -> dict:
    dirs = for_test(TEST_ID)
    pcap = timestamp_reversal_pcap(dirs["artifacts"] / "ts_reversed.pcap",
                                   n_packets=n_packets)
    result = replay(str(pcap))

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "Backwards timestamps",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "No NaN/Inf/negative IATs produced",
                "actual": result.get("skip_reason"),
                "pcap": str(pcap), "raw": result}

    bad: list[str] = []
    for r in result.get("results", []):
        for k, v in r.items():
            if isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    bad.append(f"{k} non-finite ({v})")
                if k.endswith("iat_min") and isinstance(v, (int, float)) and v < 0:
                    bad.append(f"{k} negative ({v})")

    errors = result.get("errors", [])
    if bad:
        verdict = "FAIL"
        actual  = f"{len(bad)} numeric anomalies; first: {bad[0]}"
    elif errors:
        # Errors during processing are themselves a failure of robustness.
        verdict = "FAIL"
        actual  = f"{len(errors)} per-packet errors; first: {errors[0]}"
    else:
        verdict = "PASS"
        actual  = f"{result.get('n_results', 0)} flows produced; no anomalies"
    return {"test_id": TEST_ID, "name": "Backwards timestamps",
            "severity": "HIGH", "verdict": verdict,
            "expected": "No NaN/Inf/negative IATs; no per-packet errors",
            "actual": actual, "pcap": str(pcap), "raw": {"errors_sample": errors[:5],
                                                         "n_results": result.get("n_results", 0)}}
