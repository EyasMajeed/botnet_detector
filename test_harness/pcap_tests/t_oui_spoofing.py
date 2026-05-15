"""
S1-06 MAC-OUI spoofing → Stage-1 override exploitability.

A workstation-cadence flow from an Espressif (IoT-vendor) OUI should be
routed to the IoT branch by the OUI override even though its behavioural
profile is non-IoT. We measure the routing rate.
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import spoofed_oui_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "S1-06"


def run(n_packets: int = 200) -> dict:
    dirs = for_test(TEST_ID)
    pcap = spoofed_oui_pcap(dirs["artifacts"] / "spoofed_oui.pcap",
                            n_packets=n_packets)
    result = replay(str(pcap))
    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "MAC-OUI spoofing override",
                "severity": "CRITICAL", "verdict": "SKIPPED",
                "expected": "OUI override is gated by behaviour features",
                "actual":   result.get("skip_reason"),
                "pcap": str(pcap)}
    results = result.get("results", [])
    routed_iot = sum(1 for r in results if r.get("device_type") == "iot")
    total = len(results)
    n_proc = result.get("n_packets_processed", 0)
    n_seen = result.get("n_packets_seen", 0)
    errors = result.get("errors", [])

    # Guard against false PASS: if very few flows produced results, the
    # routing-rate denominator is too small to draw any conclusion. We
    # need at least 5 flow results from a 200-packet PCAP.
    MIN_RESULTS = 5
    if total < MIN_RESULTS:
        return {"test_id": TEST_ID, "name": "MAC-OUI spoofing override",
                "severity": "CRITICAL", "verdict": "SKIPPED",
                "expected": f"≥{MIN_RESULTS} flow results to draw a conclusion",
                "actual":   (f"only {total} result(s) from {n_seen} packets "
                             f"({n_proc} processed, "
                             f"{len(errors)} per-packet errors). "
                             "Insufficient data — fix replay first."),
                "pcap": str(pcap),
                "raw": {"n_seen": n_seen, "n_processed": n_proc,
                        "n_results": total,
                        "errors_sample": errors[:3]}}

    iot_rate = routed_iot / total
    # We FAIL if the OUI override blindly forces IoT routing despite
    # workstation cadence (i.e. iot_rate is high). PASS if low (override
    # is gated, or OUI module not active and ML correctly classified).
    if iot_rate >= 0.5:
        verdict, actual = "FAIL", (
            f"{iot_rate:.0%} of spoofed-MAC flows routed to IoT branch "
            "without behavioural verification — override is exploitable.")
    else:
        verdict, actual = "PASS", (
            f"only {iot_rate:.0%} ({routed_iot}/{total}) routed to IoT; "
            "override appears gated")
    return {"test_id": TEST_ID, "name": "MAC-OUI spoofing override",
            "severity": "CRITICAL", "verdict": verdict,
            "expected": "iot routing rate < 50% on workstation-cadence flows",
            "actual":   actual, "pcap": str(pcap),
            "raw": {"n_results": total, "routed_iot": routed_iot,
                    "first_5": [{"device": r.get("device_type"),
                                 "label":  r.get("label")} for r in results[:5]]}}
