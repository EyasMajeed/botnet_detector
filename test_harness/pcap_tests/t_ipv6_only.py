"""
P-04 IPv6-only PCAP — verifies whether IPv6 traffic is silently dropped.

Expected (target behaviour): pipeline either processes v6 flows or
explicitly logs and skips them. Today, src/live/live_detector
silently drops v6 packets, and we expect to find 0 results from a
pure-v6 capture, which is the failure we want to surface.
"""

from __future__ import annotations

from pathlib import Path

from test_harness.generators.pcap_gen import ipv6_only_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "P-04"


def run(out_dir: str | None = None, n_flows: int = 30) -> dict:
    dirs = for_test(TEST_ID)
    pcap = ipv6_only_pcap(dirs["artifacts"] / "ipv6_only.pcap", n_flows=n_flows)

    result = replay(str(pcap), max_packets=10_000)

    n_results = result.get("n_results", 0)
    n_seen    = result.get("n_packets_seen", 0)

    expected = "IPv6 flows produce >=1 result OR explicit log of unsupported"
    if result.get("skipped"):
        verdict = "SKIPPED"
        actual  = result.get("skip_reason", "skipped")
    elif n_results == 0:
        verdict = "FAIL"
        actual  = (f"All IPv6 packets silently dropped: seen={n_seen}, "
                   "results=0. Confirms the IPv6 blind spot.")
    else:
        verdict = "PASS"
        actual  = f"{n_results} IPv6 flow(s) produced results"

    return {
        "test_id":  TEST_ID,
        "name":     "IPv6-only flows",
        "severity": "HIGH",
        "verdict":  verdict,
        "expected": expected,
        "actual":   actual,
        "pcap":     str(pcap),
        "raw":      result,
    }
