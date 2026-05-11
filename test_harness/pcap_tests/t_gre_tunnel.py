"""P-05 GRE-tunneled inner IP."""

from __future__ import annotations

from test_harness.generators.pcap_gen import gre_tunneled_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "P-05"


def run(n_packets: int = 100) -> dict:
    dirs = for_test(TEST_ID)
    pcap = gre_tunneled_pcap(dirs["artifacts"] / "gre.pcap", n_packets=n_packets)
    result = replay(str(pcap))

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "GRE tunnel handling",
                "severity": "MEDIUM", "verdict": "SKIPPED",
                "expected": "GRE inner IP processed or explicit log",
                "actual":   result.get("skip_reason"),
                "pcap": str(pcap), "raw": result}

    proc = result.get("n_packets_processed", 0)
    seen = result.get("n_packets_seen", 0)
    if proc == 0:
        verdict, actual = "FAIL", "GRE-encapsulated traffic produced 0 processed packets"
    else:
        verdict, actual = "PASS", f"{proc}/{seen} packets processed"
    return {"test_id": TEST_ID, "name": "GRE tunnel handling",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": "GRE traffic produces non-zero processed packets",
            "actual": actual, "pcap": str(pcap), "raw": result}
