"""
P-03 VLAN-tagged frames — verifies handling of 802.1Q.
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import vlan_tagged_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "P-03"


def run(n_packets: int = 200, vlan_id: int = 100) -> dict:
    dirs = for_test(TEST_ID)
    pcap = vlan_tagged_pcap(dirs["artifacts"] / "vlan_tagged.pcap",
                            n_packets=n_packets, vlan_id=vlan_id)
    result = replay(str(pcap))

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "VLAN-tagged frames",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "VLAN packets reach process_packet",
                "actual":   result.get("skip_reason"),
                "pcap": str(pcap), "raw": result}

    proc = result.get("n_packets_processed", 0)
    seen = result.get("n_packets_seen", 0)
    # Pass if at least 80% of VLAN packets reached the monitor's process_packet.
    pass_threshold = 0.80
    ratio = (proc / seen) if seen else 0.0
    if ratio >= pass_threshold:
        verdict, actual = "PASS", f"{proc}/{seen} ({ratio:.1%}) processed"
    else:
        verdict, actual = "FAIL", (f"only {proc}/{seen} ({ratio:.1%}) processed - "
                                   "VLAN handling likely missing")
    return {
        "test_id":  TEST_ID,
        "name":     "VLAN-tagged frames",
        "severity": "HIGH",
        "verdict":  verdict,
        "expected": f"≥{pass_threshold:.0%} of VLAN packets reach process_packet",
        "actual":   actual,
        "pcap":     str(pcap),
        "raw":      result,
    }
