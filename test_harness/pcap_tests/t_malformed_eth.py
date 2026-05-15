"""P-02 Malformed Ethernet frames — must not crash the replay."""

from __future__ import annotations

from test_harness.generators.pcap_gen import malformed_eth_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "P-02"


def run(n_packets: int = 50) -> dict:
    dirs = for_test(TEST_ID)
    pcap = malformed_eth_pcap(dirs["artifacts"] / "malformed_eth.pcap",
                              n_packets=n_packets)
    result = replay(str(pcap))

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "Malformed Ethernet frames",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "Bad frames skipped without crash",
                "actual": result.get("skip_reason"),
                "pcap": str(pcap), "raw": result}

    errors = result.get("errors", [])
    crashed = any("Traceback" in e for e in errors)
    seen = result.get("n_packets_seen", 0)
    proc = result.get("n_packets_processed", 0)
    if crashed:
        verdict = "FAIL"
        actual  = f"replay crashed on a malformed packet: {errors[0][:120]}"
    else:
        verdict = "PASS"
        actual  = (f"{proc}/{seen} packets processed; "
                   f"{len(errors)} per-packet errors logged")
    return {"test_id": TEST_ID, "name": "Malformed Ethernet frames",
            "severity": "HIGH", "verdict": verdict,
            "expected": "Replay survives; per-packet exceptions logged but contained",
            "actual": actual, "pcap": str(pcap),
            "raw": {"errors_sample": errors[:3]}}
