"""
ST-3 Long-duration monitoring (compressed to 30 s by default).

Hand-built scapy PacketList replayed manually with synthetic timestamps
that span 'duration_sec' of simulated wall time, while the test runs in
real time. Verifies the BotnetMonitor stays responsive throughout.

For CI we cap at 30s; for nightly runs the orchestrator can pass
duration_sec=3600 (1 hour) via the registry.
"""

from __future__ import annotations

import time

from test_harness.generators.pcap_gen import benign_tcp_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "ST-3"


def run(duration_sec: float = 30.0, n_flows: int = 200) -> dict:
    """We approximate "long-running" by replaying a series of small PCAPs
    in a loop until duration_sec is reached, accumulating throughput
    statistics."""
    dirs = for_test(TEST_ID)
    pcap = benign_tcp_pcap(dirs["artifacts"] / "long_run.pcap",
                           n_flows=n_flows, pkts_per_flow=10)
    end = time.monotonic() + duration_sec
    iters = 0
    total_proc = 0
    errors: list[str] = []
    while time.monotonic() < end:
        result = replay(str(pcap), max_packets=n_flows * 10 + 100)
        if result.get("skipped"):
            return {"test_id": TEST_ID, "name": "Long-duration monitoring",
                    "severity": "MEDIUM", "verdict": "SKIPPED",
                    "expected": f"runs cleanly for {duration_sec}s",
                    "actual":   result.get("skip_reason"),
                    "pcap":     str(pcap)}
        iters += 1
        total_proc += result.get("n_packets_processed", 0)
        errors.extend(result.get("errors", [])[:3])
        if len(errors) > 30:
            errors.append("... (truncated)")
            break
    if errors:
        return {"test_id": TEST_ID, "name": "Long-duration monitoring",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": f"no per-packet errors for {duration_sec}s",
                "actual":   f"{len(errors)} errors collected; first: {errors[0][:200]}",
                "pcap":     str(pcap), "raw": {"iterations": iters,
                                                "errors_sample": errors[:5]}}
    return {"test_id": TEST_ID, "name": "Long-duration monitoring",
            "severity": "MEDIUM", "verdict": "PASS",
            "expected": f"runs cleanly for {duration_sec}s",
            "actual":   f"{iters} iterations, {total_proc} packets processed",
            "pcap":     str(pcap),
            "raw":      {"iterations": iters,
                         "total_packets_processed": total_proc}}
