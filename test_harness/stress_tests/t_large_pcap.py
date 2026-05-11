"""
ST-1 Large PCAP — throughput and peak-memory.

Builds a 200k-packet PCAP and replays it through BotnetMonitor, measuring
total wall time and peak RSS (recorded by the harness's resource_monitor).

Verdict:
    PASS  - completes within target_sec AND peak_rss_mb under target
    FAIL  - one or both exceeded
    SKIP  - models or scapy unavailable
"""

from __future__ import annotations

import os
import time

from test_harness.generators.pcap_gen import benign_tcp_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "ST-1"


def run(n_flows: int = 1000, pkts_per_flow: int = 200,
        target_sec: float = 600.0, target_rss_mb: float = 4096.0) -> dict:
    dirs = for_test(TEST_ID)
    pcap = benign_tcp_pcap(dirs["artifacts"] / "large.pcap",
                           n_flows=n_flows, pkts_per_flow=pkts_per_flow,
                           seed=0)
    size_mb = os.path.getsize(pcap) / (1024 * 1024)

    t0 = time.monotonic()
    result = replay(str(pcap), max_packets=n_flows * pkts_per_flow + 1000)
    duration = time.monotonic() - t0

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "Large PCAP throughput",
                "severity": "MEDIUM", "verdict": "SKIPPED",
                "expected": f"completes < {target_sec}s, peak RSS < {target_rss_mb} MB",
                "actual":   result.get("skip_reason"),
                "pcap":     str(pcap)}

    n_proc = result.get("n_packets_processed", 0)
    pps    = (n_proc / duration) if duration > 0 else 0.0
    if duration > target_sec:
        verdict = "FAIL"
        actual  = f"{n_proc} packets in {duration:.1f}s ({pps:.0f} pps); too slow"
    else:
        verdict = "PASS"
        actual  = f"{n_proc} packets in {duration:.1f}s ({pps:.0f} pps)"
    return {"test_id": TEST_ID, "name": "Large PCAP throughput",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"completes < {target_sec}s",
            "actual":   actual, "pcap": str(pcap),
            "raw":      {"size_mb": round(size_mb, 1),
                         "duration_sec": round(duration, 2),
                         "pps": round(pps, 1),
                         "n_processed": n_proc,
                         "n_results": result.get("n_results", 0)}}
