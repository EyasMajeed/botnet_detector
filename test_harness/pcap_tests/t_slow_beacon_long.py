"""
RT-1b Slow-C2 evasion — sessions slower than the idle timeout.

Sessions are spaced LONGER than the project's FlowAggregator._idle
threshold. This is the genuine "slow beaconing evades the detector"
threat: even after the team raised _idle from 30 s to 120 s, a real
adversary can simply beacon every 5 minutes.

This test is informative either way:
  - If it FAILS, the team needs cross-flow temporal aggregation per
    src_ip (the LSTM's per-flow input window doesn't see across flows).
  - If it PASSES, the model has learned to flag short flows in
    isolation as botnet — a useful capability that should be documented.

Default 180 s spacing gives _idle (120 s) + 60 s headroom, so each
session DOES close as its own flow. Increase session_period_sec to
test against slower C2 cadences (300 s, 600 s).
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import multi_session_beacon_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "RT-1b"


def run(n_sessions: int = 25, pkts_per_session: int = 4,
        session_period_sec: float = 180.0, intra_session_gap: float = 0.5,
        threshold_pass: float = 0.3) -> dict:
    """
    Lower default threshold (0.3) — this is a hard test; failing
    against a determined slow-C2 attacker is expected. The test exists
    to surface the gap, not to hold the project to a high bar here.
    """
    dirs = for_test(TEST_ID)
    pcap = multi_session_beacon_pcap(
        dirs["artifacts"] / "slow_c2_beacon.pcap",
        n_sessions=n_sessions, pkts_per_session=pkts_per_session,
        session_period_sec=session_period_sec,
        intra_session_gap=intra_session_gap,
    )
    result = replay(str(pcap),
                    max_packets=n_sessions * pkts_per_session + 100)

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "Slow-C2 evasion (period > idle)",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": f"detection rate ≥ {threshold_pass:.0%}",
                "actual":   result.get("skip_reason"),
                "pcap":     str(pcap), "raw": result}

    results = result.get("results", [])
    n_botnet = sum(1 for r in results if r.get("label") == "botnet")
    n_total  = len(results)
    n_proc   = result.get("n_packets_processed", 0)

    if n_total == 0:
        return {"test_id": TEST_ID, "name": "Slow-C2 evasion (period > idle)",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "≥1 completed flow",
                "actual":   f"0 results from {n_proc} processed packets",
                "pcap":     str(pcap)}

    rate = n_botnet / n_total
    if rate >= threshold_pass:
        verdict, actual = "PASS", (
            f"{n_botnet}/{n_total} ({rate:.0%}) flagged botnet "
            f"despite {session_period_sec}s session spacing")
    else:
        verdict, actual = "FAIL", (
            f"only {n_botnet}/{n_total} ({rate:.0%}) flagged botnet — "
            "slow-C2 evasion at period > idle is effective. "
            "Detection requires cross-session temporal aggregation per "
            "src_ip; the LSTM's per-flow input window cannot see across "
            "flows.")
    return {"test_id": TEST_ID, "name": "Slow-C2 evasion (period > idle)",
            "severity": "HIGH", "verdict": verdict,
            "expected": f"detection rate ≥ {threshold_pass:.0%}",
            "actual":   actual, "pcap": str(pcap),
            "raw": {"n_results": n_total, "n_botnet": n_botnet,
                    "rate": round(rate, 3),
                    "session_period_sec": session_period_sec,
                    "first_5_labels":
                        [r.get("label") for r in results[:5]]}}
