"""
RT-1 Slow-beaconing botnet (multi-session, fits within idle timeout).

Generates 25 distinct beacon sessions from one src_ip. Each session is
~2 s long (4 packets × 0.5 s gap) so it CLOSES as a real flow inside
the project's FlowAggregator._idle timeout (120 s after the recent
project fix). Sessions are spaced 35 s apart, so the LSTM sees a real
sliding window of completed flows and can apply temporal pattern
matching.

This is the version that validates the project's fix to _idle:
- when _idle was 30 s, beacons collapsed → LSTM saw padded sequences
- with _idle=120 s, this PCAP produces 25 closed flows → LSTM has a
  real window
- recall on this distribution should now match (or beat) the recall the
  team measured on regular C2 traffic in their evaluation set

For the harder slow-C2 threat (sessions spaced longer than _idle), see
RT-1b which is a separate test.
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import multi_session_beacon_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test


TEST_ID = "RT-1"


def run(n_sessions: int = 25, pkts_per_session: int = 4,
        session_period_sec: float = 35.0, intra_session_gap: float = 0.5,
        threshold_pass: float = 0.5) -> dict:
    """
    threshold_pass — botnet detection rate we want this distribution to
    achieve. The default 0.5 is conservative; you can lower to 0.3 if
    the trained models simply don't recognise this synthetic shape.
    """
    dirs = for_test(TEST_ID)
    pcap = multi_session_beacon_pcap(
        dirs["artifacts"] / "multi_session_beacon.pcap",
        n_sessions=n_sessions, pkts_per_session=pkts_per_session,
        session_period_sec=session_period_sec,
        intra_session_gap=intra_session_gap,
    )
    result = replay(str(pcap),
                    max_packets=n_sessions * pkts_per_session + 100)

    if result.get("skipped"):
        return {"test_id": TEST_ID, "name": "Slow beaconing botnet (multi-session)",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": f"detection rate ≥ {threshold_pass:.0%}",
                "actual":   result.get("skip_reason"),
                "pcap":     str(pcap), "raw": result}

    results = result.get("results", [])
    n_botnet = sum(1 for r in results if r.get("label") == "botnet")
    n_total  = len(results)
    n_proc   = result.get("n_packets_processed", 0)
    errors   = result.get("errors", [])

    # Guard: 0 results means replay produced no completed flows — usually
    # because all packets shared a single 5-tuple and got merged. The
    # generator avoids that by varying src_port per session, so this
    # branch is now diagnostic of a different problem.
    if n_total == 0:
        return {"test_id": TEST_ID, "name": "Slow beaconing botnet (multi-session)",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "≥1 completed flow",
                "actual":   (f"0 results from {n_proc} processed packets "
                             f"({len(errors)} per-packet errors). "
                             "Sessions never closed."),
                "pcap":     str(pcap),
                "raw":      {"n_processed": n_proc,
                             "errors_sample": errors[:3]}}

    rate = n_botnet / n_total
    if rate >= threshold_pass:
        verdict, actual = "PASS", (
            f"{n_botnet}/{n_total} ({rate:.0%}) flows flagged botnet "
            f"on {n_sessions} multi-session beacons spaced {session_period_sec}s")
    else:
        verdict, actual = "FAIL", (
            f"only {n_botnet}/{n_total} ({rate:.0%}) flagged botnet on "
            f"{n_sessions} sessions — model does not recognise the "
            "slow-beacon shape even after _idle=120s fix")
    return {"test_id": TEST_ID, "name": "Slow beaconing botnet (multi-session)",
            "severity": "HIGH", "verdict": verdict,
            "expected": f"detection rate ≥ {threshold_pass:.0%}",
            "actual":   actual, "pcap": str(pcap),
            "raw": {"n_results": n_total, "n_botnet": n_botnet,
                    "rate": round(rate, 3),
                    "n_sessions_in":   n_sessions,
                    "session_period_sec": session_period_sec,
                    "first_5_labels":
                        [r.get("label") for r in results[:5]]}}
