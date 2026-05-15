"""
SE-A SYN flood — confirms the suspicion-scorer high-pps + SYN-no-ACK rule
fires AND, indirectly, that BotnetMonitor's per-packet path keeps up.

Note: app/suspicion_scorer is not currently invoked from the PCAP/file
path. We therefore call SuspicionScorer.score directly on a synthesised
flow dict that matches the PCAP, AND we replay the PCAP through the
monitor to catch any throughput collapse. The test is a FAIL if either
branch breaks.
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import syn_flood_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import


TEST_ID = "SE-A"


def run(n_packets: int = 5000) -> dict:
    dirs = for_test(TEST_ID)
    pcap = syn_flood_pcap(dirs["artifacts"] / "syn_flood.pcap",
                          n_packets=n_packets)

    # ── 1. Direct call to SuspicionScorer with a flow that mirrors the PCAP.
    suspicion_state: dict = {"available": False}
    susp_mod, err = soft_import("app.suspicion_scorer")
    if susp_mod is not None:
        try:
            scorer = susp_mod.SuspicionScorer()
            res = scorer.score({
                "flow_pkts_per_sec":  10_000.0,
                "flow_bytes_per_sec": 600_000.0,
                "flow_duration":      0.5,
                "dst_port":           23,
                "flag_SYN":           5_000,
                "flag_ACK":           0,
                "total_fwd_bytes":    300_000,
                "total_bwd_bytes":    0,
            })
            suspicion_state = {
                "available":     True,
                "score":         res.get("score"),
                "trigger_sniff": res.get("trigger_sniff"),
                "reasons":       res.get("reasons", [])[:5],
            }
        except Exception as e:                                   # noqa: BLE001
            suspicion_state = {"available": True,
                               "error": f"{type(e).__name__}: {e}"}

    # ── 2. Replay through the BotnetMonitor (catches throughput collapse).
    replay_result = replay(str(pcap))

    # ── 3. Verdict
    susp_ok = (
        suspicion_state.get("available")
        and isinstance(suspicion_state.get("score"), int)
        and suspicion_state["score"] >= 2
    )
    replay_ok = (
        replay_result.get("skipped")  # acceptable when models missing
        or replay_result.get("n_packets_processed", 0) >= int(n_packets * 0.5)
    )
    if susp_ok and replay_ok:
        verdict, actual = "PASS", (
            f"scorer.score={suspicion_state.get('score')}, "
            f"trigger_sniff={suspicion_state.get('trigger_sniff')}; "
            f"replay processed "
            f"{replay_result.get('n_packets_processed', 0)} packets")
    else:
        verdict = "FAIL"
        actual = (f"susp_ok={susp_ok} ({suspicion_state}); "
                  f"replay_ok={replay_ok} "
                  f"(processed={replay_result.get('n_packets_processed', 0)})")

    return {"test_id": TEST_ID, "name": "SYN flood",
            "severity": "HIGH", "verdict": verdict,
            "expected": "scorer score>=2 AND replay processes >=50% of packets",
            "actual": actual, "pcap": str(pcap),
            "raw": {"suspicion": suspicion_state,
                    "replay": {k: v for k, v in replay_result.items()
                               if k != "results"}}}
