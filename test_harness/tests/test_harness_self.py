"""
Self-tests for the harness. Verify the framework works END-TO-END
without needing the project's models. Each test is small and fast.

Run with: python -m test_harness.tests.test_harness_self
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ok(msg: str) -> None: print(f"[PASS] {msg}")
def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def test_paths():
    from test_harness.utils import paths
    assert paths.HARNESS_ROOT.exists()
    assert paths.LOGS_DIR.exists()
    _ok("paths module")


def test_generators_pcap():
    try:
        import scapy  # noqa: F401
    except ImportError:
        print("[SKIP] scapy not installed")
        return
    from test_harness.generators.pcap_gen import (
        benign_tcp_pcap, ipv6_only_pcap, vlan_tagged_pcap,
        gre_tunneled_pcap, syn_flood_pcap, slow_beacon_pcap,
        timestamp_reversal_pcap, malformed_eth_pcap, fake_magic_pcap,
        pcapng_truncated, spoofed_oui_pcap, cardinality_explosion_pcap,
    )
    tmp = Path("/tmp/harness_self_pcaps"); tmp.mkdir(exist_ok=True)
    for fn, kw in [(benign_tcp_pcap, {"n_flows": 3, "pkts_per_flow": 4}),
                   (ipv6_only_pcap, {"n_flows": 3}),
                   (vlan_tagged_pcap, {"n_packets": 10}),
                   (gre_tunneled_pcap, {"n_packets": 10}),
                   (syn_flood_pcap, {"n_packets": 100}),
                   (slow_beacon_pcap, {"n_beacons": 5, "period_sec": 1.0}),
                   (timestamp_reversal_pcap, {"n_packets": 30}),
                   (malformed_eth_pcap, {"n_packets": 6}),
                   (fake_magic_pcap, {}),
                   (pcapng_truncated, {}),
                   (spoofed_oui_pcap, {"n_packets": 10}),
                   (cardinality_explosion_pcap, {"n_destinations": 50})]:
        out = tmp / f"{fn.__name__}.pcap"
        p = fn(out, **kw) if "out" not in kw else fn(**kw)
        assert Path(p).exists() and Path(p).stat().st_size > 0, fn.__name__
    _ok("pcap generators")


def test_generators_csv():
    from test_harness.generators.flow_csv_gen import (
        S1_FEATURES, synthetic_csv, slow_beacon_csv,
    )
    assert len(S1_FEATURES) == 56
    tmp = Path("/tmp/harness_self_csvs"); tmp.mkdir(exist_ok=True)
    p = synthetic_csv(tmp / "mix.csv", n_iot=5, n_noniot=5, n_botnet=5)
    assert p.exists() and p.stat().st_size > 0
    p2 = slow_beacon_csv(tmp / "beacon.csv", n_beacons=5)
    assert p2.exists() and p2.stat().st_size > 0
    _ok("csv generators")


def test_generators_corruption():
    from test_harness.generators.file_corruption import (
        oversized_csv, header_only_csv, fuzz_random_bytes,
        utf16_bom_csv, zip_bomb_disguised_as_pcap,
        claimed_huge_pcap_header,
    )
    tmp = Path("/tmp/harness_self_corrupt"); tmp.mkdir(exist_ok=True)
    for fn, kw in [(oversized_csv, {"target_mb": 1}),
                   (header_only_csv, {}),
                   (fuzz_random_bytes, {"size_bytes": 256}),
                   (utf16_bom_csv, {"n_rows": 5}),
                   (zip_bomb_disguised_as_pcap, {}),
                   (claimed_huge_pcap_header, {})]:
        p = fn(tmp / f"{fn.__name__}.bin", **kw)
        assert Path(p).exists() and Path(p).stat().st_size > 0, fn.__name__
    _ok("corruption generators")


def test_isolation_basic():
    """Spawn a harmless dummy target and verify result round-trip.

    The harness's _payload.json -> runner_subprocess.py path imports the
    target by dotted name. We use a real module that ships with the
    harness so the child can find it via PYTHONPATH (set by isolation).
    """
    from test_harness.utils.isolation import run_isolated
    iso = run_isolated(
        test_id="SELF-001",
        target="test_harness.tests._dummy_target:hi",
        kwargs={"name": "harness"}, timeout_sec=15.0,
        monitor_resources=True,
    )
    assert iso.return_code == 0, iso
    assert iso.target_result.get("verdict") == "PASS", iso.target_result
    _ok("isolation round-trip")


def test_isolation_timeout():
    """Verify timeout actually kills the child."""
    from test_harness.utils.isolation import run_isolated
    iso = run_isolated(
        test_id="SELF-002",
        target="test_harness.tests._dummy_target:loop",
        kwargs={}, timeout_sec=2.0,
        monitor_resources=False,
    )
    assert iso.timed_out, f"expected timed_out=True, got {iso}"
    _ok("isolation timeout enforcement")


def test_isolation_crash():
    """Verify a target raising still produces a structured error record."""
    from test_harness.utils.isolation import run_isolated
    iso = run_isolated(
        test_id="SELF-003",
        target="test_harness.tests._dummy_target:boom",
        kwargs={}, timeout_sec=10.0,
        monitor_resources=False,
    )
    assert iso.crashed or iso.return_code != 0, iso
    assert "_traceback" in (iso.target_result or {}), iso.target_result
    _ok("isolation crash capture")


def test_metrics():
    from test_harness.ml_tests._metrics import metrics, threshold_sweep
    y_true  = [1, 0, 1, 0, 1, 0, 1, 0]
    y_pred  = [1, 0, 1, 0, 1, 1, 0, 0]
    y_score = [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.4, 0.3]
    m = metrics(y_true, y_pred, y_score)
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"]    <= 1
    assert 0 <= m["f1"]        <= 1
    sweep = threshold_sweep(y_true, y_score, [0.3, 0.5, 0.7])
    assert len(sweep) == 3
    _ok("metrics module")


def test_reports_with_synthetic_entries():
    """Run the report generator on hand-built entries — no project needed."""
    from test_harness.reports.generate import write_all
    import tempfile
    rd = Path(tempfile.mkdtemp(prefix="harness_report_"))
    entries = [
        {"test_id": "X-001",
         "spec":   {"test_id":"X-001","name":"demo","phase":"A",
                    "severity":"HIGH","target":"x:y","timeout_sec":30,
                    "kwargs":{},"requires_models":False,"requires_scapy":False,
                    "requires_xai":False,"requires_qt":False,
                    "monitor_resources":True,"notes":""},
         "result": {"test_id":"X-001","name":"demo","severity":"HIGH",
                    "verdict":"PASS","expected":"ok","actual":"ok"},
         "isolated":{"return_code":0,"timed_out":False,"crashed":False,
                     "duration_sec":1.0,"stdout_path":None,"stderr_path":None,
                     "resources":{"peak_cpu_pct":12,"peak_rss_mb":40}}},
        {"test_id": "X-002",
         "spec":   {"test_id":"X-002","name":"demo2","phase":"B",
                    "severity":"CRITICAL","target":"x:y","timeout_sec":30,
                    "kwargs":{},"requires_models":False,"requires_scapy":False,
                    "requires_xai":False,"requires_qt":False,
                    "monitor_resources":True,"notes":""},
         "result": {"test_id":"X-002","name":"demo2","severity":"CRITICAL",
                    "verdict":"FAIL","expected":"ok","actual":"problem"},
         "isolated":{"return_code":0,"timed_out":False,"crashed":False,
                     "duration_sec":2.0,"stdout_path":None,"stderr_path":None,
                     "resources":{"peak_cpu_pct":50,"peak_rss_mb":200}}},
    ]
    paths = write_all(rd, entries)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0, p
    _ok("report generators")


def main():
    test_paths()
    test_generators_pcap()
    test_generators_csv()
    test_generators_corruption()
    test_isolation_basic()
    test_isolation_timeout()
    test_isolation_crash()
    test_metrics()
    test_reports_with_synthetic_entries()
    print()
    print("All harness self-tests passed.")


if __name__ == "__main__":
    main()
