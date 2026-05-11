"""
configs/registry.py — Authoritative catalogue of harness tests.

Plain Python (not YAML) so the registry can stay typed and self-validating
without depending on PyYAML. Each entry declares everything the
orchestrator needs to run a single test.

Add a new test by:
    1. Implement run() in <package>/t_*.py returning the result schema.
    2. Append a TestSpec(...) to TESTS below.

Phases (from the project plan):
    A — core stability / parser safety / packet-handling robustness
    B — ML behaviour
    C — adversarial / security
    D — GUI / long-run / stress
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestSpec:
    test_id: str
    name:    str
    target:  str             # "module.path:function_name"
    phase:   str             # "A" | "B" | "C" | "D"
    severity: str            # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    timeout_sec: float = 60.0
    kwargs:  dict[str, Any] = field(default_factory=dict)
    requires_models: bool   = True
    requires_scapy:  bool   = True
    requires_xai:    bool   = False
    requires_qt:     bool   = False
    monitor_resources: bool = True
    notes:   str = ""


TESTS: list[TestSpec] = [
    # ── PHASE A — core stability ─────────────────────────────────────
    TestSpec("M-00",  "Schema drift",            "test_harness.ml_tests.t_schema_drift:run",
             "A", "CRITICAL", timeout_sec=30, requires_models=False, requires_scapy=False),
    TestSpec("F-01",  "Truncated PCAPNG",        "test_harness.parsers.t_file_handler_safety:t_pcapng_truncated",
             "A", "MEDIUM", timeout_sec=30, requires_models=False, requires_scapy=False),
    TestSpec("F-02",  "Claimed-huge PCAP",       "test_harness.parsers.t_file_handler_safety:t_pcap_huge_claimed",
             "A", "HIGH",   timeout_sec=30, requires_models=False, requires_scapy=False),
    TestSpec("F-03",  "Oversized CSV",           "test_harness.parsers.t_file_handler_safety:t_csv_oversized",
             "A", "MEDIUM", timeout_sec=120, requires_models=False, requires_scapy=False,
             kwargs={}),
    TestSpec("F-04",  "Header-only CSV",         "test_harness.parsers.t_file_handler_safety:t_csv_header_only",
             "A", "MEDIUM", timeout_sec=15, requires_models=False, requires_scapy=False),
    TestSpec("F-05",  "PCAP renamed .csv",       "test_harness.parsers.t_file_handler_safety:t_pcap_renamed_csv",
             "A", "LOW",    timeout_sec=15, requires_models=False),
    TestSpec("F-07",  "UTF-16 BOM CSV",          "test_harness.parsers.t_file_handler_safety:t_csv_utf16",
             "A", "HIGH",   timeout_sec=15, requires_models=False, requires_scapy=False),
    TestSpec("F-08",  "Random-bytes fuzz x32",   "test_harness.parsers.t_file_handler_safety:t_random_bytes_fuzz",
             "A", "MEDIUM", timeout_sec=120, requires_models=False, requires_scapy=False),
    TestSpec("F-09",  "Zip bomb as pcap",        "test_harness.parsers.t_file_handler_safety:t_zip_bomb",
             "A", "MEDIUM", timeout_sec=30, requires_models=False, requires_scapy=False),
    TestSpec("P-02",  "Malformed Ethernet",      "test_harness.pcap_tests.t_malformed_eth:run",
             "A", "HIGH",   timeout_sec=120),
    TestSpec("P-03",  "VLAN-tagged frames",      "test_harness.pcap_tests.t_vlan_tagged:run",
             "A", "HIGH",   timeout_sec=120),
    TestSpec("P-04",  "IPv6-only flows",         "test_harness.pcap_tests.t_ipv6_only:run",
             "A", "HIGH",   timeout_sec=120),
    TestSpec("P-07",  "Backwards timestamps",    "test_harness.pcap_tests.t_timestamp_reversal:run",
             "A", "HIGH",   timeout_sec=120),

    # ── PHASE B — ML validation ──────────────────────────────────────
    TestSpec("M-T1",  "Threshold sweep (synthetic)",  "test_harness.ml_tests.t_threshold_sweep:run",
             "B", "MEDIUM", timeout_sec=180),
    TestSpec("M-EF",  "Empty-flow regression",   "test_harness.ml_tests.t_empty_flow_regression:run",
             "B", "HIGH",   timeout_sec=120),
    TestSpec("M-WS",  "Window starvation",       "test_harness.ml_tests.t_window_starvation:run",
             "B", "HIGH",   timeout_sec=180),
    TestSpec("M-CAL", "Confidence calibration",  "test_harness.ml_tests.t_calibration:run",
             "B", "MEDIUM", timeout_sec=240),

    # ── PHASE C — adversarial & security ────────────────────────────
    TestSpec("P-05",  "GRE tunnel handling",     "test_harness.pcap_tests.t_gre_tunnel:run",
             "C", "MEDIUM", timeout_sec=120),
    TestSpec("S1-06", "MAC-OUI spoofing",        "test_harness.pcap_tests.t_oui_spoofing:run",
             "C", "CRITICAL", timeout_sec=120),
    TestSpec("SE-A",  "SYN flood",               "test_harness.pcap_tests.t_syn_flood:run",
             "C", "HIGH",   timeout_sec=180),
    TestSpec("SC-K",  "Kitsune key explosion",   "test_harness.pcap_tests.t_kitsune_explosion:run",
             "C", "HIGH",   timeout_sec=420),
    TestSpec("RT-1",  "Slow beaconing botnet",   "test_harness.pcap_tests.t_slow_beacon:run",
             "C", "HIGH",   timeout_sec=180),
    TestSpec("RT-1b", "Slow-C2 evasion (period > idle)",
             "test_harness.pcap_tests.t_slow_beacon_long:run",
             "C", "HIGH",   timeout_sec=240),
    TestSpec("M-ADV", "Scaler-aware FGSM",       "test_harness.ml_tests.t_adversarial_fgsm:run",
             "C", "HIGH",   timeout_sec=600, kwargs={"n_samples": 40,
                                                     "epsilons": (0.001, 0.005, 0.01)}),
    TestSpec("X-A1",  "XAI sanity battery",      "test_harness.xai_tests.t_xai_sanity:run",
             "C", "HIGH",   timeout_sec=240, requires_xai=True),

    # ── PHASE D — GUI & long-run ────────────────────────────────────
    TestSpec("G-DS",  "DetectionStore behaviour", "test_harness.gui_tests.t_detection_store:run",
             "D", "MEDIUM", timeout_sec=60, requires_models=False, requires_scapy=False,
             requires_qt=True),
    TestSpec("G-AF",  "Alert flood",              "test_harness.gui_tests.t_alert_flood:run",
             "D", "MEDIUM", timeout_sec=120, requires_models=False, requires_scapy=False,
             requires_qt=True),
    TestSpec("G-IT",  "PcapInferenceThread",      "test_harness.gui_tests.t_inference_thread:run",
             "D", "HIGH",   timeout_sec=120, requires_qt=True),
    TestSpec("G-RU",  "Repeated upload cycles",   "test_harness.gui_tests.t_repeated_upload:run",
             "D", "MEDIUM", timeout_sec=300),
    TestSpec("ST-1",  "Large PCAP throughput",    "test_harness.stress_tests.t_large_pcap:run",
             "D", "MEDIUM", timeout_sec=900),
    TestSpec("ST-2a", "Ingestion throughput (process_packet only)",
             "test_harness.stress_tests.t_ingest_throughput:run",
             "D", "MEDIUM", timeout_sec=120),
    TestSpec("ST-2b", "End-to-end throughput (with XAI)",
             "test_harness.stress_tests.t_e2e_throughput:run",
             "D", "MEDIUM", timeout_sec=600),
    TestSpec("ST-3",  "Long-duration monitoring", "test_harness.stress_tests.t_long_run:run",
             "D", "MEDIUM", timeout_sec=120, kwargs={"duration_sec": 30.0}),
    TestSpec("ST-4",  "Concurrent uploads",       "test_harness.stress_tests.t_concurrent:run",
             "D", "MEDIUM", timeout_sec=300),
]


def by_phase(phase: str) -> list[TestSpec]:
    return [t for t in TESTS if t.phase.upper() == phase.upper()]


def by_severity(severity: str) -> list[TestSpec]:
    return [t for t in TESTS if t.severity.upper() == severity.upper()]


def by_id(*ids: str) -> list[TestSpec]:
    s = {i.upper() for i in ids}
    return [t for t in TESTS if t.test_id.upper() in s]
