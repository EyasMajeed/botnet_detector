"""
SC-K Kitsune key-explosion stress test.

Replays a single-source / many-destination PCAP through the monitor and
checks how many keys the KitsuneExtractor accumulates.

Updated test (post-fix): the project's KitsuneExtractor now uses
LRUDefaultDict with MAX_STREAMS_PER_DICT = 10_000 (LRU eviction enabled).
We push 12 000 destinations through it and expect the dicts to cap out
at 10 000 — proving the LRU is actively evicting under cardinality
pressure, not just sitting unused.

Auto-detects the project's cap from extractor._max_streams when present.
"""

from __future__ import annotations

from test_harness.generators.pcap_gen import cardinality_explosion_pcap
from test_harness.pcap_tests._pcap_runner import replay
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import


TEST_ID = "SC-K"


def run(n_destinations: int = 12_000, key_cap: int | None = None) -> dict:
    """
    n_destinations : how many unique dst IPs to push through. Set this
                     comfortably above the project's MAX_STREAMS_PER_DICT
                     so we exercise eviction (default 12 000 > 10 000 cap).
    key_cap        : explicit cap to assert against. If None, auto-detect
                     from extractor._max_streams (the new attribute on
                     the patched KitsuneExtractor) and fall back to 10 000.
    """
    dirs = for_test(TEST_ID)
    pcap = cardinality_explosion_pcap(
        dirs["artifacts"] / "kitsune_keys.pcap", n_destinations=n_destinations
    )

    # We need to inspect Kitsune state AFTER the run. The cleanest
    # approach is to instantiate KitsuneExtractor directly and feed
    # packets ourselves — this matches what monitoring.py does.
    kit_mod, err = soft_import("src.live.kitsune_extractor")
    if kit_mod is None:
        return {"test_id": TEST_ID, "name": "Kitsune key explosion",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "Kitsune key count bounded by LRU",
                "actual": f"kitsune_extractor import: {err}",
                "pcap": str(pcap)}
    scapy_all, err = soft_import("scapy.all")
    if scapy_all is None:
        return {"test_id": TEST_ID, "name": "Kitsune key explosion",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "Kitsune key count bounded by LRU",
                "actual": f"scapy import: {err}",
                "pcap": str(pcap)}

    extractor = kit_mod.KitsuneExtractor()

    # Auto-detect cap from the patched extractor; fall back to 10_000.
    detected_cap = getattr(extractor, "_max_streams", None)
    if key_cap is None:
        key_cap = int(detected_cap) if detected_cap else 10_000

    pkts = scapy_all.rdpcap(str(pcap))
    for p in pkts:
        try:
            ip = p.getlayer(scapy_all.IP)
            if ip is None:
                continue
            tcp = p.getlayer(scapy_all.TCP)
            sport = int(tcp.sport) if tcp else 0
            dport = int(tcp.dport) if tcp else 0
            extractor.update(
                timestamp=float(p.time), src_mac=str(ip.src),
                src_ip=str(ip.src), dst_ip=str(ip.dst),
                src_port=sport, dst_port=dport,
                pkt_len=int(len(p)), protocol="TCP",
            )
        except Exception:
            continue

    n_hh   = len(getattr(extractor, "_hh",   {}) or {})
    n_hphp = len(getattr(extractor, "_hphp", {}) or {})
    n_h    = len(getattr(extractor, "_h",    {}) or {})

    # Replay through BotnetMonitor too — this is what production runs.
    replay_result = replay(str(pcap), max_packets=n_destinations + 1000)

    # Verdict logic:
    #   FAIL  if any per-stream dict exceeds the cap (LRU not enforced)
    #   FAIL  if dicts grew to nearly len(input) — no eviction active
    #         (only meaningful when n_destinations > cap)
    #   PASS  otherwise — cap honored
    over     = (n_hh > key_cap) or (n_hphp > key_cap)
    expected_to_evict = n_destinations > key_cap
    no_eviction       = expected_to_evict and (
        n_hh    >= n_destinations * 0.95 or
        n_hphp  >= n_destinations * 0.95
    )

    if over:
        verdict, actual = "FAIL", (
            f"Kitsune state exceeded cap: hh={n_hh}, hphp={n_hphp}, "
            f"h={n_h} (cap={key_cap}). LRU misconfigured.")
    elif no_eviction:
        verdict, actual = "FAIL", (
            f"Kitsune dicts grew to ~n_destinations ({n_hh}, {n_hphp}) "
            f"despite cap={key_cap}. Eviction is not running.")
    else:
        verdict, actual = "PASS", (
            f"hh={n_hh}, hphp={n_hphp}, h={n_h} — bounded under "
            f"cap={key_cap} after {n_destinations} unique destinations "
            f"({'eviction active' if expected_to_evict else 'no eviction needed'})")
    return {"test_id": TEST_ID, "name": "Kitsune key explosion",
            "severity": "HIGH", "verdict": verdict,
            "expected": f"hh and hphp counts ≤ {key_cap} via LRU/TTL",
            "actual":   actual, "pcap": str(pcap),
            "raw": {"hh": n_hh, "hphp": n_hphp, "h": n_h,
                    "cap_detected_from_extractor": detected_cap,
                    "cap_used": key_cap,
                    "n_destinations": n_destinations,
                    "replay_packets_processed":
                        replay_result.get("n_packets_processed", 0)}}
