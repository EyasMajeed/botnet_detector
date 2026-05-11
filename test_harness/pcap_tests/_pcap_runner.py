"""
_pcap_runner.py — Replay a PCAP through the project's BotnetMonitor.

This mirrors what app/monitor_bridge.MonitorBridge does on the live path
exactly: extract layer-3/4 fields from each Scapy packet, then call

    BotnetMonitor.process_packet(
        timestamp, src_ip, dst_ip, src_port, dst_port,
        proto, pkt_len, ttl, src_mac="", tcp_flags=0,
    )

The method returns Optional[DetectionResult] directly when a flow
completes; we collect those into a list. After the replay, we also
force-flush any still-open flows by setting aggregator._idle = -1e9
(matching app/inference_worker.PcapInferenceThread).

Returns:
    {
      "skipped": bool,
      "skip_reason": str | None,
      "n_packets_seen": int,
      "n_packets_processed": int,    # packets that reached process_packet OK
      "n_results": int,
      "results": [ { ...result_dict... }, ... ],
      "errors": [str, ...]           # exceptions caught per-packet
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from test_harness.utils.project_imports import (
    soft_import, models_state, ensure_on_path,
)


# ── Per-packet field extraction ────────────────────────────────────────────
# Returns (timestamp, src_ip, dst_ip, src_port, dst_port, proto, pkt_len,
# ttl, src_mac, tcp_flags) — or None if the packet has no IP layer.
# v6 packets ARE returned (the project's process_packet expects str
# src_ip / dst_ip, so v6 addresses pass through untouched). Whether the
# project handles them downstream is what test P-04 measures.
def _extract(scapy_all, p) -> Optional[Tuple]:
    Ether  = getattr(scapy_all, "Ether", None)
    IP     = getattr(scapy_all, "IP", None)
    IPv6   = getattr(scapy_all, "IPv6", None)
    TCP    = getattr(scapy_all, "TCP", None)
    UDP    = getattr(scapy_all, "UDP", None)
    ICMP   = getattr(scapy_all, "ICMP", None)

    if not hasattr(p, "time"):
        return None

    # Pick the IP layer. v4 first; if absent, try v6.
    ip_layer = None
    proto: int = 0
    ttl: int   = 0
    if IP is not None and IP in p:
        ip_layer = p[IP]
        try:
            proto = int(getattr(ip_layer, "proto", 0))
        except Exception:
            proto = 0
        try:
            ttl = int(getattr(ip_layer, "ttl", 0))
        except Exception:
            ttl = 0
    elif IPv6 is not None and IPv6 in p:
        ip_layer = p[IPv6]
        try:
            proto = int(getattr(ip_layer, "nh", 0))   # next header
        except Exception:
            proto = 0
        try:
            ttl = int(getattr(ip_layer, "hlim", 0))   # hop limit
        except Exception:
            ttl = 0
    else:
        return None

    src_ip = str(getattr(ip_layer, "src", ""))
    dst_ip = str(getattr(ip_layer, "dst", ""))

    # Layer-4 fields. Mirrors monitor_bridge's TCP/UDP/ICMP branches —
    # everything else falls through with zeros (acceptable; project will
    # treat as unsupported).
    src_port = dst_port = 0
    tcp_flags = 0
    if TCP is not None and TCP in p:
        tcp = p[TCP]
        try: src_port = int(tcp.sport)
        except Exception: pass
        try: dst_port = int(tcp.dport)
        except Exception: pass
        try: tcp_flags = int(tcp.flags)
        except Exception: pass
        proto = 6
    elif UDP is not None and UDP in p:
        udp = p[UDP]
        try: src_port = int(udp.sport)
        except Exception: pass
        try: dst_port = int(udp.dport)
        except Exception: pass
        proto = 17
    elif ICMP is not None and ICMP in p:
        proto = 1
    # else: leave proto as parsed from IP header

    pkt_len = int(len(p))

    src_mac = ""
    if Ether is not None and Ether in p:
        try:
            src_mac = str(p[Ether].src)
        except Exception:
            src_mac = ""

    return (float(p.time), src_ip, dst_ip, src_port, dst_port,
            proto, pkt_len, ttl, src_mac, tcp_flags)


def replay(pcap_path: str, max_packets: int = 200_000) -> Dict[str, Any]:
    pcap = Path(pcap_path)
    if not pcap.exists():
        return {"skipped": True, "skip_reason": f"pcap not found: {pcap}"}

    ensure_on_path()
    models = models_state()
    required = ["models/stage1/rf_model.pkl",
                "models/stage2/iot_cnn_lstm.pt",
                "models/stage2/noniot_cnn_lstm.pt"]
    missing = [m for m in required if not models.get(m, False)]
    if missing:
        return {"skipped": True,
                "skip_reason": f"missing model files: {missing}",
                "models_state": models}

    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return {"skipped": True, "skip_reason": f"monitoring import: {err}"}

    scapy_all, err = soft_import("scapy.all")
    if scapy_all is None:
        return {"skipped": True, "skip_reason": f"scapy import: {err}"}

    # Build the monitor.
    try:
        bm = monitoring.BotnetMonitor()
    except Exception as e:                                    # noqa: BLE001
        return {"skipped": True,
                "skip_reason": f"BotnetMonitor init failed: {e!r}"}

    rdpcap = scapy_all.rdpcap

    try:
        pkts = rdpcap(str(pcap))
    except Exception as e:                                    # noqa: BLE001
        return {"skipped": False, "n_packets_seen": 0,
                "n_packets_processed": 0, "n_results": 0,
                "results": [], "errors": [f"rdpcap: {e!r}"]}

    n_seen = len(pkts)
    n_processed = 0
    errors: List[str] = []
    detection_results: List[Any] = []

    for i, p in enumerate(pkts[:max_packets]):
        try:
            extracted = _extract(scapy_all, p)
            if extracted is None:
                continue
            (ts, src_ip, dst_ip, sp, dp, proto, pkt_len, ttl,
             src_mac, tcp_flags) = extracted
            r = bm.process_packet(
                ts, src_ip, dst_ip, sp, dp, proto, pkt_len, ttl,
                src_mac, tcp_flags,
            )
            n_processed += 1
            if r is not None:
                detection_results.append(r)
        except Exception as e:                                # noqa: BLE001
            errors.append(f"packet[{i}]: {type(e).__name__}: {e}")
            if len(errors) > 50:
                errors.append("... (truncated)")
                break

    # Force-flush every still-open flow — matches PcapInferenceThread.
    try:
        if hasattr(bm, "aggregator"):
            bm.aggregator._idle = -1e9
        if hasattr(bm, "flush_idle_flows"):
            detection_results.extend(bm.flush_idle_flows() or [])
    except Exception as e:                                    # noqa: BLE001
        errors.append(f"flush: {e!r}")

    # Convert DetectionResult objects to plain dicts.
    results: List[dict] = []
    try:
        # Same helper inference_worker uses.
        from inference_bridge import _detection_results_to_dicts  # type: ignore
        results = _detection_results_to_dicts(detection_results, 0.0)
    except Exception:
        for r in detection_results:
            try:
                results.append({
                    "label":       getattr(r, "label", "unknown"),
                    "confidence":  float(getattr(r, "s2_confidence", 0.0)),
                    "device_type": getattr(r, "device_type", "unknown"),
                    "src_ip":      getattr(r, "src_ip", ""),
                    "dst_ip":      getattr(r, "dst_ip", ""),
                    "stage1_conf": float(getattr(r, "s1_confidence", 0.0)),
                    "suspicion":   float(getattr(r, "suspicion_score", 0.0)),
                })
            except Exception:
                pass

    return {
        "skipped": False,
        "n_packets_seen":     n_seen,
        "n_packets_processed": n_processed,
        "n_results":          len(results),
        "results":            results,
        "errors":             errors,
    }
