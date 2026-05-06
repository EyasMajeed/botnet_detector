"""
═══════════════════════════════════════════════════════════════════════
 feature_metadata.py — Display names, units, and suspicion direction
 Group 07 | CPCS499 | XAI Module
═══════════════════════════════════════════════════════════════════════

Maps the raw feature names used by the Stage-1/Stage-2 models into:
  · Human-readable display names  ("Forward packet count" not "total_fwd_packets")
  · Units                         ("bytes/sec", "ms", "")
  · Suspicion direction           (does a HIGH value usually mean "more botnet"
                                   or "more benign"? +1 / -1 / 0)
  · Free-text description         (used in tooltips / detailed explanations)

The map covers two distinct feature schemas:

  1. The 56-feature unified flow schema (S1_FEATURES in monitoring.py).
     Used by Stage-1 (RF/XGBoost, IoT vs Non-IoT) and Stage-2 Non-IoT
     (CNN-LSTM, benign vs botnet).

  2. The 115-feature Kitsune AfterImage schema (FEATURE_NAMES in
     src/live/kitsune_extractor.py). Used by Stage-2 IoT (CNN-LSTM).

For Kitsune we don't write 115 entries by hand — we auto-generate
friendly labels from the structured naming convention
({stream}_{lambda}_{stat}) at lookup time.

Suspicion direction values:
  +1   higher value tends to indicate BOTNET   (e.g. flag_SYN, dns_query_count)
   0   neutral or context-dependent             (e.g. dst_port, protocol)
  -1   higher value tends to indicate BENIGN   (e.g. payload_bytes_mean for
                                                  established TLS sessions)
This is a *prior*, not ground truth — the IG attribution sign is what
actually matters per-flow. Direction is only used in the rule engine's
sanity checks ("does the model's positive attribution for THIS feature
agree with what we'd expect from domain knowledge?").
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureInfo:
    name:           str   # raw feature name (matches model schema)
    display:        str   # human-readable label
    unit:           str
    category:       str   # "flow" | "time-window" | "packet" | "tls" | "kitsune"
    suspicion_dir:  int   # +1 / 0 / -1
    description:    str


# ════════════════════════════════════════════════════════════════════════
# 56-feature unified flow schema
#   Hand-written because there are only 56 of them and many require
#   real domain expertise to label and direction correctly.
# ════════════════════════════════════════════════════════════════════════

FEATURE_META: dict[str, FeatureInfo] = {
    # ── Flow-level (40) ──────────────────────────────────────────────
    "flow_duration": FeatureInfo(
        "flow_duration", "Flow duration", "s", "flow", -1,
        "Total time the flow was active. Very short flows (<1s) often indicate scans or short bursts."),
    "total_fwd_packets": FeatureInfo(
        "total_fwd_packets", "Forward packets", "count", "flow", 0,
        "Number of packets sent by the source in this flow."),
    "total_bwd_packets": FeatureInfo(
        "total_bwd_packets", "Backward packets", "count", "flow", 0,
        "Number of packets returned to the source by the destination."),
    "total_fwd_bytes": FeatureInfo(
        "total_fwd_bytes", "Forward bytes", "bytes", "flow", 0,
        "Total bytes sent by the source."),
    "total_bwd_bytes": FeatureInfo(
        "total_bwd_bytes", "Backward bytes", "bytes", "flow", 0,
        "Total bytes returned to the source."),

    "fwd_pkt_len_min":  FeatureInfo("fwd_pkt_len_min",  "Fwd pkt min len",  "bytes", "flow", 0,
        "Smallest forward packet length."),
    "fwd_pkt_len_max":  FeatureInfo("fwd_pkt_len_max",  "Fwd pkt max len",  "bytes", "flow", 0,
        "Largest forward packet length."),
    "fwd_pkt_len_mean": FeatureInfo("fwd_pkt_len_mean", "Fwd pkt mean len", "bytes", "flow", 0,
        "Average forward packet length. Very small mean (<200) often suggests control traffic / beaconing."),
    "fwd_pkt_len_std":  FeatureInfo("fwd_pkt_len_std",  "Fwd pkt len std",  "bytes", "flow", -1,
        "Variability of forward packet length. Low variance suggests uniform / scripted traffic."),

    "bwd_pkt_len_min":  FeatureInfo("bwd_pkt_len_min",  "Bwd pkt min len",  "bytes", "flow", 0, ""),
    "bwd_pkt_len_max":  FeatureInfo("bwd_pkt_len_max",  "Bwd pkt max len",  "bytes", "flow", 0, ""),
    "bwd_pkt_len_mean": FeatureInfo("bwd_pkt_len_mean", "Bwd pkt mean len", "bytes", "flow", 0, ""),
    "bwd_pkt_len_std":  FeatureInfo("bwd_pkt_len_std",  "Bwd pkt len std",  "bytes", "flow", 0, ""),

    "flow_bytes_per_sec": FeatureInfo(
        "flow_bytes_per_sec", "Flow byte rate", "bytes/sec", "flow", +1,
        "Bytes per second across the flow. Very high rates can indicate floods or DDoS."),
    "flow_pkts_per_sec":  FeatureInfo(
        "flow_pkts_per_sec",  "Flow pkt rate",  "pkt/sec",   "flow", +1,
        "Packets per second. High packet rates often indicate scans or DoS-style traffic."),

    "flow_iat_mean": FeatureInfo("flow_iat_mean", "Flow IAT mean", "s", "flow", 0,
        "Average inter-arrival time between packets in the flow."),
    "flow_iat_std":  FeatureInfo("flow_iat_std",  "Flow IAT std",  "s", "flow", -1,
        "Variability of IAT. Low std suggests regular timing — common in C2 beacons."),
    "flow_iat_min":  FeatureInfo("flow_iat_min",  "Flow IAT min",  "s", "flow", 0, ""),
    "flow_iat_max":  FeatureInfo("flow_iat_max",  "Flow IAT max",  "s", "flow", 0, ""),

    "fwd_iat_mean": FeatureInfo("fwd_iat_mean", "Fwd IAT mean", "s", "flow", 0, ""),
    "fwd_iat_std":  FeatureInfo("fwd_iat_std",  "Fwd IAT std",  "s", "flow", 0, ""),
    "fwd_iat_min":  FeatureInfo("fwd_iat_min",  "Fwd IAT min",  "s", "flow", 0, ""),
    "fwd_iat_max":  FeatureInfo("fwd_iat_max",  "Fwd IAT max",  "s", "flow", 0, ""),

    "bwd_iat_mean": FeatureInfo("bwd_iat_mean", "Bwd IAT mean", "s", "flow", 0, ""),
    "bwd_iat_std":  FeatureInfo("bwd_iat_std",  "Bwd IAT std",  "s", "flow", 0, ""),
    "bwd_iat_min":  FeatureInfo("bwd_iat_min",  "Bwd IAT min",  "s", "flow", 0, ""),
    "bwd_iat_max":  FeatureInfo("bwd_iat_max",  "Bwd IAT max",  "s", "flow", 0, ""),

    "fwd_header_length": FeatureInfo(
        "fwd_header_length", "Fwd header length", "bytes", "flow", 0,
        "Forward TCP/IP header length. Forced to 20 in live mode."),
    "bwd_header_length": FeatureInfo(
        "bwd_header_length", "Bwd header length", "bytes", "flow", 0,
        "Backward TCP/IP header length. Forced to 20 in live mode."),

    "flag_FIN": FeatureInfo("flag_FIN", "FIN flag count", "count", "flow", 0,
        "Number of FIN flags. Normal at end of TCP sessions."),
    "flag_SYN": FeatureInfo("flag_SYN", "SYN flag count", "count", "flow", +1,
        "Number of SYN flags. Many SYNs without ACK indicates scans or SYN floods."),
    "flag_RST": FeatureInfo("flag_RST", "RST flag count", "count", "flow", +1,
        "Number of RST flags. Repeated RSTs suggest brute-force / aggressive probing."),
    "flag_PSH": FeatureInfo("flag_PSH", "PSH flag count", "count", "flow", 0, ""),
    "flag_ACK": FeatureInfo("flag_ACK", "ACK flag count", "count", "flow", 0, ""),
    "flag_URG": FeatureInfo("flag_URG", "URG flag count", "count", "flow", +1,
        "URG flags are rare in normal traffic. Often used in evasion / exploit attempts."),

    "protocol": FeatureInfo("protocol", "Protocol", "", "flow", 0,
        "IP protocol number (6=TCP, 17=UDP, 1=ICMP)."),
    "src_port": FeatureInfo("src_port", "Source port", "", "flow", 0,
        "Source port. Ephemeral ports (>1024) are normal; low ports can be services."),
    "dst_port": FeatureInfo("dst_port", "Destination port", "", "flow", 0,
        "Destination port. Risky ports (23 telnet, 2323, 1900 SSDP, 7547 TR-069) "
        "are common IoT botnet vectors."),

    "flow_active_time": FeatureInfo("flow_active_time", "Active time", "s", "flow", 0, ""),
    "flow_idle_time":   FeatureInfo("flow_idle_time",   "Idle time",   "s", "flow", 0, ""),

    # ── Time-window (6) ──────────────────────────────────────────────
    "bytes_per_sec_window": FeatureInfo(
        "bytes_per_sec_window", "Window byte rate", "bytes/sec", "time-window", +1,
        "Bytes/sec across all flows from the source in the recent window. "
        "High values relative to baseline suggest exfiltration or DDoS source."),
    "pkts_per_sec_window":  FeatureInfo(
        "pkts_per_sec_window",  "Window pkt rate",  "pkt/sec",   "time-window", +1,
        "Packets/sec from this source in the window."),
    "periodicity_score":    FeatureInfo(
        "periodicity_score",    "Periodicity",      "score",     "time-window", +1,
        "Regularity of inter-arrival times (0=irregular, 1=perfectly periodic). "
        "High scores are consistent with C2 beaconing. NOTE: forced to 0 in live mode."),
    "burst_rate":           FeatureInfo(
        "burst_rate",           "Burst rate",       "score",     "time-window", +1,
        "Fraction of flows that burst within the window. NOTE: forced to 0 in live mode."),
    "window_flow_count":    FeatureInfo(
        "window_flow_count",    "Flows in window",  "count",     "time-window", +1,
        "Total flows from the source in the recent window. NOTE: forced to 1 in live mode."),
    "window_unique_dsts":   FeatureInfo(
        "window_unique_dsts",   "Unique destinations", "count",  "time-window", +1,
        "Number of distinct destinations in the window. High counts indicate scanning. "
        "NOTE: forced to 1 in live mode."),

    # ── Packet-level (9) ─────────────────────────────────────────────
    "ttl_mean": FeatureInfo("ttl_mean", "TTL mean", "", "packet", 0,
        "Average IP TTL. Anomalous TTLs can indicate routing manipulation or spoofing."),
    "ttl_std":  FeatureInfo("ttl_std",  "TTL std",  "", "packet", 0, ""),
    "ttl_min":  FeatureInfo("ttl_min",  "TTL min",  "", "packet", 0, ""),
    "ttl_max":  FeatureInfo("ttl_max",  "TTL max",  "", "packet", 0, ""),

    "dns_query_count": FeatureInfo(
        "dns_query_count", "DNS query count", "count", "packet", +1,
        "DNS queries observed. Very high counts can indicate DNS tunneling or DGA."),

    "payload_bytes_mean": FeatureInfo("payload_bytes_mean", "Payload bytes mean", "bytes", "packet", 0, ""),
    "payload_bytes_std":  FeatureInfo("payload_bytes_std",  "Payload bytes std",  "bytes", "packet", 0, ""),
    "payload_zero_ratio": FeatureInfo(
        "payload_zero_ratio", "Zero-byte ratio", "ratio", "packet", +1,
        "Fraction of payload bytes that are zero. High in null-stuffing / tunneling. "
        "NOTE: forced to 0 in live mode (no payload inspection)."),
    "payload_entropy":    FeatureInfo(
        "payload_entropy",    "Payload entropy", "bits",  "packet", +1,
        "Shannon entropy of payload. Very high (>7.5) can indicate encryption / DNS tunneling. "
        "NOTE: forced to 0 in live mode."),

    # ── TLS (1) ──────────────────────────────────────────────────────
    "tls_features_available": FeatureInfo(
        "tls_features_available", "TLS handshake seen", "bool", "tls", 0,
        "1 if the flow contains a TLS handshake (port 443/8443/8883)."),
}


# ════════════════════════════════════════════════════════════════════════
# Kitsune (N-BaIoT) heuristic labelling — Stage-2 IoT, 115 features.
#
# Names follow the structured pattern: {stream}_{lambda}_{stat}
#   stream ∈ {MI_dir, H, HH, HH_jit, HpHp}
#   lambda ∈ {L5, L3, L1, L0.1, L0.01}    (decay factor, NOT seconds)
#   stat   ∈ {weight, mean, variance, std, magnitude, radius, covariance, pcc}
#
# Reference: Mirsky et al. (2018) "Kitsune: An Ensemble of Autoencoders
#            for Online Network Intrusion Detection", NDSS Symposium.
# ════════════════════════════════════════════════════════════════════════

_KITSUNE_STREAMS = {
    "MI_dir": "MAC+IP source rate",
    "H":      "IP source rate",
    "HH":     "channel src→dst",
    "HH_jit": "channel jitter",
    "HpHp":   "socket src:port→dst:port",
}

_KITSUNE_WINDOWS = {
    "L5":    "100ms window",
    "L3":    "500ms window",
    "L1":    "1.5s window",
    "L0.1":  "10s window",
    "L0.01": "1min window",
}


def _kitsune_friendly(name: str) -> FeatureInfo | None:
    """
    Generate a FeatureInfo on the fly for a Kitsune column name.
    Returns None if `name` doesn't match the Kitsune naming convention.
    """
    # Streams must be matched in length-descending order so "HH_jit" wins
    # over "HH" and "MI_dir" wins over "MI".
    for stream in sorted(_KITSUNE_STREAMS, key=len, reverse=True):
        prefix = stream + "_"
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        for window in sorted(_KITSUNE_WINDOWS, key=len, reverse=True):
            wpref = window + "_"
            if not tail.startswith(wpref):
                continue
            stat = tail[len(wpref):]
            display = f"{stream} {stat} ({_KITSUNE_WINDOWS[window]})"
            desc = (f"Kitsune incremental statistic — {_KITSUNE_STREAMS[stream]}, "
                    f"{stat} over a {_KITSUNE_WINDOWS[window]}.")
            return FeatureInfo(
                name=name, display=display, unit="",
                category="kitsune", suspicion_dir=0,
                description=desc,
            )
    return None


# ════════════════════════════════════════════════════════════════════════
# Public accessor functions
#   These are the only API the rest of the XAI module uses to look up
#   feature metadata. They check the hand-written map first, then fall
#   back to the Kitsune heuristic, then return a sensible default.
# ════════════════════════════════════════════════════════════════════════

def get_info(feature: str) -> FeatureInfo:
    """Return full FeatureInfo, falling back to Kitsune or a default."""
    if feature in FEATURE_META:
        return FEATURE_META[feature]
    fi = _kitsune_friendly(feature)
    if fi is not None:
        return fi
    return FeatureInfo(feature, feature, "", "unknown", 0, "")


def get_display(feature: str) -> str:
    return get_info(feature).display


def get_unit(feature: str) -> str:
    return get_info(feature).unit


def get_category(feature: str) -> str:
    return get_info(feature).category


def get_suspicion_dir(feature: str) -> int:
    return get_info(feature).suspicion_dir


def get_description(feature: str) -> str:
    return get_info(feature).description


# ════════════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    samples = [
        "flow_pkts_per_sec",            # in FEATURE_META
        "window_unique_dsts",           # in FEATURE_META, has constant note
        "MI_dir_L5_weight",             # Kitsune  → friendly name
        "HH_jit_L0.01_variance",        # Kitsune  → friendly name with the 1min window
        "made_up_feature",              # default fallback
    ]
    for name in samples:
        fi = get_info(name)
        print(f"  {name:<35}  display={fi.display:<45}  cat={fi.category:<12}  dir={fi.suspicion_dir:+d}")
