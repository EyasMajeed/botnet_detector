"""
feature_metadata.py — Human-Readable Metadata for the 56 Unified Features
==========================================================================
Group 07 | CPCS499 | XAI Module

Purpose
-------
Maps each of the 56 unified-schema features to:
  - A human-readable display name
  - A measurement unit (for the dashboard)
  - A category (flow / time-window / packet / TLS) — used for grouping
  - A suspicion direction:
        +1  → high values are suspicious (e.g. burst_rate)
        -1  → low values are suspicious  (e.g. flow_duration for scans)
         0  → context-dependent / no clear direction

This metadata is consumed by:
  - explanation_engine.py  → to write plain-English alerts
  - local_explainer.py     → to label the IG/SHAP attribution charts
  - the desktop dashboard  → to render readable feature names

How direction was assigned
--------------------------
Based on the literature review in the project report (Section 3) and
known botnet behaviour:
  - Periodic beaconing → high periodicity_score, low IAT std
  - Scanning           → many short flows, few packets, many unique dsts
  - C2 traffic         → small payloads, repeated SYN/RST, rare ports
  - DDoS               → very high pkts/sec, fwd-heavy, low IAT
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureInfo:
    name: str            # raw column name (matches ALL_FEATURES)
    display: str         # human-readable label for the GUI
    unit: str            # unit string (e.g. "bytes", "pkts/s", "")
    category: str        # "flow" | "time_window" | "packet" | "tls"
    suspicion_dir: int   # +1 high=suspicious, -1 low=suspicious, 0 unclear
    description: str     # short tooltip explanation


# ──────────────────────────────────────────────────────────────────────
# 56-feature metadata table
# Order matches src/ingestion/preprocess_from_pcap_csvs.py :: ALL_FEATURES
# ──────────────────────────────────────────────────────────────────────

FEATURE_META: dict[str, FeatureInfo] = {
    # ── Flow-level (40) ──────────────────────────────────────────────
    "flow_duration":       FeatureInfo("flow_duration",       "Flow duration",                "s",      "flow",  -1, "Total time the flow was active. Very short flows are typical of scans."),
    "total_fwd_packets":   FeatureInfo("total_fwd_packets",   "Forward packets",              "pkts",   "flow",   0, "Number of packets sent from source to destination."),
    "total_bwd_packets":   FeatureInfo("total_bwd_packets",   "Backward packets",             "pkts",   "flow",   0, "Number of packets returned from destination."),
    "total_fwd_bytes":     FeatureInfo("total_fwd_bytes",     "Forward bytes",                "bytes",  "flow",   0, "Total bytes sent from source to destination."),
    "total_bwd_bytes":     FeatureInfo("total_bwd_bytes",     "Backward bytes",               "bytes",  "flow",   0, "Total bytes returned from destination."),
    "fwd_pkt_len_min":     FeatureInfo("fwd_pkt_len_min",     "Min fwd packet length",        "bytes",  "flow",   0, "Smallest packet size sent. Tiny packets often signal C2 beacons."),
    "fwd_pkt_len_max":     FeatureInfo("fwd_pkt_len_max",     "Max fwd packet length",        "bytes",  "flow",   0, "Largest packet size sent forward."),
    "fwd_pkt_len_mean":    FeatureInfo("fwd_pkt_len_mean",    "Mean fwd packet length",       "bytes",  "flow",   0, "Average packet size sent forward."),
    "fwd_pkt_len_std":     FeatureInfo("fwd_pkt_len_std",     "Fwd packet length std",        "bytes",  "flow",  -1, "Variability of forward packet sizes. Very low std = uniform packets (bot-like)."),
    "bwd_pkt_len_min":     FeatureInfo("bwd_pkt_len_min",     "Min bwd packet length",        "bytes",  "flow",   0, "Smallest packet size received."),
    "bwd_pkt_len_max":     FeatureInfo("bwd_pkt_len_max",     "Max bwd packet length",        "bytes",  "flow",   0, "Largest packet size received."),
    "bwd_pkt_len_mean":    FeatureInfo("bwd_pkt_len_mean",    "Mean bwd packet length",       "bytes",  "flow",   0, "Average packet size received."),
    "bwd_pkt_len_std":     FeatureInfo("bwd_pkt_len_std",     "Bwd packet length std",        "bytes",  "flow",  -1, "Variability of backward packet sizes."),
    "flow_bytes_per_sec":  FeatureInfo("flow_bytes_per_sec",  "Flow byte rate",               "B/s",    "flow",  +1, "Throughput of the flow. Very high = DDoS/exfil; very low = beacon."),
    "flow_pkts_per_sec":   FeatureInfo("flow_pkts_per_sec",   "Flow packet rate",             "pkts/s", "flow",  +1, "Packets per second. Very high indicates flooding."),
    "flow_iat_mean":       FeatureInfo("flow_iat_mean",       "Mean inter-arrival time",      "s",      "flow",   0, "Average gap between packets in the flow."),
    "flow_iat_std":        FeatureInfo("flow_iat_std",        "Inter-arrival time std",       "s",      "flow",  -1, "Variation in packet timing. Very low std = periodic (bot-like)."),
    "flow_iat_min":        FeatureInfo("flow_iat_min",        "Min inter-arrival time",       "s",      "flow",   0, "Smallest gap between packets."),
    "flow_iat_max":        FeatureInfo("flow_iat_max",        "Max inter-arrival time",       "s",      "flow",   0, "Largest gap between packets."),
    "fwd_iat_mean":        FeatureInfo("fwd_iat_mean",        "Mean fwd IAT",                 "s",      "flow",   0, "Average gap between forward packets."),
    "fwd_iat_std":         FeatureInfo("fwd_iat_std",         "Fwd IAT std",                  "s",      "flow",  -1, "Variation in forward packet timing."),
    "fwd_iat_min":         FeatureInfo("fwd_iat_min",         "Min fwd IAT",                  "s",      "flow",   0, "Smallest gap between forward packets."),
    "fwd_iat_max":         FeatureInfo("fwd_iat_max",         "Max fwd IAT",                  "s",      "flow",   0, "Largest gap between forward packets."),
    "bwd_iat_mean":        FeatureInfo("bwd_iat_mean",        "Mean bwd IAT",                 "s",      "flow",   0, "Average gap between backward packets."),
    "bwd_iat_std":         FeatureInfo("bwd_iat_std",         "Bwd IAT std",                  "s",      "flow",  -1, "Variation in backward packet timing."),
    "bwd_iat_min":         FeatureInfo("bwd_iat_min",         "Min bwd IAT",                  "s",      "flow",   0, "Smallest gap between backward packets."),
    "bwd_iat_max":         FeatureInfo("bwd_iat_max",         "Max bwd IAT",                  "s",      "flow",   0, "Largest gap between backward packets."),
    "fwd_header_length":   FeatureInfo("fwd_header_length",   "Fwd header bytes",             "bytes",  "flow",   0, "Total forward header bytes."),
    "bwd_header_length":   FeatureInfo("bwd_header_length",   "Bwd header bytes",             "bytes",  "flow",   0, "Total backward header bytes."),
    "flag_FIN":            FeatureInfo("flag_FIN",            "FIN flag count",               "pkts",   "flow",   0, "Number of packets with FIN flag set."),
    "flag_SYN":            FeatureInfo("flag_SYN",            "SYN flag count",               "pkts",   "flow",  +1, "SYN packets without matching ACKs are typical of port scans."),
    "flag_RST":            FeatureInfo("flag_RST",            "RST flag count",               "pkts",   "flow",  +1, "Many RSTs suggest closed-port probing."),
    "flag_PSH":            FeatureInfo("flag_PSH",            "PSH flag count",               "pkts",   "flow",   0, "Push flag count."),
    "flag_ACK":            FeatureInfo("flag_ACK",            "ACK flag count",               "pkts",   "flow",   0, "Acknowledgement flag count."),
    "flag_URG":            FeatureInfo("flag_URG",            "URG flag count",               "pkts",   "flow",  +1, "Urgent flag is rarely used legitimately; common in evasion."),
    "protocol":            FeatureInfo("protocol",            "Protocol",                     "",       "flow",   0, "IP protocol number (6=TCP, 17=UDP)."),
    "src_port":            FeatureInfo("src_port",            "Source port",                  "",       "flow",   0, "Originating port number."),
    "dst_port":            FeatureInfo("dst_port",            "Destination port",             "",       "flow",   0, "Destination port. Risky ports (23, 2323, 1900, 6667) are botnet indicators."),
    "flow_active_time":    FeatureInfo("flow_active_time",    "Active time",                  "s",      "flow",   0, "Time the flow was actively transmitting."),
    "flow_idle_time":      FeatureInfo("flow_idle_time",      "Idle time",                    "s",      "flow",  +1, "Time the flow was idle. Long idle gaps in periodic flows = beaconing."),

    # ── Time-window (6) ──────────────────────────────────────────────
    "bytes_per_sec_window":  FeatureInfo("bytes_per_sec_window",  "Window byte rate",          "B/s",    "time_window", +1, "Bytes/sec measured over a 10s window."),
    "pkts_per_sec_window":   FeatureInfo("pkts_per_sec_window",   "Window packet rate",        "pkts/s", "time_window", +1, "Packets/sec over a 10s window. Sustained high rates are flood-like."),
    "periodicity_score":     FeatureInfo("periodicity_score",     "Periodicity score",         "",       "time_window", +1, "How regular the inter-arrival pattern is. High = bot-like beacon."),
    "burst_rate":            FeatureInfo("burst_rate",            "Burst rate",                "",       "time_window", +1, "Fraction of traffic occurring in tight bursts."),
    "window_flow_count":     FeatureInfo("window_flow_count",     "Flows in window",           "flows",  "time_window", +1, "Number of distinct flows from this source in 10s. Many flows = scan."),
    "window_unique_dsts":    FeatureInfo("window_unique_dsts",    "Unique destinations",       "hosts",  "time_window", +1, "Distinct destinations contacted in 10s. Many = scanning behaviour."),

    # ── Packet-level (9) ─────────────────────────────────────────────
    "ttl_mean":             FeatureInfo("ttl_mean",             "TTL mean",                    "",       "packet",  0, "Average IP TTL."),
    "ttl_std":              FeatureInfo("ttl_std",              "TTL std",                     "",       "packet",  +1, "TTL variation. High std = multiple hop paths (suspicious for one source)."),
    "ttl_min":              FeatureInfo("ttl_min",              "TTL min",                     "",       "packet",  0, "Smallest TTL observed in the flow."),
    "ttl_max":              FeatureInfo("ttl_max",              "TTL max",                     "",       "packet",  0, "Largest TTL observed in the flow."),
    "dns_query_count":      FeatureInfo("dns_query_count",      "DNS queries",                 "qrys",   "packet",  +1, "DNS lookups in the flow. Excessive lookups = DGA-style C2."),
    "payload_bytes_mean":   FeatureInfo("payload_bytes_mean",   "Mean payload bytes",          "bytes",  "packet",  0, "Average payload size per packet."),
    "payload_bytes_std":    FeatureInfo("payload_bytes_std",    "Payload bytes std",           "bytes",  "packet",  -1, "Variation in payload size. Low std = uniform commands (bot-like)."),
    "payload_zero_ratio":   FeatureInfo("payload_zero_ratio",   "Zero-payload ratio",          "",       "packet",  +1, "Fraction of packets with empty payload. High = control packets only."),
    "payload_entropy":      FeatureInfo("payload_entropy",      "Payload entropy",             "bits",   "packet",  +1, "Shannon entropy of payload. Very high (≈8) = encrypted/packed C2."),

    # ── TLS (1) ──────────────────────────────────────────────────────
    "tls_features_available": FeatureInfo("tls_features_available", "TLS metadata available", "bool",   "tls",    0, "Whether TLS handshake metadata was captured for this flow."),
}

# ──────────────────────────────────────────────────────────────────────
# Kitsune feature heuristic labelling (Stage-2 IoT — 115 features)
# Names follow: {stream}_{lambda}_{stat}
#   stream ∈ {MI_dir, H, HH, HH_jit, HpHp}
#   lambda ∈ {L5, L3, L1, L0.1, L0.01}    (decay factor, NOT seconds)
#   stat   ∈ {weight, mean, variance, std, magnitude, radius, covariance, pcc}
# ──────────────────────────────────────────────────────────────────────

_KITSUNE_STREAMS = {
    "MI_dir": "MAC+IP source stats",
    "H":      "IP source stats",
    "HH":     "channel stats (src→dst)",
    "HH_jit": "channel jitter",
    "HpHp":   "socket stats (src:port→dst:port)",
}

_KITSUNE_WINDOWS = {
    "L5":    "100ms window",
    "L3":    "500ms window",
    "L1":    "1.5s window",
    "L0.1":  "10s window",
    "L0.01": "1min window",
}


def _kitsune_friendly(name: str) -> FeatureInfo | None:
    """Generate a FeatureInfo on the fly for a Kitsune column name."""
    for stream in _KITSUNE_STREAMS:
        if name.startswith(stream + "_"):
            tail = name[len(stream) + 1:]
            for window in _KITSUNE_WINDOWS:
                if tail.startswith(window + "_"):
                    stat = tail[len(window) + 1:]
                    display = f"{stream} {stat} ({_KITSUNE_WINDOWS[window]})"
                    desc = (f"Kitsune incremental statistic — {_KITSUNE_STREAMS[stream]}, "
                            f"{stat} over a {_KITSUNE_WINDOWS[window]}.")
                    return FeatureInfo(
                        name=name, display=display, unit="",
                        category="kitsune", suspicion_dir=0,
                        description=desc,
                    )
    return None


# ──────────────────────────────────────────────────────────────────────
# Convenience accessors
# ──────────────────────────────────────────────────────────────────────

def get_display(feature: str) -> str:
    if feature in FEATURE_META:
        return FEATURE_META[feature].display
    fi = _kitsune_friendly(feature)
    return fi.display if fi else feature


def get_unit(feature: str) -> str:
    if feature in FEATURE_META:
        return FEATURE_META[feature].unit
    fi = _kitsune_friendly(feature)
    return fi.unit if fi else ""


def get_category(feature: str) -> str:
    if feature in FEATURE_META:
        return FEATURE_META[feature].category
    fi = _kitsune_friendly(feature)
    return fi.category if fi else "unknown"


def get_description(feature: str) -> str:
    if feature in FEATURE_META:
        return FEATURE_META[feature].description
    fi = _kitsune_friendly(feature)
    return fi.description if fi else ""


def get_suspicion_dir(feature: str) -> int:
    return FEATURE_META[feature].suspicion_dir if feature in FEATURE_META else 0



def features_in_category(category: str) -> list[str]:
    return [name for name, info in FEATURE_META.items() if info.category == category]


if __name__ == "__main__":
    # Self-check
    print(f"Total features in metadata: {len(FEATURE_META)}")
    for cat in ("flow", "time_window", "packet", "tls"):
        feats = features_in_category(cat)
        print(f"  {cat:12s}: {len(feats):3d} features")
