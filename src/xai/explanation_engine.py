"""
═══════════════════════════════════════════════════════════════════════
 explanation_engine.py — Human-readable explanation + recommendations
 Group 07 | CPCS499 | XAI Module
═══════════════════════════════════════════════════════════════════════

Consumes a `LocalExplanation` (from local_explainer.py) and produces:
  · A short summary sentence    ("This flow looks like a port scan.")
  · 2–4 bullet-point reasons    (referencing top features in plain English)
  · A recommended action list   (what the analyst should do next)
  · A severity tag              (low / medium / high / critical)
  · A behavior pattern tag      (PORT_SCAN, DDOS, C2_BEACON, DNS_TUNNEL,
                                 BRUTE_FORCE, GENERIC, UNKNOWN)

Satisfies report Section 4.4: "human-readable explanations of suspicious
behaviours + simple rule-based recommendations to guide analysts on next
steps."

═══════════════════════════════════════════════════════════════════════
 IMPORTANT — Feature availability across input modes
═══════════════════════════════════════════════════════════════════════
The current preprocessing pipeline (preprocess_from_pcap_csvs.py) and the
live monitor (monitoring.py) intentionally force several time-window and
payload features to constant values to keep training and live inference
distributions identical. The constants are:

    periodicity_score    → 0.0    (no per-window periodicity computed live)
    burst_rate           → 0.0    (no per-window burst rate computed live)
    payload_zero_ratio   → 0.0    (no payload inspection live)
    payload_entropy      → 0.0    (no payload inspection live)
    fwd_header_length    → 20.0   (TCP header default)
    bwd_header_length    → 20.0   (TCP header default)
    window_flow_count    → 1.0    (per-flow inference, no aggregation)
    window_unique_dsts   → 1.0    (per-flow inference, no aggregation)

CONSEQUENCE FOR THIS RULE ENGINE:
  · In FILE-UPLOAD and LIVE-CAPTURE modes, the eight features above are
    essentially noise. Patterns that depend ONLY on them — PORT_SCAN
    (window_unique_dsts), C2_BEACON (periodicity_score), DNS_TUNNEL
    (payload_entropy) — will rarely fire from those features alone.
  · Each pattern matcher uses _is_meaningful() to skip checks against
    the forced-constant value, preventing false-positive triggers.
  · The matchers also include checks on NON-constant features (e.g.
    SYN flag count for PORT_SCAN, dst_port=53 for DNS_TUNNEL) that
    DO carry information — those are what actually fires the patterns.
  · Patterns that don't depend on the forced-constant set — DDOS,
    BRUTE_FORCE — are reliable in all modes.

For the project defense: this is documented behaviour, not a bug. The
unified-schema constraint that lets us train one model on heterogeneous
datasets is the same constraint that limits which patterns are
recognisable from flow features alone.

═══════════════════════════════════════════════════════════════════════
 Design
═══════════════════════════════════════════════════════════════════════
We pair the *model-driven* feature attributions (from IG / SHAP) with a
*rule-based* behaviour classifier that recognises common botnet patterns:

    ┌────────────────────────┐    ┌─────────────────────────┐
    │  Top features + values │ ─→ │  Behaviour pattern      │
    │  (from IG / SHAP)      │    │  matcher (rule engine)  │
    └────────────────────────┘    └────────────┬────────────┘
                                               ↓
                                  ┌──────────────────────────┐
                                  │  Pattern-specific text   │
                                  │  + recommendations       │
                                  └──────────────────────────┘

The rule engine is deliberately simple — it asks "do the top-attributed
features collectively look like (port scan / DDoS / C2 beacon / DNS tunnel)?"
This avoids hallucinating explanations and keeps the output auditable.

For Stage-2 IoT (Kitsune features), the rule engine auto-dispatches to a
separate set of IoT-specific patterns (MIRAI_FLOOD, IOT_SCAN, SLOW_BEACON)
based on the feature names in top_features. The dispatch happens in
build_human_explanation() — flow-schema names route to the flow matchers,
Kitsune-schema names route to the Kitsune matchers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.xai.local_explainer import LocalExplanation, FeatureContribution


# ══════════════════════════════════════════════════════════════════════
# LIVE_CONSTANT_FEATURES — the eight features whose values are forced
# to fixed constants in both training and live inference. Pattern
# matchers consult this map to skip checks against features whose
# value matches the forced constant exactly.
# ══════════════════════════════════════════════════════════════════════

LIVE_CONSTANT_FEATURES: dict[str, float] = {
    "periodicity_score":  0.0,
    "burst_rate":         0.0,
    "payload_zero_ratio": 0.0,
    "payload_entropy":    0.0,
    "fwd_header_length":  20.0,
    "bwd_header_length":  20.0,
    "window_flow_count":  1.0,
    "window_unique_dsts": 1.0,
}


def _is_meaningful(feature: str, value: float, tol: float = 1e-6) -> bool:
    """
    True if `value` is *different* from the feature's forced-constant value
    (i.e. the feature actually carries information for this flow).
    For features not in LIVE_CONSTANT_FEATURES, always returns True.
    """
    if feature not in LIVE_CONSTANT_FEATURES:
        return True
    return abs(value - LIVE_CONSTANT_FEATURES[feature]) > tol


# ══════════════════════════════════════════════════════════════════════
# HumanExplanation — what the rule engine returns
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HumanExplanation:
    summary:         str                 # one sentence
    pattern:         str                 # PORT_SCAN | DDOS | C2_BEACON | ... | GENERIC
    severity:        str                 # low | medium | high | critical
    reasons:         list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _values(expl: LocalExplanation) -> dict[str, float]:
    """Return {feature: raw_value} from the top features."""
    return {f.feature: f.value for f in expl.top_features}


def _top_feature_names(expl: LocalExplanation) -> set[str]:
    return {f.feature for f in expl.top_features}


# ══════════════════════════════════════════════════════════════════════
# Pattern matchers — each returns a score in [0, 1].
# A pattern is "matched" if its score >= PATTERN_THRESHOLD.
# ══════════════════════════════════════════════════════════════════════

def _detect_port_scan(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    # window_unique_dsts and window_flow_count are forced-constant in the
    # current pipeline. They're kept here for forward-compatibility with a
    # future per-window aggregator, gated by _is_meaningful() to prevent
    # false triggering on the forced value.
    udst = v.get("window_unique_dsts", 0)
    if "window_unique_dsts" in top and _is_meaningful("window_unique_dsts", udst) and udst >= 10:
        score += 0.45
    wfc = v.get("window_flow_count", 0)
    if "window_flow_count" in top and _is_meaningful("window_flow_count", wfc) and wfc >= 20:
        score += 0.30
    # Short flows + few packets per flow + many SYNs → real signals not
    # affected by LIVE_CONSTANT_FEATURES, this is what carries the matcher.
    if v.get("flow_duration", 999) < 1.0 and v.get("total_fwd_packets", 999) <= 3:
        score += 0.25
    if v.get("flag_SYN", 0) >= 5 and v.get("flag_ACK", 0) <= 1:
        score += 0.20
    return min(score, 1.0)


def _detect_ddos(expl: LocalExplanation) -> float:
    v = _values(expl)
    score = 0.0
    # Sustained extreme packet rate — primary DDoS signal
    if v.get("flow_pkts_per_sec", 0) >= 500:
        score += 0.45
    if v.get("flow_bytes_per_sec", 0) >= 1_000_000:
        score += 0.20
    # Forward-heavy traffic: many fwd packets, very few bwd (target overwhelmed)
    fwd = v.get("total_fwd_packets", 0)
    bwd = v.get("total_bwd_packets", 1)
    if fwd >= 100 and (bwd == 0 or fwd / max(bwd, 1) >= 50):
        score += 0.20
    # Tight inter-arrival times
    if v.get("flow_iat_mean", 999) < 0.001:
        score += 0.15
    return min(score, 1.0)


def _detect_c2_beacon(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    # periodicity_score is forced to 0.0 in current modes; this branch
    # contributes 0 unless a future per-window aggregator computes it.
    pscore = v.get("periodicity_score", 0)
    if "periodicity_score" in top and _is_meaningful("periodicity_score", pscore) and pscore >= 0.7:
        score += 0.45
    # Low IAT std = regular timing — also a beacon signature, not constant
    if "flow_iat_std" in top and v.get("flow_iat_std", 999) < 0.05:
        score += 0.25
    # Small, uniform packets (control traffic) — real flow features
    if v.get("fwd_pkt_len_mean", 9999) < 200 and v.get("fwd_pkt_len_std", 9999) < 50:
        score += 0.20
    pzr = v.get("payload_zero_ratio", 0)
    if _is_meaningful("payload_zero_ratio", pzr) and pzr >= 0.5:
        score += 0.10
    return min(score, 1.0)


def _detect_dns_tunnel(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "dns_query_count" in top and v.get("dns_query_count", 0) >= 50:
        score += 0.50
    # payload_entropy is forced to 0.0 in current modes
    pent = v.get("payload_entropy", 0)
    if "payload_entropy" in top and _is_meaningful("payload_entropy", pent) and pent >= 7.5:
        score += 0.30
    if v.get("dst_port", 0) == 53 and v.get("flow_bytes_per_sec", 0) > 1000:
        score += 0.20
    return min(score, 1.0)


def _detect_brute_force(expl: LocalExplanation) -> float:
    v = _values(expl)
    score = 0.0
    syn = v.get("flag_SYN", 0)
    rst = v.get("flag_RST", 0)
    if syn >= 10 and rst >= 5:
        score += 0.40
    # Common login service ports
    LOGIN_PORTS = {22, 23, 21, 3389, 445, 1433, 3306, 5900}
    if int(v.get("dst_port", 0)) in LOGIN_PORTS:
        score += 0.30
    if v.get("flow_duration", 999) < 5.0 and syn >= 5:
        score += 0.20
    return min(score, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Kitsune-aware IoT pattern matchers
# ══════════════════════════════════════════════════════════════════════
# These run on Stage-2 IoT detections, where top_features are Kitsune
# AfterImage statistics (MI_dir_*, H_*, HH_*, HH_jit_*, HpHp_*) rather
# than the unified flow schema. The streams encode different views:
#
#   MI_dir : MAC+IP source rate                  (one device's send rate)
#   H      : IP source rate                      (any device sharing the IP)
#   HH     : channel src→dst                     (one host pair byte/pkt rate)
#   HH_jit : jitter on the channel               (timing irregularity)
#   HpHp   : socket src:port → dst:port          (one socket pair)
#
# Window suffixes encode the decay constant:
#   L5     ≈ 100ms     (very recent)
#   L3     ≈ 500ms
#   L1     ≈ 1.5s
#   L0.1   ≈ 10s
#   L0.01  ≈ 1min      (long-tail, catches slow patterns)
#
# Reference: Mirsky et al. (2018), "Kitsune: An Ensemble of Autoencoders
# for Online Network Intrusion Detection", NDSS Symposium.
# ══════════════════════════════════════════════════════════════════════

def _kitsune_topk_has_any(expl: LocalExplanation, prefixes: tuple[str, ...]) -> bool:
    """True if ANY top-feature name starts with any of the given prefixes."""
    for f in expl.top_features:
        for p in prefixes:
            if f.feature.startswith(p):
                return True
    return False


def _kitsune_topk_attribution_sum(expl: LocalExplanation,
                                   prefixes: tuple[str, ...]) -> float:
    """Sum of |attribution| for top features whose names match any prefix."""
    total = 0.0
    for f in expl.top_features:
        for p in prefixes:
            if f.feature.startswith(p):
                total += abs(f.attribution)
                break
    return total


def _detect_mirai_flood(expl: LocalExplanation) -> float:
    """
    MIRAI_FLOOD — high recent send rate from one device + high channel byte
    rate.  Mirai's signature: a single IoT device suddenly emits packets
    at very high rate to one or a few destinations.

    Top features should be dominated by:
      · MI_dir_L5_* (recent send rate from this MAC+IP source)
      · HH_L5_* / HH_L3_* (channel byte/packet rate over short windows)
    A mature flood typically pushes BOTH of these to extreme values.
    """
    score = 0.0
    # MI_dir_L5 dominant in attributions → this device is sending recently
    if _kitsune_topk_has_any(expl, ("MI_dir_L5_", "MI_dir_L3_")):
        score += 0.40
    # Channel stats also dominant → the burst is concentrated on a channel
    if _kitsune_topk_has_any(expl, ("HH_L5_", "HH_L3_", "HH_L1_")):
        score += 0.30
    # If MI_dir + HH together account for half the |attribution|, this is
    # very likely a flood.
    mi_dir_share = _kitsune_topk_attribution_sum(expl, ("MI_dir_",))
    hh_share     = _kitsune_topk_attribution_sum(expl, ("HH_",))
    total_share  = sum(abs(f.attribution) for f in expl.top_features) or 1.0
    if (mi_dir_share + hh_share) / total_share >= 0.50:
        score += 0.20
    return min(score, 1.0)


def _detect_iot_scan(expl: LocalExplanation) -> float:
    """
    IOT_SCAN — one IoT device probing many sockets. Distinct from a generic
    port scan because the device's MAC is the same throughout. Signal:

      · HpHp_L5 / HpHp_L3 weight is HIGH (many distinct socket pairs)
      · MI_dir_L5 weight is LOWER than HpHp suggests (device isn't sending
        a flood, it's sending probes — many small packets to many sockets)

    The model picking up on HpHp_* features as the main contributors is
    the strongest signal — a normal IoT device usually talks to one or two
    backend sockets, not dozens.
    """
    score = 0.0
    if _kitsune_topk_has_any(expl, ("HpHp_L5_", "HpHp_L3_")):
        score += 0.45
    # H_L5_weight high (lots of packets from this IP) is supportive but not
    # conclusive — Mirai_flood also lights up H_L5. Distinguish via HpHp
    # being among the TOP attributors, not just present.
    if _kitsune_topk_has_any(expl, ("H_L5_", "H_L3_")):
        score += 0.20
    # Tie-breaker: HpHp share of attributions
    hphp_share  = _kitsune_topk_attribution_sum(expl, ("HpHp_",))
    total_share = sum(abs(f.attribution) for f in expl.top_features) or 1.0
    if hphp_share / total_share >= 0.30:
        score += 0.20
    return min(score, 1.0)


def _detect_slow_beacon(expl: LocalExplanation) -> float:
    """
    SLOW_BEACON — long-tail correlation in the channel pair, with low
    jitter. Signature of a low-and-slow C2 beacon over a 1min window.

      · HH_L0.01_* features dominant (long-window channel statistics)
      · HH_jit_L0.01_* in the top — and the model attributes positively
        to LOW jitter values (i.e., very regular timing)
      · HH_L0.01_pcc (Pearson correlation between channel stats) high
        means the two endpoints are exchanging tightly correlated traffic
    """
    score = 0.0
    if _kitsune_topk_has_any(expl, ("HH_L0.01_", "HH_L0.1_")):
        score += 0.40
    if _kitsune_topk_has_any(expl, ("HH_jit_L0.01_", "HH_jit_L0.1_")):
        score += 0.25
    # Long-tail share dominance (L0.01 + L0.1 windows)
    longtail = _kitsune_topk_attribution_sum(expl, ("HH_L0.01_", "HH_L0.1_",
                                                     "HH_jit_L0.01_", "HH_jit_L0.1_"))
    total_share = sum(abs(f.attribution) for f in expl.top_features) or 1.0
    if longtail / total_share >= 0.40:
        score += 0.20
    return min(score, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Schema dispatch — flow-feature matchers vs Kitsune matchers
# ══════════════════════════════════════════════════════════════════════
# We auto-detect by looking at the top_features. Any Kitsune-style name
# (matches one of the five stream prefixes followed by a window suffix)
# means we're explaining a Stage-2 IoT detection. Flow names mean we're
# explaining Stage-2 Non-IoT.
#
# Mixed case can't happen (the two stages have disjoint schemas), but
# if it ever did we'd prefer Kitsune dispatch on tiebreak — IoT botnets
# have specific attack patterns that flow features can't represent.

_KITSUNE_PREFIXES = ("MI_dir_", "H_", "HH_", "HH_jit_", "HpHp_")


def _is_kitsune_explanation(expl: LocalExplanation) -> bool:
    """Returns True if the top features look like Kitsune AfterImage stats."""
    for f in expl.top_features:
        for p in _KITSUNE_PREFIXES:
            if f.feature.startswith(p):
                return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Pattern → text & recommendations
# ══════════════════════════════════════════════════════════════════════

PATTERNS = {
    "PORT_SCAN": {
        "summary": "Likely port-scan / reconnaissance behaviour.",
        "recommendations": [
            "Block or rate-limit the source IP at the firewall.",
            "Check whether the source is an authorised scanner (e.g. internal asset inventory).",
            "Audit logs of the targeted hosts for follow-up exploitation attempts.",
        ],
        "min_severity": "high",
    },
    "DDOS": {
        "summary": "Likely volumetric DDoS or flood attack.",
        "recommendations": [
            "Engage upstream rate-limiting / traffic scrubbing immediately.",
            "Identify whether the source is part of a wider distributed pattern (multiple sources hitting the same target).",
            "If internal, isolate the source host — it may be infected and used as a bot.",
        ],
        "min_severity": "critical",
    },
    "C2_BEACON": {
        "summary": "Periodic communication consistent with a Command-and-Control beacon.",
        "recommendations": [
            "Quarantine the source host and acquire a memory image for analysis.",
            "Block the destination at DNS / firewall level and look for similar beacons across the network.",
            "Hunt for related IoCs (process tree, registry, scheduled tasks) on the source host.",
        ],
        "min_severity": "high",
    },
    "DNS_TUNNEL": {
        "summary": "Likely DNS tunnelling or DGA-driven C2 over DNS.",
        "recommendations": [
            "Block resolution of the suspected domain at the recursive resolver.",
            "Inspect DNS query payloads for high entropy / Base64-like patterns.",
            "Roll affected workstations to a clean image — DNS tunnelling implies compromise.",
        ],
        "min_severity": "high",
    },
    "BRUTE_FORCE": {
        "summary": "Likely credential brute-force or login-flood attack.",
        "recommendations": [
            "Lock the targeted account(s) and require a password reset.",
            "Add the source IP to the deny-list and review for distributed brute-force attempts.",
            "Verify MFA is enforced on the targeted service.",
        ],
        "min_severity": "high",
    },
    # ── IoT (Kitsune-feature) patterns ─────────────────────────────────
    "MIRAI_FLOOD": {
        "summary": "Likely Mirai-style flood from an IoT device.",
        "recommendations": [
            "Isolate the source IoT device from the network — it is likely compromised.",
            "Capture the device for forensics (firmware image / memory dump if possible).",
            "Look for similar bursts from other IoT devices on the same VLAN — Mirai propagates laterally.",
        ],
        "min_severity": "critical",
    },
    "IOT_SCAN": {
        "summary": "IoT device probing many sockets — likely worm propagation.",
        "recommendations": [
            "Quarantine the source IoT device — it appears to be scanning for new victims.",
            "Block the device's outbound traffic at the gateway.",
            "Audit other IoT devices for indicators of compromise (open Telnet/2323, default creds).",
        ],
        "min_severity": "high",
    },
    "SLOW_BEACON": {
        "summary": "Low-and-slow IoT beacon — possible C2 over a long window.",
        "recommendations": [
            "Acquire a packet capture of the suspect device for offline analysis.",
            "Look for similar long-window beacons across other IoT devices — C2 channels often persist.",
            "If the destination resolves to a non-vendor domain, escalate to incident response.",
        ],
        "min_severity": "high",
    },
    "GENERIC": {
        "summary": "Anomalous flow flagged by the model — no specific attack pattern recognised.",
        "recommendations": [
            "Review the top contributing features manually.",
            "Cross-reference with threat intelligence on the source / destination IP.",
            "If the source is internal, monitor for repeated similar flows.",
        ],
        "min_severity": "medium",
    },
}


# ══════════════════════════════════════════════════════════════════════
# Severity calibration
# ══════════════════════════════════════════════════════════════════════

PATTERN_THRESHOLD = 0.55   # min matcher score to commit to a named pattern

# Severity ladder: low < medium < high < critical
_SEV_LADDER = ["low", "medium", "high", "critical"]


def _calibrate_severity(confidence: float, pattern_min: str) -> str:
    """
    Map (confidence, pattern's minimum severity) → final severity.

    Bias is intentionally toward HIGHER severity at moderate confidence —
    the project's evaluation rules emphasise recall (catching real botnets)
    over precision (avoiding false alarms). A 0.6-confidence detection that
    matches a known attack pattern should not be downgraded to low.
    """
    base = _SEV_LADDER.index(pattern_min)

    if confidence >= 0.95:
        bumped = base + 1                 # very confident → upgrade one tier
    elif confidence >= 0.80:
        bumped = base                     # confident → keep
    elif confidence >= 0.60:
        bumped = base                     # moderate → still keep (recall priority)
    else:
        bumped = max(0, base - 1)         # low confidence → downgrade one tier

    bumped = min(bumped, len(_SEV_LADDER) - 1)
    return _SEV_LADDER[bumped]


# ══════════════════════════════════════════════════════════════════════
# Reason generation
# ══════════════════════════════════════════════════════════════════════

def _format_value(feature: str, value: float) -> str:
    """Format a feature value in a human-friendly way."""
    if feature in {"flag_FIN", "flag_SYN", "flag_RST", "flag_PSH",
                   "flag_ACK", "flag_URG",
                   "total_fwd_packets", "total_bwd_packets",
                   "dns_query_count",
                   "window_flow_count", "window_unique_dsts"}:
        return f"{int(value)}"
    if feature in {"src_port", "dst_port", "protocol"}:
        return f"{int(value)}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _build_reasons(expl: LocalExplanation, max_reasons: int = 4) -> list[str]:
    """Convert top-K feature contributions into bullet-point sentences."""
    reasons: list[str] = []
    for f in expl.top_features[:max_reasons]:
        sign = "+" if f.attribution >= 0 else ""
        val_str = _format_value(f.feature, f.value)
        reasons.append(
            f"{f.display} = {val_str}  "
            f"({f.direction}, contribution {sign}{f.attribution:.3f})"
        )
    return reasons


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def build_human_explanation(expl: LocalExplanation) -> HumanExplanation:
    """
    Run the appropriate pattern matchers (flow-schema or Kitsune-schema
    depending on what's in expl.top_features), pick the highest-scoring
    pattern (if any beats PATTERN_THRESHOLD), and assemble the analyst-
    facing summary.

    For benign predictions we still produce a (low-severity) explanation so
    the analyst can audit the model's reasoning. The recommendations list
    will be empty for benign cases.
    """
    if expl.prediction != "botnet":
        return HumanExplanation(
            summary  = "Flow appears benign — no suspicious pattern detected.",
            pattern  = "GENERIC",
            severity = "low",
            reasons  = _build_reasons(expl, max_reasons=3),
            recommendations = [],
        )

    # Auto-dispatch: IoT detections produce Kitsune-named top features,
    # Non-IoT detections produce flow-schema names. Different matchers.
    if _is_kitsune_explanation(expl):
        scores = {
            "MIRAI_FLOOD": _detect_mirai_flood(expl),
            "IOT_SCAN":    _detect_iot_scan(expl),
            "SLOW_BEACON": _detect_slow_beacon(expl),
        }
    else:
        scores = {
            "PORT_SCAN":   _detect_port_scan(expl),
            "DDOS":        _detect_ddos(expl),
            "C2_BEACON":   _detect_c2_beacon(expl),
            "DNS_TUNNEL":  _detect_dns_tunnel(expl),
            "BRUTE_FORCE": _detect_brute_force(expl),
        }
    best_pattern, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score >= PATTERN_THRESHOLD:
        info = PATTERNS[best_pattern]
        return HumanExplanation(
            summary  = info["summary"],
            pattern  = best_pattern,
            severity = _calibrate_severity(expl.confidence, info["min_severity"]),
            reasons  = _build_reasons(expl),
            recommendations = list(info["recommendations"]),
        )

    # Fallback — no specific pattern matched confidently
    info = PATTERNS["GENERIC"]
    return HumanExplanation(
        summary  = info["summary"],
        pattern  = "GENERIC",
        severity = _calibrate_severity(expl.confidence, info["min_severity"]),
        reasons  = _build_reasons(expl),
        recommendations = list(info["recommendations"]),
    )


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Synthesise a port-scan-like attribution to prove the matcher works.
    fake = LocalExplanation(
        prediction="botnet",
        confidence=0.88,
        method="integrated_gradients",
        raw_attributions={},
        top_features=[
            FeatureContribution("window_unique_dsts", "Unique destinations",
                                value=42.0, attribution=+0.35,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("window_flow_count", "Flows in window",
                                value=120.0, attribution=+0.28,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("flow_duration", "Flow duration",
                                value=0.04, attribution=-0.20,
                                direction="↓ pushed toward benign"),
            FeatureContribution("total_fwd_packets", "Forward packets",
                                value=2, attribution=-0.15,
                                direction="↓ pushed toward benign"),
            FeatureContribution("flag_SYN", "SYN flag count",
                                value=18, attribution=+0.10,
                                direction="↑ pushed toward botnet"),
        ],
    )
    h = build_human_explanation(fake)
    print(f"Pattern : {h.pattern}")
    print(f"Severity: {h.severity}")
    print(f"Summary : {h.summary}\n")
    print("Reasons:")
    for r in h.reasons:
        print(f"  · {r}")
    print("\nRecommendations:")
    for r in h.recommendations:
        print(f"  → {r}")

    # Test the LIVE_CONSTANT guard: window features stuck at constants
    # should NOT trigger PORT_SCAN.
    print("\n" + "="*60)
    print("Guard test: constant-valued window features should NOT fire PORT_SCAN")
    print("="*60)
    fake_const = LocalExplanation(
        prediction="botnet",
        confidence=0.65,
        method="integrated_gradients",
        raw_attributions={},
        top_features=[
            FeatureContribution("window_unique_dsts", "Unique destinations",
                                value=1.0, attribution=+0.20,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("window_flow_count", "Flows in window",
                                value=1.0, attribution=+0.18,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("flow_duration", "Flow duration",
                                value=15.0, attribution=-0.10,
                                direction="↓ pushed toward benign"),
            FeatureContribution("total_fwd_packets", "Forward packets",
                                value=50, attribution=-0.08,
                                direction="↓ pushed toward benign"),
        ],
    )
    h2 = build_human_explanation(fake_const)
    print(f"Pattern: {h2.pattern}  (expect: GENERIC)")
    assert h2.pattern == "GENERIC", "BUG: guard didn't suppress false PORT_SCAN!"
    print("PASS — constant-valued window features correctly did NOT fire PORT_SCAN.")