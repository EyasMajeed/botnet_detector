"""
explanation_engine.py — Human-Readable Explanations & Recommendations
======================================================================
Group 07 | CPCS499 | XAI Module

Purpose
-------
Consumes a `LocalExplanation` (from local_explainer.py) and produces:
  · A short summary sentence  ("This flow looks like a port scan.")
  · 2–4 bullet-point reasons    (referencing top features in plain English)
  · A recommended action list   (what the analyst should do next)
  · A severity tag              (low / medium / high / critical)

This satisfies report Section 4.4: "human-readable explanations of suspicious
behaviours + simple rule-based recommendations to guide analysts on next
steps."

Design
------
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

Patterns recognised
-------------------
  PORT_SCAN     : many flows in a window + many unique dsts + few packets/flow
  DDOS          : sustained very-high pkts/sec + fwd-heavy + low IAT
  C2_BEACON     : high periodicity_score + low IAT std + small uniform packets
  DNS_TUNNEL    : high dns_query_count + high payload_entropy
  BRUTE_FORCE   : many SYN flags + many connections to same dst + repeated RSTs
  GENERIC       : fallback — uses the top-3 attributions verbatim
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from src.xai.local_explainer import LocalExplanation, FeatureContribution


# ══════════════════════════════════════════════════════════════════════
# Output schema
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HumanExplanation:
    """The final analyst-facing object."""
    summary:         str            = ""
    pattern:         str            = "GENERIC"
    severity:        str            = "low"        # low|medium|high|critical
    reasons:         list[str]      = field(default_factory=list)
    recommendations: list[str]      = field(default_factory=list)
    confidence:      float          = 0.0
    prediction:      str            = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# Severity calibration
# ══════════════════════════════════════════════════════════════════════
# Rationale (recall-focused per project's evaluation rules):
#  - We bias UPWARD on severity at lower confidence to minimise missed
#    alerts (false negatives). The dashboard analyst is the final filter.
#  - "critical" reserved for confidence ≥ 0.9 AND a recognised pattern,
#    so we do not cry wolf on every flagged flow.

def _severity(prediction: str, confidence: float, pattern: str) -> str:
    if prediction == "benign":
        return "low"
    if confidence >= 0.90 and pattern != "GENERIC":
        return "critical"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


# ══════════════════════════════════════════════════════════════════════
# Pattern matchers
# ══════════════════════════════════════════════════════════════════════
# Each matcher inspects:
#   1. The top-attributed features (whether they appear in the top-K)
#   2. The actual feature values  (whether they exceed thresholds)
# It returns a confidence score 0–1 for that pattern.
#
# Thresholds are chosen from common botnet-detection literature; tune
# during CPCS499 once you have validation set numbers in hand.
# ══════════════════════════════════════════════════════════════════════

def _values(expl: LocalExplanation) -> dict[str, float]:
    """Map feature name → observed value (from the top-K in this explanation)."""
    return {f.feature: f.value for f in expl.top_features}


def _top_feature_names(expl: LocalExplanation, k: int = 5) -> set[str]:
    return {f.feature for f in expl.top_features[:k]}


def _detect_port_scan(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "window_unique_dsts" in top and v.get("window_unique_dsts", 0) >= 10:
        score += 0.45
    if "window_flow_count" in top and v.get("window_flow_count", 0) >= 20:
        score += 0.30
    # Short flows + few packets per flow
    if v.get("flow_duration", 999) < 1.0 and v.get("total_fwd_packets", 999) <= 3:
        score += 0.25
    return min(score, 1.0)


def _detect_ddos(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "flow_pkts_per_sec" in top and v.get("flow_pkts_per_sec", 0) >= 1000:
        score += 0.40
    if "pkts_per_sec_window" in top and v.get("pkts_per_sec_window", 0) >= 1000:
        score += 0.30
    # Heavily fwd-skewed (asymmetric attack)
    fwd = v.get("total_fwd_packets", 0)
    bwd = v.get("total_bwd_packets", 1)
    if fwd > 0 and bwd >= 0 and (fwd / max(bwd, 1)) >= 10:
        score += 0.20
    if v.get("flow_iat_mean", 1.0) < 0.001:
        score += 0.10
    return min(score, 1.0)


def _detect_c2_beacon(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "periodicity_score" in top and v.get("periodicity_score", 0) >= 0.7:
        score += 0.45
    if "flow_iat_std" in top and v.get("flow_iat_std", 999) < 0.05:
        score += 0.20
    # Small, uniform packets (control traffic)
    if v.get("fwd_pkt_len_mean", 9999) < 200 and v.get("fwd_pkt_len_std", 9999) < 50:
        score += 0.20
    if v.get("payload_zero_ratio", 0) >= 0.5:
        score += 0.15
    return min(score, 1.0)


def _detect_dns_tunnel(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "dns_query_count" in top and v.get("dns_query_count", 0) >= 50:
        score += 0.50
    if "payload_entropy" in top and v.get("payload_entropy", 0) >= 7.5:
        score += 0.30
    if v.get("dst_port", 0) == 53 and v.get("flow_bytes_per_sec", 0) > 1000:
        score += 0.20
    return min(score, 1.0)


def _detect_brute_force(expl: LocalExplanation) -> float:
    v = _values(expl)
    top = _top_feature_names(expl)
    score = 0.0
    if "flag_SYN" in top and v.get("flag_SYN", 0) >= 5:
        score += 0.30
    if "flag_RST" in top and v.get("flag_RST", 0) >= 3:
        score += 0.30
    # Common brute-force target ports
    if int(v.get("dst_port", 0)) in {22, 23, 3389, 5900, 21, 445}:
        score += 0.40
    return min(score, 1.0)


# Registry: pattern_name → (matcher_fn, default_summary, default_recs)
PATTERNS = {
    "PORT_SCAN": {
        "matcher": _detect_port_scan,
        "summary": "Likely port-scan / reconnaissance behaviour.",
        "recommendations": [
            "Block or rate-limit the source IP at the firewall.",
            "Check whether the source is an authorised scanner (e.g. internal asset inventory).",
            "Audit logs of the targeted hosts for follow-up exploitation attempts.",
        ],
    },
    "DDOS": {
        "matcher": _detect_ddos,
        "summary": "Likely volumetric DDoS or flood attack.",
        "recommendations": [
            "Engage upstream DDoS mitigation (provider scrubbing or rate-limiting).",
            "Drop traffic from the source at the perimeter if not legitimate.",
            "Confirm whether the target service is degraded; consider failover.",
        ],
    },
    "C2_BEACON": {
        "matcher": _detect_c2_beacon,
        "summary": "Periodic communication consistent with a C2 (command-and-control) beacon.",
        "recommendations": [
            "Quarantine the source endpoint and capture a memory image for forensics.",
            "Investigate the destination IP/domain on threat-intelligence feeds.",
            "Search the network for other hosts contacting the same destination.",
        ],
    },
    "DNS_TUNNEL": {
        "matcher": _detect_dns_tunnel,
        "summary": "Likely DNS tunnelling or DGA-driven C2 over DNS.",
        "recommendations": [
            "Inspect DNS query names for high-entropy or unusually long subdomains.",
            "Force DNS through a filtering resolver and block the suspicious domain.",
            "Investigate the source endpoint for DNS-tunnel client malware.",
        ],
    },
    "BRUTE_FORCE": {
        "matcher": _detect_brute_force,
        "summary": "Likely credential brute-force / login flood.",
        "recommendations": [
            "Lock or throttle the targeted accounts; enable account-lockout policies.",
            "Block the source IP if not an authorised admin path.",
            "Review authentication logs for any successful login from that source.",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════
# Generic fallback
# ══════════════════════════════════════════════════════════════════════

def _generic_explanation(expl: LocalExplanation) -> tuple[str, list[str], list[str]]:
    """When no pattern matches: just report the top-3 features verbatim."""
    summary = (
        f"Model classified this flow as {expl.prediction} "
        f"({expl.confidence*100:.1f}% confidence). "
        "No specific attack pattern recognised; review the contributing features."
    )
    reasons = [
        f"{f.display} = {f.value:.4g} ({f.direction}, contribution {f.attribution:+.3f})"
        for f in expl.top_features[:3]
    ]
    recs = [
        "Manually review the flow's full feature set for context.",
        "Cross-reference the source/destination IPs with threat intelligence.",
        "If the alert is a false positive, label this flow and use it for retraining.",
    ]
    return summary, reasons, recs


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

PATTERN_THRESHOLD = 0.55   # min matcher score to commit to a named pattern


def build_human_explanation(expl: LocalExplanation) -> HumanExplanation:
    """
    Convert a `LocalExplanation` into an analyst-ready `HumanExplanation`.

    For benign predictions, returns a short note without recommendations.
    For botnet predictions, runs the rule engine and selects the highest-
    scoring pattern (if any beats PATTERN_THRESHOLD).
    """
    # ── Benign branch ───────────────────────────────────────────────
    if expl.prediction == "benign":
        return HumanExplanation(
            summary    = f"Flow appears benign ({expl.confidence*100:.1f}% confidence).",
            pattern    = "BENIGN",
            severity   = "low",
            reasons    = [
                f"{f.display} = {f.value:.4g} ({f.direction})"
                for f in expl.top_features[:2]
            ],
            recommendations = [],
            confidence = expl.confidence,
            prediction = expl.prediction,
        )

    # ── Botnet branch: try each pattern ─────────────────────────────
    pattern_scores = {name: spec["matcher"](expl) for name, spec in PATTERNS.items()}
    best_pattern, best_score = max(pattern_scores.items(), key=lambda kv: kv[1])

    if best_score >= PATTERN_THRESHOLD:
        spec = PATTERNS[best_pattern]
        summary = spec["summary"]
        recs    = spec["recommendations"]
        # Reasons: cite the top-3 features that the model itself relied on
        reasons = [
            f"{f.display} = {f.value:.4g} ({f.direction}, contribution {f.attribution:+.3f})"
            for f in expl.top_features[:3]
        ]
    else:
        best_pattern = "GENERIC"
        summary, reasons, recs = _generic_explanation(expl)

    return HumanExplanation(
        summary        = summary,
        pattern        = best_pattern,
        severity       = _severity(expl.prediction, expl.confidence, best_pattern),
        reasons        = reasons,
        recommendations= recs,
        confidence     = expl.confidence,
        prediction     = expl.prediction,
    )


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Construct a fake LocalExplanation that LOOKS like a port scan,
    # to verify the rule engine fires correctly.
    fake = LocalExplanation(
        prediction="botnet",
        confidence=0.88,
        method="integrated_gradients",
        raw_attributions={},
        top_features=[
            FeatureContribution("window_unique_dsts", "Unique destinations",
                                value=42, attribution=+0.35,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("window_flow_count",  "Flows in window",
                                value=120, attribution=+0.28,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("flow_duration",      "Flow duration",
                                value=0.04, attribution=-0.20,
                                direction="↑ pushed toward botnet"),
            FeatureContribution("total_fwd_packets",  "Forward packets",
                                value=2,  attribution=-0.15,
                                direction="↑ pushed toward botnet"),
        ],
    )

    h = build_human_explanation(fake)
    print("=" * 60)
    print(f"PATTERN  : {h.pattern}")
    print(f"SEVERITY : {h.severity}")
    print(f"SUMMARY  : {h.summary}")
    print(f"\nReasons:")
    for r in h.reasons:
        print(f"  · {r}")
    print(f"\nRecommendations:")
    for r in h.recommendations:
        print(f"  → {r}")
    print("=" * 60)
