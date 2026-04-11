"""
suspicion_scorer.py  —  Lightweight anomaly scorer (Section 7.3.4)
===================================================================
Runs on raw flow dicts BEFORE any ML inference.
No models, no dependencies — just fast threshold + statistical checks.

Scoring:
    0  → Normal   — forward directly to merge/correlate
    1  → Mild     — mark for frequent export; no sniffing
   ≥2  → Suspicious — trigger packet sniffing for this flow

Usage:
    from app.suspicion_scorer import SuspicionScorer
    scorer = SuspicionScorer()
    result = scorer.score(flow_dict)
    # result = {"score": int, "trigger_sniff": bool, "reasons": [str]}
"""

from __future__ import annotations
import math
from collections import deque
from typing import Optional

# ── 7.3.4.A  Hard-threshold indicators ────────────────────────────────────────
RISKY_PORTS: set[int] = {23, 2323, 1900, 5555, 9999, 4444, 6667, 31337}
HIGH_PKT_RATE   = 500          # pkts/sec  → flood-like activity
SHORT_FLOW_DUR  = 0.05         # seconds   → scan-like (very short repetitive)
LARGE_BYTES     = 50_000       # bytes     → unusually large transfer

# ── Rolling window size for statistical baselines ─────────────────────────────
WINDOW = 200   # keep last N flows for rolling mean/std


class SuspicionScorer:
    """
    Stateful scorer that maintains rolling statistics of recent flows
    so thresholds adapt to the current network baseline (Section 7.3.4.B).
    """

    def __init__(self) -> None:
        self._pps_history:   deque[float] = deque(maxlen=WINDOW)
        self._bps_history:   deque[float] = deque(maxlen=WINDOW)
        self._dur_history:   deque[float] = deque(maxlen=WINDOW)

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, flow: dict) -> dict:
        """
        Score a single flow dict.

        Expected keys (all optional — missing values default to 0):
            flow_pkts_per_sec, flow_bytes_per_sec, flow_duration,
            total_fwd_bytes, total_bwd_bytes,
            flag_SYN, flag_ACK, flag_FIN, flag_RST,
            dst_port, src_port,
            total_fwd_packets, total_bwd_packets

        Returns:
            {
                "score":         int,
                "trigger_sniff": bool,
                "reasons":       list[str]
            }
        """
        score   = 0
        reasons: list[str] = []

        pps  = float(flow.get("flow_pkts_per_sec",  0) or 0)
        bps  = float(flow.get("flow_bytes_per_sec", 0) or 0)
        dur  = float(flow.get("flow_duration",      0) or 0)
        dst  = int(  flow.get("dst_port",           0) or 0)
        src  = int(  flow.get("src_port",           0) or 0)
        syn  = int(  flow.get("flag_SYN",           0) or 0)
        ack  = int(  flow.get("flag_ACK",           0) or 0)
        rst  = int(  flow.get("flag_RST",           0) or 0)
        fwd_b = float(flow.get("total_fwd_bytes",   0) or 0)
        bwd_b = float(flow.get("total_bwd_bytes",   0) or 0)
        total_bytes = fwd_b + bwd_b

        # ── A. Threshold-based indicators ─────────────────────────────────────

        if pps > HIGH_PKT_RATE:
            score += 2
            reasons.append(f"High packet rate ({pps:.0f} pkt/s > {HIGH_PKT_RATE})")

        if 0 < dur < SHORT_FLOW_DUR:
            score += 1
            reasons.append(f"Very short flow ({dur:.4f}s — scan-like)")

        if syn and not ack:
            score += 2
            reasons.append("SYN without ACK (possible SYN flood / half-open scan)")

        if rst and not ack:
            score += 1
            reasons.append("RST without ACK (port scan indicator)")

        if dst in RISKY_PORTS or src in RISKY_PORTS:
            port = dst if dst in RISKY_PORTS else src
            score += 1
            reasons.append(f"Risky port used ({port})")

        if total_bytes > LARGE_BYTES and dur < 2.0:
            score += 2
            reasons.append(
                f"Large bytes ({total_bytes:,.0f}) in short duration ({dur:.2f}s)"
            )

        # ── B. Statistical deviation indicators ───────────────────────────────
        mean_pps, std_pps = self._stats(self._pps_history)
        mean_bps, std_bps = self._stats(self._bps_history)
        mean_dur, std_dur = self._stats(self._dur_history)

        if mean_pps > 0 and pps > mean_pps + 2 * std_pps:
            score += 2
            reasons.append(
                f"pkt/s ({pps:.1f}) > mean+2σ ({mean_pps + 2*std_pps:.1f})"
            )

        if mean_bps > 0 and bps > mean_bps + 2 * std_bps:
            score += 2
            reasons.append(
                f"bytes/s ({bps:.0f}) > mean+2σ ({mean_bps + 2*std_bps:.0f})"
            )

        if mean_dur > 0 and dur > 0 and dur < mean_dur - 2 * std_dur:
            score += 1
            reasons.append(
                f"Duration ({dur:.3f}s) << typical baseline ({mean_dur:.3f}s)"
            )

        # ── Update rolling history ─────────────────────────────────────────────
        if pps > 0:  self._pps_history.append(pps)
        if bps > 0:  self._bps_history.append(bps)
        if dur > 0:  self._dur_history.append(dur)

        return {
            "score":         score,
            "trigger_sniff": score >= 2,
            "reasons":       reasons,
        }

    def reset_baseline(self) -> None:
        """Clear rolling history (call when switching interfaces or sessions)."""
        self._pps_history.clear()
        self._bps_history.clear()
        self._dur_history.clear()

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stats(hist: deque) -> tuple[float, float]:
        """Return (mean, std) of a deque; returns (0, 1) when empty."""
        if len(hist) < 5:
            return 0.0, 1.0
        n    = len(hist)
        mean = sum(hist) / n
        var  = sum((x - mean) ** 2 for x in hist) / n
        return mean, math.sqrt(var) or 1.0
