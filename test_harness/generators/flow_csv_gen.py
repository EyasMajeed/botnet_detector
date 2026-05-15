"""
flow_csv_gen.py — Synthetic CSV generators aligned to the project's
56-feature S1_FEATURES schema.

We mirror exactly what app/inference_bridge.run_file_inference reads:
the CSV must have S1_FEATURES columns, plus optional src_ip and label.

This generator is also used to feed the cross-dataset tests with
controlled distributions (class imbalance, port memorisation ablation,
empty-flow ratio, etc.).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Optional

# The 56 features used by Stage-1 — copied verbatim from
# models/stage1/s1_scaler.json so the harness does not depend on the
# project being importable. Any rename in the project is caught by the
# schema-drift test (ml_tests/t_schema_drift).
S1_FEATURES: list[str] = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG",
    "protocol", "src_port", "dst_port",
    "flow_active_time", "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "window_flow_count", "window_unique_dsts",
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max",
    "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
    "tls_features_available",
]
assert len(S1_FEATURES) == 56


# ── Random sampling helpers ────────────────────────────────────────────────

def _benign_iot_row(rng: random.Random) -> dict:
    pps = rng.uniform(0.5, 5.0)
    bps = pps * rng.uniform(40, 200)
    dur = rng.uniform(1.0, 60.0)
    iat = 1.0 / max(pps, 1e-3)
    return {
        "flow_duration": dur,
        "total_fwd_packets": int(pps * dur * 0.6),
        "total_bwd_packets": int(pps * dur * 0.4),
        "total_fwd_bytes":   int(bps * dur * 0.6),
        "total_bwd_bytes":   int(bps * dur * 0.4),
        "fwd_pkt_len_min": 40,  "fwd_pkt_len_max": 200,
        "fwd_pkt_len_mean": 80, "fwd_pkt_len_std": 20,
        "bwd_pkt_len_min": 40,  "bwd_pkt_len_max": 200,
        "bwd_pkt_len_mean": 80, "bwd_pkt_len_std": 20,
        "flow_bytes_per_sec": bps, "flow_pkts_per_sec": pps,
        "flow_iat_mean": iat, "flow_iat_std": iat * 0.2,
        "flow_iat_min":  iat * 0.5, "flow_iat_max": iat * 2.0,
        "fwd_iat_mean": iat, "fwd_iat_std": iat * 0.2,
        "fwd_iat_min":  iat * 0.5, "fwd_iat_max": iat * 2.0,
        "bwd_iat_mean": iat, "bwd_iat_std": iat * 0.2,
        "bwd_iat_min":  iat * 0.5, "bwd_iat_max": iat * 2.0,
        "fwd_header_length": 20, "bwd_header_length": 20,
        "flag_FIN": 1, "flag_SYN": 1, "flag_RST": 0, "flag_PSH": 1,
        "flag_ACK": 1, "flag_URG": 0,
        "protocol": 6,
        "src_port":  rng.randint(40000, 60000),
        "dst_port":  rng.choice([1883, 8883, 5683, 5353]),  # IoT-ish
        "flow_active_time": dur, "flow_idle_time": 0.0,
        "bytes_per_sec_window": bps, "pkts_per_sec_window": pps,
        "periodicity_score": rng.uniform(0.4, 0.7),
        "burst_rate": rng.uniform(0.0, 0.2),
        "window_flow_count": 1, "window_unique_dsts": 1,
        "ttl_mean": 64, "ttl_std": 0, "ttl_min": 64, "ttl_max": 64,
        "dns_query_count": 0,
        "payload_bytes_mean": 60, "payload_bytes_std": 20,
        "payload_zero_ratio": 0.05, "payload_entropy": 4.5,
        "tls_features_available": 0,
    }


def _benign_noniot_row(rng: random.Random) -> dict:
    pps = rng.uniform(20, 200)
    bps = pps * rng.uniform(200, 1500)
    dur = rng.uniform(0.1, 30.0)
    iat = 1.0 / max(pps, 1e-3)
    return {
        "flow_duration": dur,
        "total_fwd_packets": int(pps * dur * 0.6),
        "total_bwd_packets": int(pps * dur * 0.4),
        "total_fwd_bytes":   int(bps * dur * 0.6),
        "total_bwd_bytes":   int(bps * dur * 0.4),
        "fwd_pkt_len_min": 60,  "fwd_pkt_len_max": 1500,
        "fwd_pkt_len_mean": 600, "fwd_pkt_len_std": 300,
        "bwd_pkt_len_min": 60,  "bwd_pkt_len_max": 1500,
        "bwd_pkt_len_mean": 600, "bwd_pkt_len_std": 300,
        "flow_bytes_per_sec": bps, "flow_pkts_per_sec": pps,
        "flow_iat_mean": iat, "flow_iat_std": iat * 0.5,
        "flow_iat_min":  iat * 0.1, "flow_iat_max": iat * 5.0,
        "fwd_iat_mean": iat, "fwd_iat_std": iat * 0.5,
        "fwd_iat_min":  iat * 0.1, "fwd_iat_max": iat * 5.0,
        "bwd_iat_mean": iat, "bwd_iat_std": iat * 0.5,
        "bwd_iat_min":  iat * 0.1, "bwd_iat_max": iat * 5.0,
        "fwd_header_length": 20, "bwd_header_length": 20,
        "flag_FIN": 1, "flag_SYN": 1, "flag_RST": 0, "flag_PSH": 1,
        "flag_ACK": 1, "flag_URG": 0,
        "protocol": 6,
        "src_port":  rng.randint(40000, 60000),
        "dst_port":  rng.choice([80, 443, 8080]),
        "flow_active_time": dur * 0.95, "flow_idle_time": dur * 0.05,
        "bytes_per_sec_window": bps, "pkts_per_sec_window": pps,
        "periodicity_score": rng.uniform(0.0, 0.3),
        "burst_rate": rng.uniform(0.0, 0.4),
        "window_flow_count": rng.randint(1, 5),
        "window_unique_dsts": rng.randint(1, 4),
        "ttl_mean": 64, "ttl_std": 1, "ttl_min": 63, "ttl_max": 64,
        "dns_query_count": rng.randint(0, 3),
        "payload_bytes_mean": 500, "payload_bytes_std": 200,
        "payload_zero_ratio": 0.02, "payload_entropy": 7.0,
        "tls_features_available": 1,
    }


def _botnet_noniot_row(rng: random.Random) -> dict:
    """Periodic small packets with high SYN/ACK ratio and unusual port."""
    pps = rng.uniform(200, 800)
    bps = pps * rng.uniform(40, 200)
    dur = rng.uniform(0.05, 5.0)
    iat = 1.0 / max(pps, 1e-3)
    return {
        "flow_duration": dur,
        "total_fwd_packets": int(pps * dur),
        "total_bwd_packets": int(pps * dur * 0.05),
        "total_fwd_bytes":   int(bps * dur),
        "total_bwd_bytes":   int(bps * dur * 0.05),
        "fwd_pkt_len_min": 40,  "fwd_pkt_len_max": 80,
        "fwd_pkt_len_mean": 50, "fwd_pkt_len_std":  5,
        "bwd_pkt_len_min": 0,   "bwd_pkt_len_max":  0,
        "bwd_pkt_len_mean": 0,  "bwd_pkt_len_std": 0,
        "flow_bytes_per_sec": bps, "flow_pkts_per_sec": pps,
        "flow_iat_mean": iat, "flow_iat_std": iat * 0.05,
        "flow_iat_min":  iat * 0.95, "flow_iat_max": iat * 1.05,
        "fwd_iat_mean": iat, "fwd_iat_std": iat * 0.05,
        "fwd_iat_min":  iat * 0.95, "fwd_iat_max": iat * 1.05,
        "bwd_iat_mean": 0.0, "bwd_iat_std": 0.0,
        "bwd_iat_min":  0.0, "bwd_iat_max": 0.0,
        "fwd_header_length": 20, "bwd_header_length": 0,
        "flag_FIN": 0, "flag_SYN": int(pps * dur * 0.9), "flag_RST": 1,
        "flag_PSH": 0, "flag_ACK": int(pps * dur * 0.1), "flag_URG": 0,
        "protocol": 6,
        "src_port": rng.randint(40000, 60000),
        "dst_port": rng.choice([23, 2323, 6667, 31337, 8443]),
        "flow_active_time": dur, "flow_idle_time": 0.0,
        "bytes_per_sec_window": bps, "pkts_per_sec_window": pps,
        "periodicity_score": rng.uniform(0.85, 0.99),
        "burst_rate": rng.uniform(0.7, 0.99),
        "window_flow_count": rng.randint(20, 200),
        "window_unique_dsts": rng.randint(10, 200),
        "ttl_mean": 64, "ttl_std": 0, "ttl_min": 64, "ttl_max": 64,
        "dns_query_count": 0,
        "payload_bytes_mean": 0, "payload_bytes_std": 0,
        "payload_zero_ratio": 1.0, "payload_entropy": 0.0,
        "tls_features_available": 0,
    }


def _empty_flow_row() -> dict:
    """Documented CIC-IDS-2018 empty-flow shape: zero forward bytes,
    zero backward packets — what the Non-IoT model returns ~0.0068 for."""
    return {f: 0.0 for f in S1_FEATURES} | {"protocol": 6, "src_port": 12345, "dst_port": 80}


# ── Public API ─────────────────────────────────────────────────────────────

def write_csv(rows: Iterable[dict], out: Path,
              extra_cols: Optional[list[str]] = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = extra_cols or []
    fields = S1_FEATURES + extra_cols
    rows = list(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0) for k in fields})
    return out


def synthetic_csv(out: Path, n_iot: int, n_noniot: int, n_botnet: int,
                  n_empty: int = 0, seed: int = 0,
                  with_src_ip: bool = True,
                  unique_src_ips: Optional[int] = None) -> Path:
    """
    Compose a CSV with explicit per-class counts.

    `unique_src_ips` controls how many distinct source IPs appear — small
    values exercise the per-src_ip Stage-2 sliding window; very large
    values trigger the window-starvation regime.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    label_col = "class_label"
    extra = ["src_ip", label_col] if with_src_ip else [label_col]

    def _ip_for(idx: int) -> str:
        if unique_src_ips is None or unique_src_ips <= 0:
            return f"10.0.{(idx >> 8) & 0xFF}.{idx & 0xFF}"
        return f"10.0.{((idx % unique_src_ips) >> 8) & 0xFF}.{((idx % unique_src_ips) & 0xFF) or 1}"

    idx = 0
    for i in range(n_iot):
        r = _benign_iot_row(rng); r["src_ip"] = _ip_for(idx); r[label_col] = "benign"; rows.append(r); idx += 1
    for i in range(n_noniot):
        r = _benign_noniot_row(rng); r["src_ip"] = _ip_for(idx); r[label_col] = "benign"; rows.append(r); idx += 1
    for i in range(n_botnet):
        r = _botnet_noniot_row(rng); r["src_ip"] = _ip_for(idx); r[label_col] = "botnet"; rows.append(r); idx += 1
    for i in range(n_empty):
        r = _empty_flow_row(); r["src_ip"] = _ip_for(idx); r[label_col] = "botnet"; rows.append(r); idx += 1

    rng.shuffle(rows)
    return write_csv(rows, out, extra_cols=extra)


def slow_beacon_csv(out: Path, n_beacons: int = 60,
                    period_sec: float = 35.0, seed: int = 0) -> Path:
    """One-row-per-beacon CSV — each row is a full 'flow' to mimic the way
    the upload-path defeats the LSTM on slow beaconing (each beacon ends
    up a separate flow because of the 30 s idle timeout)."""
    rng = random.Random(seed)
    rows = []
    for k in range(n_beacons):
        r = _botnet_noniot_row(rng)
        r["flow_duration"]      = 0.05
        r["total_fwd_packets"]  = 1
        r["flow_pkts_per_sec"]  = 20.0
        r["periodicity_score"]  = 0.99
        r["burst_rate"]         = 0.0
        r["window_flow_count"]  = 1
        r["src_ip"]             = "10.0.0.7"
        r["class_label"]        = "botnet"
        rows.append(r)
    return write_csv(rows, out, extra_cols=["src_ip", "class_label"])
