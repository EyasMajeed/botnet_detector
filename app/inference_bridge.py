"""
inference_bridge.py — File / single-flow inference for the GUI.

Used by:
    UploadPage._run_detection → run_file_inference(info)        (batch)
    MonitorPage._on_flow      → run_inference(flow_dict)        (single)

Pipeline (file path):
    1. Read CSV with pandas
    2. Per row: build a 56-feature dict and run Stage-1 RF (with scaler)
    3. For Non-IoT rows: maintain a per-src_ip sliding window of 20
       feature rows and run Stage-2 NonIoT CNN-LSTM
    4. For IoT rows: label "unknown" — Stage-2 IoT requires raw packet
       Kitsune sequences which a CSV cannot supply. Use the live
       monitoring page (BotnetMonitor) for IoT detection.

We deliberately reuse the wrapper classes from monitoring.py
(Stage1Classifier, Stage2NonIoTDetector) instead of the half-finished
loaders in models/stage1/classifier.py — those bypass the StandardScaler
and produce wrong predictions.
"""

from __future__ import annotations

# Force torch single-threaded BEFORE the monitoring import pulls torch in.
# Same defence-in-depth as monitor_bridge.py (avoids macOS PyQt6+OpenMP
# segfault when torch.load runs after Qt has initialised libomp).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")

import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Resolve project root and put it on sys.path so `from monitoring import …`
# works whether the GUI is launched from app/ or from the project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from file_handler import FileFormat   # for format guards in run_file_inference

# Heavy imports are deferred to first use — keeps GUI startup snappy
_stage1     = None    # monitoring.Stage1Classifier
_stage2_nin = None    # monitoring.Stage2NonIoTDetector
_S1_FEATURES: List[str] = []
_NONIOT_SEQ_LEN: int = 20

USE_REAL_INFERENCE = True


# ═════════════════════════════════════════════════════════════════════════════
# Lazy loaders (real models from monitoring.py)
# ═════════════════════════════════════════════════════════════════════════════

def _get_stage1():
    """Load Stage-1 RF/XGB once. Reuses monitoring.py's properly-scaled wrapper."""
    global _stage1, _S1_FEATURES
    if _stage1 is None:
        from monitoring import (Stage1Classifier as _S1, MODEL_S1_RF,
                                SCALER_S1_JSON, S1_FEATURES)
        _stage1 = _S1(MODEL_S1_RF, SCALER_S1_JSON)
        _S1_FEATURES = list(S1_FEATURES)
    return _stage1


def _get_stage2_noniot():
    """Load Stage-2 Non-IoT CNN-LSTM once."""
    global _stage2_nin, _NONIOT_SEQ_LEN
    if _stage2_nin is None:
        from monitoring import (Stage2NonIoTDetector as _S2, MODEL_S2_NONIOT,
                                NONIOT_SEQ_LEN)
        _stage2_nin = _S2(MODEL_S2_NONIOT)
        _NONIOT_SEQ_LEN = int(NONIOT_SEQ_LEN)
    return _stage2_nin


def _ensure_features_loaded() -> List[str]:
    """Make sure _S1_FEATURES is populated even if Stage-1 was loaded before."""
    if not _S1_FEATURES:
        _get_stage1()
    return _S1_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(flow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Stage-1 + Stage-2 NonIoT on a single flow dict (used by UI fall-backs).
    For real-time monitoring use BotnetMonitor (monitor_bridge.py) instead —
    it carries the per-IP sliding-window state we don't have here.
    """
    t0  = time.perf_counter()
    s1  = _get_stage1()
    s2  = _get_stage2_noniot()
    feat = _flow_to_feat_dict(flow)
    device, s1_conf = s1.stage1_predict(feat)
    if device == "noniot":
        # Single row, no temporal context — Stage-2 will internally pad to 20.
        row    = s2.stage2_preprocess_non_iot(feat)
        seq    = np.stack([row])
        label, s2_conf = s2.stage2_predict(seq)
    else:
        label, s2_conf = "unknown", 0.0
    return {
        "label":       label,
        "confidence":  float(s2_conf),
        "device_type": device,
        "stage1_conf": float(s1_conf),
        "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


def run_file_inference(info: Any) -> List[Dict[str, Any]]:
    """
    Run full Stage-1 + Stage-2 NonIoT pipeline on every row of an uploaded
    CSV. Returns a list of result dicts, one per input row.

    Each result dict:
        {
          "row":          int,
          "src_ip":       str,
          "dst_ip":       str,
          "src_port":     int,
          "dst_port":     int,
          "protocol":     "TCP" | "UDP" | "ICMP" | str,
          "device_type":  "iot" | "noniot",
          "label":        "botnet" | "benign" | "unknown",
          "confidence":   float,        # Stage-2 sigmoid (0.0 for IoT/unknown)
          "stage1_conf":  float,
          "latency_ms":   float,        # avg per-row wall-clock (filled at end)
        }
    """
    # ── Validation ───────────────────────────────────────────────────────
    if not getattr(info, "is_valid", False):
        raise ValueError("Cannot run inference on an invalid file.")
    if info.format not in (
        FileFormat.CSV_UNIFIED,
        FileFormat.CSV_GENERIC,
        FileFormat.CSV_CICFLOW,
        FileFormat.CSV_CTU13,
        FileFormat.CSV_UNSW,
        FileFormat.NETFLOW_CSV,
    ):
        raise ValueError(
            f"Unsupported file format for inference: {info.format}. "
            "Only CSV flow exports are supported. PCAP files should be replayed "
            "through monitoring.py --pcap (live pipeline) for full Stage-1 + "
            "Stage-2 IoT/NonIoT detection."
        )

    # ── Read CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(info.path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}") from e
    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    # ── Prepare models ───────────────────────────────────────────────────
    s1   = _get_stage1()
    s2   = _get_stage2_noniot()
    feats = _ensure_features_loaded()

    # Strip whitespace from column names; downstream lookup is case-sensitive.
    df.columns = [c.strip() for c in df.columns]

    # Format-specific column aliasing: map dataset-specific names to the unified
    # 56-feature schema. Anything not aliased is filled with 0.0 per-row.
    df = _alias_columns(df, info.format)

    # Warn (don't fail) if too few unified features are present after aliasing.
    matched = sum(1 for c in feats if c in df.columns)
    if matched < 0.4 * len(feats):
        raise ValueError(
            f"CSV has only {matched}/{len(feats)} of the 56 unified features. "
            "Convert it with data_processing/process_*.py first, or upload a "
            "CSV produced by your team's preprocessing pipeline."
        )

    # ── Inference loop ──────────────────────────────────────────────────
    t0           = time.perf_counter()
    results:     List[Dict[str, Any]] = []
    noniot_bufs: Dict[str, deque]     = defaultdict(lambda: deque(maxlen=_NONIOT_SEQ_LEN))

    for i, row in df.iterrows():
        feat = {c: _safe_float(row.get(c, 0.0)) for c in feats}

        # ── Stage-1 ──────────────────────────────────────────────────────
        try:
            device, s1_conf = s1.stage1_predict(feat)
        except Exception as e:
            device, s1_conf = "noniot", 0.0
            print(f"[run_file_inference] Stage-1 failed on row {i}: {e!r}")

        # ── Stage-2 ──────────────────────────────────────────────────────
        if device == "noniot":
            try:
                row_vec = s2.stage2_preprocess_non_iot(feat)
                # Use src_ip as the per-flow conversation key; fall back to a
                # per-row id so rows without IPs each get their own (un-padded)
                # window — the model self-pads internally.
                src_ip_key = str(row.get("src_ip", "") or f"row_{i}")
                noniot_bufs[src_ip_key].append(row_vec)
                seq = np.stack(list(noniot_bufs[src_ip_key]))
                label, s2_conf = s2.stage2_predict(seq)
            except Exception as e:
                label, s2_conf = "unknown", 0.0
                print(f"[run_file_inference] Stage-2 NonIoT failed on row {i}: {e!r}")
        else:
            # IoT: Stage-2 IoT needs Kitsune packet-level sequences not in CSV.
            label, s2_conf = "unknown", 0.0

        # ── Build result dict ───────────────────────────────────────────
        proto_n = int(_safe_float(row.get("protocol", 0)))
        proto   = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(
            proto_n, str(proto_n) if proto_n else "—")

        results.append({
            "row":         int(i) + 1,
            "src_ip":      str(row.get("src_ip", "") or ""),
            "dst_ip":      str(row.get("dst_ip", "") or ""),
            "src_port":    int(_safe_float(row.get("src_port", 0))),
            "dst_port":    int(_safe_float(row.get("dst_port", 0))),
            "protocol":    proto,
            "device_type": device,
            "label":       label,
            "confidence":  float(s2_conf),
            "stage1_conf": float(s1_conf),
        })

    # Distribute total wall-clock latency evenly per row (avg)
    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms   = round(total_ms / max(len(results), 1), 2)
    for r in results:
        r["latency_ms"] = avg_ms
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _safe_float(v: Any) -> float:
    """Coerce anything to float, returning 0.0 on failure (NaN, None, str)."""
    try:
        f = float(v)
        if not np.isfinite(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _flow_to_feat_dict(flow: Dict[str, Any]) -> Dict[str, float]:
    """
    Map an arbitrary flow dict (from LiveCaptureThread or upload row) to the
    56-feature dict Stage-1 expects. Missing keys default to 0.0.
    """
    feats = _ensure_features_loaded()
    out: Dict[str, float] = {}
    for k in feats:
        out[k] = _safe_float(flow.get(k, 0.0))
    return out


def _alias_columns(df: pd.DataFrame, fmt: FileFormat) -> pd.DataFrame:
    """
    Best-effort column rename so dataset-specific exports work with the
    56-feature unified schema. Returns a renamed copy when aliases applied,
    otherwise the original df unchanged.

    NOTE: this does NOT recompute features — it only renames matching columns.
    For full coverage on raw CIC-IDS / CTU-13 dumps, run them through
    data_processing/process_cicids2017.py or process_ctu13.py first.
    """
    aliases: Dict[str, Dict[str, str]] = {
        # CICFlowMeter standard → unified
        "cicflow": {
            "Flow Duration":              "flow_duration",
            "Total Fwd Packets":          "total_fwd_packets",
            "Total Backward Packets":     "total_bwd_packets",
            "Total Length of Fwd Packets":"total_fwd_bytes",
            "Total Length of Bwd Packets":"total_bwd_bytes",
            "Fwd Packet Length Min":      "fwd_pkt_len_min",
            "Fwd Packet Length Max":      "fwd_pkt_len_max",
            "Fwd Packet Length Mean":     "fwd_pkt_len_mean",
            "Fwd Packet Length Std":      "fwd_pkt_len_std",
            "Bwd Packet Length Min":      "bwd_pkt_len_min",
            "Bwd Packet Length Max":      "bwd_pkt_len_max",
            "Bwd Packet Length Mean":     "bwd_pkt_len_mean",
            "Bwd Packet Length Std":      "bwd_pkt_len_std",
            "Flow Bytes/s":               "flow_bytes_per_sec",
            "Flow Packets/s":             "flow_pkts_per_sec",
            "Flow IAT Mean":              "flow_iat_mean",
            "Flow IAT Std":               "flow_iat_std",
            "Flow IAT Min":               "flow_iat_min",
            "Flow IAT Max":               "flow_iat_max",
            "Fwd IAT Mean":               "fwd_iat_mean",
            "Fwd IAT Std":                "fwd_iat_std",
            "Fwd IAT Min":                "fwd_iat_min",
            "Fwd IAT Max":                "fwd_iat_max",
            "Bwd IAT Mean":               "bwd_iat_mean",
            "Bwd IAT Std":                "bwd_iat_std",
            "Bwd IAT Min":                "bwd_iat_min",
            "Bwd IAT Max":                "bwd_iat_max",
            "Fwd Header Length":          "fwd_header_length",
            "Bwd Header Length":          "bwd_header_length",
            "FIN Flag Count":             "flag_FIN",
            "SYN Flag Count":             "flag_SYN",
            "RST Flag Count":             "flag_RST",
            "PSH Flag Count":             "flag_PSH",
            "ACK Flag Count":             "flag_ACK",
            "URG Flag Count":             "flag_URG",
            "Protocol":                   "protocol",
            "Source Port":                "src_port",
            "Destination Port":           "dst_port",
            "Source IP":                  "src_ip",
            "Destination IP":             "dst_ip",
        },
        # CTU-13 binetflow → unified
        "ctu13": {
            "Dur":      "flow_duration",
            "Proto":    "protocol",
            "SrcAddr":  "src_ip",
            "Sport":    "src_port",
            "DstAddr":  "dst_ip",
            "Dport":    "dst_port",
            "TotPkts":  "total_fwd_packets",
            "TotBytes": "total_fwd_bytes",
        },
    }
    table = (aliases["cicflow"] if fmt == FileFormat.CSV_CICFLOW
             else aliases["ctu13"] if fmt == FileFormat.CSV_CTU13
             else None)
    if table is None:
        return df
    rename = {src: dst for src, dst in table.items() if src in df.columns}
    if not rename:
        return df
    return df.rename(columns=rename)