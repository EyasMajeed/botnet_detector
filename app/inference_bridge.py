"""
inference_bridge.py  —  Single integration point for ML inference
==================================================================
Two public entry points used by the GUI:

    run_inference(flow_dict)        — single live flow (from monitor_page.py)
    run_file_inference(file_info)   — uploaded file (from upload_page.py)

Both go through Stage-1 (RF) and the Stage-2 IoT CNN-LSTM when
USE_REAL_INFERENCE is True. Non-IoT flows currently short-circuit to
"benign" because the Non-IoT detector is wired up in Phase C — see TODO.

Feature list reference: models/stage1/classifier.py → ALL_FEATURES (56).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Make project root importable so `models.*` resolves regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.stage1.classifier import Stage1Classifier
from models.stage2.iot_detector import Stage2Detector as Stage2IoTDetector
from models.stage2.noniot_detector_cnnlstm import Stage2Detector as Stage2NonIoTDetector
from file_handler import FileFormat


# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_PATH_S1        = ROOT / "models" / "stage1" / "rf_model.pkl"
_MODEL_PATH_S2_IOT    = ROOT / "models" / "stage2" / "iot_cnn_lstm.pt"
_MODEL_PATH_S2_NONIOT = ROOT / "models" / "stage2" / "noniot_cnn_lstm.pt"

# ── Flip to False to use the random stub instead of the trained models ────────
USE_REAL_INFERENCE = True


# ── Lazy singletons ───────────────────────────────────────────────────────────
_stage1: Stage1Classifier | None = None
_stage2_iot: Stage2IoTDetector | None = None
_stage2_noniot: Stage2NonIoTDetector | None = None


def _get_stage1() -> Stage1Classifier:
    global _stage1
    if _stage1 is None:
        _stage1 = Stage1Classifier.load(_MODEL_PATH_S1)
    return _stage1


def _get_stage2_iot() -> Stage2IoTDetector:
    global _stage2_iot
    if _stage2_iot is None:
        _stage2_iot = Stage2IoTDetector.load(_MODEL_PATH_S2_IOT)
    return _stage2_iot


def _get_stage2_noniot() -> Stage2NonIoTDetector:
    global _stage2_noniot
    if _stage2_noniot is None:
        _stage2_noniot = Stage2NonIoTDetector.load(_MODEL_PATH_S2_NONIOT)
    return _stage2_noniot


# ══════════════════════════════════════════════════════════════════════════════
# Phase-A feature normalisation — coerces live-flow dicts into the unified schema
# ══════════════════════════════════════════════════════════════════════════════

# Minimal protocol-string → number mapping. Matches process_ctu13.py PROTO_MAP
# and pcap_to_csv.py conventions used at training time.
_PROTO_MAP = {
    "tcp": 6, "udp": 17, "icmp": 1, "arp": 0, "igmp": 2,
    "esp": 50, "gre": 47, "ipv6-icmp": 58, "icmpv6": 58,
    "unknown": 0, "": 0, "none": 0, "other": 0,
}


def _coerce_protocol(value) -> int:
    """Map 'TCP'/'tcp'/6/'6'/None → an integer protocol number for the RF."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    if s.isdigit():
        return int(s)
    return _PROTO_MAP.get(s, 0)


def _normalize_for_stage1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame safe to feed into Stage1Classifier.predict().

    Specifically:
      1. Coerce 'protocol' from string ('TCP', 'UDP', ...) to integer.
      2. Coerce any object-dtype feature column to numeric (NaN → 0).
      3. Leave missing features alone — Stage1Classifier._align() zero-fills them.
    """
    df = df.copy()

    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].apply(_coerce_protocol)

    # Anything still object-typed (e.g. 'tcp_state' as a string) → numeric or 0.
    for col in df.columns:
        if df[col].dtype == object and col not in ("src_ip", "dst_ip", "timestamp"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Public API #1 — single live flow (called by monitor_page.py per packet group)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(flow: dict) -> dict:
    """
    Run Stage-1 + Stage-2 inference on a single flow dict from live capture.

    Returns:
        {
            "label":       "botnet" | "benign",
            "confidence":  float in [0.0, 1.0],
            "device_type": "iot" | "noniot",
            "stage1_conf": float,
            "latency_ms":  float,
        }
    """
    t0 = time.perf_counter()

    if USE_REAL_INFERENCE:
        result = _real_flow_inference(flow)
    else:
        result = _stub_inference(flow)

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


def _real_flow_inference(flow: dict) -> dict:
    """
    Real Stage-1 + Stage-2 inference on a single flow dict.

    Two optional sequence keys may be attached by live_capture.py:
      _kitsune_seq : (KITSUNE_SEQ_LEN, 115) numpy array — for Stage-2 IoT
      _noniot_seq  : (NONIOT_SEQ_LEN, 48)   numpy array — for Stage-2 Non-IoT

    Each is consumed only when the corresponding device_type from Stage-1
    routes to it. If the appropriate sequence is absent (buffer warming up),
    the result falls back to Stage-1 confidence as a placeholder.
    """
    # Strip non-feature keys before handing the row to Stage-1.
    # Sequence arrays are numpy and would confuse pandas; pop both.
    kitsune_seq = flow.pop("_kitsune_seq", None) if isinstance(flow, dict) else None
    noniot_seq  = flow.pop("_noniot_seq",  None) if isinstance(flow, dict) else None

    df = _normalize_for_stage1(pd.DataFrame([flow]))

    stage1 = _get_stage1()
    device_type, stage1_conf = stage1.predict(df)

    stage2_ran = False

    if device_type == "iot":
        if kitsune_seq is not None:
            stage2 = _get_stage2_iot()
            label, confidence = stage2.predict_sequence(kitsune_seq)
            stage2_ran = True
        else:
            # IoT-classified but Kitsune buffer for this src_ip not yet full.
            label = "benign"
            confidence = float(stage1_conf)
    else:  # device_type == "noniot"
        if noniot_seq is not None:
            stage2 = _get_stage2_noniot()
            label, confidence = stage2.predict_sequence(noniot_seq)
            stage2_ran = True
        else:
            # Non-IoT-classified but flow buffer for this src_ip not yet full.
            label = "benign"
            confidence = float(stage1_conf)

    return {
        "label":       label,
        "confidence":  float(confidence),
        "device_type": device_type,
        "stage1_conf": float(stage1_conf),
        "stage2_ran":  stage2_ran,
    }


def _stub_inference(flow: dict) -> dict:
    """
    Demo stub — returns plausible-looking random results.
    Used when USE_REAL_INFERENCE is False (UI development without trained models).
    """
    dst = int(flow.get("dst_port", 0) or 0)
    BOTNET_PORTS = {4444, 9999, 6667, 31337, 2323, 23}
    base_botnet_prob = 0.65 if dst in BOTNET_PORTS else 0.15

    is_botnet  = random.random() < base_botnet_prob
    device_iot = random.random() < 0.45

    return {
        "label":       "botnet" if is_botnet else "benign",
        "confidence":  round(random.uniform(0.72, 0.99), 3),
        "device_type": "iot" if device_iot else "noniot",
        "stage1_conf": round(random.uniform(0.80, 0.99), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API #2 — uploaded file (called by upload_page.py)
# ══════════════════════════════════════════════════════════════════════════════

_SUPPORTED_FILE_FORMATS = (
    FileFormat.CSV_UNIFIED,
    FileFormat.CSV_GENERIC,
    FileFormat.CSV_CICFLOW,
    FileFormat.CSV_CTU13,
    FileFormat.CSV_UNSW,
    FileFormat.NETFLOW_CSV,
)


def run_file_inference(info: Any) -> list[dict[str, Any]]:
    """
    Run inference on every row of an uploaded CSV file.

    Args:
        info: FileInfo object from app.file_handler.load_file().

    Returns:
        List of per-row result dicts (same schema as run_inference()).
    """
    if not getattr(info, "is_valid", False):
        raise ValueError("Cannot run inference on an invalid file.")

    if info.format not in _SUPPORTED_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file format for inference: {info.format}. "
            "Only CSV flow exports are supported."
        )

    try:
        df = pd.read_csv(info.path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV for inference: {exc}") from exc

    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    df = _normalize_for_stage1(df)

    t0 = time.perf_counter()
    stage1 = _get_stage1()
    device_types, stage1_confs = stage1.predict(df)

    # Stage1Classifier.predict returns scalars for a 1-row DataFrame and lists
    # for multi-row. Normalise to lists so the loop below is uniform.
    if isinstance(device_types, str):
        device_types = [device_types]
        stage1_confs = [stage1_confs]

    results: list[dict[str, Any]] = []
    for i, (dt, conf) in enumerate(zip(device_types, stage1_confs)):
        if dt == "iot":
            # Stage-2 IoT works per-row via the seq_len-pad trick in its predict().
            stage2 = _get_stage2_iot()
            label, confidence = stage2.predict(df.iloc[[i]])
        else:
            # Stage-2 Non-IoT — for uploaded files we don't have a pre-built
            # per-src_ip rolling buffer, so we call its single-row predict()
            # which zero-pads to seq_len. This is less accurate than the
            # rolling-buffer path used in live monitoring, but better than
            # short-circuiting every flow to "benign".
            stage2 = _get_stage2_noniot()
            label, confidence = stage2.predict(df.iloc[[i]])

        results.append({
            "row":         i + 1,
            "device_type": dt,
            "label":       label,
            "confidence":  float(confidence),
            "stage1_conf": float(conf),
        })

    latency = round((time.perf_counter() - t0) * 1000, 2)
    for result in results:
        result["latency_ms"] = latency

    return results
