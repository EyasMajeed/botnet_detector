"""
inference_bridge.py  —  Single integration point for ML inference
==================================================================
This is the ONLY file that needs to change when the preprocessing
pipeline and trained models are ready.

Current state: STUB — returns plausible random results so the
monitoring UI works end-to-end today.

Integration checklist (for the preprocessing team):
    1. Implement _real_inference() below
    2. Flip USE_REAL_INFERENCE = True
    3. Done — the rest of the app is untouched

Flow dict keys expected by the real pipeline (all 56 features):
    See src/preprocessing_pipeline/config.py  →  ALL_FEATURES
"""

from __future__ import annotations
import random
import time
from typing import Any

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.stage1.classifier import Stage1Classifier
from models.stage2.iot_detector import Stage2Detector
from file_handler import FileFormat

from src.xai import ExplainerBundle, explain_flow
_xai_bundle: ExplainerBundle | None = None


# ── Model loading ─────────────────────────────────────────────────────────────
_MODEL_PATH_S1 = ROOT / "models" / "stage1" / "rf_model.pkl"
_MODEL_PATH_S2 = ROOT / "models" / "stage2" / "iot_cnn_lstm.pt"

_stage1: Stage1Classifier | None = None
_stage2: Stage2Detector | None = None

def _get_stage1() -> Stage1Classifier:
    global _stage1
    if _stage1 is None:
        _stage1 = Stage1Classifier.load(_MODEL_PATH_S1)
    return _stage1

def _get_stage2() -> Stage2Detector:
    global _stage2
    if _stage2 is None:
        _stage2 = Stage2Detector.load(_MODEL_PATH_S2)
    return _stage2

# ── Flip this when the real pipeline is ready ─────────────────────────────────
USE_REAL_INFERENCE = True


# ── Public API ─────────────────────────────────────────────────────────────────

def run_inference(flow: dict) -> dict:
    """
    Run Stage-1 + Stage-2 inference on a single flow dict.

    Args:
        flow:  Raw flow dictionary from LiveCaptureThread or pcap_parser.
               The bridge handles feature alignment internally.

    Returns:
        {
            "label":       "botnet" | "benign",
            "confidence":  float  in [0.0, 1.0],
            "device_type": "iot"  | "noniot",
            "stage1_conf": float,   # IoT vs Non-IoT confidence
            "latency_ms":  float,   # wall-clock inference time
        }
    """
    t0 = time.perf_counter()

    if USE_REAL_INFERENCE:
        result = run_file_inference(flow)
    else:
        result = _stub_inference(flow)

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ── Stub (active now) ──────────────────────────────────────────────────────────

def _stub_inference(flow: dict) -> dict:
    """
    Realistic-looking random results for UI development.
    Botnet probability is influenced by suspicion score so the
    demo feels coherent with the scorer output.
    """
    # Use dst_port as a weak heuristic to make stub feel realistic
    dst = int(flow.get("dst_port", 0) or 0)
    BOTNET_PORTS = {4444, 9999, 6667, 31337, 2323, 23}
    base_botnet_prob = 0.65 if dst in BOTNET_PORTS else 0.15

    is_botnet   = random.random() < base_botnet_prob
    device_iot  = random.random() < 0.45
    confidence  = round(random.uniform(0.72, 0.99), 3)
    stage1_conf = round(random.uniform(0.80, 0.99), 3)

    return {
        "label":       "botnet" if is_botnet else "benign",
        "confidence":  confidence,
        "device_type": "iot" if device_iot else "noniot",
        "stage1_conf": stage1_conf,
    }


def run_file_inference(info: Any) -> list[dict[str, Any]]:
    """Run inference on an uploaded file using Stage-1 model."""
    if not getattr(info, "is_valid", False):
        raise ValueError("Cannot run inference on an invalid file.")
    
    # Load file as DataFrame
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
            "Only CSV flow exports are supported."
        )
    
    try:
        df = pd.read_csv(info.path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV for inference: {exc}") from exc
    
    if df.empty:
        raise ValueError("CSV file contains no data rows.")
    
    # Run Stage-1 on the entire DataFrame
    t0 = time.perf_counter()
    stage1 = _get_stage1()
    device_types, stage1_confs = stage1.predict(df)
    
    # # For now, assume all are benign if not IoT (no Stage-2 for non-IoT)
    # results = []
    # for i, (dt, conf) in enumerate(zip(device_types, stage1_confs)):
    #     if dt == "iot":
    #         # Would need Stage-2 here, but for simplicity, mark as unknown
    #         label = "unknown"  # Since Stage-2 is only for IoT
    #         confidence = 0.5
    #     else:
    #         label = "benign"
    #         confidence = 0.95
        
    #     results.append({
    #         "row": i + 1,
    #         "device_type": dt,
    #         "label": label,
    #         "confidence": float(confidence),
    #         "stage1_conf": float(conf),
    #     })
    
    # latency = round((time.perf_counter() - t0) * 1000, 2)
    # for result in results:
    #     result["latency_ms"] = latency
    
    # return results
    """
    Real inference using Stage-1 and Stage-2 models.
    """
    # Convert flow dict to DataFrame
    df = pd.DataFrame([flow])
    
    # Stage-1: IoT vs Non-IoT
    stage1 = _get_stage1()
    device_type, stage1_conf = stage1.predict(df)
    
    # Stage-2: Botnet detection (only for IoT flows)
    if device_type == "iot":
        stage2 = _get_stage2()
        label, confidence = stage2.predict(df)
    else:
        # For non-IoT, assume benign (no Stage-2 model for non-IoT yet)
        label = "benign"
        confidence = 0.95  # High confidence for benign non-IoT

    if label != "benign":  # only explain detections, save compute on benign
        try:
            xai = explain_flow(
                flow_df       = df_window,    # the window passed to Stage-2
                stage1_label  = device_type,  # "iot" | "noniot"
                stage2_label  = label,
                bundle        = _get_xai_bundle(),
                top_k         = 8,
            )
        except Exception as e:
            # Don't let XAI errors break detection — log and continue
            print(f"  [XAI] Warning: explanation failed: {e}")
            xai = None
    else:
        xai = None

    return {
        "label": label,
        "confidence": float(confidence),
        "device_type": device_type,
        "stage1_conf": float(stage1_conf),
        "xai":          xai,
    }

def _get_xai_bundle() -> ExplainerBundle:
    global _xai_bundle
    if _xai_bundle is None:
        _xai_bundle = ExplainerBundle(
            stage1_classifier = _get_stage1(),
            iot_detector      = _get_stage2(),     # your existing IoT loader
            noniot_detector   = None,              # add when you have one
        )
    return _xai_bundle


