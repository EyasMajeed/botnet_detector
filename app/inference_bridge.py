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

# ── Flip this when the real pipeline is ready ─────────────────────────────────
USE_REAL_INFERENCE = False


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
        result = _real_inference(flow)
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


# ── Real implementation (fill this in) ────────────────────────────────────────

def _real_inference(flow: dict) -> dict:
    """
    TODO — implement when preprocessing pipeline + models are ready.

    Suggested implementation:
        1. Import full_feature_pipeline from feature_utils
        2. Convert flow dict → single-row DataFrame
        3. Run full_feature_pipeline(df, normalize=True)
        4. Pass feature vector to Stage-1 RF → get device_type + conf
        5. Route to Stage-2 IoT or Non-IoT CNN-LSTM → get label + conf
        6. Return structured result dict

    Example skeleton:
        import pandas as pd
        from src.preprocessing_pipeline.features.feature_utils import full_feature_pipeline
        from models.stage1.classifier import Stage1Classifier
        from models.stage2.cnn_lstm import Stage2Detector

        _stage1 = Stage1Classifier.load("models/stage1/rf_model.pkl")
        _stage2_iot    = Stage2Detector.load("models/stage2/iot_cnn_lstm.pt")
        _stage2_noniot = Stage2Detector.load("models/stage2/noniot_cnn_lstm.pt")

        df, _ = full_feature_pipeline(pd.DataFrame([flow]))
        device_type, s1_conf = _stage1.predict(df)
        model = _stage2_iot if device_type == "iot" else _stage2_noniot
        label, confidence = model.predict(df)
        return {
            "label": label, "confidence": confidence,
            "device_type": device_type, "stage1_conf": s1_conf,
        }
    """
    raise NotImplementedError(
        "Real inference not implemented yet. Set USE_REAL_INFERENCE = False "
        "to keep using the stub, or implement _real_inference()."
    )
