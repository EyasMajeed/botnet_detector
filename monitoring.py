"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         monitoring.py  —  Live Hybrid Botnet Detection Pipeline            ║
║         Group 07 | CPCS499  |  Hybrid AI-Based Botnet Detection            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE                                                               ║
║                                                                              ║
║  Every packet (parallel, silent):                                            ║
║      KitsuneExtractor.update() → MinMaxScaler → per-src_ip deque           ║
║      (buffer fills continuously; no inference fires here)                   ║
║                                                                              ║
║  On flow completion (FIN/RST/idle) — Stage-1 gates Stage-2:                ║
║      FlowRecord → flow_feature_extractor() → 56-feature dict               ║
║               │                                                              ║
║          Stage-1 RF  (+ StandardScaler if s1_scaler.json exists)           ║
║               │ "iot" | "noniot"                                             ║
║               ├── "iot"    → read Kitsune deque (already filled)           ║
║               │              → Stage-2 IoT CNN-LSTM → "botnet"|"benign"    ║
║               └── "noniot" → stage2_preprocess_non_iot()                   ║
║                              → Stage-2 Non-IoT CNN-LSTM → "botnet"|"benign"║
║                                                                              ║
║  WHY KITSUNE STILL RUNS PER-PACKET:                                         ║
║      Stage-1 fires on flow completion (up to 30s after the first packet).  ║
║      Kitsune's exponential-decay windows must accumulate continuously or    ║
║      the buffer is empty at routing time. Kitsune runs silently on every    ║
║      packet; inference only fires when Stage-1 routes to "iot".            ║
║                                                                              ║
║  STAGE-1 SCALER NOTE:                                                        ║
║  classifier.py trains on an already-normalised CSV (preprocess_from_        ║
║  pcap_csvs.py applies StandardScaler before saving).  The scaler was        ║
║  never exported.  Run export_s1_scaler.py ONCE to generate                  ║
║  s1_scaler.json, then restart monitoring.py.                                 ║
║                                                                              ║
║  If s1_scaler.json is absent the RF still runs but on raw live values       ║
║  instead of the normalised values it trained on — accuracy will drop.        ║
║  This is logged as a WARNING, not a hard failure, so you can at least        ║
║  verify the rest of the pipeline works while you generate the scaler.        ║
║                                                                              ║
║  SETUP:                                                                      ║
║    python3 export_s1_scaler.py          → models/stage1/s1_scaler.json     ║
║    python3 src/ingestion/preprocess_nbaiot.py → models/stage2/iot_scaler.json║
║    sudo python3 monitoring.py --iface en0                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from scapy.all import sniff, Ether, IP, TCP, UDP, ICMP
    _SCAPY_OK = True
except ImportError:
    _SCAPY_OK = False

# ── KitsuneExtractor (project-local) ─────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_cand / "src" / "live" / "kitsune_extractor.py").exists():
        sys.path.insert(0, str(_cand / "src" / "live"))
        break

try:
    from kitsune_extractor import KitsuneExtractor, FEATURE_NAMES as KITSUNE_FEATURE_NAMES
    _KITSUNE_OK = True
except ImportError:
    _KITSUNE_OK = False
    KITSUNE_FEATURE_NAMES: List[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("monitoring.log", mode="a"),
    ],
)
log = logging.getLogger("BotnetMonitor")
for _lib in ("urllib3", "scapy.runtime", "scapy.loading"):
    logging.getLogger(_lib).setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

def _ep(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default)))

ROOT            = _HERE
MODEL_S1_RF     = _ep("MODEL_S1_RF",    ROOT / "models/stage1/rl_model.json")
SCALER_S1_JSON  = _ep("SCALER_S1_JSON", ROOT / "models/stage1/s1_scaler.json")
MODEL_S2_IOT    = _ep("MODEL_S2_IOT",   ROOT / "models/stage2/iot_cnn_lstm.pt")
SCALER_S2_IOT   = _ep("SCALER_S2_IOT",  ROOT / "models/stage2/iot_scaler.json")
MODEL_S2_NONIOT = _ep("MODEL_S2_NONIOT",ROOT / "models/stage2/noniot_cnn_lstm.pt")

# Live-calibrated scaler path (written by LiveScalerCalibrator after 500 flows)
SCALER_S2_NONIOT_LIVE = _ep(
    "SCALER_S2_NONIOT_LIVE",
    ROOT / "models/stage2/noniot_scaler_live.json",
)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

IOT_SEQ_LEN       = 20
NONIOT_SEQ_LEN    = 20
FLOW_IDLE_TIMEOUT = 30.0
ALERT_COOLDOWN    = 10.0
MAX_TRACKED_IPS   = 500

SUSP_PKT_RATE  = 2000.0   # raised: TLS bursts / HTTP2 multiplex can exceed 500 pkt/s legitimately
SUSP_FLOW_DUR  = 0.1      # lowered: 0.5s catches DNS/QUIC/API calls; 0.1s targets real scans
SUSP_BPS       = 1e6
SUSP_THRESHOLD = 3.0
RISKY_PORTS    = {23, 2323, 1900, 7547, 5555}

# ── Stage-2 Non-IoT threshold safety override ─────────────────────────────────
# The noniot_cnn_lstm.pt checkpoint stores threshold=0.0100 — essentially a
# near-zero gate that labels every flow with sigmoid > 1% as botnet.
# We cannot retrain right now, so any checkpoint threshold below 0.30 is
# replaced with this value at load time.  0.50 is the statistical decision
# boundary for a binary sigmoid and is the correct default for an uncalibrated model.
NONIOT_THRESHOLD_OVERRIDE: float = 0.50

# ── Live scaler calibration ───────────────────────────────────────────────────
# Number of completed NonIoT flows whose raw feature vectors are accumulated
# before a live StandardScaler is fitted and persisted.
N_CALIBRATION_FLOWS: int = 500


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

S1_FEATURES: List[str] = [
    "flow_duration",
    "total_fwd_packets","total_bwd_packets",
    "total_fwd_bytes","total_bwd_bytes",
    "fwd_pkt_len_min","fwd_pkt_len_max","fwd_pkt_len_mean","fwd_pkt_len_std",
    "bwd_pkt_len_min","bwd_pkt_len_max","bwd_pkt_len_mean","bwd_pkt_len_std",
    "flow_bytes_per_sec","flow_pkts_per_sec",
    "flow_iat_mean","flow_iat_std","flow_iat_min","flow_iat_max",
    "fwd_iat_mean","fwd_iat_std","fwd_iat_min","fwd_iat_max",
    "bwd_iat_mean","bwd_iat_std","bwd_iat_min","bwd_iat_max",
    "fwd_header_length","bwd_header_length",
    "flag_FIN","flag_SYN","flag_RST","flag_PSH","flag_ACK","flag_URG",
    "protocol","src_port","dst_port",
    "flow_active_time","flow_idle_time",
    "bytes_per_sec_window","pkts_per_sec_window",
    "periodicity_score","burst_rate",
    "window_flow_count","window_unique_dsts",
    "ttl_mean","ttl_std","ttl_min","ttl_max",
    "dns_query_count",
    "payload_bytes_mean","payload_bytes_std",
    "payload_zero_ratio","payload_entropy",
    "tls_features_available",
]

S2_NONIOT_FEATURES_FALLBACK: List[str] = [
    "flow_duration",
    "total_fwd_packets","total_bwd_packets",
    "total_fwd_bytes","total_bwd_bytes",
    "fwd_pkt_len_min","fwd_pkt_len_max","fwd_pkt_len_mean","fwd_pkt_len_std",
    "bwd_pkt_len_min","bwd_pkt_len_max","bwd_pkt_len_mean","bwd_pkt_len_std",
    "flow_bytes_per_sec","flow_pkts_per_sec",
    "flow_iat_mean","flow_iat_std","flow_iat_min","flow_iat_max",
    "fwd_iat_mean","fwd_iat_std","fwd_iat_min","fwd_iat_max",
    "bwd_iat_mean","bwd_iat_std","bwd_iat_min","bwd_iat_max",
    "fwd_header_length","bwd_header_length",
    "flag_FIN","flag_SYN","flag_RST","flag_PSH","flag_ACK","flag_URG",
    "protocol","src_port","dst_port",
    "flow_active_time","flow_idle_time",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlowKey:
    """
    Canonical bidirectional flow key.
    Sorted so A→B and B→A always map to the same record, fixing the
    unidirectional aggregation bug that zeroed all bwd_* features.
    """
    ip_lo: str; ip_hi: str; port_lo: int; port_hi: int; proto: int

    @staticmethod
    def make(src_ip: str, dst_ip: str, sport: int, dport: int, proto: int) -> "FlowKey":
        a, b = (src_ip, sport), (dst_ip, dport)
        lo, hi = (a, b) if a <= b else (b, a)
        return FlowKey(lo[0], hi[0], lo[1], hi[1], proto)

    def __str__(self):
        return f"{self.ip_lo}:{self.port_lo}<->{self.ip_hi}:{self.port_hi}/{self.proto}"

    def __hash__(self):
        return hash((self.ip_lo, self.ip_hi, self.port_lo, self.port_hi, self.proto))

    def __eq__(self, o):
        return (self.ip_lo, self.ip_hi, self.port_lo, self.port_hi, self.proto) == \
               (o.ip_lo, o.ip_hi, o.port_lo, o.port_hi, o.proto)


@dataclass
class FlowRecord:
    key: FlowKey; start_time: float; last_seen: float
    proto: int; src_port: int; dst_port: int
    fwd_pkts: int = 0; fwd_bytes: int = 0
    fwd_times: List[float] = field(default_factory=list)
    fwd_sizes: List[int]   = field(default_factory=list)
    bwd_pkts: int = 0; bwd_bytes: int = 0
    bwd_times: List[float] = field(default_factory=list)
    bwd_sizes: List[int]   = field(default_factory=list)
    ttl_samples: List[int] = field(default_factory=list)
    flag_FIN: int = 0; flag_SYN: int = 0; flag_RST: int = 0
    flag_PSH: int = 0; flag_ACK: int = 0; flag_URG: int = 0


@dataclass
class DetectionResult:
    flow_id: str; src_ip: str; dst_ip: str
    device_type: str; label: str
    s1_confidence: float; s2_confidence: float
    suspicion_score: float; latency_ms: float; alerted: bool
    timestamp: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: flow_feature_extractor
# ══════════════════════════════════════════════════════════════════════════════

def _iat_stats(times: List[float]) -> Tuple[float, float, float, float]:
    if len(times) < 2:
        return 0.0, 0.0, 0.0, 0.0
    d = np.diff(times)
    return float(d.mean()), float(d.std()), float(d.min()), float(d.max())


def _len_stats(sizes: List[int]) -> Tuple[float, float, float, float]:
    if not sizes:
        return 0.0, 0.0, 0.0, 0.0
    a = np.array(sizes, dtype=float)
    return float(a.min()), float(a.max()), float(a.mean()), float(a.std())


def flow_feature_extractor(rec: FlowRecord,
                            window_flow_count: int = 1,
                            window_unique_dsts: int = 1) -> Dict[str, float]:
    """
    Converts a completed bidirectional FlowRecord to the 56-feature dict
    expected by Stage-1.

    All statistics are computed from real per-packet data (timestamps, sizes,
    TTL samples). No features default to zero due to unidirectional aggregation
    — FlowKey.make() ensures forward and backward packets share one record.
    """
    duration = max(rec.last_seen - rec.start_time, 1e-6)
    all_times = sorted(rec.fwd_times + rec.bwd_times)
    fim, fis, fimn, fimx = _iat_stats(all_times)
    fwm, fws, fwmn, fwmx = _iat_stats(rec.fwd_times)
    bwm, bws, bwmn, bwmx = _iat_stats(rec.bwd_times)
    fln, flx, flm, fls   = _len_stats(rec.fwd_sizes)
    bln, blx, blm, bls   = _len_stats(rec.bwd_sizes)
    ttl = np.array(rec.ttl_samples, dtype=float) if rec.ttl_samples else np.array([64.0])
    tb, tp = rec.fwd_bytes + rec.bwd_bytes, rec.fwd_pkts + rec.bwd_pkts

    return {
        "flow_duration": duration,
        "total_fwd_packets": float(rec.fwd_pkts),
        "total_bwd_packets": float(rec.bwd_pkts),
        "total_fwd_bytes":   float(rec.fwd_bytes),
        "total_bwd_bytes":   float(rec.bwd_bytes),
        "fwd_pkt_len_min": fln, "fwd_pkt_len_max": flx,
        "fwd_pkt_len_mean": flm, "fwd_pkt_len_std": fls,
        "bwd_pkt_len_min": bln, "bwd_pkt_len_max": blx,
        "bwd_pkt_len_mean": blm, "bwd_pkt_len_std": bls,
        "flow_bytes_per_sec": tb / duration,
        "flow_pkts_per_sec":  tp / duration,
        "flow_iat_mean": fim, "flow_iat_std": fis,
        "flow_iat_min":  fimn, "flow_iat_max": fimx,
        "fwd_iat_mean":  fwm, "fwd_iat_std":  fws,
        "fwd_iat_min":   fwmn, "fwd_iat_max":  fwmx,
        "bwd_iat_mean":  bwm, "bwd_iat_std":  bws,
        "bwd_iat_min":   bwmn, "bwd_iat_max":  bwmx,
        "fwd_header_length": 20.0, "bwd_header_length": 20.0,
        "flag_FIN": float(rec.flag_FIN), "flag_SYN": float(rec.flag_SYN),
        "flag_RST": float(rec.flag_RST), "flag_PSH": float(rec.flag_PSH),
        "flag_ACK": float(rec.flag_ACK), "flag_URG": float(rec.flag_URG),
        "protocol": float(rec.proto),
        "src_port": float(rec.src_port), "dst_port": float(rec.dst_port),
        "flow_active_time": duration, "flow_idle_time": 0.0,
        "bytes_per_sec_window": tb / duration,
        "pkts_per_sec_window":  tp / duration,
        "periodicity_score": 0.0, "burst_rate": 0.0,
        "window_flow_count":  float(window_flow_count),
        "window_unique_dsts": float(window_unique_dsts),
        "ttl_mean": float(ttl.mean()), "ttl_std": float(ttl.std()),
        "ttl_min":  float(ttl.min()),  "ttl_max": float(ttl.max()),
        "dns_query_count": float(rec.dst_port == 53 or rec.src_port == 53),
        "payload_bytes_mean": flm, "payload_bytes_std": fls,
        "payload_zero_ratio": 0.0, "payload_entropy": 0.0,
        "tls_features_available": float(
            rec.dst_port in {443, 8443, 8883} or rec.src_port in {443, 8443, 8883}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: suspicion_scoring
# ══════════════════════════════════════════════════════════════════════════════

def suspicion_scoring(feat: Dict[str, float]) -> Tuple[float, bool]:
    """
    Lightweight rule-based anomaly score on a completed flow.
    Informational only — Stage-1 and Stage-2 always run regardless of score.

    Tuning history (Group 07 — CPCS499):
      v2: raised SUSP_PKT_RATE 500→2000, lowered SUSP_FLOW_DUR 0.5→0.1.
          Added TLS exemption on pkt-rate rule (+2.0 → +0.5 for encrypted
          flows) and skipped SYN-ratio check for web ports (80/443/8080/8443)
          to eliminate false positives from TLS resumption / HTTP-2 bursts.
    """
    score    = 0.0
    dst_port = int(feat.get("dst_port", 0))
    is_tls   = (feat.get("tls_features_available", 0.0) == 1.0
                and dst_port in {443, 8443, 8883})
    is_web   = dst_port in {80, 443, 8080, 8443}

    # ── Packet-rate check ────────────────────────────────────────────────────
    # TLS flows that exceed the threshold get a reduced contribution (+0.5)
    # because HTTP/2 multiplexing and TLS session resumption legitimately
    # produce short bursts well above 500 pkt/s.  Real flood traffic operates
    # at tens of thousands of pkt/s and is caught by the raised threshold.
    if feat.get("flow_pkts_per_sec", 0) > SUSP_PKT_RATE:
        score += 0.5 if is_tls else 2.0

    # ── Short-duration check ─────────────────────────────────────────────────
    if feat.get("flow_duration", 1) < SUSP_FLOW_DUR:
        score += 1.5

    # ── High bandwidth check ─────────────────────────────────────────────────
    if feat.get("flow_bytes_per_sec", 0) > SUSP_BPS:
        score += 2.0

    # ── SYN-flood indicator ──────────────────────────────────────────────────
    # Skip for standard web ports: legitimate TCP connection establishment to
    # any HTTP/HTTPS server always produces a high SYN/(SYN+ACK) ratio on
    # short flows (the server ACK may not appear in the captured window).
    if not is_web:
        syn, ack = feat.get("flag_SYN", 0), feat.get("flag_ACK", 0)
        if (syn + ack) > 0 and syn / (syn + ack) > 0.9:
            score += 1.5

    # ── Risky port check ─────────────────────────────────────────────────────
    if dst_port in RISKY_PORTS:
        score += 1.0

    # ── High window-flow-count (C2 beaconing / scanning indicator) ──────────
    if feat.get("window_flow_count", 0) > 50:
        score += 1.5

    return score, score >= SUSP_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: packet_feature_extractor
# ══════════════════════════════════════════════════════════════════════════════

def packet_feature_extractor(src_ip: str,
                              iot_bufs: Dict[str, deque],
                              s2_iot: "Stage2IoTDetector") -> Optional[np.ndarray]:
    """
    Returns the current Kitsune sequence for src_ip as a (IOT_SEQ_LEN, 115)
    array if the buffer is full, else None.

    This replaces the old PacketBuffer which was never populated and returned
    only synthetic random values.  Kitsune is the correct packet-level
    enrichment for IoT detection: its 115 incremental statistics are updated
    per-packet in process_packet() and this function reads the accumulated
    buffer when needed.
    """
    buf = iot_bufs.get(src_ip)
    if buf is None or len(buf) < IOT_SEQ_LEN:
        return None
    return np.stack(list(buf))   # (IOT_SEQ_LEN, 115)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: Stage1Classifier  (stage1_preprocess + stage1_predict)
# ══════════════════════════════════════════════════════════════════════════════

class Stage1Classifier:
    """
    Wraps the trained Random Forest for IoT vs Non-IoT classification.

    SCALER BEHAVIOUR:
    classifier.py trains on an already-normalised CSV.  The scaler was never
    saved.  This class handles three cases:

      Case A — s1_scaler.json EXISTS (best):
        StandardScaler is applied before calling predict_proba().
        Live features → z = (x - mean) / scale → RF → probabilities.
        This exactly reproduces training conditions.

      Case B — s1_scaler.json MISSING (degraded):
        RF runs on raw live feature values.  Since live and training
        distributions differ, split thresholds will be off, but the RF
        may still produce useful predictions for high-contrast cases.
        A WARNING is logged every 100 flows as a reminder.

    To generate s1_scaler.json, run once:
        python3 export_s1_scaler.py
    """

    def __init__(self, model_path: Path, scaler_path: Optional[Path] = None):
        # ── Load model (RF pickle OR XGBoost portable JSON) ──────────────
        if not model_path.exists():
            raise FileNotFoundError(
                f"Stage-1 model missing: {model_path}\n"
                "Train it with: python3 models/stage1/classifier.py"
            )

        # Auto-detect format by file suffix:
        #   .json  → XGBoost portable format (cross-platform-safe)
        #   .pkl   → RandomForest pickle (default, sklearn-portable)
        is_xgb_json = str(model_path).lower().endswith(".json")

        if is_xgb_json:
            # ── XGBoost portable load (avoids macOS segfault) ───────────
            # Companion meta pickle holds: LabelEncoder, feature_names,
            # init_params for XGBClassifier reconstruction.
            meta_path = model_path.with_name(
                model_path.stem + "_meta.pkl"
            )
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Stage-1 XGBoost meta pickle missing: {meta_path}\n"
                    "It must sit next to the .json file. "
                    "Re-run classifier.py to regenerate both."
                )

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError(
                    "[Stage1] XGBoost JSON model detected but xgboost "
                    "package is not installed. Install with: pip install xgboost"
                ) from e

            # Reconstruct the XGBClassifier with original training params,
            # then load the Booster state from the portable JSON.
            init_params = meta.get("init_params", {})
            xgb = XGBClassifier(**init_params)
            xgb.load_model(str(model_path))
            xgb._le = None  # internal sklearn-XGBoost compat shim

            # Force n_classes_ so sklearn-style predict_proba works
            xgb.n_classes_ = meta.get("n_classes", 2)
            xgb.classes_   = np.arange(xgb.n_classes_)

            self._rf = xgb
            self._le = meta.get("label_encoder", None)
            log.info("[Stage1] XGBoost loaded from portable JSON: %s", model_path)
        else:
            # ── RandomForest pickle load (legacy default path) ──────────
            with open(model_path, "rb") as f:
                saved = pickle.load(f)

            if isinstance(saved, dict):
                self._rf = saved["model"]
                # Prefer the LabelEncoder stored alongside the RF so we get the
                # original string classes ("iot", "noniot") rather than the
                # encoded integers (0, 1) that live in rf.classes_.
                self._le = saved.get("label_encoder", None)
            else:
                # Legacy pkl that is the bare RF object itself
                self._rf = saved
                self._le = None

        if self._le is not None:
            # le.classes_ is ["iot", "noniot"] (the original string labels)
            le_classes = list(self._le.classes_)
            if "iot" not in le_classes:
                raise ValueError(
                    f"[Stage1] LabelEncoder classes do not contain 'iot': {le_classes}"
                )
            self._iot_idx = le_classes.index("iot")
            log.info("[Stage1] RF loaded: %s  (LabelEncoder classes=%s, iot_idx=%d)",
                     model_path, le_classes, self._iot_idx)
        else:
            # Fallback: try to read string classes directly from the RF.
            # This only works if the RF was trained without encoding (rare).
            rf_classes = list(self._rf.classes_)
            if "iot" in rf_classes:
                self._iot_idx = rf_classes.index("iot")
                log.warning(
                    "[Stage1] No LabelEncoder found in pkl. "
                    "Falling back to rf.classes_=%s  (may be integers if label-encoded).",
                    rf_classes,
                )
            else:
                # rf.classes_ contains integers (0, 1) — default iot to index 0
                self._iot_idx = 0
                log.warning(
                    "[Stage1] No LabelEncoder found and rf.classes_=%s are not strings. "
                    "Defaulting iot_idx=0. Re-save the model with Stage1Classifier.save() "
                    "to embed the LabelEncoder.",
                    rf_classes,
                )
            log.info("[Stage1] RF loaded: %s  (rf.classes_=%s, iot_idx=%d)",
                     model_path, rf_classes, self._iot_idx)

        # ── Load scaler (optional) ────────────────────────────────────────
        self._mean:  Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._features: List[str] = S1_FEATURES
        self._has_scaler = False
        self._no_scaler_warn_count = 0

        if scaler_path and scaler_path.exists():
            with open(scaler_path) as f:
                s = json.load(f)
            self._mean     = np.array(s["mean"],  dtype=np.float32)
            self._scale    = np.array(s["scale"], dtype=np.float32)
            self._scale    = np.where(self._scale == 0, 1.0, self._scale)
            self._features = s.get("features", S1_FEATURES)
            self._has_scaler = True
            log.info("[Stage1] Scaler loaded: %s  (%d features)", scaler_path, len(self._features))
        else:
            log.warning(
                "[Stage1] ⚠  s1_scaler.json not found at %s\n"
                "         The RF was trained on NORMALISED data. Running on raw\n"
                "         live features will degrade Stage-1 accuracy.\n"
                "         FIX: python3 export_s1_scaler.py  (takes ~10 seconds)",
                scaler_path,
            )

    def stage1_preprocess(self, feat: Dict[str, float]) -> np.ndarray:
        """
        Builds the feature vector in training column order.
        Applies StandardScaler if available; passes raw values otherwise.
        """
        row = np.array([feat.get(f, 0.0) for f in self._features], dtype=np.float32)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        if self._has_scaler:
            row = (row - self._mean) / self._scale
        return row

    def stage1_predict(self, feat: Dict[str, float]) -> Tuple[str, float]:
        """Returns (decoded_string_label, confidence).

        The predicted integer class index is decoded back to its original
        string label via self._le.inverse_transform() when a LabelEncoder is
        available.  This avoids hard-coding "iot"/"noniot" and remains correct
        even if the encoder's class ordering changes between training runs.
        """
        # Remind user every 100 flows if scaler is missing:
        if not self._has_scaler:
            self._no_scaler_warn_count += 1
            if self._no_scaler_warn_count % 100 == 1:
                log.warning("[Stage1] Still running WITHOUT scaler. "
                            "Run export_s1_scaler.py to fix this.")

        X        = self.stage1_preprocess(feat).reshape(1, -1)
        proba    = self._rf.predict_proba(X)[0]          # shape: (n_classes,)
        iot_prob = float(proba[self._iot_idx])

        if iot_prob >= 0.5:
            predicted_int = self._iot_idx
            confidence    = iot_prob
        else:
            # Index of the highest-probability non-IoT class
            predicted_int = int(np.argmax(proba))
            if predicted_int == self._iot_idx:
                # Edge case: argmax still landed on IoT despite iot_prob < 0.5.
                # Pick the next best class.
                sorted_idx    = np.argsort(proba)[::-1]
                predicted_int = int(next(i for i in sorted_idx if i != self._iot_idx))
            confidence = float(proba[predicted_int])

        # Decode integer → original string label
        if self._le is not None:
            label: str = str(self._le.inverse_transform([predicted_int])[0])
        else:
            # No encoder: rf.classes_ may already hold strings, or ints
            raw   = self._rf.classes_[predicted_int]
            label = str(raw)

        return label, confidence


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: Stage2IoTDetector  (stage2_preprocess_iot + stage2_predict)
# ══════════════════════════════════════════════════════════════════════════════

class _IotCnnLstm(nn.Module if _TORCH_OK else object):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU())
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


class Stage2IoTDetector:
    """
    Stage-2 CNN-LSTM for IoT traffic using real Kitsune per-packet sequences.
    HARD FAILURE if iot_scaler.json or iot_cnn_lstm.pt is missing.
    """

    def __init__(self, model_path: Path, scaler_path: Path):
        if not _TORCH_OK:
            raise ImportError("torch required: pip install torch")
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"IoT scaler missing: {scaler_path}\n"
                "Run: python3 src/ingestion/preprocess_nbaiot.py"
            )
        with open(scaler_path) as f:
            sc = json.load(f)
        self._feat_min = np.array(sc["min"], dtype=np.float32)
        self._feat_max = np.array(sc["max"], dtype=np.float32)
        if self._feat_max.max() < 2.0:
            raise ValueError(
                "iot_scaler.json is from already-normalised data (max ≤ 1.0). "
                "Re-run preprocess_nbaiot.py to get the raw-range scaler."
            )
        denom = self._feat_max - self._feat_min
        denom[denom == 0] = 1.0
        self._denom = denom
        log.info("[Stage2-IoT] Scaler loaded (%d features, raw max=%.1f)",
                 len(sc["min"]), float(self._feat_max.max()))

        if not model_path.exists():
            raise FileNotFoundError(
                f"IoT CNN-LSTM missing: {model_path}\n"
                "Train: python3 models/stage2/iot_detector.py"
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model = _IotCnnLstm(ckpt["n_features"])
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._threshold = float(ckpt.get("threshold", 0.52))
        log.info("[Stage2-IoT] Model loaded (threshold=%.4f)", self._threshold)

    def stage2_preprocess_iot(self, raw_vec: np.ndarray) -> np.ndarray:
        """MinMax scale one raw Kitsune vector per packet. Called in process_packet()."""
        return np.clip((raw_vec - self._feat_min) / self._denom, 0.0, 1.0).astype(np.float32)

    def stage2_predict(self, seq: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on (IOT_SEQ_LEN, 115) — a REAL temporal sequence,
        one row per packet. NOT tiled.
        """
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self._model(x)).item()
        return ("botnet" if prob >= self._threshold else "benign"), float(prob)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: LiveScalerCalibrator
# ══════════════════════════════════════════════════════════════════════════════

class LiveScalerCalibrator:
    """
    Accumulates raw Non-IoT flow feature vectors from live traffic and, once
    N_CALIBRATION_FLOWS samples have been collected, fits a StandardScaler on
    them and persists the result to ``noniot_scaler_live.json``.

    WHY THIS IS NEEDED
    ──────────────────
    The training pipeline (preprocess_from_pcap_csvs.py → clean_and_normalize())
    applied StandardScaler *before* saving stage2_noniot_botnet.csv.  The
    raw CSV was never persisted, so noniot_scaler.json was derived from an
    already-normalised distribution.  Its mean values (≈ 0–0.71) and scale
    values (≈ 0.001–1.0) are effectively an identity transform and cannot
    undo the distortion seen in live raw feature values (e.g.
    flow_bytes_per_sec ≈ 150 000).  Feeding those values directly into the
    CNN-LSTM collapses sigmoid outputs to 0.0 for ~87 % of flows.

    This calibrator solves the problem without retraining: it observes the
    actual live feature distribution and derives a scaler that matches what
    the model expected at training time.

    LIFECYCLE
    ─────────
    1. Stage2NonIoTDetector instantiates one LiveScalerCalibrator.
    2. Every time stage2_preprocess_non_iot() is called it first passes the
       *raw* feature row to calibrator.record(raw_row, feature_cols) before
       any scaling is applied.
    3. Once N_CALIBRATION_FLOWS rows have been recorded, calibrator.fit()
       is called automatically.  The fitted scaler is written to
       noniot_scaler_live.json and a log message advises the operator to
       restart monitoring.py so the new scaler is loaded.
    4. On the next start, Stage2NonIoTDetector.load() detects
       noniot_scaler_live.json (preferred over noniot_scaler.json) and
       loads the live-calibrated parameters.

    THREAD-SAFETY
    ─────────────
    record() is called from a single thread (the packet-processing loop), so
    no locking is required.
    """

    def __init__(self, out_path: Path, n_target: int = N_CALIBRATION_FLOWS):
        self._out_path  = out_path
        self._n_target  = n_target
        self._buf: deque = deque(maxlen=n_target)
        self._done      = False
        log.info(
            "[Calibration] LiveScalerCalibrator initialised — will fit on %d NonIoT flows. "
            "Live scaler will be saved to: %s",
            n_target, out_path,
        )

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def n_collected(self) -> int:
        return len(self._buf)

    def record(self, raw_row: np.ndarray, feature_cols: List[str]) -> None:
        """
        Store one raw (unscaled) feature row.

        Parameters
        ----------
        raw_row     : 1-D float32 array, shape (n_features,) — values BEFORE
                      any StandardScaler transform is applied.
        feature_cols: column names in the same order as raw_row, used to
                      populate the JSON "features" field for audit purposes.
        """
        if self._done:
            return
        self._buf.append(raw_row.copy())
        if len(self._buf) >= self._n_target:
            self._fit(feature_cols)

    def _fit(self, feature_cols: List[str]) -> None:
        """Fit StandardScaler on the accumulated buffer and persist to JSON."""
        from sklearn.preprocessing import StandardScaler as _SS

        X = np.stack(list(self._buf), axis=0)  # (N_CALIBRATION_FLOWS, n_features)
        scaler = _SS()
        scaler.fit(X)

        # Guard against zero-variance features (constant columns) which
        # sklearn resolves by setting scale_ = 1.0 internally.  We mirror
        # that here so the JSON is always safe to apply.
        scale = scaler.scale_.copy()
        scale[scale == 0] = 1.0

        payload = {
            "features" : feature_cols,
            "mean"     : scaler.mean_.tolist(),
            "scale"    : scale.tolist(),
            "note"     : (
                f"StandardScaler fitted on {self._n_target} live NonIoT flow feature "
                "vectors (raw, before any normalisation) by LiveScalerCalibrator. "
                "This file takes precedence over noniot_scaler.json when present."
            ),
        }

        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._out_path, "w") as fh:
            json.dump(payload, fh, indent=2)

        self._done = True
        log.warning(
            "[Calibration] NonIoT scaler fitted on %d live flows. "
            "Saved to %s. "
            "Restart monitoring.py to use it.",
            self._n_target, self._out_path,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: Stage2NonIoTDetector  (stage2_preprocess_non_iot + stage2_predict)
# ══════════════════════════════════════════════════════════════════════════════

class _NonIotCnnLstm(nn.Module if _TORCH_OK else object):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU())
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, dropout=0.3)
        # Layer order MUST match checkpoint keys:
        # head.0=Linear(128,64)  head.1=ReLU  head.2=Dropout  head.3=Linear(64,1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),   # head.0
            nn.ReLU(),            # head.1
            nn.Dropout(0.4),      # head.2
            nn.Linear(64, 1),     # head.3
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


class Stage2NonIoTDetector:
    """
    Stage-2 CNN-LSTM for Non-IoT traffic.
    Input: sliding window of FLOW ROWS (each row = one completed flow's features).
    Feature list and threshold read from .pt checkpoint.
    HARD FAILURE if model file is missing.

    SCALER BEHAVIOUR
    ────────────────
    The Non-IoT CNN-LSTM was trained on StandardScaler-normalised feature
    rows.  Without the scaler, raw live values (e.g. flow_bytes_per_sec
    = 150 000) push the sigmoid output to 0.0 for ~87 % of flows.

      Priority 1 — noniot_scaler_live.json  (live-calibrated by LiveScalerCalibrator)
        Fitted on the first N_CALIBRATION_FLOWS=500 observed NonIoT flows.
        Automatically written by the embedded calibrator.  Restart monitoring.py
        after it appears to activate it.

      Priority 2 — noniot_scaler.json  (static export from export_noniot_scaler.py)
        Used as a fallback when the live scaler has not yet been generated.
        If this was exported from already-normalised data its mean/scale values
        will be near-identity (mean ≈ 0–0.71, scale ≈ 0.001–1.0) and will
        not help.  The live calibrator corrects this automatically.

      Priority 3 — no scaler  (degraded mode)
        Raw values passed directly.  A WARNING is emitted every 100 flows.
        The pipeline does NOT crash so the rest of the architecture remains
        testable.

    THRESHOLD OVERRIDE
    ──────────────────
    Checkpoints saved with threshold < 0.30 are overridden to
    NONIOT_THRESHOLD_OVERRIDE (default 0.50) at load time.  The raw
    checkpoint value is logged as a WARNING.
    """

    def __init__(self, model_path: Path,
                 scaler_path: Optional[Path] = None,
                 live_scaler_path: Optional[Path] = None):
        if not _TORCH_OK:
            raise ImportError("torch required: pip install torch")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Non-IoT CNN-LSTM missing: {model_path}\n"
                "Train: python3 models/stage2/noniot_detector_cnnlstm.py"
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._n_features   = ckpt["n_features"]
        self._seq_len      = ckpt.get("seq_len", NONIOT_SEQ_LEN)
        self._feature_cols = ckpt.get("feature_cols", S2_NONIOT_FEATURES_FALLBACK)
        self._model = _NonIotCnnLstm(self._n_features)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        # ── Threshold override ────────────────────────────────────────────────
        # Some checkpoint files are saved with dangerously low threshold values
        # (e.g. 0.01) that cause the model to label virtually every flow as
        # botnet.  We detect this and replace it with NONIOT_THRESHOLD_OVERRIDE.
        self._threshold = float(ckpt.get("threshold", 0.5))
        if self._threshold < 0.30:
            log.warning(
                "[Stage2-NonIoT] Checkpoint threshold %.4f is dangerously low "
                "(labels almost everything as botnet). "
                "Overriding to NONIOT_THRESHOLD_OVERRIDE=%.2f.",
                self._threshold, NONIOT_THRESHOLD_OVERRIDE,
            )
            self._threshold = NONIOT_THRESHOLD_OVERRIDE

        log.info("[Stage2-NonIoT] Model loaded (%d features, threshold=%.4f, seq=%d)",
                 self._n_features, self._threshold, self._seq_len)

        # ── Load StandardScaler ───────────────────────────────────────────────
        # Priority 1: noniot_scaler_live.json  (live-calibrated, most accurate)
        # Priority 2: noniot_scaler.json        (static export, may be near-identity)
        # Priority 3: no scaler                 (degraded mode, emits warnings)
        self._s2_mean:  Optional[np.ndarray] = None
        self._s2_scale: Optional[np.ndarray] = None
        self._has_scaler = False
        self._no_scaler_warn_count = 0

        # Resolve default paths relative to the model file
        if live_scaler_path is None:
            live_scaler_path = SCALER_S2_NONIOT_LIVE
        if scaler_path is None:
            scaler_path = model_path.parent / "noniot_scaler.json"

        # Pick the best available scaler
        chosen_path: Optional[Path] = None
        scaler_label = ""
        if live_scaler_path.exists():
            chosen_path  = live_scaler_path
            scaler_label = "LIVE-CALIBRATED"
        elif scaler_path.exists():
            chosen_path  = scaler_path
            scaler_label = "static export"

        if chosen_path is not None:
            with open(chosen_path) as f:
                sc = json.load(f)
            # JSON "features" list is authoritative — overrides checkpoint list
            if "features" in sc:
                self._feature_cols = sc["features"]
            self._s2_mean  = np.array(sc["mean"],  dtype=np.float32)
            self._s2_scale = np.array(sc["scale"], dtype=np.float32)
            # Guard: replace any residual zero-scale values to avoid div/0
            self._s2_scale = np.where(self._s2_scale == 0, 1.0, self._s2_scale)
            self._has_scaler = True
            log.info("[Stage2-NonIoT] Scaler loaded (%s): %s  (%d features, "
                     "mean_range=[%.4f, %.4f], scale_range=[%.4f, %.4f])",
                     scaler_label, chosen_path, len(self._feature_cols),
                     float(self._s2_mean.min()), float(self._s2_mean.max()),
                     float(self._s2_scale.min()), float(self._s2_scale.max()))
        else:
            log.warning(
                "[Stage2-NonIoT] ⚠  No scaler found.\n"
                "         Checked (1) %s  [live-calibrated, preferred]\n"
                "                  (2) %s  [static export]\n"
                "         The CNN-LSTM was trained on NORMALISED data. Running on\n"
                "         raw live features will collapse sigmoid outputs (~87%% → 0.0).\n"
                "         LiveScalerCalibrator will auto-generate option (1) after\n"
                "         %d NonIoT flows have been observed. Restart to apply it.",
                live_scaler_path, scaler_path, N_CALIBRATION_FLOWS,
            )

        # ── Live calibration ──────────────────────────────────────────────────
        # Always instantiate the calibrator.  It is a no-op once done or if the
        # live scaler already exists (is_done becomes True immediately).
        self._calibrator = LiveScalerCalibrator(live_scaler_path)
        if live_scaler_path.exists():
            # Mark calibration as complete so record() becomes a no-op.
            self._calibrator._done = True
            log.info("[Calibration] Live scaler already exists — calibrator disabled.")

    def stage2_preprocess_non_iot(self, feat: Dict[str, float]) -> np.ndarray:
        """
        Convert a completed flow's feature dict to one (n_features,) row.

        Column order is taken from the JSON "features" list when the scaler is
        loaded (authoritative), otherwise falls back to S2_NONIOT_FEATURES_FALLBACK.

        Two things happen in sequence:

        1. The *raw* row (before any scaling) is forwarded to the
           LiveScalerCalibrator.  Once 500 such rows have been accumulated the
           calibrator fits a StandardScaler on the live distribution and saves
           it to noniot_scaler_live.json.  The operator is then prompted to
           restart monitoring.py so the new scaler is loaded.

        2. StandardScaler normalisation (z = (x - mean) / scale) is applied
           when a scaler is available, exactly matching the training-time
           transform.  Without a scaler the raw row is passed directly (with a
           periodic warning).
        """
        # Remind user every 100 flows if scaler is missing
        if not self._has_scaler:
            self._no_scaler_warn_count += 1
            if self._no_scaler_warn_count % 100 == 1:
                log.warning(
                    "[Stage2-NonIoT] Still running WITHOUT scaler (%d flows seen). "
                    "LiveScalerCalibrator needs %d/%d flows before it can fit. "
                    "Restart monitoring.py once calibration completes.",
                    self._no_scaler_warn_count,
                    self._calibrator.n_collected,
                    N_CALIBRATION_FLOWS,
                )

        # Build raw feature vector BEFORE any scaling
        raw_row = np.array(
            [feat.get(c, 0.0) for c in self._feature_cols], dtype=np.float32
        )
        raw_row = np.nan_to_num(raw_row, nan=0.0, posinf=0.0, neginf=0.0)

        # Feed raw row into the calibrator (no-op once calibration is done)
        self._calibrator.record(raw_row, self._feature_cols)

        # Apply scaler if available
        if self._has_scaler:
            return (raw_row - self._s2_mean) / self._s2_scale

        return raw_row

    def stage2_predict(self, seq: np.ndarray) -> Tuple[str, float]:
        """Run CNN-LSTM on (NONIOT_SEQ_LEN, n_features) sliding flow-row window."""
        if len(seq) < self._seq_len:
            pad = np.zeros((self._seq_len - len(seq), self._n_features), np.float32)
            seq = np.vstack([pad, seq])
        seq = seq[-self._seq_len:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self._model(x)).item()
        return ("botnet" if prob >= self._threshold else "benign"), float(prob)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: FlowAggregator
# ══════════════════════════════════════════════════════════════════════════════

class FlowAggregator:
    """
    Accumulates packets into completed bidirectional FlowRecords.
    Uses canonical (sorted) FlowKey — fixes the unidirectional aggregation bug.
    Exports on TCP FIN/RST or after FLOW_IDLE_TIMEOUT seconds of inactivity.
    """

    def __init__(self, idle_timeout: float = FLOW_IDLE_TIMEOUT):
        self._flows: Dict[FlowKey, FlowRecord] = {}
        self._idle  = idle_timeout
        self._ip_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._ip_dsts:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    def process_packet(self, ts: float,
                       src_ip: str, dst_ip: str,
                       sport: int, dport: int,
                       proto: int, pkt_len: int,
                       ttl: int, tcp_flags: int = 0) -> Optional[FlowRecord]:
        key    = FlowKey.make(src_ip, dst_ip, sport, dport, proto)
        is_fwd = (src_ip, sport) <= (dst_ip, dport)

        if key not in self._flows:
            if len(self._flows) >= MAX_TRACKED_IPS * 20:
                return None
            self._flows[key] = FlowRecord(
                key=key, start_time=ts, last_seen=ts, proto=proto,
                src_port=sport if is_fwd else dport,
                dst_port=dport if is_fwd else sport,
            )

        rec = self._flows[key]
        rec.last_seen = ts

        if is_fwd:
            rec.fwd_pkts += 1; rec.fwd_bytes += pkt_len
            rec.fwd_times.append(ts); rec.fwd_sizes.append(pkt_len)
        else:
            rec.bwd_pkts += 1; rec.bwd_bytes += pkt_len
            rec.bwd_times.append(ts); rec.bwd_sizes.append(pkt_len)

        rec.ttl_samples.append(ttl)

        if proto == 6:
            rec.flag_FIN |= int(bool(tcp_flags & 0x01))
            rec.flag_SYN |= int(bool(tcp_flags & 0x02))
            rec.flag_RST |= int(bool(tcp_flags & 0x04))
            rec.flag_PSH |= int(bool(tcp_flags & 0x08))
            rec.flag_ACK |= int(bool(tcp_flags & 0x10))
            rec.flag_URG |= int(bool(tcp_flags & 0x20))
            if tcp_flags & 0x01 or tcp_flags & 0x04:
                return self._export(key, ts)
        return None

    def flush_idle(self, now: Optional[float] = None) -> List[FlowRecord]:
        now = now or time.time()
        expired = [k for k, r in self._flows.items() if now - r.last_seen > self._idle]
        return [r for k in expired for r in [self._export(k, now)] if r]

    def _export(self, key: FlowKey, now: float) -> Optional[FlowRecord]:
        rec = self._flows.pop(key, None)
        if rec is None:
            return None
        rec.last_seen = now
        self._ip_times[rec.key.ip_lo].append(now)
        self._ip_dsts[rec.key.ip_lo].append(rec.key.ip_hi)
        return rec

    def window_stats(self, src_ip: str, window_sec: float = 60.0) -> Tuple[int, int]:
        cutoff = time.time() - window_sec
        times  = self._ip_times.get(src_ip, deque())
        dsts   = list(self._ip_dsts.get(src_ip, deque()))
        recent = [t for t in times if t > cutoff]
        return len(recent), len(set(dsts[-len(recent):]) if recent else [])


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: BotnetMonitor
# ══════════════════════════════════════════════════════════════════════════════

class BotnetMonitor:
    """
    Two-stage live botnet detection pipeline.

    Stage-1 gates Stage-2 — all inference flows through one path per flow.

    KITSUNE (per-packet, silent):
        Every packet: KitsuneExtractor → MinMax scale → per-src_ip deque.
        The buffer fills continuously. No inference fires here.
        Kitsune must run per-packet because its exponential-decay windows
        need continuous updates — waiting for Stage-1 would leave the buffer
        empty at routing time (Stage-1 fires up to 30s after the first packet).

    FLOW COMPLETION (per completed flow):
        FIN/RST/idle → FlowAggregator → flow_feature_extractor → Stage-1 RF.

        Stage-1 routes to one of two Stage-2 branches:

        "iot"    → read the Kitsune deque already accumulated for src_ip
                   → Stage-2 IoT CNN-LSTM → "botnet"|"benign"
                   (one result per completed flow, not per N packets)

        "noniot" → stage2_preprocess_non_iot → per-src_ip flow-row deque
                   → Stage-2 Non-IoT CNN-LSTM → "botnet"|"benign"
    """

    ALERT_MIN_CONF = 0.7

    def __init__(self,
                 s1_model_path:      Path = MODEL_S1_RF,
                 s1_scaler_path:     Path = SCALER_S1_JSON,
                 s2_iot_model_path:  Path = MODEL_S2_IOT,
                 s2_iot_scaler_path: Path = SCALER_S2_IOT,
                 s2_noniot_path:     Path = MODEL_S2_NONIOT):

        log.info("═" * 60)
        log.info("  Initialising BotnetMonitor")
        log.info("═" * 60)

        if not _KITSUNE_OK:
            raise ImportError(
                "KitsuneExtractor not importable.\n"
                "Ensure kitsune_extractor.py is in src/live/ relative to this file."
            )

        self.kitsune     = KitsuneExtractor()
        self._iot_bufs:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=IOT_SEQ_LEN))
        self._noniot_bufs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=NONIOT_SEQ_LEN))
        self.aggregator  = FlowAggregator()

        # Stage-1: scaler is OPTIONAL — warns but does not crash if missing.
        self.s1        = Stage1Classifier(s1_model_path, s1_scaler_path)
        # Stage-2: HARD FAILURE if model/scaler missing.
        self.s2_iot    = Stage2IoTDetector(s2_iot_model_path, s2_iot_scaler_path)
        self.s2_noniot = Stage2NonIoTDetector(s2_noniot_path)

        self._last_alert: Dict[str, float] = {}
        self._results:    List[DetectionResult] = []
        self._stats = {"packets": 0, "flows_completed": 0,
                       "iot": 0, "noniot": 0,
                       "botnet": 0, "benign": 0,
                       "alerts": 0, "suspicious_flows": 0}
        log.info("  BotnetMonitor ready.")

    def process_packet(self, timestamp: float,
                       src_ip: str, dst_ip: str,
                       src_port: int, dst_port: int,
                       proto: int, pkt_len: int, ttl: int,
                       src_mac: str = "", tcp_flags: int = 0) -> Optional[DetectionResult]:
        self._stats["packets"] += 1

        # ── Kitsune: silent per-packet accumulation ───────────────────────────
        # Runs on every packet regardless of device type. No inference fires
        # here. The buffer fills continuously so it is ready when Stage-1
        # routes a completed flow to the "iot" branch.
        proto_str = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto, "OTHER")
        raw_kit = self.kitsune.update(
            timestamp=timestamp, src_mac=src_mac or src_ip,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            pkt_len=pkt_len, protocol=proto_str,
        )
        scaled_kit = self.s2_iot.stage2_preprocess_iot(raw_kit)
        self._iot_bufs[src_ip].append(scaled_kit)

        # ── Flow aggregation: fires inference on completion via _process_flow ─
        completed = self.aggregator.process_packet(
            timestamp, src_ip, dst_ip, src_port, dst_port, proto, pkt_len, ttl, tcp_flags)
        return self._process_flow(completed) if completed else None

    def flush_idle_flows(self) -> List[DetectionResult]:
        results = []
        for rec in self.aggregator.flush_idle():
            r = self._process_flow(rec)
            if r:
                results.append(r)
        return results

    def _process_flow(self, rec: FlowRecord) -> Optional[DetectionResult]:
        t0 = time.perf_counter()
        self._stats["flows_completed"] += 1
        src_ip = rec.key.ip_lo

        # ── Feature extraction & suspicion scoring ────────────────────────────
        wcount, wdsts = self.aggregator.window_stats(src_ip)
        feat = flow_feature_extractor(rec, wcount, wdsts)

        susp, is_susp = suspicion_scoring(feat)
        if is_susp:
            self._stats["suspicious_flows"] += 1
            log.info("  [!] Suspicious flow %s (score=%.1f)", str(rec.key), susp)

        # ── Stage-1: IoT vs Non-IoT routing ──────────────────────────────────
        device_type, s1_conf = self.s1.stage1_predict(feat)

        # ── Stage-2: gated by Stage-1 result ─────────────────────────────────
        if device_type == "iot":
            # Read the Kitsune buffer that has been filling silently per-packet.
            # Inference fires here — once per completed flow — not on a packet
            # counter. If the buffer is not yet full (fewer than IOT_SEQ_LEN
            # packets seen from this IP), we defer and return no result.
            self._stats["iot"] += 1
            buf = self._iot_bufs[src_ip]
            if len(buf) < IOT_SEQ_LEN:
                log.debug("  [IoT] %-18s Kitsune buffer not full yet (%d/%d) — deferring",
                          src_ip, len(buf), IOT_SEQ_LEN)
                return None
            seq = np.stack(list(buf))   # (IOT_SEQ_LEN, 115) — real temporal sequence
            label, s2_conf = self.s2_iot.stage2_predict(seq)

        else:  # "noniot"
            # Append this flow's feature row to the sliding window and infer.
            self._stats["noniot"] += 1
            row = self.s2_noniot.stage2_preprocess_non_iot(feat)
            self._noniot_bufs[src_ip].append(row)
            seq = np.stack(list(self._noniot_bufs[src_ip]))
            label, s2_conf = self.s2_noniot.stage2_predict(seq)

        lat_ms  = (time.perf_counter() - t0) * 1000
        alerted = self._maybe_alert(src_ip, label, s2_conf, rec.last_seen, device_type)
        self._stats["botnet" if label == "botnet" else "benign"] += 1

        result = DetectionResult(
            flow_id=str(rec.key), src_ip=src_ip, dst_ip=rec.key.ip_hi,
            device_type=device_type, label=label,
            s1_confidence=round(s1_conf, 4), s2_confidence=round(s2_conf, 4),
            suspicion_score=round(susp, 2), latency_ms=round(lat_ms, 2),
            alerted=alerted, timestamp=rec.last_seen,
        )
        self._results.append(result)
        log.debug("  [Flow] %-22s dev=%-6s label=%-7s s1=%.2f s2=%.2f lat=%.1fms",
                  src_ip, device_type, label, s1_conf, s2_conf, lat_ms)
        return result

    def _maybe_alert(self, src_ip: str, label: str, conf: float,
                     now: float, device_type: str) -> bool:
        if label != "botnet" or conf < self.ALERT_MIN_CONF:
            return False
        if now - self._last_alert.get(src_ip, 0.0) < ALERT_COOLDOWN:
            return False
        self._last_alert[src_ip] = now
        self._stats["alerts"] += 1
        log.warning("  🚨 ALERT  %-20s  device=%-6s  label=BOTNET  conf=%.3f",
                    src_ip, device_type, conf)
        return True

    def print_summary(self, final: bool = False) -> None:
        s   = self._stats
        tag = "FINAL SUMMARY" if final else "── Stats ──"
        log.info(
            "%s  pkts=%d  flows=%d  IoT=%d  NonIoT=%d  "
            "botnet=%d  benign=%d  alerts=%d  suspicious=%d",
            tag, s["packets"], s["flows_completed"],
            s["iot"], s["noniot"], s["botnet"], s["benign"],
            s["alerts"], s["suspicious_flows"],
        )

    def save_results(self, path: str = "detection_results.csv") -> None:
        if not self._results:
            log.info("No results to save.")
            return
        pd.DataFrame([vars(r) for r in self._results]).to_csv(path, index=False)
        log.info("Results saved → %s (%d rows)", path, len(self._results))


# ══════════════════════════════════════════════════════════════════════════════
# LIVE CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

def run_live(monitor: BotnetMonitor, iface: Optional[str],
             duration: Optional[float] = None) -> None:
    """
    Capture live packets. promisc=False is intentional on macOS:
    promiscuous mode requires a network extension entitlement.
    Without it you see only your own machine's traffic — correct for
    a single-host agent.
    """
    if not _SCAPY_OK:
        raise ImportError("scapy required: pip install scapy")
    log.info("Live capture: iface=%s  promisc=False", iface or "auto")
    last_flush = [time.time()]

    def handler(pkt):
        if IP not in pkt:
            return
        ip  = pkt[IP]; t = float(pkt.time)
        mac = pkt[Ether].src if Ether in pkt else ip.src
        if TCP in pkt:
            tcp = pkt[TCP]
            monitor.process_packet(t, ip.src, ip.dst, tcp.sport, tcp.dport,
                                   6, len(pkt), ip.ttl, mac, int(tcp.flags))
        elif UDP in pkt:
            udp = pkt[UDP]
            monitor.process_packet(t, ip.src, ip.dst, udp.sport, udp.dport,
                                   17, len(pkt), ip.ttl, mac, 0)
        elif ICMP in pkt:
            monitor.process_packet(t, ip.src, ip.dst, 0, 0,
                                   1, len(pkt), ip.ttl, mac, 0)
        now = time.time()
        if now - last_flush[0] > 5.0:
            monitor.flush_idle_flows(); last_flush[0] = now

    try:
        sniff(iface=iface, prn=handler, store=False,
              timeout=duration, promisc=False, filter="ip")
    except KeyboardInterrupt:
        pass
    monitor.flush_idle_flows()


# ══════════════════════════════════════════════════════════════════════════════
# PCAP REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def run_pcap(monitor: BotnetMonitor, pcap_path: str) -> None:
    if not _SCAPY_OK:
        raise ImportError("scapy required: pip install scapy")
    from scapy.all import rdpcap
    log.info("Replaying PCAP: %s", pcap_path)
    for pkt in rdpcap(pcap_path):
        if IP not in pkt:
            continue
        ip = pkt[IP]; t = float(pkt.time)
        mac = pkt[Ether].src if Ether in pkt else ip.src
        if TCP in pkt:
            tcp = pkt[TCP]
            monitor.process_packet(t, ip.src, ip.dst, tcp.sport, tcp.dport,
                                   6, len(pkt), ip.ttl, mac, int(tcp.flags))
        elif UDP in pkt:
            udp = pkt[UDP]
            monitor.process_packet(t, ip.src, ip.dst, udp.sport, udp.dport,
                                   17, len(pkt), ip.ttl, mac, 0)
    monitor.flush_idle_flows()


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(monitor: BotnetMonitor, n_packets: int = 2000,
                   botnet_ratio: float = 0.15) -> None:
    """Synthetic packets for pipeline structure testing. Not for accuracy eval."""
    import random
    random.seed(42); np.random.seed(42)
    iot_ips    = ["192.168.1.101", "192.168.1.102"]
    noniot_ips = ["10.0.0.10", "10.0.0.11"]
    servers    = ["8.8.8.8", "1.1.1.1", "93.184.216.34"]
    t0 = time.time(); last_flush = t0

    log.info("Simulation: %d packets (%.0f%% botnet)", n_packets, botnet_ratio * 100)
    for i in range(n_packets):
        t = t0 + i * 0.01
        is_bot = random.random() < botnet_ratio
        is_iot = random.random() < 0.5
        src  = random.choice(iot_ips if is_iot else noniot_ips)
        dst  = (f"10.{random.randint(0,255)}.{random.randint(0,255)}.1"
                if is_bot else random.choice(servers))
        monitor.process_packet(
            t, src, dst,
            random.randint(1024, 65535),
            random.choice([23, 2323] if is_bot else [80, 443]),
            6,
            random.randint(40, 100) if is_bot else random.randint(200, 1400),
            random.randint(28, 35)  if is_bot else random.randint(54, 64),
            f"aa:bb:cc:{i%255:02x}:ee:ff",
            0x02 if i % 30 == 0 else (0x10 if i % 5 else 0x11),
        )
        if t - last_flush > 5.0:
            monitor.flush_idle_flows(); last_flush = t
        if i > 0 and i % 500 == 0:
            monitor.print_summary()
    monitor.flush_idle_flows()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Group 07 — Live Botnet Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick-start:
  1. python3 export_s1_scaler.py                  generate s1_scaler.json
  2. sudo python3 monitoring.py --iface en0        live capture
     python3 monitoring.py --pcap capture.pcap    replay PCAP
     python3 monitoring.py --simulate             structure test (no live capture needed)

Environment overrides:
  MODEL_S1_RF  SCALER_S1_JSON  MODEL_S2_IOT  SCALER_S2_IOT  MODEL_S2_NONIOT
        """,
    )
    ap.add_argument("--iface",    type=str,   default=None)
    ap.add_argument("--pcap",     type=str,   default=None)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--n-pkts",  type=int,   default=2000)
    ap.add_argument("--output",  type=str,   default="detection_results.csv")
    ap.add_argument("--debug",   action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    monitor = BotnetMonitor()

    try:
        if args.simulate:
            run_simulation(monitor, n_packets=args.n_pkts)
        elif args.pcap:
            run_pcap(monitor, args.pcap)
        else:
            run_live(monitor, args.iface, args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.print_summary(final=True)
        monitor.save_results(args.output)


if __name__ == "__main__":
    main()
