"""
inference_bridge.py — File / single-flow inference for the GUI.

Used by:
    UploadPage._run_detection → run_file_inference(info)        (batch)
    MonitorPage._on_flow      → run_inference(flow_dict)        (single)

File-format routing:
    PCAP / PCAPNG  →  Stage-1 + Stage-2 IoT/NonIoT (full pipeline)
    CSV (any)      →  Stage-1 + Stage-2 NonIoT only (IoT rows = 'unknown'
                       because Stage-2 IoT needs raw packet sequences)

We deliberately reuse the wrapper classes from monitoring.py
(Stage1Classifier, Stage2NonIoTDetector, BotnetMonitor) instead of the
half-finished loaders in models/stage1/classifier.py — those bypass the
StandardScaler and produce wrong predictions.
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
    if not _S1_FEATURES:
        _get_stage1()
    return _S1_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(flow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Stage-1 + Stage-2 NonIoT on a single flow dict (UI fall-back path).
    For real-time monitoring use BotnetMonitor (monitor_bridge.py) — it
    carries per-IP sliding-window state we don't have here.
    """
    t0 = time.perf_counter()
    s1 = _get_stage1()
    s2 = _get_stage2_noniot()
    feat = _flow_to_feat_dict(flow)
    device, s1_conf = s1.stage1_predict(feat)
    if device == "noniot":
        row = s2.stage2_preprocess_non_iot(feat)
        seq = np.stack([row])
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
    Run the full Stage-1 + Stage-2 pipeline over an uploaded file.

    Routing:
        PCAP / PCAPNG  →  _run_pcap_inference()  (Stage-1 + Stage-2 IoT + NonIoT)
        CSV (any)      →  _run_csv_inference()   (Stage-1 + Stage-2 NonIoT only)

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
          "confidence":   float,        # Stage-2 sigmoid (0.0 for IoT-from-CSV)
          "stage1_conf":  float,
          "latency_ms":   float,
        }
    """
    if not getattr(info, "is_valid", False):
        raise ValueError("Cannot run inference on an invalid file.")

    if info.format in (FileFormat.PCAP, FileFormat.PCAPNG):
        return _run_pcap_inference(info.path)

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
            "Supported formats: PCAP, PCAPNG, CSV (unified / CICFlowMeter / "
            "CTU-13 / UNSW-NB15 / NetFlow CSV)."
        )
    return _run_csv_inference(info)


def _run_csv_inference(info: Any) -> List[Dict[str, Any]]:
    """
    CSV path: Stage-1 RF + Stage-2 NonIoT CNN-LSTM. IoT-classified rows
    return label='unknown' because Stage-2 IoT needs raw packet-level
    Kitsune statistics not available from a flow-level CSV.
    """
    try:
        df = pd.read_csv(info.path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}") from e
    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    s1    = _get_stage1()
    s2    = _get_stage2_noniot()
    feats = _ensure_features_loaded()

    df.columns = [c.strip() for c in df.columns]
    df = _alias_columns(df, info.format)

    matched = sum(1 for c in feats if c in df.columns)
    if matched < 0.4 * len(feats):
        raise ValueError(
            f"CSV has only {matched}/{len(feats)} of the 56 unified features. "
            "Convert it with data_processing/process_*.py first, or upload a "
            "CSV produced by your team's preprocessing pipeline."
        )

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    noniot_bufs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_NONIOT_SEQ_LEN))

    for i, row in df.iterrows():
        feat = {c: _safe_float(row.get(c, 0.0)) for c in feats}
        try:
            device, s1_conf = s1.stage1_predict(feat)
        except Exception as e:
            device, s1_conf = "noniot", 0.0
            print(f"[run_file_inference] Stage-1 failed on row {i}: {e!r}")

        if device == "noniot":
            try:
                row_vec = s2.stage2_preprocess_non_iot(feat)
                src_ip_key = str(row.get("src_ip", "") or f"row_{i}")
                noniot_bufs[src_ip_key].append(row_vec)
                seq = np.stack(list(noniot_bufs[src_ip_key]))
                label, s2_conf = s2.stage2_predict(seq)
            except Exception as e:
                label, s2_conf = "unknown", 0.0
                print(f"[run_file_inference] Stage-2 NonIoT failed on row {i}: {e!r}")
        else:
            label, s2_conf = "unknown", 0.0

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

    avg_ms = round((time.perf_counter() - t0) * 1000 / max(len(results), 1), 2)
    for r in results:
        r["latency_ms"] = avg_ms
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _safe_float(v: Any) -> float:
    try:
        f = float(v)
        if not np.isfinite(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _flow_to_feat_dict(flow: Dict[str, Any]) -> Dict[str, float]:
    feats = _ensure_features_loaded()
    return {k: _safe_float(flow.get(k, 0.0)) for k in feats}


def _alias_columns(df: pd.DataFrame, fmt: FileFormat) -> pd.DataFrame:
    """
    Best-effort column rename so dataset-specific exports work with the
    56-feature unified schema. Returns a renamed copy when aliases applied.

    NOTE: this does NOT recompute features — it only renames matching columns.
    For full coverage on raw CIC-IDS / CTU-13 dumps, run them through
    data_processing/process_cicids2017.py or process_ctu13.py first.
    """
    aliases: Dict[str, Dict[str, str]] = {
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


# ═════════════════════════════════════════════════════════════════════════════
# PCAP inference
# ═════════════════════════════════════════════════════════════════════════════

def _run_pcap_inference(pcap_path: str) -> List[Dict[str, Any]]:
    """
    Replay a PCAP / PCAPNG file through the full BotnetMonitor pipeline.

    This gives us proper Stage-1 + Stage-2 IoT/NonIoT detection because we
    have raw packets, which means Kitsune (the IoT Stage-2 feature
    extractor) can build its 115-dim packet-level statistics.

    Implementation notes:
        - We construct a FRESH BotnetMonitor per call. BotnetMonitor carries
          per-IP buffers + flow aggregator state that would leak between
          uploads if cached. The 3-5s model-loading cost per upload is
          acceptable; if it becomes a bottleneck, add a reset() method to
          BotnetMonitor that wipes state but keeps loaded weights.

        - flush_idle_flows() expires flows whose last_seen is older than
          FLOW_IDLE_TIMEOUT (30s) relative to time.time(). After processing
          a PCAP with arbitrary timestamps, that may or may not catch every
          open flow. We force-flush by setting the aggregator's idle window
          to a huge negative number, which makes every open flow expired on
          the next scan regardless of timestamps.

        - This blocks the main thread until the PCAP is fully processed.
          For a small (<10MB) PCAP that's fine. For very large files,
          consider running in a QThread (with the main-thread torch.load
          caveat from monitor_bridge.py applied — i.e. construct the
          BotnetMonitor on the main thread first, then hand it to the
          worker for the sniff loop).
    """
    try:
        from monitoring import BotnetMonitor
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Ether
    except Exception as e:
        raise RuntimeError(
            f"PCAP inference unavailable: {e!r}. "
            "Install scapy and ensure monitoring.py + all model artifacts are in place."
        ) from e

    monitor = BotnetMonitor()        # loads RF + 2 CNN-LSTMs (~3-5s)
    detection_results = []

    t0 = time.perf_counter()
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read PCAP: {e}") from e

    for pkt in packets:
        if IP not in pkt:
            continue
        ip   = pkt[IP]
        ts   = float(pkt.time)
        mac  = pkt[Ether].src if Ether in pkt else ip.src
        plen = len(pkt)
        try:
            if TCP in pkt:
                tcp = pkt[TCP]
                r = monitor.process_packet(
                    ts, ip.src, ip.dst, int(tcp.sport), int(tcp.dport),
                    6, plen, int(ip.ttl), mac, int(tcp.flags),
                )
            elif UDP in pkt:
                udp = pkt[UDP]
                r = monitor.process_packet(
                    ts, ip.src, ip.dst, int(udp.sport), int(udp.dport),
                    17, plen, int(ip.ttl), mac, 0,
                )
            elif ICMP in pkt:
                r = monitor.process_packet(
                    ts, ip.src, ip.dst, 0, 0,
                    1, plen, int(ip.ttl), mac, 0,
                )
            else:
                continue
        except Exception as e:
            print(f"[_run_pcap_inference] process_packet error: {e!r}")
            continue
        if r is not None:
            detection_results.append(r)

    # Force-flush every still-open flow.
    # _idle = -1e9 makes the predicate `(now - last_seen) > _idle` always true.
    try:
        monitor.aggregator._idle = -1e9
        detection_results.extend(monitor.flush_idle_flows())
    except Exception as e:
        print(f"[_run_pcap_inference] final flush error: {e!r}")

    return _detection_results_to_dicts(detection_results, t0)


def _detection_results_to_dicts(detection_results: List[Any],
                                t0: float) -> List[Dict[str, Any]]:
    """
    Convert a list of monitoring.DetectionResult into the GUI's result-dict
    format. Shared by the sync (_run_pcap_inference) and async
    (inference_worker.PcapInferenceThread) paths so they emit identical shapes.

    `t0` is a perf_counter() snapshot taken when packet processing started;
    it's used to compute an average per-row latency stamped onto every dict.
    """
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(detection_results, 1):
        src_port = dst_port = 0
        proto    = "—"
        try:
            left, _, rest = r.flow_id.partition("<->")
            right, _, p   = rest.partition("/")
            _, _, sp = left.rpartition(":")
            _, _, dp = right.rpartition(":")
            src_port = int(sp); dst_port = int(dp)
            proto    = {"6":"TCP","17":"UDP","1":"ICMP"}.get(p, p)
        except Exception:
            # Malformed flow_id — leave ports/proto as defaults
            pass
        out.append({
            "row":         i,
            "src_ip":      r.src_ip,
            "dst_ip":      r.dst_ip,
            "src_port":    src_port,
            "dst_port":    dst_port,
            "protocol":    proto,
            "device_type": r.device_type,
            "label":       r.label,
            "confidence":  float(r.s2_confidence),
            "stage1_conf": float(r.s1_confidence),
        })

    avg_ms = round((time.perf_counter() - t0) * 1000 / max(len(out), 1), 2)
    for o in out:
        o["latency_ms"] = avg_ms
    return out