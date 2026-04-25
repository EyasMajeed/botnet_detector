"""
════════════════════════════════════════════════════════════════════════
 live_detector.py  —  Real-time Botnet Detection Pipeline
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 Sniffs live packets, extracts N-BaIoT features via Kitsune, and
 runs the CNN-LSTM to detect botnet-infected IoT devices in real time.

 SETUP (run in this order):
   1. python3 src/ingestion/preprocess_nbaiot.py
      → produces stage2_iot_botnet.csv AND models/stage2/iot_scaler.json

   2. python3 models/stage2/iot_detector.py
      → trains and saves models/stage2/iot_cnn_lstm.pt

   3. sudo python3 src/live/live_detector.py --interface en0 --duration 60
      → starts live detection

 IMPORTANT — SCALER:
   iot_scaler.json contains the min/max of RAW N-BaIoT features
   (before normalization). The live Kitsune extractor produces raw
   values (weight≈15, mean≈100, variance≈1000) which must be scaled
   with the SAME min/max used during training preprocessing.
   The scaler is produced automatically by preprocess_nbaiot.py.

 REQUIREMENTS:
   pip install scapy numpy torch
   Run with admin/root (required for packet capture).
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

try:
    from kitsune_extractor import KitsuneExtractor, FEATURE_NAMES
except ImportError:
    from src.live.kitsune_extractor import KitsuneExtractor, FEATURE_NAMES

try:
    from scapy.all import sniff, Ether, IP, TCP, UDP, ICMP
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False
    print("[WARN] scapy not installed. Run: pip install scapy")


# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

SEQ_LEN        = 20
INFER_EVERY    = 20       # run inference every N new packets per device
ALERT_COOLDOWN = 10.0     # seconds between repeated alerts for same src_ip
MAX_DEVICES    = 500


# ════════════════════════════════════════════════════════════════════════
# CNN-LSTM MODEL DEFINITION
# (must match architecture in iot_detector.py exactly)
# ════════════════════════════════════════════════════════════════════════

class _CnnLstm(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, stride=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.lstm  = torch.nn.LSTM(256, 128, 2,
                                   batch_first=True, dropout=0.3)
        self.head  = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


# ════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """
    Load CNN-LSTM + scaler from disk.
    The scaler JSON must contain min/max of RAW (pre-normalization) features.
    """
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model = _CnnLstm(ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold    = float(ckpt.get("threshold", 0.07))
    feature_cols = ckpt.get("feature_cols", FEATURE_NAMES)

    # Load scaler — must be from RAW data (saved by preprocess_nbaiot.py)
    scaler_path = Path(model_path).parent / "iot_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"\n  iot_scaler.json not found at {scaler_path}\n\n"
            "  Run preprocessing first to generate it:\n"
            "    python3 src/ingestion/preprocess_nbaiot.py\n\n"
            "  This file contains the min/max of RAW N-BaIoT features\n"
            "  and is required for correct live feature normalization.")

    with open(scaler_path) as f:
        scaler_data = json.load(f)

    # Validate scaler is from raw data (not the already-normalized CSV)
    raw_max = np.array(scaler_data["max"])
    if raw_max.max() < 2.0:
        raise ValueError(
            "\n  iot_scaler.json appears to be from already-normalized data\n"
            "  (all max values are <= 1.0). This means the scaler was\n"
            "  incorrectly exported from the processed CSV instead of raw data.\n\n"
            "  Fix: rerun preprocess_nbaiot.py to regenerate both the CSV\n"
            "  and the scaler from raw N-BaIoT files.")

    print(f"  Model loaded    : {model_path}")
    print(f"  Threshold       : {threshold:.4f}")
    print(f"  Scaler loaded   : {scaler_path}")
    print(f"  Raw feature range sample:")
    feats = scaler_data.get("features", FEATURE_NAMES)
    mins  = scaler_data["min"]
    maxs  = scaler_data["max"]
    for i in range(min(4, len(feats))):
        print(f"    {feats[i]:<35} "
              f"raw min={mins[i]:>10.3f}  raw max={maxs[i]:>12.3f}")

    return model, threshold, feature_cols, scaler_data


def normalise(features: np.ndarray, scaler_data: dict) -> np.ndarray:
    """
    Apply min-max scaling using raw feature ranges from scaler_data.
    Reproduces exactly what MinMaxScaler did during preprocessing.

    scaled = (raw - min) / (max - min)   clipped to [0, 1]
    """
    mins  = np.array(scaler_data["min"],  dtype=np.float32)
    maxs  = np.array(scaler_data["max"],  dtype=np.float32)
    scale = maxs - mins
    scale[scale == 0] = 1.0     # constant features → stay 0
    scaled = (features - mins) / scale
    return np.clip(scaled, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════
# LIVE DETECTOR
# ════════════════════════════════════════════════════════════════════════

class LiveDetector:
    """
    Captures live packets → Kitsune features → CNN-LSTM inference.
    Thread-safe; packet capture and inference run in the scapy thread.
    """

    def __init__(self,
                 model_path : str,
                 interface  : Optional[str] = None,
                 on_alert   : Optional[Callable] = None,
                 verbose    : bool = True):
        if not _SCAPY_AVAILABLE:
            raise RuntimeError("scapy not installed. Run: pip install scapy")

        print(f"\n  Initialising LiveDetector...")
        self.model, self.threshold, self.feature_cols, self.scaler = \
            load_model(model_path)

        self.interface  = interface
        self.on_alert   = on_alert
        self.verbose    = verbose

        self.extractor  = KitsuneExtractor()

        # Per-device sliding window buffer
        self.buffers    : dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SEQ_LEN))
        self.pkt_count  : dict[str, int]   = defaultdict(int)
        self.last_alert : dict[str, float] = {}
        self._stop      = threading.Event()
        self._stats     = {"packets": 0, "inferences": 0, "alerts": 0}

    # ── Public API ────────────────────────────────────────────────────

    def start(self, duration: Optional[float] = None):
        """Start live packet capture (blocking). Ctrl-C to stop."""
        print(f"\n  Interface : {self.interface or 'auto'}")
        print(f"  SEQ_LEN   : {SEQ_LEN}")
        print(f"  Threshold : {self.threshold:.4f}")
        print(f"  Duration  : {duration or 'infinite'}s")
        print(f"\n  Listening... (Ctrl-C to stop)\n")

        try:
            sniff(
                iface       = self.interface,
                prn         = self._process_packet,
                store       = False,
                timeout     = duration,
                promisc     = False,   # required on macOS Wi-Fi (BPF limitation)
                stop_filter = lambda _: self._stop.is_set(),
            )
        except KeyboardInterrupt:
            pass
        self._print_summary()

    def stop(self):
        self._stop.set()

    def get_status(self) -> dict:
        """Poll from dashboard thread."""
        return {**self._stats, "devices": len(self.buffers)}

    # ── Packet handler ────────────────────────────────────────────────

    def _process_packet(self, pkt) -> None:
        try:
            if IP not in pkt:
                return

            ip   = pkt[IP]
            t    = float(pkt.time)
            src  = ip.src
            dst  = ip.dst
            size = len(pkt)
            src_mac = pkt[Ether].src if Ether in pkt else src

            if TCP in pkt:
                sport, dport, proto = pkt[TCP].sport, pkt[TCP].dport, "TCP"
            elif UDP in pkt:
                sport, dport, proto = pkt[UDP].sport, pkt[UDP].dport, "UDP"
            elif ICMP in pkt:
                sport, dport, proto = 0, 0, "ICMP"
            else:
                sport, dport, proto = 0, 0, "OTHER"
        except Exception:
            return

        self._stats["packets"] += 1

        if src not in self.buffers and len(self.buffers) >= MAX_DEVICES:
            return

        # 1. Extract 115 raw Kitsune features
        raw_feat = self.extractor.update(
            timestamp=t, src_mac=src_mac,
            src_ip=src, dst_ip=dst,
            src_port=sport, dst_port=dport,
            pkt_len=size, protocol=proto)

        # 2. Normalise using RAW scaler (min/max of pre-normalization data)
        scaled_feat = normalise(raw_feat, self.scaler)

        # 3. Append to device buffer
        self.buffers[src].append(scaled_feat)
        self.pkt_count[src] += 1

        # 4. Run inference when buffer is full
        if (len(self.buffers[src]) == SEQ_LEN and
                self.pkt_count[src] % INFER_EVERY == 0):
            self._infer(src, t)

    # ── Inference ─────────────────────────────────────────────────────

    def _infer(self, src_ip: str, t: float) -> None:
        seq = np.stack(list(self.buffers[src_ip]))       # (20, 115)
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1,20,115)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        label = "botnet" if prob >= self.threshold else "benign"
        self._stats["inferences"] += 1

        if self.verbose:
            icon = "🚨 BOTNET" if label == "botnet" else "   benign"
            print(f"  {icon}  {src_ip:<18}  "
                  f"conf={prob:.4f}  pkts={self.pkt_count[src_ip]}")

        if label == "botnet":
            last = self.last_alert.get(src_ip, 0.0)
            if t - last >= ALERT_COOLDOWN:
                self.last_alert[src_ip] = t
                self._stats["alerts"] += 1
                if self.on_alert:
                    self.on_alert(src_ip, float(prob), t)

    # ── Summary ───────────────────────────────────────────────────────

    def _print_summary(self):
        print(f"\n{'─'*52}")
        print(f"  Capture complete")
        print(f"{'─'*52}")
        print(f"  Packets captured  : {self._stats['packets']:,}")
        print(f"  Inferences run    : {self._stats['inferences']:,}")
        print(f"  Botnet alerts     : {self._stats['alerts']:,}")
        print(f"  Devices tracked   : {len(self.buffers):,}")
        print(f"  Kitsune streams   : {self.extractor.n_streams:,}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def _export_scaler(csv_path: str, model_dir: str) -> None:
    """
    Read the processed CSV and save raw min/max scaler to iot_scaler.json.
    Run this ONCE after preprocessing and before live capture.
    """
    import pandas as pd
    print(f"  Loading {csv_path} to compute scaler...")
    META = {"class_label", "attack_type", "device_name", "src_ip", "seq_index"}
    df = pd.read_csv(csv_path, low_memory=False)
    feat = [c for c in df.columns if c not in META]
    scaler_data = {
        "features": feat,
        "min": df[feat].min().tolist(),
        "max": df[feat].max().tolist(),
        "note": "Min/max of RAW N-BaIoT features before MinMaxScaler.",
    }
    out = Path(model_dir) / "iot_scaler.json"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        import json; json.dump(scaler_data, f, indent=2)
    print(f"  Scaler saved: {out}")
    print(f"  Features: {len(feat)}")
    raw_max = max(scaler_data["max"])
    if raw_max < 2.0:
        print("  [WARN] All max values <= 1.0 — CSV may already be normalized.")
        print("         Re-run preprocess_nbaiot.py to get the correct raw scaler.")
    else:
        print(f"  OK — raw max={raw_max:.2f} (confirms pre-normalization values)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Live IoT Botnet Detector — CNN-LSTM + Kitsune")
    ap.add_argument("--interface", "-i", default=None,
                    help="Network interface (e.g. en0, eth0).")
    ap.add_argument("--model",     "-m",
                    default="models/stage2/iot_cnn_lstm.pt")
    ap.add_argument("--duration",  "-d", type=float, default=None,
                    help="Capture seconds (default: infinite).")
    ap.add_argument("--export-scaler", action="store_true",
                    help="Export scaler from processed CSV and exit.")
    ap.add_argument("--csv",
                    default="data/processed/stage2_iot_botnet.csv",
                    help="Processed CSV (used with --export-scaler).")
    ap.add_argument("--model-dir", default="models/stage2",
                    help="Model directory (used with --export-scaler).")
    args = ap.parse_args()

    if args.export_scaler:
        _export_scaler(args.csv, args.model_dir)
        import sys; sys.exit(0)

    def alert(src_ip, conf, ts):
        print(f"\n  *** ALERT: {src_ip} is BOTNET  "
              f"(conf={conf:.4f}  t={ts:.1f}) ***\n")

    detector = LiveDetector(
        model_path = args.model,
        interface  = args.interface,
        on_alert   = alert,
        verbose    = True,
    )
    detector.start(duration=args.duration)
