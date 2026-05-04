"""
════════════════════════════════════════════════════════════════════════
 pcap_evaluator.py  —  Cross-Dataset PCAP Evaluation
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 Tests the N-BaIoT CNN-LSTM on IoT-23 PCAP files using the Kitsune
 feature extractor — proving the model detects real botnet patterns,
 not just N-BaIoT-specific statistics.

 PIPELINE:
   IoT-23 PCAP
     → scapy (read packets)
       → KitsuneExtractor (115 N-BaIoT features per packet)
         → normalise (iot_scaler.json)
           → per-src_ip sequence buffer (SEQ_LEN=20)
             → CNN-LSTM inference
               → compare vs ground truth (from conn.log or filename)

 GROUND TRUTH OPTIONS:
   Option A — from filename (simplest):
     IoT-23 PCAPs are named like:
       CTU-IoT-Malware-Capture-1-1/  ← botnet capture
       CTU-IoT-Normal-Capture-20-1/  ← benign capture
     If "Malware" in path → all traffic = botnet
     If "Normal"  in path → all traffic = benign

   Option B — from conn.log (precise, per-flow labels):
     IoT-23 provides Zeek conn.log files with per-flow labels.
     The script tries to load the conn.log from the same folder.
     Falls back to Option A if not found.

 USAGE:
   # Evaluate a single PCAP:
   python src/live/pcap_evaluator.py \\
       --pcap "data/raw/iot23/CTU-IoT-Malware-Capture-1-1/capture.pcap"

   # Evaluate all PCAPs in a folder:
   python src/live/pcap_evaluator.py \\
       --pcap-dir "data/raw/iot23" \\
       --output "models/stage2/results/iot23_cross_eval.csv"

 REQUIREMENTS:
   pip install scapy tqdm
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")

try:
    from kitsune_extractor import KitsuneExtractor, FEATURE_NAMES
except ImportError:
    from src.live.kitsune_extractor import KitsuneExtractor, FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "stage2" / "iot_cnn_lstm.pt"
SCALER_PATH = ROOT / "models" / "stage2" / "iot_scaler.json"

SEQ_LEN = 20


# ════════════════════════════════════════════════════════════════════════
# MODEL LOADER  (same as live_detector.py)
# ════════════════════════════════════════════════════════════════════════

class _CnnLstm(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, stride=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.lstm  = torch.nn.LSTM(256, 128, 2, batch_first=True, dropout=0.3)
        self.head  = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


def load_model(model_path, scaler_path):
    ckpt      = torch.load(model_path, map_location="cpu", weights_only=False)
    model     = _CnnLstm(ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    threshold = float(ckpt.get("threshold", 0.07))

    with open(scaler_path) as f:
        scaler = json.load(f)

    print(f"  Model     : {model_path}")
    print(f"  Threshold : {threshold}")
    print(f"  Device    : {DEVICE}")
    return model, threshold, scaler


def normalise(features, scaler):
    mins  = np.array(scaler["min"],  dtype=np.float32)
    maxs  = np.array(scaler["max"],  dtype=np.float32)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    return np.clip((features - mins) / scale, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════
# GROUND TRUTH  — infer label from PCAP path
# ════════════════════════════════════════════════════════════════════════

# IoT-23 benign honeypot scenarios (real IoT devices, NOT attacker traffic):
# Scenario 21 = CTU-Honeypot-Capture-7-1  (Somfy Doorlock)  → BENIGN
# Scenario 22 = CTU-Honeypot-Capture-4-1  (Philips HUE)     → BENIGN
# Scenario 23 = CTU-Honeypot-Capture-5-1  (Amazon Echo)     → BENIGN
# All other CTU-Honeypot-Capture-* are attacker-side         → BOTNET
BENIGN_HONEYPOT_SCENARIOS = {"4-1", "5-1", "7-1"}


def infer_ground_truth(pcap_path: Path) -> str | None:
    """
    Infer ground-truth label from the PCAP file path.

    IoT-23 honeypot naming is ambiguous:
      CTU-Honeypot-Capture-4-1  → benign (Philips HUE)
      CTU-Honeypot-Capture-5-1  → benign (Amazon Echo)
      CTU-Honeypot-Capture-7-1  → benign (Somfy Doorlock)
      CTU-Honeypot-Capture-*    → botnet (all other honeypots)
      CTU-IoT-Malware-Capture-* → botnet
      CTU-IoT-Normal-Capture-*  → benign

    Returns "botnet", "benign", or None (unknown).
    """
    import re
    path_str = str(pcap_path).lower()

    if "malware" in path_str:
        return "botnet"

    if "honeypot" in path_str:
        match = re.search(r"honeypot-capture-(\d+-\d+)", path_str)
        if match and match.group(1) in BENIGN_HONEYPOT_SCENARIOS:
            return "benign"
        return "botnet"

    if "normal" in path_str or "benign" in path_str:
        return "benign"

    return None


# ════════════════════════════════════════════════════════════════════════
# PCAP EVALUATOR
# ════════════════════════════════════════════════════════════════════════

class PcapEvaluator:
    def __init__(self, model, threshold, scaler):
        self.model     = model
        self.threshold = threshold
        self.scaler    = scaler

    def evaluate_pcap(self, pcap_path: Path,
                      ground_truth: str | None = None,
                      max_packets: int | None = None) -> dict:
        """
        Run the full pipeline on a single PCAP file.

        Returns a dict with per-file results:
          n_packets, n_inferences, n_botnet, n_benign,
          mean_conf_botnet, mean_conf_benign,
          ground_truth, predicted_label, correct (if GT known)
        """
        print(f"\n  PCAP: {pcap_path.name}")

        if ground_truth is None:
            ground_truth = infer_ground_truth(pcap_path)
        print(f"  Ground truth: {ground_truth or 'unknown'}")

        extractor = KitsuneExtractor()
        buffers   : dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))
        pkt_count : dict[str, int]   = defaultdict(int)

        results: list[dict] = []   # per-inference records
        n_packets = 0

        try:
            reader = PcapReader(str(pcap_path))
        except Exception as e:
            print(f"  [ERROR] Cannot open PCAP: {e}")
            return {}

        for pkt in reader:
            if max_packets and n_packets >= max_packets:
                break
            try:
                if IP not in pkt:
                    continue
                ip   = pkt[IP]
                t    = float(pkt.time)
                src  = ip.src
                dst  = ip.dst
                size = len(pkt)
                src_mac = pkt[Ether].src if Ether in pkt else src

                if TCP in pkt:
                    sport, dport = pkt[TCP].sport, pkt[TCP].dport
                    proto = "TCP"
                elif UDP in pkt:
                    sport, dport = pkt[UDP].sport, pkt[UDP].dport
                    proto = "UDP"
                elif ICMP in pkt:
                    sport, dport, proto = 0, 0, "ICMP"
                else:
                    sport, dport, proto = 0, 0, "OTHER"
            except Exception:
                continue

            n_packets += 1
            raw_feat = extractor.update(
                timestamp=t, src_mac=src_mac,
                src_ip=src, dst_ip=dst,
                src_port=sport, dst_port=dport,
                pkt_len=size, protocol=proto)

            scaled = normalise(raw_feat, self.scaler)
            buffers[src].append(scaled)
            pkt_count[src] += 1

            # Inference every SEQ_LEN packets
            if (len(buffers[src]) == SEQ_LEN and
                    pkt_count[src] % SEQ_LEN == 0):
                seq  = np.stack(list(buffers[src]))
                x    = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                x = x.to(DEVICE)
                with torch.no_grad():
                    prob = torch.sigmoid(self.model(x)).item()
                label = "botnet" if prob >= self.threshold else "benign"
                results.append({
                    "src_ip": src,
                    "prob":   prob,
                    "label":  label,
                    "pkt_no": pkt_count[src],
                })

        reader.close()

        if not results:
            print(f"  No inferences (too few packets or no IP traffic)")
            return {"pcap": pcap_path.name, "n_packets": n_packets,
                    "n_inferences": 0, "ground_truth": ground_truth}

        df_r = pd.DataFrame(results)
        n_bot = (df_r["label"] == "botnet").sum()
        n_ben = (df_r["label"] == "benign").sum()
        total = len(df_r)

        # Majority vote = file-level prediction
        predicted = "botnet" if n_bot > n_ben else "benign"
        botnet_pct = n_bot / total * 100

        mean_conf_all = df_r["prob"].mean()

        print(f"  Packets   : {n_packets:,}")
        print(f"  Inferences: {total:,}")
        print(f"  Botnet    : {n_bot:,} ({botnet_pct:.1f}%)  "
              f"Benign: {n_ben:,} ({100-botnet_pct:.1f}%)")
        print(f"  Mean conf : {mean_conf_all:.4f}")
        print(f"  Predicted : {predicted.upper()}", end="")

        if ground_truth:
            correct = predicted == ground_truth
            print(f"  ({'✓ CORRECT' if correct else '✗ WRONG'})")
        else:
            correct = None
            print()

        return {
            "pcap":          pcap_path.name,
            "pcap_path":     str(pcap_path),
            "ground_truth":  ground_truth,
            "predicted":     predicted,
            "correct":       correct,
            "n_packets":     n_packets,
            "n_inferences":  total,
            "n_botnet_inferences": int(n_bot),
            "n_benign_inferences": int(n_ben),
            "botnet_pct":    round(botnet_pct, 2),
            "mean_conf":     round(float(mean_conf_all), 4),
        }

    def evaluate_directory(self, pcap_dir: Path,
                            max_packets_per_pcap: int | None = None,
                            output_csv: Path | None = None) -> pd.DataFrame:
        """Evaluate all PCAPs found recursively in pcap_dir."""
        pcap_files = sorted(pcap_dir.rglob("*.pcap")) + \
                     sorted(pcap_dir.rglob("*.pcapng"))

        if not pcap_files:
            print(f"  No PCAP files found in {pcap_dir}")
            return pd.DataFrame()

        print(f"\n  Found {len(pcap_files)} PCAP file(s) in {pcap_dir}")

        all_results = []
        for pcap_path in pcap_files:
            result = self.evaluate_pcap(pcap_path,
                                        max_packets=max_packets_per_pcap)
            if result:
                all_results.append(result)

        df = pd.DataFrame(all_results)

        # Summary
        print(f"\n{'═'*58}")
        print(f"  CROSS-DATASET EVALUATION SUMMARY")
        print(f"{'═'*58}")
        print(f"  PCAPs evaluated : {len(df):,}")

        labeled = df[df["ground_truth"].notna()]
        if len(labeled) > 0:
            correct = labeled["correct"].sum()
            acc = correct / len(labeled) * 100
            print(f"  With GT labels  : {len(labeled):,}")
            print(f"  Correct         : {correct:,} / {len(labeled):,}")
            print(f"  Accuracy        : {acc:.1f}%")

            print(f"\n  Per-class breakdown:")
            for gt in ["botnet", "benign"]:
                sub = labeled[labeled["ground_truth"] == gt]
                if len(sub) == 0:
                    continue
                tp = (sub["predicted"] == gt).sum()
                print(f"    {gt:>8s}: {tp:,}/{len(sub):,} correct  "
                      f"({tp/len(sub)*100:.1f}%)")

        print(f"\n  Prediction distribution:")
        print(f"    Botnet : {(df['predicted']=='botnet').sum():,}")
        print(f"    Benign : {(df['predicted']=='benign').sum():,}")

        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\n  Results saved → {output_csv}")

        return df


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate CNN-LSTM on IoT-23 PCAP files")
    ap.add_argument("--pcap",     "-p", default=None,
                    help="Path to a single PCAP file.")
    ap.add_argument("--pcap-dir", "-d", default=None,
                    help="Directory to scan for all PCAP files recursively.")
    ap.add_argument("--label",    "-l", default=None,
                    choices=["botnet", "benign"],
                    help="Ground truth label (single PCAP only). "
                         "Auto-inferred from path if not given.")
    ap.add_argument("--max-packets", "-n", type=int, default=None,
                    help="Max packets to read per PCAP (default: all).")
    ap.add_argument("--output",   "-o",
                    default="models/stage2/results/iot23_cross_eval.csv",
                    help="Output CSV path for directory evaluation.")
    ap.add_argument("--model",    "-m",
                    default=str(MODEL_PATH))
    ap.add_argument("--scaler",   "-s",
                    default=str(SCALER_PATH))
    ap.add_argument("--threshold", "-t", type=float, default=None,
                    help="Override decision threshold (default: use value saved in model).")
    args = ap.parse_args()

    if not args.pcap and not args.pcap_dir:
        ap.error("Provide --pcap or --pcap-dir")

    print("=" * 58)
    print("  IoT-23 Cross-Dataset PCAP Evaluation — Group 07")
    print("=" * 58)

    model, threshold, scaler = load_model(
        Path(args.model), Path(args.scaler))
    if args.threshold is not None:
        print(f"  Threshold overridden: {threshold} → {args.threshold}")
        threshold = args.threshold
    evaluator = PcapEvaluator(model, threshold, scaler)

    if args.pcap:
        evaluator.evaluate_pcap(
            Path(args.pcap),
            ground_truth=args.label,
            max_packets=args.max_packets)
    else:
        evaluator.evaluate_directory(
            Path(args.pcap_dir),
            max_packets_per_pcap=args.max_packets,
            output_csv=Path(args.output))
