"""
=============================================================================
CTU-13 Binetflow Processor  —  Group 07 | CPCS499 Botnet Detection
=============================================================================
Actual directory structure (confirmed from screenshot):
    data/raw/ctu13/
        1/
            capture20110810.binetflow   ← this is what we read
            botnet-capture-20110810-neris.pcap
        2/
            capture20110811.binetflow
        3/ ... 13/

Binetflow columns (Argus bidirectional):
    StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
    State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

Label mapping:
    flow=Background-*   → EXCLUDED (unlabeled — NOT benign)
    flow=From-Botnet-*  → 1 (botnet)
    flow=Normal-*       → 0 (benign)
    flow=LEGITIMATE     → 0 (benign)

CRITICAL: Do NOT normalise here.
          The StandardScaler in noniot_detector_cnnlstm.py must be
          fitted on RAW values (TotBytes in millions, Dur in seconds).

Output: data/processed/ctu13_processed.csv

Usage:
    Windows : python  data_processing/process_ctu13.py
    macOS   : python3 data_processing/process_ctu13.py

Dependencies:
    Windows : pip  install pandas numpy tqdm
    macOS   : pip3 install pandas numpy tqdm
=============================================================================
"""

import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROTO_MAP = {
    "tcp": 6, "udp": 17, "icmp": 1, "arp": 0,
    "igmp": 2, "esp": 50, "gre": 47, "ipv6-icmp": 58,
}

STATE_MAP = {
    "FIN": 0, "INT": 1, "CON": 2, "REQ": 3, "RST": 4,
    "ECO": 5, "URH": 6, "URN": 7, "URP": 8, "TXD": 9,
    "NNS": 10, "NNO": 11, "ACC": 12,
}


def label_binetflow(raw: str) -> int:
    """Returns 1=botnet, 0=benign, -1=background(exclude)."""
    s = str(raw).strip().lower()
    if "background" in s:
        return -1
    if "botnet" in s or "malicious" in s:
        return 1
    if "normal" in s or "legitimate" in s:
        return 0
    return -1  # unknown → treat as background (safe)


def port_to_int(val) -> float:
    """CTU-13 ports can be hex strings like '0x1A2B'."""
    try:
        s = str(val).strip().lower()
        return float(int(s, 16) if s.startswith("0x") else s)
    except (ValueError, TypeError):
        return 0.0


def process_binetflow(path: str) -> "pd.DataFrame | None":
    log.info(f"  Reading: {path}")
    raw = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            raw = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.warning(f"  Cannot read {path}: {e}")
            return None
    if raw is None:
        log.warning(f"  All encodings failed: {path}")
        return None

    raw.columns = raw.columns.str.strip()

    label_col = next((c for c in raw.columns
                      if c.strip().lower() == "label"), None)
    if label_col is None:
        log.warning(f"  No Label column — skipping {path}")
        return None

    raw["_lbl"] = raw[label_col].apply(label_binetflow)
    before = len(raw)
    raw = raw[raw["_lbl"] >= 0].copy()
    n_bg = before - len(raw)

    if raw.empty:
        log.warning(f"  No labeled rows after background removal in {path}")
        return None

    log.info(f"  background_excluded={n_bg:,}  "
             f"benign={(raw._lbl==0).sum():,}  "
             f"botnet={(raw._lbl==1).sum():,}")

    def _num(col, default=0.0):
        if col in raw.columns:
            return pd.to_numeric(raw[col], errors="coerce").fillna(default)
        return pd.Series([default] * len(raw), index=raw.index)

    dur       = _num("Dur")
    tot_pkts  = _num("TotPkts")
    tot_bytes = _num("TotBytes")
    src_bytes = _num("SrcBytes")
    bwd_bytes = (tot_bytes - src_bytes).clip(lower=0)
    fwd_pkts  = (tot_pkts / 2.0).round()
    bwd_pkts  = (tot_pkts - fwd_pkts).clip(lower=0)
    safe_dur  = dur.replace(0, np.nan)
    safe_fp   = fwd_pkts.replace(0, np.nan)
    safe_bp   = bwd_pkts.replace(0, np.nan)

    out = pd.DataFrame()
    out["src_ip"]      = raw.get("SrcAddr", pd.Series(["0.0.0.0"]*len(raw))).values
    out["timestamp"]   = raw.get("StartTime", pd.Series([None]*len(raw))).values
    out["device_type"] = "noniot"
    out["class_label"] = raw["_lbl"].values

    # ── Flow features (RAW — no normalisation) ────────────────────────────
    out["flow_duration"]      = dur.values
    out["total_fwd_packets"]  = fwd_pkts.values
    out["total_bwd_packets"]  = bwd_pkts.values
    out["total_fwd_bytes"]    = src_bytes.values        # SrcBytes = forward
    out["total_bwd_bytes"]    = bwd_bytes.values
    out["protocol_num"]       = raw.get("Proto", pd.Series(["tcp"]*len(raw))).apply(
                                    lambda p: PROTO_MAP.get(str(p).lower().strip(), 0)).values
    out["tcp_state"]          = raw.get("State", pd.Series(["UNK"]*len(raw))).apply(
                                    lambda s: STATE_MAP.get(str(s).strip().upper(), -1)).values
    out["src_port"]           = raw.get("Sport", pd.Series([0]*len(raw))).apply(port_to_int).values
    out["dst_port"]           = raw.get("Dport", pd.Series([0]*len(raw))).apply(port_to_int).values

    # ── Derived rate features (still raw, computed from raw bytes/pkts) ───
    out["bytes_per_second"]   = (tot_bytes / safe_dur).fillna(0).values
    out["packets_per_second"] = (tot_pkts  / safe_dur).fillna(0).values
    out["fwd_pkt_len_mean"]   = (src_bytes / safe_fp).fillna(0).values
    out["bwd_pkt_len_mean"]   = (bwd_bytes / safe_bp).fillna(0).values

    # ── Columns binetflow doesn't have — fill 0 ───────────────────────────
    for col in ["fwd_pkt_len_std", "bwd_pkt_len_std",
                "flow_iat_mean",   "flow_iat_std",
                "fwd_iat_mean",    "bwd_iat_mean",
                "fwd_syn_flag_cnt","fwd_ack_flag_cnt",
                "fwd_rst_flag_cnt","fwd_psh_flag_cnt",
                "tls_features_available"]:
        out[col] = 0.0

    out.replace([np.inf, -np.inf], 0, inplace=True)
    out.fillna(0, inplace=True)
    return out


def process_ctu13(input_dir: str, output_path: str) -> None:
    files = sorted(set(
        glob.glob(os.path.join(input_dir, "**", "*.binetflow"), recursive=True) +
        glob.glob(os.path.join(input_dir, "*.binetflow"))
    ))

    if not files:
        log.error(
            f"\nNo .binetflow files found in '{input_dir}'.\n"
            "Expected: data/raw/ctu13/1/capture20110810.binetflow\n"
            "          data/raw/ctu13/2/capture20110811.binetflow  ... etc.\n"
            "Download: https://www.stratosphereips.org/datasets-ctu13"
        )
        return

    log.info(f"Found {len(files)} .binetflow file(s) in CTU-13.")
    dfs = []
    for f in tqdm(files, desc="CTU-13 scenarios"):
        df = process_binetflow(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        log.error("All files failed. Check .binetflow format.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # ── RAW scale verification — THE CRITICAL CHECK ───────────────────────
    exclude = {"class_label","protocol_num","tcp_state",
               "src_port","dst_port","tls_features_available"}
    feat_cols = [c for c in combined.select_dtypes(include=[np.number]).columns
                 if c not in exclude]
    max_vals = combined[feat_cols].max()

    log.info("\n=== RAW SCALE CHECK (top-5 max values — MUST be >> 1.0) ===")
    log.info("\n" + max_vals.sort_values(ascending=False).head(5).to_string())

    if max_vals.max() <= 1.0:
        log.error(
            "STOP: max feature value <= 1.0 → data is ALREADY NORMALISED.\n"
            "Use original Stratosphere .binetflow files (not pre-processed versions)."
        )
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.to_csv(output_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"SAVED : {output_path}")
    log.info(f"Rows  : {len(combined):,}")
    log.info(f"Benign: {(combined.class_label==0).sum():,}")
    log.info(f"Botnet: {(combined.class_label==1).sum():,}")
    log.info(f"Max raw value: {max_vals.max():.2f}  (must be >> 1.0)")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/ctu13/",
                        help="Root CTU-13 dir containing 1/ 2/ ... 13/ subdirs")
    parser.add_argument("--output", default="data/processed/ctu13_processed.csv")
    args = parser.parse_args()
    process_ctu13(args.input, args.output)