"""
=============================================================================
CIC-IDS-2017 Processor  —  Group 07 | CPCS499 Botnet Detection
=============================================================================
Actual file (confirmed from screenshots + error output):
    data/raw/cicids2017/friday_flows.csv

This file was produced by your team's step2_preprocess.py / load_friday_csv().
It already uses the unified schema column names AND has class_label as a
string column ("benign" / "botnet") — NOT CICFlowMeter's "Label" column.

Columns present (from error output showing first 10 cols):
    flow_duration, total_fwd_packets, total_bwd_packets,
    total_fwd_bytes, total_bwd_bytes, fwd_pkt_len_min,
    fwd_pkt_len_max, fwd_pkt_len_mean, fwd_pkt_len_std,
    bwd_pkt_len_min, ...  [53 columns total]

Two supported formats — the script auto-detects:
  FORMAT A: class_label column exists (string: "benign"/"botnet")
            → your team's pre-processed CSV (friday_flows.csv)
  FORMAT B: Label column exists (string: "BENIGN"/"Bot"/...)
            → raw CICFlowMeter output

CRITICAL: Do NOT normalise here.
          The StandardScaler in noniot_detector_cnnlstm.py is fitted on
          RAW values. friday_flows.csv must have values >> 1.0.

Output: data/processed/cicids2017_processed.csv

Usage:
    Windows : python  data_processing/process_cicids2017.py
    macOS   : python3 data_processing/process_cicids2017.py

    # With explicit path to your file:
    python3 data_processing/process_cicids2017.py \
        --input data/raw/cicids2017/friday_flows.csv

Dependencies:
    Windows : pip  install pandas numpy
    macOS   : pip3 install pandas numpy
=============================================================================
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def detect_label_format(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect the label column and format.

    Returns:
        (label_col, format_name)
        format_name: "unified"  — class_label with "benign"/"botnet" strings
                     "cicflow"  — Label with "BENIGN"/"Bot" strings
                     "unknown"  — cannot detect

    Priority: class_label > Label > label (case-insensitive search)
    """
    col_lower = {c.strip().lower(): c for c in df.columns}

    # Format A: your team's unified pre-processed CSV
    if "class_label" in col_lower:
        return col_lower["class_label"], "unified"

    # Format B: raw CICFlowMeter output
    if "label" in col_lower:
        return col_lower["label"], "cicflow"

    return None, "unknown"


def encode_unified_label(raw: str) -> int:
    """
    Encode your team's class_label strings.
    Values: "benign" → 0, "botnet" → 1, anything else → -1 (exclude)
    """
    s = str(raw).strip().lower()
    if s == "benign":
        return 0
    if s in ("botnet", "bot", "malicious"):
        return 1
    return -1


def encode_cicflow_label(raw: str) -> int:
    """
    Encode raw CICFlowMeter Label strings.
    Values: "BENIGN" → 0, "Bot" → 1, others (DDoS etc) → -1 (exclude)
    """
    s = str(raw).strip().lower()
    if s == "benign":
        return 0
    if s == "bot":
        return 1
    return -1  # DDoS, PortScan, FTP-Patator → not our task


def process_cicids2017(input_path: str, output_path: str) -> None:

    # ── Locate the file ───────────────────────────────────────────────────
    # If a directory was passed, look for friday_flows.csv inside it
    if os.path.isdir(input_path):
        candidates = [
            os.path.join(input_path, "friday_flows.csv"),
            os.path.join(input_path, "Friday-WorkingHours-Afternoon-Bot.pcap_ISCX.csv"),
            os.path.join(input_path, "Friday.csv"),
            os.path.join(input_path, "friday.csv"),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            log.error(
                f"\nDirectory given: {input_path}\n"
                "No recognised CIC-IDS-2017 CSV found inside.\n"
                "Expected: friday_flows.csv\n"
                "Run explicitly:\n"
                "  python3 data_processing/process_cicids2017.py \\\n"
                "    --input data/raw/cicids2017/friday_flows.csv"
            )
            return
        input_path = found
        log.info(f"Auto-detected file: {input_path}")

    if not os.path.exists(input_path):
        log.error(f"File not found: {input_path}")
        return

    # ── Load ──────────────────────────────────────────────────────────────
    log.info(f"Loading: {input_path}")
    raw = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            raw = pd.read_csv(input_path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.error(f"Cannot read file: {e}")
            return
    if raw is None:
        log.error("All encodings failed.")
        return

    # Strip leading/trailing whitespace from ALL column names
    raw.columns = raw.columns.str.strip()
    log.info(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns")
    log.info(f"First 10 columns: {list(raw.columns[:10])}")

    # ── Detect label format ───────────────────────────────────────────────
    label_col, fmt = detect_label_format(raw)

    if fmt == "unknown":
        log.error(
            "Cannot find a label column.\n"
            f"Available columns: {list(raw.columns)}\n"
            "Expected: 'class_label' (your unified format) or 'Label' (CICFlowMeter)"
        )
        return

    log.info(f"Detected format: {fmt}  |  label column: '{label_col}'")
    log.info(f"\nRaw label distribution:\n{raw[label_col].value_counts().to_string()}")

    # ── Encode labels ──────────────────────────────────────────────────────
    encoder = encode_unified_label if fmt == "unified" else encode_cicflow_label
    raw["_lbl"] = raw[label_col].apply(encoder)

    before = len(raw)
    raw = raw[raw["_lbl"] >= 0].copy()
    n_excluded = before - len(raw)

    log.info(f"\nAfter label filtering:")
    log.info(f"  Excluded (non-botnet/benign): {n_excluded:,}")
    log.info(f"  Benign  : {(raw._lbl == 0).sum():,}")
    log.info(f"  Botnet  : {(raw._lbl == 1).sum():,}")

    if (raw._lbl == 1).sum() == 0:
        log.error(
            "\nZero botnet rows!\n"
            "friday_flows.csv may have been produced from only the morning PCAP.\n"
            "The Bot traffic is in: Friday-WorkingHours-Afternoon-Bot.pcap_ISCX.csv\n"
            "Check your file using: python3 -c \"\n"
            "  import pandas as pd\n"
            "  df = pd.read_csv('data/raw/cicids2017/friday_flows.csv')\n"
            "  print(df.columns.tolist())\n"
            "  if 'class_label' in df.columns: print(df.class_label.value_counts())\n\""
        )
        return

    # ── Build output DataFrame ────────────────────────────────────────────
    out = pd.DataFrame()

    # Metadata
    out["src_ip"]      = raw.get("src_ip",
                         raw.get("Source IP",
                         pd.Series(["0.0.0.0"] * len(raw)))).values
    out["timestamp"]   = raw.get("timestamp",
                         raw.get("Timestamp",
                         pd.Series([None] * len(raw)))).values
    out["device_type"] = "noniot"
    out["class_label"] = raw["_lbl"].values

    # ── Copy ALL existing numeric columns (they are already unified names) ─
    # friday_flows.csv already has unified column names from your pipeline.
    # Just copy them all over.
    meta_cols = {label_col, "src_ip", "Source IP", "timestamp", "Timestamp",
                 "device_type", "class_label", "_lbl"}
    for col in raw.columns:
        if col in meta_cols or col == "_lbl":
            continue
        if pd.api.types.is_numeric_dtype(raw[col]):
            safe_name = col.strip().lower().replace(" ", "_").replace("/", "_per_")
            if safe_name not in out.columns:
                out[safe_name] = pd.to_numeric(raw[col], errors="coerce").fillna(0).values
        elif col.strip().lower() not in {"label", "class_label"}:
            # Skip string columns that aren't labels
            pass

    # ── Ensure mandatory unified columns exist ─────────────────────────────
    for col in ["tls_features_available", "tcp_state", "protocol_num",
                "src_port", "dst_port", "bytes_per_second", "packets_per_second"]:
        if col not in out.columns:
            out[col] = 0.0

    # ── Replace inf / nan ──────────────────────────────────────────────────
    out.replace([np.inf, -np.inf], 0, inplace=True)
    out.fillna(0, inplace=True)

    # ── CRITICAL: raw scale verification ──────────────────────────────────
    exclude = {"class_label", "protocol_num", "tcp_state",
               "src_port", "dst_port", "tls_features_available"}
    feat_cols = [c for c in out.select_dtypes(include=[np.number]).columns
                 if c not in exclude]
    max_vals = out[feat_cols].max()

    log.info("\n=== RAW SCALE CHECK (top-5 max values — MUST be >> 1.0) ===")
    top5 = max_vals.sort_values(ascending=False).head(5)
    log.info("\n" + top5.to_string())

    if max_vals.max() <= 1.0:
        log.error(
            "\nSTOP: All features <= 1.0 — data is ALREADY NORMALISED.\n"
            "friday_flows.csv was normalised before saving (MinMaxScaler).\n"
            "You need the raw CICFlowMeter output, not the normalised version.\n"
            "Check src/ingestion/step2_preprocess.py — it calls clean_and_normalize()\n"
            "before saving. You need features BEFORE that step."
        )
        return

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"SAVED : {output_path}")
    log.info(f"Rows  : {len(out):,}")
    log.info(f"Benign: {(out.class_label==0).sum():,}")
    log.info(f"Botnet: {(out.class_label==1).sum():,}")
    log.info(f"Max raw value: {max_vals.max():.2f}  (must be >> 1.0)")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process CIC-IDS-2017 friday_flows.csv.\n"
            "Auto-detects both unified schema format (your team's) and "
            "raw CICFlowMeter format."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/cicids2017/friday_flows.csv",
        help=(
            "Path to friday_flows.csv OR directory containing it. "
            "Default: data/raw/cicids2017/friday_flows.csv"
        )
    )
    parser.add_argument(
        "--output",
        default="data/processed/cicids2017_processed.csv"
    )
    args = parser.parse_args()
    process_cicids2017(args.input, args.output)