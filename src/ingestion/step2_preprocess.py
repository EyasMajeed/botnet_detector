"""
════════════════════════════════════════════════════════════════════════
 Step 2 — Combine & Normalise Non-IoT Datasets
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT:
   --ctu13-dir       data/raw/CTU-13/             (binetflow files)
   --friday-csv      data/raw/CIC-IDS-2017/friday_flows.csv  (from step1)

 OUTPUT:
   --output          data/processed/stage2_noniot_botnet.csv

 WHAT IT DOES:
   Loads each source, maps to the 49-feature unified schema (raw values),
   concatenates, then fits a single global MinMax scaler [0,1] over ALL
   49 numeric features.  src_ip and timestamp are preserved unscaled for
   per-device temporal sequence construction in the detector.
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse, gc, glob, json, sys, warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# ── Unified schema ─────────────────────────────────────────────────────
NUMERIC_FEATURES: List[str] = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK",
    "protocol", "src_port", "dst_port",
    "flow_active_time", "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "ttl_mean", "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
]
META_COLS     = ["class_label", "device_type", "src_ip", "timestamp"]
OUTPUT_COLS   = NUMERIC_FEATURES + META_COLS


def _clean_num(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0).astype(np.float32))


def _ensure_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = _clean_num(df[col])
    if "class_label" not in df.columns:
        raise ValueError(f"[{name}] missing class_label")
    df["class_label"] = df["class_label"].astype(str).str.lower().str.strip()
    df["device_type"] = df.get("device_type", pd.Series("noniot", index=df.index))
    df["src_ip"]      = df["src_ip"].astype(str) if "src_ip" in df.columns else "unknown"
    df["timestamp"]   = pd.to_numeric(df.get("timestamp", 0), errors="coerce").fillna(0.0)
    before = len(df)
    df = df[df["class_label"].isin(["benign", "botnet"])].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  [{name}] dropped {dropped:,} rows with unknown labels")
    return df[OUTPUT_COLS].copy()


# ══════════════════════════════════════════════════════════════════════
# SOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_friday_csv(path: Path) -> pd.DataFrame:
    """Load the pre-labeled friday_flows.csv produced by step1."""
    print(f"\n── Loading CIC-IDS-2017 Friday flows from {path.name} ──")
    if not path.exists():
        print(f"  [skip] file not found: {path}")
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} rows loaded")
    result = _ensure_schema(df, "CIC-IDS2017-Friday")
    vc = result["class_label"].value_counts()
    print(f"  botnet={vc.get('botnet',0):,}  benign={vc.get('benign',0):,}")

    # Verify we actually have temporal mixing per device
    mixed = 0
    for ip, grp in result.groupby("src_ip"):
        if grp["class_label"].nunique() > 1:
            mixed += 1
    print(f"  ✅ {mixed} devices have BOTH labels (temporal transitions for CNN-LSTM)")
    return result


def _ctu13_label(lbl: str) -> str:
    s = str(lbl).lower()
    if "botnet" in s: return "botnet"
    if "normal" in s: return "benign"
    return "background"


def _ctu13_flags(state: pd.Series) -> pd.DataFrame:
    s = state.fillna("").astype(str).str.upper()
    first = s.str.split("_").str[0].fillna("")
    return pd.DataFrame({
        "flag_FIN": first.str.contains("F").astype(float),
        "flag_SYN": first.str.contains("S").astype(float),
        "flag_RST": first.str.contains("R").astype(float),
        "flag_PSH": first.str.contains("P").astype(float),
        "flag_ACK": first.str.contains("A").astype(float),
    })


def _ctu13_port(series: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v): return 0.0
        s = str(v).strip()
        if not s: return 0.0
        try: return float(int(s, 16)) if s.lower().startswith("0x") else float(s)
        except: return 0.0
    return series.apply(one).astype(np.float32)


def _ctu13_proto(proto: pd.Series) -> pd.Series:
    p = proto.fillna("").astype(str).str.lower().str.strip()
    return p.map({"tcp":6,"udp":17,"icmp":1,"arp":0,"igmp":2}).fillna(0.0).astype(np.float32)


def _parse_ts(series: pd.Series) -> pd.Series:
    as_num = pd.to_numeric(series, errors="coerce")
    if as_num.notna().mean() > 0.9:
        return as_num.fillna(0.0).astype(np.float64)
    try:
        dt     = pd.to_datetime(series, errors="coerce", utc=False)
        dt_ns  = dt.astype("datetime64[ns]").to_numpy()
        as_int = dt_ns.view("int64")
        valid  = pd.notna(dt).to_numpy()
        secs   = np.where(valid, as_int / 1e9, 0.0)
        return pd.Series(secs, index=series.index, dtype=np.float64)
    except Exception:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float64)


def load_ctu13(ctu_dir: Path) -> pd.DataFrame:
    print(f"\n── Loading CTU-13 from {ctu_dir} ──")
    files  = sorted(glob.glob(str(ctu_dir / "**" / "*.binetflow"), recursive=True))
    files += sorted(glob.glob(str(ctu_dir / "**" / "*.csv"),       recursive=True))
    if not files:
        print("  [warn] no files found"); return pd.DataFrame(columns=OUTPUT_COLS)

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, low_memory=False, encoding="latin-1")

        cols = {c.lower(): c for c in df.columns}
        def col(*names):
            for n in names:
                if n.lower() in cols: return cols[n.lower()]
            return None

        lc = col("Label")
        if lc is None: continue

        out = pd.DataFrame(index=df.index)
        out["class_label"] = df[lc].apply(_ctu13_label)
        out = out[out["class_label"] != "background"]
        if out.empty: continue

        idx = out.index
        dur      = _clean_num(df.loc[idx, col("Dur","Duration")]) if col("Dur","Duration") else pd.Series(0.0, index=idx)
        tot_pkts = _clean_num(df.loc[idx, col("TotPkts")])  if col("TotPkts") else pd.Series(0.0, index=idx)
        tot_bytes= _clean_num(df.loc[idx, col("TotBytes")]) if col("TotBytes") else pd.Series(0.0, index=idx)
        src_bytes= _clean_num(df.loc[idx, col("SrcBytes")]) if col("SrcBytes") else pd.Series(0.0, index=idx)
        bwd_bytes= (tot_bytes - src_bytes).clip(lower=0)

        fwd_pkts = (tot_pkts / 2.0).round()
        bwd_pkts = (tot_pkts - fwd_pkts).clip(lower=0)
        safe_dur = dur.replace(0, np.nan)
        safe_fp  = fwd_pkts.replace(0, np.nan)
        safe_bp  = bwd_pkts.replace(0, np.nan)

        fwd_mean = (src_bytes / safe_fp).fillna(0.0)
        bwd_mean = (bwd_bytes / safe_bp).fillna(0.0)
        iat_mean = (dur / (tot_pkts - 1).replace(0, np.nan)).fillna(0.0)
        bps      = (tot_bytes / safe_dur).fillna(0.0)
        pps      = (tot_pkts  / safe_dur).fillna(0.0)

        out["device_type"]      = "noniot"
        out["src_ip"]           = df.loc[idx, col("SrcAddr","Src IP")].astype(str) if col("SrcAddr","Src IP") else "unknown"
        out["timestamp"]        = _parse_ts(df.loc[idx, col("StartTime","Start Time")]) if col("StartTime","Start Time") else 0.0
        out["flow_duration"]    = dur
        out["total_fwd_packets"]= fwd_pkts
        out["total_bwd_packets"]= bwd_pkts
        out["total_fwd_bytes"]  = src_bytes
        out["total_bwd_bytes"]  = bwd_bytes
        for c in ("fwd_pkt_len_min","fwd_pkt_len_max","fwd_pkt_len_mean"): out[c] = fwd_mean
        out["fwd_pkt_len_std"]  = 0.0
        for c in ("bwd_pkt_len_min","bwd_pkt_len_max","bwd_pkt_len_mean"): out[c] = bwd_mean
        out["bwd_pkt_len_std"]  = 0.0
        out["flow_bytes_per_sec"]= bps
        out["flow_pkts_per_sec"] = pps
        for c in ("flow_iat_mean","fwd_iat_mean","bwd_iat_mean"): out[c] = iat_mean
        for c in ("flow_iat_std","flow_iat_min","flow_iat_max",
                  "fwd_iat_std","fwd_iat_min","fwd_iat_max",
                  "bwd_iat_std","bwd_iat_min","bwd_iat_max"): out[c] = 0.0
        out["fwd_header_length"] = 40.0 * fwd_pkts
        out["bwd_header_length"] = 40.0 * bwd_pkts
        if col("State","Flags"):
            flg = _ctu13_flags(df.loc[idx, col("State","Flags")])
            for c in flg.columns: out[c] = flg[c]
        else:
            for c in ("flag_FIN","flag_SYN","flag_RST","flag_PSH","flag_ACK"): out[c] = 0.0
        out["protocol"] = _ctu13_proto(df.loc[idx, col("Proto","Protocol")]) if col("Proto","Protocol") else 0.0
        out["src_port"] = _ctu13_port(df.loc[idx, col("Sport","Src Port")]) if col("Sport","Src Port") else 0.0
        out["dst_port"] = _ctu13_port(df.loc[idx, col("Dport","Dst Port")]) if col("Dport","Dst Port") else 0.0
        out["flow_active_time"]     = dur
        out["flow_idle_time"]       = 0.0
        out["bytes_per_sec_window"] = bps
        out["pkts_per_sec_window"]  = pps
        out["periodicity_score"]    = (1.0 / iat_mean.replace(0, np.nan)).fillna(0.0)
        out["burst_rate"]           = pps
        for c in ("ttl_mean","dns_query_count","payload_bytes_mean",
                  "payload_bytes_std","payload_zero_ratio","payload_entropy"): out[c] = 0.0

        frames.append(out.reset_index(drop=True))
        vc = out["class_label"].value_counts()
        print(f"  {Path(fp).name}: {len(out):,} rows  "
              f"(botnet={vc.get('botnet',0):,}  benign={vc.get('benign',0):,})")

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLS)
    return _ensure_schema(pd.concat(frames, ignore_index=True), "CTU-13")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctu13-dir",   type=Path, default=None)
    ap.add_argument("--friday-csv",  type=Path, default=None,
                    help="friday_flows.csv produced by step1")
    ap.add_argument("--output",      type=Path, required=True)
    ap.add_argument("--scaler-out",  type=Path, default=None)
    ap.add_argument("--subsample-botnet", type=int, default=None)
    ap.add_argument("--subsample-benign", type=int, default=None)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    if not any([args.ctu13_dir, args.friday_csv]):
        ap.error("Provide at least one of --ctu13-dir or --friday-csv")

    np.random.seed(args.seed)
    print("═" * 60)
    print("  Step 2 — Combine & Normalise Non-IoT Datasets")
    print("═" * 60)

    sources = []
    if args.friday_csv:
        sources.append(load_friday_csv(args.friday_csv))
    if args.ctu13_dir:
        sources.append(load_ctu13(args.ctu13_dir))

    sources = [s for s in sources if not s.empty]
    if not sources:
        print("[error] no data loaded"); sys.exit(1)

    print("\n── Concatenating ──")
    combined = pd.concat(sources, ignore_index=True)
    del sources; gc.collect()
    print(f"  Combined: {len(combined):,} rows")
    vc = combined["class_label"].value_counts()
    print(f"  Balance: {vc.to_dict()}")

    # Count devices with temporal transitions (important for CNN-LSTM)
    mixed_devs = sum(
        1 for _, g in combined.groupby("src_ip")
        if g["class_label"].nunique() > 1
    )
    print(f"  Devices with both labels (temporal transitions): {mixed_devs}")

    # Optional subsampling
    if args.subsample_botnet and vc.get("botnet", 0) > args.subsample_botnet:
        bot = combined[combined["class_label"]=="botnet"].sample(n=args.subsample_botnet, random_state=args.seed)
        combined = pd.concat([bot, combined[combined["class_label"]=="benign"]], ignore_index=True)
    if args.subsample_benign and (combined["class_label"]=="benign").sum() > args.subsample_benign:
        ben = combined[combined["class_label"]=="benign"].sample(n=args.subsample_benign, random_state=args.seed)
        combined = pd.concat([combined[combined["class_label"]=="botnet"], ben], ignore_index=True)

    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print("\n── Fitting global MinMax scaler ──")
    X = combined[NUMERIC_FEATURES].values.astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    combined[NUMERIC_FEATURES] = X_scaled
    print(f"  Scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined[OUTPUT_COLS].to_csv(args.output, index=False)
    mb = args.output.stat().st_size / 1e6
    print(f"\n  ✓ Saved: {args.output}  ({mb:.1f} MB | {len(combined):,} rows)")

    if args.scaler_out:
        args.scaler_out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"features": NUMERIC_FEATURES,
                   "data_min": scaler.data_min_.tolist(),
                   "data_max": scaler.data_max_.tolist()},
                  open(args.scaler_out, "w"), indent=2)
        print(f"  ✓ Scaler: {args.scaler_out}")
    print("═" * 60)


if __name__ == "__main__":
    main()