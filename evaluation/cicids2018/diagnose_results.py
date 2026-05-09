"""
═══════════════════════════════════════════════════════════════════════════
 diagnose_results.py
 Group 07 | CPCS499  —  Hybrid AI-Based Botnet Detection
═══════════════════════════════════════════════════════════════════════════

 Purpose
 ───────
 Follow-up analysis for evaluate_external_csv.py results.
 Answers ONE specific question: WHY does recall plateau on this CSV?

 What it produces
 ────────────────
   1. Probability-distribution plot   (TP / FN / TN / FP histograms)
        → shows whether FN flows are "borderline" or "confidently benign"
   2. FN vs TP breakdown by Dst Port  (top 15)
        → shows which destination ports the model misses entirely
   3. FN vs TP breakdown by Protocol
        → identifies if a whole protocol family is missed
   4. Feature-statistic comparison    (FN flows vs TP flows)
        → shows in which features the missed botnets differ from caught ones
   5. Markdown summary report

 Why this matters
 ────────────────
 If FN flows are concentrated near probability 0, threshold tuning won't
 help — the model has a representation gap, not a calibration gap. The
 per-port and per-feature breakdowns then tell you which subset of botnet
 behavior the model failed to learn.

 Save location
 ─────────────
   <project_root>/evaluation/diagnose_results.py

 Usage
 ─────
 Windows:
   python evaluation\\diagnose_results.py ^
       --predictions evaluation\\results\\predictions_Friday-02-03-2018.csv ^
       --csv data\\raw\\cicids2018\\Friday-02-03-2018.csv ^
       --max_rows 200000

 macOS:
   python3 evaluation/diagnose_results.py \\
       --predictions evaluation/results/predictions_Friday-02-03-2018.csv \\
       --csv data/raw/cicids2018/Friday-02-03-2018.csv \\
       --max_rows 200000

 Note: --max_rows MUST match what you used when generating predictions.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Protocol number → name, for human-readable output
PROTO_NAMES = {0: "HOPOPT", 1: "ICMP", 6: "TCP", 17: "UDP", 47: "GRE",
               50: "ESP", 51: "AH", 58: "ICMPv6", 132: "SCTP"}


def _print_header(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PROBABILITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def plot_probability_distribution(preds: pd.DataFrame, out_path: Path) -> dict:
    """
    Histogram of stage2_prob, separated by (true × predicted).

    INTERPRETATION GUIDE
    ────────────────────
      • If FN distribution is concentrated near prob=0:
          → model is "confidently wrong" on those flows
          → threshold tuning will NOT recover them
          → root cause is representation / feature gap
      • If FN distribution spans broadly with mass near threshold:
          → model is "uncertain" on those flows
          → lowering threshold may help (but check FP cost)
    """
    is_pos     = preds["true_label"] == "Botnet"
    is_pred_pos = preds["predicted_label"] == "Botnet"

    tp_probs = preds.loc[is_pos & is_pred_pos,  "stage2_prob"].values
    fn_probs = preds.loc[is_pos & ~is_pred_pos, "stage2_prob"].values
    tn_probs = preds.loc[~is_pos & ~is_pred_pos, "stage2_prob"].values
    fp_probs = preds.loc[~is_pos & is_pred_pos,  "stage2_prob"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    bins = np.linspace(0, 1, 41)

    axes[0, 0].hist(tp_probs, bins=bins, color="#2ca02c")
    axes[0, 0].set_title(f"TP — true Bot, predicted Bot   (n={len(tp_probs):,})")
    axes[0, 1].hist(fn_probs, bins=bins, color="#d62728")
    axes[0, 1].set_title(f"FN — true Bot, predicted Benign   (n={len(fn_probs):,})")
    axes[1, 0].hist(tn_probs, bins=bins, color="#1f77b4")
    axes[1, 0].set_title(f"TN — true Benign, predicted Benign   (n={len(tn_probs):,})")
    axes[1, 1].hist(fp_probs, bins=bins, color="#ff7f0e")
    axes[1, 1].set_title(f"FP — true Benign, predicted Bot   (n={len(fp_probs):,})")

    for ax in axes.flat:
        ax.set_xlabel("Stage-2 probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    fig.suptitle("Stage-2 probability distribution by outcome", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Summary stats — the key question is "where is the FN mass?"
    fn_quartiles = (np.percentile(fn_probs, [25, 50, 75, 90, 99]).tolist()
                    if len(fn_probs) else [0, 0, 0, 0, 0])
    fn_below_01  = float((fn_probs < 0.01).mean()) if len(fn_probs) else 0.0
    fn_below_10  = float((fn_probs < 0.10).mean()) if len(fn_probs) else 0.0

    print(f"\n  FN probability quartiles  [25%, 50%, 75%, 90%, 99%]:")
    print(f"    {fn_quartiles}")
    print(f"  Fraction of FN with prob < 0.01 : {fn_below_01:.1%}")
    print(f"  Fraction of FN with prob < 0.10 : {fn_below_10:.1%}")

    if fn_below_01 > 0.5:
        verdict = "CONFIDENTLY WRONG on FN — threshold tuning won't help."
    elif fn_below_10 > 0.7:
        verdict = "FN mass concentrated low — small threshold gain possible, limited."
    else:
        verdict = "FN mass spread — threshold tuning may have meaningful effect."
    print(f"  Diagnostic verdict: {verdict}")

    return {
        "fn_quartiles":       fn_quartiles,
        "fn_below_01":        fn_below_01,
        "fn_below_10":        fn_below_10,
        "verdict":            verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. PER-PORT BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def per_port_breakdown(joined: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """For each Dst Port: TP, FN, recall on that port, total botnet flows."""
    bot = joined[joined["true_label"] == "Botnet"].copy()
    if bot.empty:
        return pd.DataFrame()

    bot["caught"] = (bot["predicted_label"] == "Botnet").astype(int)
    g = bot.groupby("Dst Port", dropna=False).agg(
        botnet_flows=("caught", "size"),
        tp=("caught", "sum"),
    )
    g["fn"]     = g["botnet_flows"] - g["tp"]
    g["recall"] = g["tp"] / g["botnet_flows"]
    g = g.sort_values("botnet_flows", ascending=False).head(top_n)
    return g


# ═══════════════════════════════════════════════════════════════════════════
# 3. PER-PROTOCOL BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def per_protocol_breakdown(joined: pd.DataFrame) -> pd.DataFrame:
    bot = joined[joined["true_label"] == "Botnet"].copy()
    if bot.empty:
        return pd.DataFrame()

    bot["caught"] = (bot["predicted_label"] == "Botnet").astype(int)
    g = bot.groupby("Protocol", dropna=False).agg(
        botnet_flows=("caught", "size"),
        tp=("caught", "sum"),
    )
    g["fn"]     = g["botnet_flows"] - g["tp"]
    g["recall"] = g["tp"] / g["botnet_flows"]
    g["proto_name"] = g.index.map(lambda p: PROTO_NAMES.get(int(p), f"#{int(p)}")
                                   if pd.notna(p) else "?")
    return g.sort_values("botnet_flows", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE COMPARISON FN vs TP
# ═══════════════════════════════════════════════════════════════════════════

# Features most likely to explain why some flows are caught and others aren't.
# Picked to span size, rate, duration, and direction asymmetry.
COMPARE_FEATS = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "Init Fwd Win Byts",
]

def feature_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    bot = joined[joined["true_label"] == "Botnet"].copy()
    if bot.empty:
        return pd.DataFrame()

    tp = bot[bot["predicted_label"] == "Botnet"]
    fn = bot[bot["predicted_label"] == "Benign"]
    rows = []
    for col in COMPARE_FEATS:
        if col not in joined.columns:
            continue
        tp_vals = pd.to_numeric(tp[col], errors="coerce").dropna()
        fn_vals = pd.to_numeric(fn[col], errors="coerce").dropna()
        if tp_vals.empty or fn_vals.empty:
            continue
        rows.append({
            "feature":  col,
            "tp_median": float(tp_vals.median()),
            "fn_median": float(fn_vals.median()),
            "tp_p90":   float(tp_vals.quantile(0.90)),
            "fn_p90":   float(fn_vals.quantile(0.90)),
            "ratio_median": (
                float(tp_vals.median() / fn_vals.median())
                if fn_vals.median() != 0 else float("inf")
            ),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions_*.csv produced by evaluate_external_csv.py")
    ap.add_argument("--csv", required=True,
                    help="Path to original CICFlowMeter CSV (same one used for evaluation).")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Must match the value used during evaluation.")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    preds_path = Path(args.predictions)
    csv_path   = Path(args.csv)
    out_dir    = Path(args.out_dir) if args.out_dir else preds_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    # ─── Load predictions ───────────────────────────────────────────────
    _print_header(f"Loading predictions: {preds_path}")
    preds = pd.read_csv(preds_path, low_memory=False)
    print(f"  Rows: {len(preds):,}")

    # ─── Load original CSV (we need Dst Port, Protocol, raw features) ──
    _print_header(f"Loading original CSV: {csv_path}")
    df = pd.read_csv(csv_path, nrows=args.max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Rows: {len(df):,}")

    if len(preds) != len(df):
        raise ValueError(
            f"Row count mismatch: predictions={len(preds):,}, "
            f"raw csv={len(df):,}. Re-run with the same --max_rows."
        )

    # Stitch by row index
    joined = pd.concat([df.reset_index(drop=True),
                        preds.reset_index(drop=True)], axis=1)

    # ─── 1. Probability distribution ────────────────────────────────────
    _print_header("Diagnostic 1 — Probability distribution")
    prob_path = out_dir / f"prob_dist_{stem}.png"
    prob_summary = plot_probability_distribution(preds, prob_path)
    print(f"\n  Saved: {prob_path}")

    # ─── 2. Per-port breakdown ──────────────────────────────────────────
    _print_header("Diagnostic 2 — Per-Dst-Port recall (top 15 by flow count)")
    port_df = per_port_breakdown(joined)
    if not port_df.empty:
        print(port_df.to_string(float_format=lambda v: f"{v:.4f}"))
        port_df.to_csv(out_dir / f"per_port_{stem}.csv")
    else:
        print("  No botnet flows present.")

    # ─── 3. Per-protocol breakdown ──────────────────────────────────────
    _print_header("Diagnostic 3 — Per-protocol recall")
    proto_df = per_protocol_breakdown(joined)
    if not proto_df.empty:
        print(proto_df.to_string(float_format=lambda v: f"{v:.4f}"))
        proto_df.to_csv(out_dir / f"per_protocol_{stem}.csv")

    # ─── 4. Feature comparison ──────────────────────────────────────────
    _print_header("Diagnostic 4 — Feature statistics: FN flows vs TP flows")
    feat_df = feature_comparison(joined)
    if not feat_df.empty:
        print(feat_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        feat_df.to_csv(out_dir / f"feature_comparison_{stem}.csv", index=False)
        print("\n  Interpretation: large |tp_median - fn_median| ratios identify")
        print("  features whose distribution shifts between caught and missed botnet")
        print("  flows — these are candidates the model failed to learn from.")

    # ─── 5. Markdown report ─────────────────────────────────────────────
    md_path = out_dir / f"diagnostic_report_{stem}.md"
    with open(md_path, "w") as fp:
        fp.write(f"# Diagnostic Report — {csv_path.name}\n\n")
        fp.write("## 1. Probability Distribution Verdict\n\n")
        fp.write(f"- FN probability quartiles `[25, 50, 75, 90, 99]`: "
                 f"`{prob_summary['fn_quartiles']}`\n")
        fp.write(f"- Fraction of FN with prob < 0.01: "
                 f"**{prob_summary['fn_below_01']:.1%}**\n")
        fp.write(f"- Fraction of FN with prob < 0.10: "
                 f"**{prob_summary['fn_below_10']:.1%}**\n")
        fp.write(f"- Verdict: **{prob_summary['verdict']}**\n\n")
        fp.write(f"![Probability distribution](prob_dist_{stem}.png)\n\n")

        if not port_df.empty:
            fp.write("## 2. Per-Destination-Port Recall\n\n")
            fp.write(port_df.to_markdown(floatfmt=".4f"))
            fp.write("\n\n")
        if not proto_df.empty:
            fp.write("## 3. Per-Protocol Recall\n\n")
            fp.write(proto_df.to_markdown(floatfmt=".4f"))
            fp.write("\n\n")
        if not feat_df.empty:
            fp.write("## 4. Feature Comparison FN vs TP\n\n")
            fp.write(feat_df.to_markdown(index=False, floatfmt=".2f"))
            fp.write("\n")

    _print_header("Done")
    print(f"  Probability distribution : {prob_path}")
    print(f"  Per-port CSV             : {out_dir / f'per_port_{stem}.csv'}")
    print(f"  Per-protocol CSV         : {out_dir / f'per_protocol_{stem}.csv'}")
    print(f"  Feature comparison CSV   : {out_dir / f'feature_comparison_{stem}.csv'}")
    print(f"  Markdown report          : {md_path}")
    print()


if __name__ == "__main__":
    main()