"""
════════════════════════════════════════════════════════════════════════
 N-BaIoT Preprocessing  →  stage2_iot_botnet.csv + iot_scaler.json
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 INPUT  : data/raw/detection+of+iot+botnet+attacks+n+baiot/
            <device_folder>/
              benign_traffic.csv
              gafgyt_attacks.rar
              mirai_attacks.rar

 OUTPUT : data/processed/stage2_iot_botnet.csv   (normalized features)
          models/stage2/iot_scaler.json           (raw min/max for live inference)

 CRITICAL — WHY WE SAVE THE SCALER:
   The processed CSV stores features normalized to [0,1].
   The live Kitsune extractor produces RAW feature values
   (e.g. weight=15.9, mean=102.1, variance=1017.9).
   To feed live features into the trained CNN-LSTM correctly,
   we must apply the SAME normalization that was applied during
   training. iot_scaler.json stores the min/max of each feature
   BEFORE normalization so the live pipeline can reproduce it exactly.

 REQUIREMENTS (macOS):
   brew install unar
   pip install rarfile scikit-learn
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw" / "detection+of+iot+botnet+attacks+n+baiot"
OUTPUT_DIR  = ROOT / "data" / "processed"
MODEL_DIR   = ROOT / "models" / "stage2"
OUTPUT_CSV  = OUTPUT_DIR / "stage2_iot_botnet.csv"
SCALER_JSON = MODEL_DIR  / "iot_scaler.json"   # ← saved BEFORE normalization

BENIGN_FILE = "benign_traffic.csv"
ATTACK_RARS = {
    "gafgyt_attacks.rar": "gafgyt",
    "mirai_attacks.rar":  "mirai",
}


# ════════════════════════════════════════════════════════════════════════
# RAR EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def _check_unar() -> str:
    unar = shutil.which("unar")
    if unar:
        return unar
    raise RuntimeError(
        "\n  'unar' not found.\n"
        "  macOS: brew install unar\n"
        "  Linux: sudo apt-get install unar")


def extract_rar(rar_path: Path, unar_bin: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="nbaiot_"))
    result = subprocess.run(
        [unar_bin, "-o", str(tmp), "-D", str(rar_path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"unar failed on {rar_path.name}:\n{result.stderr}")
    return tmp


def load_csvs_from_rar(rar_path: Path, prefix: str,
                       unar_bin: str) -> list[tuple[str, pd.DataFrame]]:
    results = []
    tmp = None
    try:
        tmp = extract_rar(rar_path, unar_bin)
        for csv_path in sorted(tmp.rglob("*.csv")):
            attack_type = f"{prefix}_{csv_path.stem}"
            df = pd.read_csv(csv_path, low_memory=False)
            results.append((attack_type, df))
            print(f"      {csv_path.name:<25} {len(df):>8,} rows")
    except Exception as e:
        print(f"      [WARN] {rar_path.name}: {e}")
    finally:
        if tmp and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return results


# ════════════════════════════════════════════════════════════════════════
# PER-DEVICE LOADER
# ════════════════════════════════════════════════════════════════════════

def load_device(device_dir: Path, unar_bin: str) -> pd.DataFrame | None:
    device_name = device_dir.name
    frames: list[pd.DataFrame] = []

    benign_path = device_dir / BENIGN_FILE
    if not benign_path.exists():
        print(f"    [SKIP] benign_traffic.csv not found")
        return None
    df_benign = pd.read_csv(benign_path, low_memory=False)
    df_benign["class_label"] = "benign"
    df_benign["attack_type"] = "benign"
    frames.append(df_benign)
    print(f"    benign_traffic.csv            {len(df_benign):>8,} rows")

    for rar_name, prefix in ATTACK_RARS.items():
        rar_path = device_dir / rar_name
        if not rar_path.exists():
            continue
        print(f"    {rar_name}:")
        for attack_type, df_atk in load_csvs_from_rar(rar_path, prefix, unar_bin):
            df_atk["class_label"] = "botnet"
            df_atk["attack_type"] = attack_type
            frames.append(df_atk)

    combined = pd.concat(frames, ignore_index=True)
    combined["device_name"] = device_name
    combined["src_ip"]      = device_name
    combined["seq_index"]   = np.arange(len(combined), dtype=np.int64)

    b   = (combined["class_label"] == "benign").sum()
    bot = (combined["class_label"] == "botnet").sum()
    print(f"    → {len(combined):,}  benign={b:,}  botnet={bot:,}")
    return combined


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  N-BaIoT PREPROCESSING  —  Group 07")
    print("  Outputs: stage2_iot_botnet.csv + iot_scaler.json")
    print("=" * 64)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DIR}")

    unar_bin = _check_unar()
    print(f"\n  unar: {unar_bin}")

    device_dirs = sorted([d for d in RAW_DIR.iterdir()
                          if d.is_dir() and not d.name.startswith(".")])
    print(f"\n  Found {len(device_dirs)} devices:")
    for d in device_dirs:
        print(f"    {d.name}")

    # ── Load all devices ──────────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []
    for device_dir in device_dirs:
        print(f"\n  ── {device_dir.name} ──")
        df_dev = load_device(device_dir, unar_bin)
        if df_dev is not None:
            all_frames.append(df_dev)

    if not all_frames:
        raise RuntimeError("No data loaded.")

    print("\n  Combining...")
    df = pd.concat(all_frames, ignore_index=True)
    print(f"  Total: {len(df):,} rows")

    META = {"class_label", "attack_type", "device_name", "src_ip", "seq_index"}
    feat = [c for c in df.columns if c not in META]
    print(f"  Features: {len(feat)}")

    vc = df["class_label"].value_counts()
    print(f"\n  Class distribution:")
    for cls, cnt in vc.items():
        print(f"    {cls:>8s}: {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")

    # ── Validate ─────────────────────────────────────────────────────
    print("\n  Validating...")
    n_inf = np.isinf(df[feat].values).sum()
    n_nan = df[feat].isna().sum().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"  Replacing {n_inf:,} inf + {n_nan:,} NaN with medians")
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
        df[feat] = df[feat].fillna(df[feat].median())

    # ── Deduplicate ───────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=feat, keep="first")
    print(f"  Removed {n_before - len(df):,} exact duplicate rows")

    # ── Save raw scaler BEFORE normalization ──────────────────────────
    # This is the critical step for live inference correctness.
    # iot_scaler.json stores the min/max of RAW Kitsune feature values.
    # The live pipeline applies: scaled = (raw - min) / (max - min)
    # which exactly reproduces the training normalization.
    print("\n  Computing and saving raw scaler (before normalization)...")
    raw_min = df[feat].min().tolist()
    raw_max = df[feat].max().tolist()
    scaler_data = {
        "features": feat,
        "min":      raw_min,
        "max":      raw_max,
        "note":     ("Min/max of RAW N-BaIoT features before MinMaxScaler. "
                     "Apply to live Kitsune extractor output to match training scale.")
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_JSON, "w") as f:
        json.dump(scaler_data, f, indent=2)
    print(f"  Scaler saved → {SCALER_JSON}")

    # Sanity check: show a few raw ranges
    print("  Sample raw feature ranges:")
    for i, f_name in enumerate(feat[:6]):
        print(f"    {f_name:<35} "
              f"min={raw_min[i]:>12.4f}  max={raw_max[i]:>12.4f}")

    # ── Normalize to [0, 1] ───────────────────────────────────────────
    print("\n  Normalizing features to [0, 1]...")
    scaler = MinMaxScaler()
    df[feat] = scaler.fit_transform(df[feat])

    # ── Final stats ───────────────────────────────────────────────────
    vc2 = df["class_label"].value_counts()
    print(f"\n  Final distribution ({len(df):,} rows):")
    for cls, cnt in vc2.items():
        print(f"    {cls:>8s}: {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")

    # ── Save CSV ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = feat + [c for c in
                       ["class_label", "attack_type", "device_name",
                        "src_ip", "seq_index"] if c in df.columns]
    print(f"\n  Saving CSV → {OUTPUT_CSV}")
    df[out_cols].to_csv(OUTPUT_CSV, index=False)

    size_mb = OUTPUT_CSV.stat().st_size / 1e6
    print(f"\n{'='*64}")
    print(f"  DONE")
    print(f"  CSV      : {OUTPUT_CSV}  ({size_mb:.1f} MB)")
    print(f"  Scaler   : {SCALER_JSON}  ← used by live_detector.py")
    print(f"  Rows     : {len(df):,}")
    print(f"  Features : {len(feat)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()