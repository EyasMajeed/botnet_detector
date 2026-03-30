"""
run_all_preprocessing.py — Master Preprocessing Runner
═══════════════════════════════════════════════════════
Runs all three dataset preprocessors in sequence and produces
three training-ready CSV files:

  data/processed/
    stage1_iot_vs_noniot.csv     ← UNSW-NB15 + CICIDS2017 (IoT vs Non-IoT)
    stage2_iot_botnet.csv        ← IoT-23           (Benign vs Botnet - IoT)
    stage2_noniot_botnet.csv     ← CTU-13            (Benign vs Botnet - Non-IoT)

USAGE:
  python run_all_preprocessing.py

BEFORE RUNNING:
  1. Place your datasets in the correct directories:

     data/raw/
     ├── ctu13/                         # CTU-13 binetflow or CSV files
     │   ├── capture20110810.binetflow
     │   └── ...
     ├── iot23/                          # IoT-23 conn.log.labeled files
     │   ├── CTU-IoT-Malware-Capture-1-1/
     │   │   └── conn.log.labeled
     │   └── ...
     └── unsw_cicids2017/               # UNSW-NB15 + CICIDS2017
         ├── unsw/                       # UNSW-NB15 CSV files
         │   ├── UNSW-NB15_1.csv
         │   └── ...
         └── cicids/                     # CICIDS2017 CICFlowMeter CSVs
             ├── Monday-WorkingHours.pcap_Flow.csv
             └── ...

  2. If you have PCAP files instead of CSVs, first convert them:
     python -m src.ingestion.pcap_parser -i path/to/pcaps/ -o output.csv

  3. Install dependencies:
     pip install -r requirements.txt
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RAW_CTU13, RAW_IOT23, RAW_UNSW_CIC, PROCESSED_DIR


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     BOTNET DETECTOR — DATASET PREPROCESSING PIPELINE        ║
║                                                              ║
║  Group 07 | CPCS498/499 Graduation Project                  ║
║  AI-Based Botnet Detection Using Hybrid Deep Learning       ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_data_directories():
    """Verify data directories exist and contain files."""
    issues = []

    for name, path in [("CTU-13", RAW_CTU13),
                       ("IoT-23", RAW_IOT23),
                       ("UNSW+CICIDS", RAW_UNSW_CIC)]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            issues.append(f"  {name}: Directory created at {path} — add your data files")
        else:
            file_count = len(list(path.rglob("*")))
            if file_count == 0:
                issues.append(f"  {name}: Directory {path} is empty — add your data files")
            else:
                print(f"  ✓ {name}: {file_count} files found in {path}")

    if issues:
        print("\n  ⚠ Issues detected:")
        for issue in issues:
            print(issue)
        print()
        return False
    return True


def run_preprocessor(name: str, module_path: str) -> bool:
    """Import and run a preprocessor module."""
    start = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_path)
        mod.main()
        elapsed = time.time() - start
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit:
        print(f"\n  ✗ {name} exited (data may be missing)")
        return False
    except Exception as e:
        print(f"\n  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_banner()

    # Check directories
    print("Checking data directories ...")
    check_data_directories()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1) CTU-13 → Stage-2 Non-IoT
    print("\n" + "─" * 60)
    results["CTU-13"] = run_preprocessor(
        "CTU-13 (Stage-2 Non-IoT)",
        "src.ingestion.preprocess_ctu13"
    )

    # 2) IoT-23 → Stage-2 IoT
    print("\n" + "─" * 60)
    results["IoT-23"] = run_preprocessor(
        "IoT-23 (Stage-2 IoT)",
        "src.ingestion.preprocess_iot23"
    )

    # 3) UNSW + CICIDS → Stage-1
    print("\n" + "─" * 60)
    results["UNSW+CICIDS"] = run_preprocessor(
        "UNSW-NB15 + CICIDS2017 (Stage-1)",
        "src.ingestion.preprocess_unsw_cicids"
    )

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name:20s} : {status}")

    # List output files
    print("\nOutput files:")
    for f in sorted(PROCESSED_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:35s} ({size_mb:.1f} MB)")

    successful = sum(results.values())
    print(f"\n{successful}/{len(results)} preprocessors completed successfully.")

    if successful == len(results):
        print("\n  All datasets preprocessed! Ready for model training.")
    else:
        print("\n  Some datasets could not be processed.")
        print("  Please check the error messages above and ensure")
        print("  the raw data files are in the correct directories.")


if __name__ == "__main__":
    main()
