# AI-Based Botnet Detection Using Hybrid Deep Learning Models

> **Group 07 · CPCS498 / CPCS499 · King Abdulaziz University**
> Hybrid two-stage detection of botnet activity in IoT and non-IoT network traffic,
> with explainable AI and a desktop SOC dashboard.

A graduation project that detects botnet command-and-control, scanning, and DDoS
behaviour in network traffic by combining a lightweight Stage-1 device-type
classifier (Random Forest) with two specialised Stage-2 CNN-LSTM detectors —
one tuned for IoT device traffic, one for non-IoT host traffic. Every botnet
alert is accompanied by a post-hoc explanation (SHAP for Stage-1, Integrated
Gradients for Stage-2) so that an analyst can see *why* a flow was flagged, not
just *that* it was.

---

## Table of Contents

1. [Why this project](#why-this-project)
2. [System architecture](#system-architecture)
3. [Evaluation results](#evaluation-results)
4. [Project structure](#project-structure)
5. [Quick start](#quick-start)
6. [Usage](#usage)
   - [Launch the desktop GUI](#1-launch-the-desktop-gui)
   - [Analyse a PCAP or CSV file](#2-analyse-a-pcap-or-csv-file)
   - [Run live packet capture](#3-run-live-packet-capture-cli)
   - [Train the models from scratch](#4-train-the-models-from-scratch)
7. [Datasets](#datasets)
8. [Explainable AI (XAI)](#explainable-ai-xai)
9. [Test harness](#test-harness)
10. [Troubleshooting](#troubleshooting)
11. [Team](#team)
12. [License & academic use](#license--academic-use)

---

## Why this project

Botnets remain one of the most common origins of DDoS, credential theft, and
ransomware delivery; IoT-targeting families (Mirai, Bashlite, Gafgyt, and their
descendants) make the problem worse because the victims are unmonitored devices
on consumer networks. Existing IDS rules detect known signatures well but tend
to miss novel C2 patterns and encrypted-tunnel beaconing.

This project asks whether a **deep learning pipeline that respects the
structural difference between IoT and non-IoT traffic** can detect such activity
on commodity hardware, in real time, and with explanations a security analyst
can act on. The design choices — two stages, separate detectors per traffic
type, recall-first thresholding, post-hoc XAI — all flow from that question.

The project prioritises **Recall** (minimising false negatives) over raw
accuracy. A missed botnet costs more than a false alarm an analyst can dismiss
in seconds, and recall is explicitly tracked against a target of ≥ 0.85 during
threshold selection in both Stage-2 trainers.

---

## System architecture

```
                ┌──────────────────────────────────────────────┐
                │  Input layer  (PCAP / CSV / NetFlow / live)  │
                └─────────────────────┬────────────────────────┘
                                      │
                ┌─────────────────────▼────────────────────────┐
                │  Feature extraction  (flow stats + time-     │
                │  window rates + TLS metadata + Kitsune)      │
                └─────────────────────┬────────────────────────┘
                                      │
                ┌─────────────────────▼────────────────────────┐
                │  STAGE 1  ·  Random Forest                   │
                │  IoT vs Non-IoT routing decision             │
                │  56 features, StandardScaler                 │
                └──────┬──────────────────────────┬────────────┘
                       │ IoT                      │ Non-IoT
            ┌──────────▼────────┐       ┌─────────▼─────────┐
            │ STAGE 2 · IoT     │       │ STAGE 2 · Non-IoT │
            │ CNN-LSTM          │       │ CNN-LSTM          │
            │ (115 Kitsune,     │       │ (46 flow feats,   │
            │  seq_len = 20)    │       │  seq_len = 20)    │
            └──────────┬────────┘       └─────────┬─────────┘
                       │                          │
                ┌──────▼──────────────────────────▼────────────┐
                │  XAI  ·  SHAP (Stage-1)  +  IG (Stage-2)     │
                │  Top-K features, severity, recommendations   │
                └─────────────────────┬────────────────────────┘
                                      │
                ┌─────────────────────▼────────────────────────┐
                │  PyQt6 desktop dashboard  +  CSV / PDF       │
                └──────────────────────────────────────────────┘
```

| Stage | Model | Input | Purpose |
|---|---|---|---|
| **1** | Random Forest (XGBoost baseline) | 56 flow features | Route every flow to the correct Stage-2 detector |
| **2-IoT** | CNN-LSTM | 20 × 115 Kitsune packet sequence | Benign vs Botnet on IoT-device traffic |
| **2-Non-IoT** | CNN-LSTM | 20 × 46 flow sequence | Benign vs Botnet on host / server traffic |
| **XAI** | SHAP TreeExplainer + Integrated Gradients | per-flow attributions | Analyst-readable explanation per detection |

Why two Stage-2 detectors and not one? IoT and non-IoT traffic have
structurally different feature distributions (IoT is dominated by short,
periodic, fixed-port telemetry; non-IoT is bursty, variable-length, multi-port).
A single model averaged across both populations under-fits each. The Stage-1
router lets each Stage-2 model specialise.

---

## Evaluation results

All numbers below are from the held-out test split of the training corpus and
are reproducible by re-running the corresponding training script.

### Stage-1 — IoT vs Non-IoT routing

| Metric | Random Forest (production) | XGBoost (baseline) |
|---|---|---|
| Accuracy | 96.70% | 99.56% |
| Precision (weighted) | 96.77% | 99.57% |
| Recall (weighted) | 96.70% | 99.56% |
| F1 (weighted) | 96.61% | 99.57% |
| AUC-ROC | 0.9989 | 0.9990 |
| IoT recall (per-class) | 85.48% | 99.58% |
| Non-IoT recall (per-class) | 99.74% | 99.56% |

Random Forest was chosen as the production Stage-1 model despite XGBoost's
higher accuracy because XGBoost's pickle format segfaults on macOS Apple
Silicon. The trade-off is documented in detail in `Stage1_Chapter.docx` §5.

### Stage-2 IoT detector (CNN-LSTM)

| Metric | Value |
|---|---|
| Accuracy | 99.91% |
| Precision (botnet) | 99.95% |
| **Recall (botnet)** | **99.92%** |
| Recall (benign) | 99.88% |
| F1-score | 99.93% |
| AUC-ROC | ≈ 1.0000 |
| Threshold | 0.52 |

Training corpus: N-BaIoT (Mirai + Bashlite flooding) + IoT-23 (scanning + C&C),
3.98 M sequences combined, ~87% botnet.

### Stage-2 Non-IoT detector (CNN-LSTM)

| Metric | Value |
|---|---|
| Accuracy | 99.54% |
| Precision (botnet) | 99.62% |
| **Recall (botnet)** | **99.19%** |
| F1-score | 99.41% |
| AUC-ROC | 0.9998 |
| Threshold | 0.766 |

Training corpus: CTU-13 + CIC-IDS-2017 Friday + CIC-IDS-2018 Friday 02-03,
1.85 M flows after merging.

### Reading the confusion matrix

For every detector, the confusion matrix has the same orientation:

|  | Predicted Benign | Predicted Botnet |
|---|---|---|
| **Actual Benign** | TN | FP (false alarm) |
| **Actual Botnet** | FN (**missed botnet**) | TP |

The project's priority metric is **Recall** = TP / (TP + FN) — i.e. how often
we catch a real botnet flow. A false positive (FP) interrupts an analyst
briefly; a false negative (FN) lets an infected device keep operating.

---

## Project structure

```
botnet_project/
├── monitoring.py                 # Two-stage live pipeline + BotnetMonitor class
├── requirements.txt              # Top-level pip dependencies
├── README.md                     # This file
│
├── app/                          # PyQt6 desktop SOC dashboard
│   ├── mockApp.py                #   ▶ GUI entry point
│   ├── monitor_bridge.py         #   QThread wrapper around BotnetMonitor
│   ├── live_capture.py           #   Scapy-based live capture thread
│   ├── upload_page.py            #   PCAP/CSV upload & inference UI
│   ├── inference_bridge.py       #   File → Stage-1 → Stage-2 dispatcher
│   ├── report_generator.py       #   PDF report writer (reportlab)
│   ├── file_handler.py           #   Format detection + safe loading
│   ├── detection_store.py        #   Persistent per-session results
│   ├── app_settings.py           #   Threshold / window-size settings
│   ├── theme.py                  #   Single source of truth for UI colours
│   └── setup_live_capture.py     #   Helper: lists interfaces, tests sniff
│
├── models/
│   ├── stage1/
│   │   ├── classifier.py         # Train RF + XGB, save rf_model.pkl
│   │   ├── rf_model.pkl          # Production Stage-1 model
│   │   ├── s1_scaler.json        # StandardScaler params (mean + scale)
│   │   └── results/              # confusion_matrix.png, comparison_report.json
│   └── stage2/
│       ├── iot_detector.py            # Train Stage-2 IoT CNN-LSTM
│       ├── noniot_detector_cnnlstm.py # Train Stage-2 Non-IoT CNN-LSTM
│       ├── iot_cnn_lstm.pt            # Production IoT detector
│       ├── noniot_cnn_lstm.pt         # Production Non-IoT detector
│       ├── iot_scaler.json
│       ├── noniot_scaler.json
│       └── results/                   # iot_metrics.json, noniot_metrics.json
│
├── data_processing/              # Dataset preprocessors (RAW → CSV)
│   ├── process_ctu13.py          # CTU-13 binetflow → CSV
│   ├── process_cicids2017.py     # CIC-IDS-2017 daily CSVs → unified
│   ├── pcap_to_csv.py            # Generic PCAP → CICFlowMeter-style CSV (tshark)
│   ├── merge_stage2_noniot.py    # CTU-13 + CIC-IDS → stage2_noniot_botnet.csv
│   ├── preprocess_from_pcap_csvs.py  # Stage-1 multi-source unifier (v6)
│   └── README.md                 # Per-step usage and download URLs
│
├── src/
│   ├── ingestion/                # Per-dataset PCAP → Kitsune features
│   │   ├── preprocess_nbaiot.py  # N-BaIoT processor (with MinMax)
│   │   ├── pcap_to_csv.py        # Shared PCAP loader
│   │   └── pcap_to_nbaiot_features.py
│   ├── live/                     # Live Kitsune extractor + LiveDetector
│   │   ├── kitsune_extractor.py
│   │   └── live_detector.py
│   └── xai/                      # Explainable-AI module
│       ├── local_explainer.py    # Stage1/Stage2 per-flow explainers
│       ├── explanation_engine.py # Rule-based pattern recogniser
│       ├── global_importance.py  # Offline aggregate-importance plots
│       └── feature_metadata.py   # Friendly names + units + directions
│
├── evaluation/                   # Cross-dataset + ablation studies
│   ├── cnn_test/                 # CNN-only ablation (no LSTM)
│   ├── lstm_test/                # LSTM-only ablation (no CNN)
│   ├── ablation_comparison/      # Aggregator → bar chart + markdown
│   ├── cicids2018/               # External CIC-IDS-2018 OOD evaluation
│   └── ETF/                      # External ETF IoT dataset evaluation
│
├── test_harness/                 # Automated QA framework
│   ├── orchestrator.py           # Entry point
│   ├── configs/registry.py       # Test catalogue (phases A–D)
│   ├── generators/               # Synthetic PCAP / CSV makers
│   ├── pcap_tests/, parsers/, ml_tests/, xai_tests/, gui_tests/,
│   └── README.md
│
└── data/                         # (gitignored)
    ├── raw/                      # Downloaded datasets (see "Datasets" below)
    └── processed/                # Outputs of data_processing/*.py
```

---

## Quick start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10 – 3.13** | The codebase uses PEP-604 unions and `list[str]` generics. |
| OS | Windows 10/11 **or** macOS 12+ (Intel / Apple Silicon) | Both targets validated. |
| Wireshark / tshark | Latest | Required for `data_processing/pcap_to_csv.py`. Not a pip package. |
| Npcap (Windows only) | Latest, WinPcap-compat mode | Required for live capture under Scapy. |

### Install

```bash
# Clone (or copy) the project
cd botnet_project

# Create a virtual environment (recommended)
python -m venv venv          # macOS/Linux: python3 -m venv venv
# Activate:
#   Windows : venv\Scripts\activate
#   macOS   : source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt           # Windows
pip3 install -r requirements.txt          # macOS
```

A quick smoke check that the install succeeded:

```bash
python  -c "import numpy, pandas, sklearn, torch, matplotlib, scapy, PyQt6, reportlab; print('OK')"   # Windows
python3 -c "import numpy, pandas, sklearn, torch, matplotlib, scapy, PyQt6, reportlab; print('OK')"   # macOS
```

> **Note on trained models.** The repository ships with the trained artifacts under `models/stage1/` and `models/stage2/`. If they are missing, run the training pipeline (see [Train the models from scratch](#4-train-the-models-from-scratch)).

---

## Usage

### 1. Launch the desktop GUI

**Windows**

```powershell
python app\mockApp.py
```

**macOS**

```bash
python3 app/mockApp.py
```

What you get: a six-page SOC-style dashboard with Dashboard, Upload, Monitor,
Results, Reports, and Settings pages. The Upload page accepts PCAP / PCAPNG /
CSV. The Monitor page starts and stops live capture. The Results page shows
per-flow XAI panels. The Reports page exports CSV and PDF reports.

### 2. Analyse a PCAP or CSV file

Open the GUI → **Upload** page → drag a file (or click *Browse*) →
**Run Detection**. The file is routed through Stage-1 and the appropriate
Stage-2 detector, and every flow becomes a row in the **Results** table with
an XAI explanation attached.

Supported formats end-to-end:

| Extension | Stage-1 | Stage-2 IoT | Stage-2 Non-IoT | Notes |
|---|---|---|---|---|
| `.pcap`, `.pcapng` | ✓ | ✓ | ✓ | Full pipeline via `BotnetMonitor` |
| `.csv` (CICFlowMeter-format) | ✓ | — | ✓ | IoT rows return `unknown` (needs raw packets) |
| `.binetflow`, `.nfcapd` | — | — | — | Pre-process with `data_processing/process_*.py` first |

### 3. Run live packet capture (CLI)

For a no-GUI capture (e.g. for headless evaluation):

**Windows** (run PowerShell **as Administrator**):

```powershell
python monitoring.py --interface "Wi-Fi" --duration 60
```

**macOS / Linux**:

```bash
sudo python3 monitoring.py --interface en0 --duration 60
```

Useful flags (`python monitoring.py --help` for the full list):

| Flag | Default | Effect |
|---|---|---|
| `--interface NAME` | auto | Network interface to sniff |
| `--duration SECS` | infinite | Stop after this many seconds |
| `--iot-threshold τ` | 0.70 | Stage-1 probability cutoff for IoT routing |
| `--no-oui` | off | Disable MAC-vendor (OUI) override on Stage-1 |
| `--simulate N` | off | Bypass capture and inject N synthetic packets |

The CLI writes per-flow predictions to `detection_results.csv` in the project
root. To list available interfaces first:

```bash
python  app/setup_live_capture.py     # Windows
python3 app/setup_live_capture.py     # macOS
```

### 4. Train the models from scratch

The full pipeline runs in three stages. Each step writes its outputs under
`data/processed/`, `models/stage1/`, or `models/stage2/`.

**Step A — Preprocess datasets**

```bash
# 1. Place downloaded datasets under data/raw/ (see Datasets section)
# 2. CTU-13 → CSV
python  data_processing/process_ctu13.py        # Windows
python3 data_processing/process_ctu13.py        # macOS

# 3. CIC-IDS-2017 daily CSVs → unified
python  data_processing/process_cicids2017.py

# 4. Merge into the Stage-2 Non-IoT training set
python  data_processing/merge_stage2_noniot.py

# 5. N-BaIoT + IoT-23 → Stage-2 IoT training set
python  src/ingestion/preprocess_nbaiot.py
python  src/ingestion/pcap_to_nbaiot_features.py

# 6. Build the unified Stage-1 dataset (IoT vs Non-IoT)
python  data_processing/preprocess_from_pcap_csvs.py
```

**Step B — Train Stage-1**

```bash
python  models/stage1/classifier.py             # Windows
python3 models/stage1/classifier.py             # macOS
```

Outputs: `models/stage1/rf_model.pkl`, `s1_scaler.json`, plus
confusion-matrix and feature-importance plots under `models/stage1/results/`.

**Step C — Train Stage-2**

```bash
# Non-IoT CNN-LSTM
python  models/stage2/noniot_detector_cnnlstm.py
# IoT CNN-LSTM (requires N-BaIoT + IoT-23 preprocessed)
python  models/stage2/iot_detector.py
```

Outputs: `noniot_cnn_lstm.pt`, `iot_cnn_lstm.pt`, the matching `*_scaler.json`,
and per-detector `metrics.json` / ROC / training-curve PNGs under
`models/stage2/results/`.

**Step D — (optional) Run ablation studies**

The ablation scripts under `evaluation/cnn_test/` and `evaluation/lstm_test/`
train CNN-only and LSTM-only variants with identical splits / scalers /
hyperparameters, then `evaluation/ablation_comparison/aggregate_metrics.py`
produces side-by-side bar charts for the report.

---

## Datasets

All datasets are publicly available. None are committed to the repository.
Place each download under `data/raw/<name>/` exactly as the per-dataset
preprocessor expects (see `data_processing/README.md` for the layout).

| Dataset | Used by | License | Download |
|---|---|---|---|
| **N-BaIoT** | Stage-2 IoT (training) | CC BY 4.0 | <https://archive.ics.uci.edu/dataset/442/detection_of_iot_botnet_attacks_n_baiot> |
| **IoT-23** | Stage-2 IoT (training) | CC0 | <https://www.stratosphereips.org/datasets-iot23> |
| **CTU-13** | Stage-2 Non-IoT (training) | Research-only | <https://www.stratosphereips.org/datasets-ctu13> |
| **CIC-IDS-2017** | Stage-2 Non-IoT (training) | Free academic | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| **CIC-IDS-2018** | Stage-2 Non-IoT (training & OOD) | Free academic | <https://www.unb.ca/cic/datasets/ids-2018.html> |
| **ETF IoT Botnet** | Cross-dataset evaluation only | Mendeley Data | <https://data.mendeley.com/datasets/nbs66kvx6n> |
| **IEEE Mirai Botnet** | Stage-1 evaluation only | IEEE DataPort | <https://ieee-dataport.org/open-access/iot-network-intrusion-dataset> |

> **Storage note.** Combined raw downloads exceed **50 GB**. After
> preprocessing, the processed CSVs typically take 4–8 GB.

---

## Explainable AI (XAI)

Every Stage-2 detection produces a structured explanation:

- **Top-K features** (default K = 8) ranked by absolute attribution.
- **Direction**: did the feature push the prediction toward botnet (↑) or toward benign (↓)?
- **Severity** (`low` / `medium` / `high` / `critical`) chosen by the rule engine in `src/xai/explanation_engine.py`.
- **Plain-text summary** plus 1–3 short reasons (e.g. *“High packet rate and repeated SYN packets — periodic beaconing consistent with C&C.”*).
- **Recommendation** (e.g. *“Block source IP and isolate device.”*) — surfaced in the GUI chip and the PDF report.

| Layer | Stage-1 | Stage-2 IoT | Stage-2 Non-IoT |
|---|---|---|---|
| Method | SHAP TreeExplainer (exact) | Integrated Gradients | Integrated Gradients |
| Latency / flow | < 1 ms | ≈ 95 ms | ≈ 85 ms |
| Dependency | `shap` (optional) | `torch.autograd` only | `torch.autograd` only |

Method choice is justified in detail in `XAI_Chapter.docx`; the short version is
that SHAP TreeExplainer is exact on tree ensembles, and IG is the only post-hoc
attribution method that satisfies Implementation Invariance on the multiplicative
gates of an LSTM (Sundararajan et al., ICML 2017; Ancona et al., ICLR 2018).

To compute aggregate (offline) feature importance for the report:

```bash
python  src/xai/global_importance.py            # Windows
python3 src/xai/global_importance.py            # macOS
```

This emits SHAP / IG / permutation-importance plots under `xai_results/`.

---

## Test harness

The repository ships with an automated QA framework under `test_harness/`.
It generates synthetic and malformed traffic, replays it through the project's
own modules, captures crashes, and produces machine-readable reports. It is
**separate** from the main project dependencies.

```bash
# Install harness extras (small superset of main requirements)
pip install -r test_harness/requirements.txt

# Run everything
python -m test_harness.orchestrator

# Or filter by phase / severity / test ID
python -m test_harness.orchestrator --phase A
python -m test_harness.orchestrator --severity CRITICAL
python -m test_harness.orchestrator --test-ids P-04 M-EF
```

Phases:

| Phase | Coverage |
|---|---|
| **A** | Core stability — parser safety, packet handling, schema integrity, IPv6 / VLAN robustness |
| **B** | ML behaviour — threshold sweep, calibration, drift detection |
| **C** | Adversarial / security — FGSM perturbations, malformed PCAPs |
| **D** | GUI / long-run / stress — DetectionStore concurrency, alert flood, packet flood |

See `test_harness/README.md` for the per-test contract and CI recipe.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: monitoring` | Running from the wrong directory | All scripts assume the project root as CWD: `cd botnet_project` first. |
| `FileNotFoundError: models/stage1/rf_model.pkl` | Models not yet trained / not present | Either train the models (see [section 4](#4-train-the-models-from-scratch)) or copy the trained artifacts in. |
| `PermissionError` on live capture | Raw sockets need elevation | **Windows**: install **Npcap** with *WinPcap-compatible mode*, run terminal as Administrator. **macOS / Linux**: `sudo python3 …`. |
| `xgboost.core.XGBoostError: segmentation fault` on macOS | Documented Apple-Silicon XGBoost regression | Production uses Random Forest. Keep XGBoost installed for Windows training only; the live pipeline ignores `xgb_model.pkl` if it fails to load. |
| `tshark: command not found` | tshark not installed (not a pip package) | **Windows**: install Wireshark MSI (`choco install wireshark`). **macOS**: `brew install wireshark`. |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` (headless Linux CI) | No display attached | `export QT_QPA_PLATFORM=offscreen` before running. |
| Capture starts but no flows appear | Wrong interface selected | Run `python app/setup_live_capture.py` — it lists interfaces, tests permissions, and prints the exact constructor line to change. |
| `RuntimeError: torch.load() segfault` on macOS at GUI startup | PyTorch / Qt threading conflict | Already handled: `monitor_bridge.ensure_monitor()` constructs `BotnetMonitor` on the main thread before the QThread starts. If you bypass that path, replicate the same pattern. |
| `AssertionError: SCALER SANITY CHECK FAILED` during training | Input CSV is already normalised (max ≤ 1.0) | Regenerate the processed CSV from the raw dataset using the matching `data_processing/process_*.py` script. The trainer fits StandardScaler on raw values. |

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Iyas Majeed | 2236567 | — |
| Omar Alsiary | 2236983 | — |
| Zeyad Alghamdi | 2237000 | — |

**Supervisor:** Dr. Wajdi Aljedaibi

Course: **CPCS498 — Graduation Project I** (System Design, completed),
**CPCS499 — Graduation Project II** (Implementation, in progress).

Faculty of Computing and Information Technology, King Abdulaziz University.

---

## License & academic use

This project is academic coursework. The codebase is shared with the supervisor
and examiners for evaluation purposes. External datasets are governed by their
own licenses (see the [Datasets](#datasets) section) — please respect them.

Third-party libraries retain their original licenses (see
[`requirements.txt`](requirements.txt) for the full list).

When citing this work, please reference the final report:

> Majeed, I., Alsiary, O., Alghamdi, Z. (2026). *Hybrid AI-Based Botnet
> Detection Using Two-Stage CNN-LSTM Models with Explainable AI.* Graduation
> Project, Faculty of Computing and Information Technology, King Abdulaziz
> University.
