# Data Processing Pipeline — Stage-2 Non-IoT Botnet Detector
## Group 07 | CPCS499 | AI-Based Botnet Detection

---

## ⚠️ CRITICAL: DO NOT NORMALISE BEFORE SAVING CSV

The root cause of the scaler bug (scale_max ~ 1) was normalising data
before fitting the StandardScaler. Every script in this folder outputs
**RAW, UN-NORMALISED** features. The scaler is fitted exclusively inside
`noniot_detector_cnnlstm.py`.

---

## Folder Structure

```
botnet_project/
├── data/
│   ├── raw/
│   │   ├── ctu13/          ← put CTU-13 CSV files here
│   │   └── cicids2017/     ← put CIC-IDS-2017 per-day CSVs here
│   └── processed/
│       ├── ctu13_processed.csv
│       ├── cicids2017_processed.csv
│       └── stage2_noniot_botnet.csv   ← final training input
├── models/
│   └── stage2/
│       ├── noniot_scaler.json
│       └── noniot_cnn_lstm.pt
└── data_processing/
    ├── process_ctu13.py
    ├── process_cicids2017.py
    ├── merge_stage2_noniot.py
    └── verify_pipeline.py
```

---

## Step 0: Install Dependencies

**Windows:**
```
pip install pandas numpy scikit-learn tqdm torch
```

**macOS:**
```
pip3 install pandas numpy scikit-learn tqdm torch
```

---

## Step 1: Download Datasets

### CTU-13
- Official: https://www.stratosphereips.org/datasets-ctu13
- Kaggle:   search "CTU-13 botnet dataset"
- Format:   CSV with columns: StartTime, Dur, Proto, SrcAddr, Sport, Dir,
            DstAddr, Dport, State, TotPkts, TotBytes, SrcBytes, Label
- Place files in: `data/raw/ctu13/`

### CIC-IDS-2017
- Official: https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle:   search "CIC-IDS-2017 intrusion detection"
- Format:   Per-day CSVs (Monday.csv … Friday.csv) — CICFlowMeter output
- Relevant: Friday-WorkingHours-Afternoon-Bot.pcap_ISCX.csv contains Bot rows
- Place files in: `data/raw/cicids2017/`

---

## Step 2: Process CTU-13

**Windows:**
```
python data_processing/process_ctu13.py --input data/raw/ctu13/ --output data/processed/ctu13_processed.csv
```

**macOS:**
```
python3 data_processing/process_ctu13.py --input data/raw/ctu13/ --output data/processed/ctu13_processed.csv
```

**Expected output:**
```
[INFO] Found 13 CTU-13 file(s) to process.
[INFO]   Reading: capture20110818.binetflow
[INFO]   → 400,000 rows | botnet=35,000 (8.7%)
...
[INFO] === RAW SCALE CHECK (top-5 max values) ===
total_fwd_bytes     245,890,234.00    ← MUST be >> 1.0
flow_duration             86,400.00
...
[INFO] Saved 1,234,567 rows → data/processed/ctu13_processed.csv
```

**If you see `All features <= 1.0`:** You have a pre-normalised version.
Download the original binetflow CSVs from Stratosphere.

---

## Step 3: Process CIC-IDS-2017

**Windows:**
```
python data_processing/process_cicids2017.py --input data/raw/cicids2017/ --output data/processed/cicids2017_processed.csv
```

**macOS:**
```
python3 data_processing/process_cicids2017.py --input data/raw/cicids2017/ --output data/processed/cicids2017_processed.csv
```

**Expected output:**
```
[INFO] Label dist: benign=200,000  botnet=1,956
...
[INFO] === RAW SCALE CHECK (top-5 max values) ===
total_fwd_bytes     999,999,999.00
flow_iat_mean             500,000.00
...
[INFO] Saved 201,956 rows → data/processed/cicids2017_processed.csv
```

---

## Step 4: Merge into stage2_noniot_botnet.csv

**Windows:**
```
python data_processing/merge_stage2_noniot.py
```

**macOS:**
```
python3 data_processing/merge_stage2_noniot.py
```

With custom paths:
```
python3 data_processing/merge_stage2_noniot.py \
    --ctu13   data/processed/ctu13_processed.csv \
    --cicids  data/processed/cicids2017_processed.csv \
    --output  data/processed/stage2_noniot_botnet.csv \
    --balance 5.0
```

**Expected output:**
```
[INFO] Merged: 1,436,523 rows total | benign=1,200,000 | botnet=236,523
[INFO] === RAW SCALE CHECK (MUST show values >> 1.0) ===
total_fwd_bytes     999,999,999.00   ← PASS
flow_duration            86,400.00   ← PASS
...
[INFO] Raw scale OK. max_feature_value = 999999999.00
[INFO] SAVED: data/processed/stage2_noniot_botnet.csv
```

---

## Step 5: Verify Pipeline

**Windows:**
```
python data_processing/verify_pipeline.py
```

**macOS:**
```
python3 data_processing/verify_pipeline.py
```

All 3 checks must PASS before running training:
```
✅ PASS Both classes present.
✅ PASS Raw scale looks correct (max = 999999999.00).
✅ PASS No NaN values.
```

---

## Step 6: Train the Non-IoT CNN-LSTM

**Windows:**
```
python models/stage2/noniot_detector_cnnlstm.py
```

**macOS:**
```
python3 models/stage2/noniot_detector_cnnlstm.py
```

---

## Step 7: Post-training Verification

```
python3 data_processing/verify_pipeline.py \
    --scaler models/stage2/noniot_scaler.json \
    --model  models/stage2/noniot_cnn_lstm.pt
```

**Expected:**
```
scale_max : 245890234.00   ← PASS (was ~1.96 before fix)
threshold : 0.42           ← PASS if 0.30–0.70
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `No CSV files found` | Wrong --input path | Check directory name |
| `All features <= 1.0` | Pre-normalised CSV | Use original dataset |
| `No label column found` | CSV format differs | Check column names |
| `Zero botnet rows` | Label filter too strict | Check Label values in raw CSV |
| `scale_max <= 100 after training` | CSV was normalised | Re-run from Step 0 |

---

## Dataset Notes for Report

| Dataset | Role | Botnet Types |
|---------|------|-------------|
| CTU-13 | C2/spam/scan botnets | Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay |
| CIC-IDS-2017 (Bot day) | Real botnet traffic | ARES botnet |

Both contribute to **Stage-2 Non-IoT** generalization across:
- C&C communication patterns (CTU-13)
- Modern HTTP-based botnet behavior (CIC-IDS-2017)
