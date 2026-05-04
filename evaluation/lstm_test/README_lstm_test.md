# LSTM-only Ablation — Stage-2 Botnet Detector

## Group 07 | CPCS499 | AI-Based Botnet Detection

This folder contains the **LSTM-only** ablation: the same recurrent block
(2-layer LSTM with hidden size 128, dropout 0.3) as the production hybrid
CNN-LSTM, with the convolutional spine removed entirely. The LSTM
consumes the raw normalised feature sequence directly. Everything else
(dataset, splits, scaler, hyperparameters, training loop, threshold
selection, evaluation) is reused unchanged from the hybrid pipeline so
that any difference in test-set metrics is attributable to the CNN
removal alone.

---

## Why this exists

Pairs with `evaluation/cnn_test/` to provide the empirical justification
for the hybrid CNN-LSTM choice. Without an LSTM-only baseline, we cannot
distinguish "the CNN matters" from "anything beats LSTM alone".

---

## Files

| File | Purpose |
|---|---|
| `lstm_only_noniot.py` | LSTM-only training+evaluation on `data/processed/stage2_noniot_botnet.csv` |
| `lstm_only_iot.py`    | LSTM-only training+evaluation on `data/processed/stage2_iot_combined.csv` |
| `results/noniot/`     | Auto-created on first Non-IoT run |
| `results/iot/`        | Auto-created on first IoT run |

The scripts import the proven data-prep / training / threshold /
evaluation helpers from `models/stage2/noniot_detector_cnnlstm.py` and
`models/stage2/iot_detector.py`. Only the model class is local.

---

## Architecture

```
Input  (B, seq_len=20, n_features)
   └── LSTM(input=n_features, hidden=128, layers=2, dropout=0.3, batch_first)
                ^---- input dim is n_features (NOT 256), since
                      there is no CNN compression first
   └── h_n[-1]                                   →  (B, 128)
   └── Linear(128, 64) → ReLU → Dropout(0.4) → Linear(64, 1)  (logit)
```

**Parameter count** for the LSTM block scales with `n_features`:
- Non-IoT (~49 features): ~130K params (lighter than hybrid ~510K)
- IoT (~115 features): ~275K params (lighter than hybrid ~520K)

---

## Running

### Step 0 — Ensure the hybrid models have been trained first

```
python3 models/stage2/noniot_detector_cnnlstm.py
python3 models/stage2/iot_detector.py
```

### Step 1 — Install dependencies

**Windows**
```
pip install pandas numpy scikit-learn torch matplotlib
```

**macOS**
```
pip3 install pandas numpy scikit-learn torch matplotlib
```

### Step 2 — Run the LSTM-only ablations

**Windows**
```
python evaluation\lstm_test\lstm_only_noniot.py
python evaluation\lstm_test\lstm_only_iot.py
```

**macOS**
```
python3 evaluation/lstm_test/lstm_only_noniot.py
python3 evaluation/lstm_test/lstm_only_iot.py
```

### Step 3 — Aggregate and compare

```
python3 evaluation/ablation_comparison/aggregate_metrics.py
```

---

## Expected outputs (per branch)

```
evaluation/lstm_test/results/<branch>/
├── metrics.json          # accuracy, precision, recall, f1, auc_roc, threshold, n_parameters
├── training_curves.png
├── confusion_matrix.png
├── roc_curve.png
├── model.pt              # trained checkpoint (PyTorch)
└── scaler.json           # StandardScaler (Non-IoT only; IoT uses the production MinMaxScaler)
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: stage2_noniot_botnet.csv` | Pipeline not run | Run `data_processing/merge_stage2_noniot.py` first |
| `FileNotFoundError: stage2_iot_combined.csv` | Pipeline not run | Run `src/ingestion/preprocess_nbaiot.py` then `data_processing/combine_datasets.py` |
| Long training on CPU | No GPU | Lower `EPOCHS` in the source module if needed; cuDNN gives a big speedup on GPU |
| Loss goes to NaN immediately | Learning rate too high for raw-feature LSTM input | Already mitigated via `clip_grad_norm_(model.parameters(), 1.0)` in the IoT loop and `BCEWithLogitsLoss + pos_weight` in both branches; if NaN persists, halve LR |

---

## Reading the results

In a properly trained ablation, the **LSTM-only model is expected to
underperform on Precision** because it has no spatial-pattern extractor
to recognise local feature combinations (e.g., "high SYN count AND short
duration AND suspicious port"). The LSTM has to learn these from raw
features through long credit-assignment paths. Recall may be acceptable
because the LSTM still picks up temporal patterns.

If LSTM-only matches the hybrid, the local CNN combinations were not
critical for that branch — useful for justifying a leaner production
model on that side. If LSTM-only collapses far below the hybrid, the
spatial extractor is doing real work.