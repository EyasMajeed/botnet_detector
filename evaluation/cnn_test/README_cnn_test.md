# CNN-only Ablation — Stage-2 Botnet Detector

## Group 07 | CPCS499 | AI-Based Botnet Detection

This folder contains the **CNN-only** ablation: the same Conv1d spine as
the production hybrid CNN-LSTM, with the LSTM block replaced by
`AdaptiveAvgPool1d` over the temporal dimension. Everything else
(dataset, splits, scaler, hyperparameters, training loop, threshold
selection, evaluation) is reused unchanged from the hybrid pipeline so
that any difference in test-set metrics is attributable to the LSTM
removal alone.

---

## Why this exists

The CS498 final report (Section 7.2) claims the hybrid CNN-LSTM is justified
because "using the two models together provides stronger detection
performance than using CNN or LSTM alone." This claim was previously
unverified in the repository. This script provides the empirical evidence.

---

## Files

| File | Purpose |
|---|---|
| `cnn_only_noniot.py` | CNN-only training+evaluation on `data/processed/stage2_noniot_botnet.csv` |
| `cnn_only_iot.py`    | CNN-only training+evaluation on `data/processed/stage2_iot_combined.csv` |
| `results/noniot/`    | Auto-created on first Non-IoT run (metrics.json, plots, model.pt, scaler.json) |
| `results/iot/`       | Auto-created on first IoT run |

The scripts **import** the proven data-prep / training / threshold /
evaluation helpers from `models/stage2/noniot_detector_cnnlstm.py` and
`models/stage2/iot_detector.py`, then redirect the scaler-write path so
that running the ablation does **not** clobber the production
`noniot_scaler.json`. Only the model class is local.

---

## Architecture (Non-IoT and IoT identical, CNN block reused verbatim)

```
Input  (B, seq_len=20, n_features)
   └── permute → Conv1d(n_feat → 128, k=3, p=1) → BN → ReLU → MaxPool(2)
                 Conv1d(128    → 256, k=3, p=1) → BN → ReLU
   └── AdaptiveAvgPool1d(1) → squeeze            ←  replaces LSTM
   └── Linear(256, 64) → ReLU → Dropout(0.4) → Linear(64, 1)  (logit)
```

**Parameter count** is significantly lower than the hybrid (no LSTM = no
recurrent gates, ~400K fewer parameters). The aggregator records the
exact count for the comparison table.

---

## Running

### Step 0 — Ensure the hybrid models have been trained first

The aggregator pulls hybrid metrics from
`models/stage2/results/{iot,noniot}_metrics.json`. If those files do not
yet exist, run the production training scripts first:

**Windows**
```
python models\stage2\noniot_detector_cnnlstm.py
python models\stage2\iot_detector.py
```

**macOS**
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

### Step 2 — Run the CNN-only ablations

**Windows**
```
python evaluation\cnn_test\cnn_only_noniot.py
python evaluation\cnn_test\cnn_only_iot.py
```

**macOS**
```
python3 evaluation/cnn_test/cnn_only_noniot.py
python3 evaluation/cnn_test/cnn_only_iot.py
```

### Step 3 — Aggregate and compare

```
python3 evaluation/ablation_comparison/aggregate_metrics.py
```

This produces `comparison_table.csv`, `comparison_bars.png`, and
`comparison_summary.md` in `evaluation/ablation_comparison/results/`.

---

## Expected outputs (per branch)

```
evaluation/cnn_test/results/<branch>/
├── metrics.json          # accuracy, precision, recall, f1, auc_roc, threshold, n_parameters
├── training_curves.png   # loss + val accuracy + val recall vs. epoch
├── confusion_matrix.png  # test-set confusion matrix
├── roc_curve.png         # test-set ROC curve
├── model.pt              # trained checkpoint (PyTorch)
└── scaler.json           # StandardScaler (Non-IoT only; IoT uses the production MinMaxScaler)
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: stage2_noniot_botnet.csv` | Pipeline not run | Run `data_processing/merge_stage2_noniot.py` first |
| `FileNotFoundError: stage2_iot_combined.csv` | Pipeline not run | Run `src/ingestion/preprocess_nbaiot.py` then `data_processing/combine_datasets.py` |
| `ModuleNotFoundError: models.stage2.noniot_detector_cnnlstm` | Wrong working directory | Run from project root, OR the script auto-fixes via `sys.path.insert` |
| `AssertionError: SCALER SANITY CHECK FAILED` | CSV is already normalised | Regenerate the Non-IoT CSV from raw flows |
| Slow training on CPU | No GPU | LSTM-only and hybrid both use cuDNN-fused LSTM on GPU; CPU is acceptable but slower |

---

## Reading the results

In a properly trained ablation, the **CNN-only model is expected to
underperform on Recall** (priority metric) because it loses the ability
to model botnet C2 beaconing periodicity, slow-scan timing patterns, and
sequential connection bursts that the LSTM captures. Precision usually
holds up because local feature combinations (high SYN ratio, suspicious
port + duration combinations) are still captured by the CNN.

If CNN-only matches or beats the hybrid on a given branch, that is itself
a useful finding: it means the temporal signal in that dataset is weak
and the project could use a leaner model in production for that branch.