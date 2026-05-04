# CSE-CIC-IDS-2018 External Evaluation
**Group 07 | CPCS499 — AI-Based Botnet Detection**

This folder contains everything needed to evaluate the trained Stage-2
Non-IoT models against the CSE-CIC-IDS-2018 dataset — an external dataset
never seen during training, used to measure out-of-distribution
generalisation and to validate the hybrid CNN-LSTM architecture choice
against CNN-only and LSTM-only ablations.

---

## Folder layout

```
evaluation/cicids2018/
├── README.md                        ← this file
├── evaluate_external_csv.py         ← hybrid-only evaluator (Stage-1 + Stage-2)
├── evaluate_ablations_cicids.py     ← ablation evaluator (hybrid + CNN-only + LSTM-only)
├── stratified_eval.py               ← active vs empty flow stratification
├── diagnose_results.py              ← FN analysis, per-port/protocol breakdown
└── results/
    ├── summary_Friday-02-03-2018.json
    ├── stratified_summary_Friday-02-03-2018.json
    └── ablations/
        ├── Friday-02-03-2018/       ← ablation artefacts for Day 1
        ├── 02-15-2018/              ← ablation artefacts for Day 2
        └── _lstm_collapse_under_tiling.png
```

---

## Datasets

Download the daily CSV files from the
[UNB CSE-CIC-IDS-2018 page](https://www.unb.ca/cic/datasets/ids-2018.html)
or from Kaggle. Place them at:

```
data/raw/cicids2018/
├── Friday-02-03-2018.csv    ← Ares botnet  (used for evaluation)
├── 02-15-2018.csv           ← Bot family   (used for evaluation)
└── 02-14-2018.csv           ← Infiltration (NOT used — different attack type)
```

**Do not commit these files.** They are large and publicly available.
Add the following to `.gitignore`:

```
data/raw/cicids2018/*.csv
evaluation/cicids2018/results/predictions_*.csv
```

### Why 02-14 is excluded

The February 14 file contains Infiltration attacks — lateral movement and
data exfiltration, not botnet C2 traffic. Our Stage-2 models were trained
to detect botnet command-and-control behaviour and correctly classify
Infiltration flows as benign. Using that file to evaluate botnet detection
is a category error. Only the two botnet days (Friday-02-03 and 02-15)
are valid comparison points.

---

## Scripts

### `evaluate_ablations_cicids.py` — main ablation evaluator

Runs all three architectures (hybrid, CNN-only, LSTM-only) on the same
CSV in one pass. Applies active/empty stratification using the same
definition as `stratified_eval.py`. Produces per-stratum metrics,
confusion matrices, threshold sweeps, and bar charts.

```bash
# macOS — Day 1 (Ares botnet)
python3 evaluation/cicids2018/evaluate_ablations_cicids.py \
    --csv data/raw/cicids2018/Friday-02-03-2018.csv \
    --max_rows 200000 \
    --out_dir evaluation/cicids2018/results/ablations/Friday-02-03-2018

# macOS — Day 2 (Bot family)
python3 evaluation/cicids2018/evaluate_ablations_cicids.py \
    --csv data/raw/cicids2018/02-15-2018.csv \
    --max_rows 200000 \
    --out_dir evaluation/cicids2018/results/ablations/02-15-2018
```

```cmd
:: Windows — Day 1
python evaluation\cicids2018\evaluate_ablations_cicids.py ^
    --csv data\raw\cicids2018\Friday-02-03-2018.csv ^
    --max_rows 200000 ^
    --out_dir evaluation\cicids2018\results\ablations\Friday-02-03-2018

:: Windows — Day 2
python evaluation\cicids2018\evaluate_ablations_cicids.py ^
    --csv data\raw\cicids2018\02-15-2018.csv ^
    --max_rows 200000 ^
    --out_dir evaluation\cicids2018\results\ablations\02-15-2018
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--csv` | required | Path to CICFlowMeter CSV |
| `--max_rows` | None | Cap rows read (200,000 recommended for first run) |
| `--out_dir` | `results/ablations` | Where to write artefacts |
| `--strata` | `all,active,empty` | Which subsets to report |
| `--skip` | — | Comma list of archs to skip, e.g. `--skip lstm-only` |

### `evaluate_external_csv.py` — hybrid-only evaluator

Runs the full end-to-end pipeline (Stage-1 IoT/Non-IoT classifier →
Stage-2 hybrid CNN-LSTM) on a single CSV. Produces a threshold sweep,
confusion matrix, and summary JSON. Use this for the single-model
production evaluation; use `evaluate_ablations_cicids.py` for the
architecture comparison.

```bash
python3 evaluation/cicids2018/evaluate_external_csv.py \
    --csv data/raw/cicids2018/Friday-02-03-2018.csv \
    --max_rows 200000 \
    --threshold 0.30
```

### `stratified_eval.py` — active vs empty stratification

Splits per-row predictions into active flows (payload present in at
least one direction) and empty flows (TCP control-packet residue with
no payload). Run after `evaluate_external_csv.py` to explain the
unstratified recall plateau.

```bash
python3 evaluation/cicids2018/stratified_eval.py \
    --predictions evaluation/cicids2018/results/predictions_Friday-02-03-2018.csv \
    --csv data/raw/cicids2018/Friday-02-03-2018.csv \
    --max_rows 200000
```

**Active flow definition** (identical across all scripts):
`empty = (TotLen Fwd Pkts == 0) AND (Tot Bwd Pkts == 0)`

### `diagnose_results.py` — false-negative analysis

Analyses the false-negative population: probability quartiles,
per-port and per-protocol recall, and a feature comparison between
true positives and false negatives. Run after `evaluate_external_csv.py`.

```bash
python3 evaluation/cicids2018/diagnose_results.py \
    --predictions evaluation/cicids2018/results/predictions_Friday-02-03-2018.csv \
    --csv data/raw/cicids2018/Friday-02-03-2018.csv
```

---

## Results

### In-distribution test set (held-out from training data)

All three architectures are statistically equivalent on data drawn from
the same distribution as training. Differences are in the third decimal
place and within random-split noise.

| Architecture | Parameters | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN-LSTM | 456,321 | 0.9962 | 0.9919 | 0.9941 | 0.99982 |
| CNN-only | 133,633 | 0.9961 | 0.9924 | 0.9943 | 0.99971 |
| LSTM-only | 230,529 | 0.9975 | 0.9871 | 0.9923 | 0.99982 |

The architecture choice cannot be justified from in-distribution metrics
alone. All three saturate the measurable ceiling.

### Out-of-distribution: active-flow AUC across two botnet days

AUC is threshold-independent and is the cleanest comparison metric on
OOD data where calibration drift between training and test distributions
is expected.

| Architecture | Friday-02-03 (Ares) | 02-15 (Bot family) | Wins |
|---|---:|---:|---:|
| **Hybrid CNN-LSTM** | **0.9778** | **0.7871** | 2 / 2 |
| CNN-only | 0.8172 | 0.7769 | 0 / 2 |
| LSTM-only | 0.7175 | 0.4664 | 0 / 2 |

The hybrid outranks both ablations on both botnet evaluation days. The
pattern is consistent across two different botnet families, two different
testbed years, and two different class distributions.

### Friday-02-03 active flows — detailed (n = 114,614)

At best threshold meeting recall ≥ 0.85 on a uniform 0.10–0.90 sweep:

| Architecture | thr | Precision | Recall | F1 | FP |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN-LSTM | 0.70 | **0.9961** | 0.9635 | **0.9795** | **308** |
| CNN-only | 0.90 | 0.8752 | **0.9752** | 0.9225 | 11,388 |
| LSTM-only | 0.90 | 0.8820 | 0.9728 | 0.9252 | 10,602 |

The hybrid produces ~35× fewer false alarms than either ablation while
remaining within three recall points. This is the operational metric that
matters for a deployed security tool.

### 02-15 active flows — detailed (n = 157,869)

| Architecture | thr | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN-LSTM | 0.10 | 0.6002 | **0.7803** | **0.6785** | **0.7871** |
| CNN-only | 0.90 | 0.5044 | 0.7899 | 0.6156 | 0.7769 |
| LSTM-only | 0.20 | 0.2619 | 0.9502 | 0.4107 | 0.4664 |

All three models struggle on this day because the 02-15 Bot family is
more distant from the CTU-13 + CIC-IDS-2017 training distribution.
However the ranking is preserved: hybrid best, LSTM-only worst.
LSTM-only's AUC of 0.466 (below random) confirms it is inverting its
ranking on this unseen traffic.

### Empty-flow sanity check

Empty flows are TCP control-packet residue (no payload). A well-behaved
model should assign them ~0 probability regardless of their inherited
host-level label.

| Architecture | Mean P(bot\|true-bot empty) | Mean P(bot\|true-benign empty) | Behaves correctly? |
|---|---:|---:|---:|
| Hybrid CNN-LSTM | 0.0068 | 0.0061 | ✓ |
| CNN-only | 0.0000 | 0.0225 | ✓ |
| LSTM-only | 0.9376 | 0.8574 | ✗ |

Hybrid and CNN-only treat empty flows as feature-less and correctly
output ~0 probability. LSTM-only — lacking a CNN frontend — cannot
recognise the absence of distinguishing features and flags nearly
everything as botnet. This is a second, independent argument for
retaining the CNN block in the architecture.

---

## Key findings for the thesis

**Finding 1 — Architecture choice is invisible in-distribution.**
All three models achieve F1 ≈ 0.994 and AUC > 0.999 on held-out test
data from the training distribution. In-distribution metrics alone cannot
justify the hybrid.

**Finding 2 — The hybrid generalises better across botnet families.**
On two unseen botnet days the hybrid achieves the highest AUC in both
cases (0.978 and 0.787). Both ablations degrade significantly, with
LSTM-only falling below random on the second day (AUC = 0.466).

**Finding 3 — The hybrid produces far fewer false alarms.**
On Friday-02-03, at comparable recall, the hybrid produces 308 false
positives versus ~11,000 for the ablations. A 35× reduction in false
alarm volume is operationally decisive for a security tool.

**Finding 4 — The CNN frontend is necessary for handling empty flows.**
LSTM-only fails to recognise feature-less flows as benign (AUC = 0.378
on empty flows — worse than random). Both architectures that include a
Conv block pass this sanity check. The CNN block provides the inductive
bias the recurrent block alone cannot supply.

**Thesis framing:** The hybrid CNN-LSTM is the correct architecture
choice not because it outperforms ablations on in-distribution data —
it does not — but because it is the only configuration that maintains
both discriminative ranking (AUC) and precision (low false-alarm volume)
when evaluated on traffic it was never trained on. Since a deployed
botnet detector will always encounter new traffic, OOD generalisation
is the property that matters.

---

## Known limitations

**Tile-collapse on CICIDS2018.** Published CICIDS2018 CSVs do not
preserve `src_ip`, so the evaluator tiles each flow into a constant
length-20 sequence. Constant input drives an LSTM hidden state to a
fixed point within ~5 timesteps (see `results/ablations/_lstm_collapse_under_tiling.png`),
meaning the LSTM's temporal contribution is partially silenced. The
hybrid's OOD advantage is therefore likely an underestimate of its true
advantage on properly-sequenced traffic. Re-extracting the CSVs from
the original PCAPs using `src/ingestion/pcap_to_csv.py` would recover
real per-IP sequence ordering and is the recommended next step.

**Single evaluation day per botnet family.** Two botnet days were
evaluated. Adding further days or additional datasets would strengthen
the generalisation claim.

**Hyperparameters tuned for the hybrid.** The training schedule was
optimised for the hybrid and applied unchanged to the ablations. Ablation
numbers are therefore a lower bound on what each architecture could
achieve with its own tuning.