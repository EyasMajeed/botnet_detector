# CICIDS2018 ablation -- `02-15-2018`

- Rows evaluated: **200,000**
- Ground truth: **52,498 botnet** / **147,502 benign**

> **Caveat.** Constant-tile sequences (no `src_ip`) collapse the LSTM into a fixed-point computation. These numbers compare the models' *feature-space* generalisation on each stratum, not their temporal modelling. Re-extract the day from PCAP for a temporal comparison.

> **Why stratify?** CICFlowMeter splits one logical TCP conversation into multiple flow records on FIN/RST/idle-timeout, producing empty residue records that inherit the host-level `Bot` label without containing any C2 behaviour. The model correctly assigns these flows ~0 probability — but they drag the unstratified recall down. The **ACTIVE** subset is the meaningful comparison; **EMPTY** is included as a sanity check (all three architectures should score ~0 there).

## stratum: ALL  (n=200,000, botnet=52,498, benign=147,502)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.7077 | 0.3850 | 0.4987 | 0.6905 | 20,214 | 32,284 | 8,348 | 139,154 |
| cnn-only | 0.77 | 0.4958 | 0.5478 | 0.5205 | 0.6712 | 28,757 | 23,741 | 29,250 | 118,252 |
| lstm-only | 0.74 | 0.2786 | 0.7796 | 0.4105 | 0.4531 | 40,930 | 11,568 | 105,978 | 41,524 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.6002 | 0.5411 | 0.5691 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.90 | 0.5043 | 0.5478 | 0.5252 |
| lstm-only | recall>=0.85, max precision | 0.20 | 0.2927 | 0.9654 | 0.4492 |

## stratum: ACTIVE  (n=157,869, botnet=36,405, benign=121,464)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.7077 | 0.5553 | 0.6223 | 0.7871 | 20,214 | 16,191 | 8,347 | 113,117 |
| cnn-only | 0.77 | 0.5003 | 0.7899 | 0.6126 | 0.7769 | 28,757 | 7,648 | 28,725 | 92,739 |
| lstm-only | 0.74 | 0.2344 | 0.6822 | 0.3489 | 0.4664 | 24,837 | 11,568 | 81,129 | 40,335 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.6002 | 0.7803 | 0.6785 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.90 | 0.5044 | 0.7899 | 0.6156 |
| lstm-only | recall>=0.85, max precision | 0.20 | 0.2619 | 0.9502 | 0.4107 |

## stratum: EMPTY  (n=42,131, botnet=16,093, benign=26,038)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.7272 | 0 | 16,093 | 1 | 26,037 |
| cnn-only | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.4592 | 0 | 16,093 | 525 | 25,513 |
| lstm-only | 0.74 | 0.3931 | 1.0000 | 0.5643 | 0.3090 | 16,093 | 0 | 24,849 | 1,189 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.3933 | 1.0000 | 0.5645 |

## How to read this

- **Recall is the project's primary metric** (minimise false negatives).
- The *training-time threshold* row shows operational behaviour: each model was tuned to hit recall>=0.85 on its own validation set; this row reveals whether that calibration generalises out-of-distribution.
- The *uniform sweep* row is the fair architecture comparison: same threshold grid, same data, same scaler-per-model logic. Differences are attributable to architecture only.
- The **ACTIVE** stratum corresponds to the 114,614-flow active subset cited in the CS499 thesis, where the hybrid scored P=0.984 R=0.977. Compare ablation rows to that baseline.