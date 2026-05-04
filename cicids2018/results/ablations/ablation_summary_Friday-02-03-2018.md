# CICIDS2018 ablation -- `Friday-02-03-2018`

- Rows evaluated: **200,000**
- Ground truth: **162,906 botnet** / **37,094 benign**

> **Caveat.** Constant-tile sequences (no `src_ip`) collapse the LSTM into a fixed-point computation. These numbers compare the models' *feature-space* generalisation on each stratum, not their temporal modelling. Re-extract the day from PCAP for a temporal comparison.

> **Why stratify?** CICFlowMeter splits one logical TCP conversation into multiple flow records on FIN/RST/idle-timeout, producing empty residue records that inherit the host-level `Bot` label without containing any C2 behaviour. The model correctly assigns these flows ~0 probability — but they drag the unstratified recall down. The **ACTIVE** subset is the meaningful comparison; **EMPTY** is included as a sanity check (all three architectures should score ~0 there).

## stratum: ALL  (n=200,000, botnet=162,906, benign=37,094)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.9972 | 0.4179 | 0.5890 | 0.7285 | 68,075 | 94,831 | 193 | 36,901 |
| cnn-only | 0.77 | 0.8735 | 0.4873 | 0.6256 | 0.5876 | 79,380 | 83,526 | 11,494 | 25,600 |
| lstm-only | 0.74 | 0.9088 | 0.9865 | 0.9461 | 0.6809 | 160,703 | 2,203 | 16,123 | 20,971 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.60 | 0.9929 | 0.4863 | 0.6528 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.90 | 0.8752 | 0.4873 | 0.6260 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.9199 | 0.9855 | 0.9516 |

## stratum: ACTIVE  (n=114,614, botnet=81,399, benign=33,215)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.9972 | 0.8363 | 0.9097 | 0.9778 | 68,075 | 13,324 | 192 | 33,023 |
| cnn-only | 0.77 | 0.8745 | 0.9752 | 0.9221 | 0.8172 | 79,380 | 2,019 | 11,388 | 21,827 |
| lstm-only | 0.74 | 0.8621 | 0.9747 | 0.9150 | 0.7175 | 79,341 | 2,058 | 12,690 | 20,525 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | recall>=0.85, max precision | 0.70 | 0.9961 | 0.9635 | 0.9795 |
| cnn-only | recall>=0.85, max precision | 0.90 | 0.8752 | 0.9752 | 0.9225 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.8820 | 0.9728 | 0.9252 |

## stratum: EMPTY  (n=85,386, botnet=81,507, benign=3,879)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.6062 | 0 | 81,507 | 1 | 3,878 |
| cnn-only | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.4689 | 0 | 81,507 | 106 | 3,773 |
| lstm-only | 0.74 | 0.9595 | 0.9982 | 0.9785 | 0.3783 | 81,362 | 145 | 3,433 | 446 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.9601 | 0.9982 | 0.9788 |

## How to read this

- **Recall is the project's primary metric** (minimise false negatives).
- The *training-time threshold* row shows operational behaviour: each model was tuned to hit recall>=0.85 on its own validation set; this row reveals whether that calibration generalises out-of-distribution.
- The *uniform sweep* row is the fair architecture comparison: same threshold grid, same data, same scaler-per-model logic. Differences are attributable to architecture only.
- The **ACTIVE** stratum corresponds to the 114,614-flow active subset cited in the CS499 thesis, where the hybrid scored P=0.984 R=0.977. Compare ablation rows to that baseline.