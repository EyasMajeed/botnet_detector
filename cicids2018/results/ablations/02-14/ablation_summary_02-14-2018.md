# CICIDS2018 ablation -- `02-14-2018`

- Rows evaluated: **200,000**
- Ground truth: **199,734 botnet** / **266 benign**

> **Caveat.** Constant-tile sequences (no `src_ip`) collapse the LSTM into a fixed-point computation. These numbers compare the models' *feature-space* generalisation on each stratum, not their temporal modelling. Re-extract the day from PCAP for a temporal comparison.

> **Why stratify?** CICFlowMeter splits one logical TCP conversation into multiple flow records on FIN/RST/idle-timeout, producing empty residue records that inherit the host-level `Bot` label without containing any C2 behaviour. The model correctly assigns these flows ~0 probability — but they drag the unstratified recall down. The **ACTIVE** subset is the meaningful comparison; **EMPTY** is included as a sanity check (all three architectures should score ~0 there).

## stratum: ALL  (n=200,000, botnet=199,734, benign=266)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.8600 | 0.0011 | 0.0022 | 0.2198 | 215 | 199,519 | 35 | 231 |
| cnn-only | 0.77 | 0.9962 | 0.0596 | 0.1125 | 0.5696 | 11,903 | 187,831 | 45 | 221 |
| lstm-only | 0.74 | 0.9988 | 0.9619 | 0.9800 | 0.6573 | 192,122 | 7,612 | 235 | 31 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.9907 | 0.0204 | 0.0399 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.10 | 0.9962 | 0.0596 | 0.1125 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.9988 | 0.9259 | 0.9610 |

## stratum: ACTIVE  (n=199,875, botnet=199,733, benign=142)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.8600 | 0.0011 | 0.0022 | 0.3435 | 215 | 199,518 | 35 | 107 |
| cnn-only | 0.77 | 0.9962 | 0.0596 | 0.1125 | 0.4736 | 11,903 | 187,830 | 45 | 97 |
| lstm-only | 0.74 | 0.9994 | 0.9619 | 0.9803 | 0.6942 | 192,121 | 7,612 | 111 | 31 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.9907 | 0.0204 | 0.0399 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.10 | 0.9962 | 0.0596 | 0.1125 |
| lstm-only | recall>=0.85, max precision | 0.90 | 0.9995 | 0.9259 | 0.9613 |

## stratum: EMPTY  (n=125, botnet=1, benign=124)

**At each model's own training-time threshold**

| Architecture | thr | Precision | Recall | F1 | AUC | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 | 1 | 0 | 124 |
| cnn-only | 0.77 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 | 1 | 0 | 124 |
| lstm-only | 0.74 | 0.0080 | 1.0000 | 0.0159 | 0.5000 | 1 | 0 | 124 | 0 |

**At each model's best threshold in the uniform sweep**

| Architecture | basis | thr | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| hybrid | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| cnn-only | max F1 (recall target unattainable in sweep) | 0.10 | 0.0000 | 0.0000 | 0.0000 |
| lstm-only | recall>=0.85, max precision | 0.10 | 0.0080 | 1.0000 | 0.0159 |

## How to read this

- **Recall is the project's primary metric** (minimise false negatives).
- The *training-time threshold* row shows operational behaviour: each model was tuned to hit recall>=0.85 on its own validation set; this row reveals whether that calibration generalises out-of-distribution.
- The *uniform sweep* row is the fair architecture comparison: same threshold grid, same data, same scaler-per-model logic. Differences are attributable to architecture only.
- The **ACTIVE** stratum corresponds to the 114,614-flow active subset cited in the CS499 thesis, where the hybrid scored P=0.984 R=0.977. Compare ablation rows to that baseline.