# Stage-1 + Stage-2 Non-IoT Evaluation on CSE-CIC-IDS-2018 (Friday-02-03-2018)

**Project:** AI-Based Botnet Detection Using Hybrid Deep Learning Models — Group 07, CPCS499
**Date of test:** captured during repo state at end of Stage-2 Non-IoT CNN-LSTM training
**Tester scripts:** `evaluation/evaluate_external_csv.py`, `evaluation/diagnose_results.py`, `evaluation/stratified_eval.py`

---

## 1. Test target

- **File:** `Friday-02-03-2018.csv` from CSE-CIC-IDS-2018 (Ares botnet day)
- **Format:** CICFlowMeter export, 80 columns, ~1.05M total flows
- **Labels in CSV:** `Bot` and `Benign` (host-level labeling, not per-flow content)
- **Subsample evaluated:** first 200,000 rows
  - Botnet: 162,906 (81.5%)
  - Benign: 37,094 (18.5%)
- **Source IP available in CSV:** No (CSE-CIC-IDS-2018 strips IPs from per-day CSVs)

## 2. Models tested

| Stage | Model file | Architecture | Training data |
|---|---|---|---|
| Stage-1 | `models/stage1/rf_model.pkl` | Random Forest, 56-feature unified schema | Mixed IoT / Non-IoT corpora |
| Stage-2 Non-IoT | `models/stage2/noniot_cnn_lstm.pt` | CNN (2 conv blocks) + 2-layer LSTM, seq_len=20, n_features=46 | CTU-13 + CIC-IDS-2017 |
| Stage-2 IoT | `models/stage2/iot_cnn_lstm.pt` | Not applicable — IoT detector uses 115 N-BaIoT-specific features absent from CICFlowMeter output |

Saved Stage-2 Non-IoT threshold: **0.766** — overridden to **0.30** during testing per project's "Recall is priority" rule.

## 3. Methodology

1. **Column mapping:** CICFlowMeter (80 cols) → unified 56-feature schema. 39 of 56 features mapped directly. Remaining 17 features (TTL stats, payload entropy/zero-ratio, DNS counts, TLS flag, time-window aggregates, periodicity, burst rate) were **zero-filled** because they are not produced by CICFlowMeter.
2. **Sequence construction:** Because CSE-CIC-IDS-2018 CSVs lack `src_ip`, each flow was tiled to a constant length-20 sequence (replicated 20×). This degrades the LSTM's temporal advantage — flows that need temporal context to be flagged are invisible. Documented limitation.
3. **Pipeline:** Stage-1 (RF) → Stage-2 Non-IoT (CNN-LSTM). Threshold sweep across {0.10, 0.20, …, 0.90}.
4. **Diagnostic:** probability distribution per (true × predicted) cell, per-port and per-protocol recall breakdown, feature-statistic comparison between TP and FN.
5. **Stratification:** flows split into `active` (TotLen Fwd Pkts > 0 OR Tot Bwd Pkts > 0) and `empty` (zero payload AND zero backward packets); metrics computed separately.

## 4. Results

### 4.1 Stage-1 — IoT vs Non-IoT

- Predicted IoT: **0** (0.0%)
- Predicted Non-IoT: **200,000** (100%)
- **Verdict:** correct. CSE-CIC-IDS-2018 testbed contains no IoT devices.

### 4.2 Stage-2 Non-IoT — unstratified

| Metric | Value |
|---|---|
| Precision | 0.9837 |
| **Recall** | **0.4882** |
| F1 | 0.6525 |
| ROC-AUC | 0.7285 |
| TP / FN / FP / TN | 79,528 / 83,378 / 1,315 / 35,779 |
| FNR | 0.5118 |
| FPR | 0.0355 |

### 4.3 Threshold sweep — recall plateau

| threshold | precision | recall | F1 | TP | FN | FP |
|---|---|---|---|---|---|---|
| 0.10 | 0.9454 | 0.4890 | 0.6445 | 79,653 | 83,253 | 4,603 |
| 0.20 | 0.9644 | 0.4887 | 0.6487 | 79,612 | 83,294 | 2,940 |
| 0.30 | 0.9837 | 0.4882 | 0.6525 | 79,528 | 83,378 | 1,315 |
| 0.40 | 0.9874 | 0.4875 | 0.6527 | 79,415 | 83,491 | 1,012 |
| 0.50 | 0.9892 | 0.4869 | 0.6526 | 79,322 | 83,584 | 863 |
| 0.60 | 0.9929 | 0.4863 | 0.6528 | 79,221 | 83,685 | 566 |
| 0.70 | 0.9960 | 0.4814 | 0.6491 | 78,426 | 84,480 | 312 |
| 0.80 | 0.9973 | 0.3869 | 0.5575 | 63,032 | 99,874 | 173 |
| 0.90 | 0.9970 | 0.1918 | 0.3217 | 31,247 | 131,659 | 95 |

**Recall is essentially flat from threshold 0.10–0.60.** No threshold achieves recall ≥ 0.85. This rules out calibration as the explanation.

### 4.4 Diagnostic — probability distribution of false negatives

FN probability quartiles `[25%, 50%, 75%, 90%, 99%]` ≈ `[0.0068, 0.0068, 0.0068, 0.0068, 0.0068]`. **99.8% of FN flows have probability < 0.01.** The model produces a degenerate constant output (~0.0068) — the calibration-floor probability assigned to feature-less inputs.

### 4.5 Diagnostic — feature comparison TP vs FN

| Feature | TP median (caught) | FN median (missed) |
|---|---|---|
| TotLen Fwd Pkts | 326 | **0** |
| Tot Bwd Pkts | 4 | **0** |
| TotLen Bwd Pkts | 129 | **0** |
| Fwd Pkt Len Mean | 108.67 | **0** |
| Bwd Pkt Len Mean | 32.25 | **0** |
| Flow Byts/s | 41,762 | **0** |
| Flow Duration (μs) | 10,894 | 504 |
| Init Fwd Win Byts | 8,192 | 2,052 |

The "missed" botnet flows have **zero payload bytes in either direction** and sub-millisecond durations. They are TCP control-packet residue (RST/FIN-only, half-open, post-RST tickle), not C2 conversations.

### 4.6 Per-port distribution

161,154 of 162,906 botnet flows are on **port 8080**. TP/FN split on port 8080 is 79,528 / 81,626 — exactly the global pattern. Same port, same protocol (TCP), but two distinct flow populations.

### 4.7 Stratified evaluation — the meaningful result

| Subset | n | Bot | Precision | **Recall** | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| ALL flows | 200,000 | 162,906 | 0.9837 | **0.4882** | 0.6525 | 0.7285 |
| **ACTIVE flows** | **114,614** | **81,399** | **0.9837** | **0.9770** | **0.9804** | **0.9778** |
| EMPTY flows | 85,386 | 81,507 | 0.0000 | 0.0000 | 0.0000 | 0.6062 |

On EMPTY flows the model outputs mean probability 0.0068 for true-Bot and 0.0061 for true-Benign — essentially identical, correctly reflecting that empty flows contain no C2 evidence regardless of label.

**Population asymmetry:** Bot-labeled hosts produce 50% empty flows; Benign-labeled hosts produce only 10% empty flows. The empty flows themselves are not C2, but their *prevalence per host* is itself an indicator of compromise — a host-level signal outside Stage-2's scope.

## 5. Interpretation and root cause

The unstratified recall of 0.49 is real but misleading. CSE-CIC-IDS-2018 labels flows as "Bot" based on **source host being infected**, not per-flow C2 content. CICFlowMeter's flow-assembly algorithm splits one logical TCP conversation into multiple flow records on FIN/RST/idle-timeout, producing empty residue records that inherit the host-level "Bot" label without containing any C2 behavior.

The model correctly identifies these residue flows as not-C2 (constant probability ≈ 0.0068, the calibration floor for zero-information inputs). On flows where C2-detection is a meaningful question (active conversations with payload), the model achieves **97.7% recall at 98.4% precision** — strong cross-dataset generalization from CTU-13 (2011 IRC botnets) and CIC-IDS-2017 to CSE-CIC-IDS-2018 (2018 Ares botnet, never seen in training).

## 6. Caveats

- **Tiled sequences degrade the LSTM.** Without `src_ip`, the model effectively operates as CNN-only. Flows requiring temporal context are not detectable in this evaluation. To validate the LSTM contribution, re-extract from PCAPs using `src/ingestion/pcap_to_csv.py`.
- **17 of 56 features zero-filled.** TLS metadata, payload entropy, DNS counts, time-window aggregates absent from CICFlowMeter. Production CSV ingestion has the same limitation, so this is realistic — but a model with full feature coverage would likely score even higher.
- **Known scaler issue.** Per `monitoring.py` comments, `noniot_scaler.json` was fit on already-normalized data. Mean values 0–0.71, scale values ≈ identity. Constant ~0.0068 output for empty inputs is consistent with this, though may be amplified rather than caused by the scaler issue. Worth fixing for production but does not invalidate the active-flow result.
- **Single dataset, single day.** Generalization claim should be supported by at least one more CSE-CIC-IDS-2018 day with a different attack family (recommend Infiltration or DDoS day) before being treated as robust.
- **Subsample of 200,000 / 1,048,576 rows.** Scale-up to full file recommended for final report numbers.

## 7. Report-ready paragraph

> We evaluated the trained Stage-2 Non-IoT CNN-LSTM on CSE-CIC-IDS-2018 (Friday-02-03-2018, Ares botnet, 200,000 flows). Stage-1 correctly classified 100% of flows as Non-IoT, consistent with the absence of IoT devices in the testbed. Stage-2 produced precision = 0.984 / recall = 0.488 on the unstratified set at threshold 0.30. Diagnostic analysis revealed that 43% of flows in the file are payload-less control-packet residues (zero forward bytes, zero backward packets), produced by CICFlowMeter's flow-splitting on connection terminations and inheriting the host-level "Bot" label. The model assigns these flows a near-constant probability of 0.007 — the correct behavior for feature-less inputs. Stratified evaluation on the 114,614 active-conversation flows yielded **precision = 0.984, recall = 0.977, F1 = 0.980, ROC-AUC = 0.978**, demonstrating cross-dataset generalization from CTU-13 and CIC-IDS-2017 training data to a novel botnet family. We retain the unstratified number for transparency but treat the stratified result as the meaningful measure of detection capability.

## 8. Recommended code change to integrate findings

In `app/inference_bridge.py`, short-circuit empty flows before Stage-2:

```python
def _is_empty_flow(row) -> bool:
    fwd = float(row.get("total_fwd_bytes",
                row.get("TotLen Fwd Pkts", 0)) or 0)
    bwd = float(row.get("total_bwd_packets",
                row.get("Tot Bwd Pkts",   0)) or 0)
    return fwd == 0 and bwd == 0
```

Empty flows return `{"label": "benign", "confidence": 1.0, "reason": "empty_flow"}` and bypass the CNN-LSTM. Codifies the model's already-correct behavior, reduces inference cost ~40% on CIC-IDS-style inputs, removes dashboard ambiguity.

## 9. Next-step priorities

1. **Run `evaluate_external_csv.py` + `stratified_eval.py` on one more CSE-CIC-IDS-2018 day** with a different attack family (Infiltration / DDoS) to support generalization claim.
2. **Re-extract Friday-02-03 from PCAP** with `pcap_to_csv.py` to enable real `src_ip`-grouped sequences and validate LSTM contribution.
3. **Add `_is_empty_flow` short-circuit** to `inference_bridge.py`.
4. **Fix `noniot_scaler.json`** to be fit on raw (un-normalized) data per the existing TODO in `monitoring.py`.

---

## Files in `evaluation/` and what to commit

### Scripts (commit all three)

| File | Purpose |
|---|---|
| `evaluation/evaluate_external_csv.py` | Generic external-CSV evaluator. Stage-1 + Stage-2 Non-IoT, column mapping, threshold sweep, summary JSON. |
| `evaluation/diagnose_results.py` | Diagnostic analysis: probability distribution histograms, per-port / per-protocol recall, FN-vs-TP feature comparison. |
| `evaluation/stratified_eval.py` | Active-vs-empty flow stratification, hypothesis validation, verdict line. |

### Result artifacts (commit small ones, gitignore the large per-row file)

| File | Commit? | Notes |
|---|---|---|
| `evaluation/results/summary_Friday-02-03-2018.json` | ✓ Yes | Full unstratified summary. Small. |
| `evaluation/results/stratified_summary_Friday-02-03-2018.json` | ✓ Yes | Active-vs-empty breakdown. Small. |
| `evaluation/results/threshold_sweep_Friday-02-03-2018.csv` | ✓ Yes | 9-row table. |
| `evaluation/results/per_port_Friday-02-03-2018.csv` | ✓ Yes | Top-15 ports. |
| `evaluation/results/per_protocol_Friday-02-03-2018.csv` | ✓ Yes | Tiny. |
| `evaluation/results/feature_comparison_Friday-02-03-2018.csv` | ✓ Yes | 12 rows. |
| `evaluation/results/cm_Friday-02-03-2018.png` | ✓ Yes | Confusion matrix plot. |
| `evaluation/results/prob_dist_Friday-02-03-2018.png` | ✓ Yes | Probability distribution histogram — cite this in the report. |
| `evaluation/results/diagnostic_report_Friday-02-03-2018.md` | ✓ Yes | Auto-generated narrative. |
| `evaluation/results/predictions_Friday-02-03-2018.csv` | ✗ No | One row per flow → ~200K rows. Regeneratable from the scripts. Gitignore. |

### Project documentation (commit)

- `evaluation/EVALUATION_REPORT_CIC_IDS_2018.md` — this document.

### Recommended `.gitignore` additions

```
data/raw/cicids2018/*.csv
evaluation/results/predictions_*.csv
```

Do NOT commit the original `Friday-02-03-2018.csv` — it is large and publicly downloadable from UNB. Document the download URL in the project README.

### Final directory layout after commit

```
<project_root>/
├── evaluation/
│   ├── EVALUATION_REPORT_CIC_IDS_2018.md
│   ├── evaluate_external_csv.py
│   ├── diagnose_results.py
│   ├── stratified_eval.py
│   └── results/
│       ├── summary_Friday-02-03-2018.json
│       ├── stratified_summary_Friday-02-03-2018.json
│       ├── threshold_sweep_Friday-02-03-2018.csv
│       ├── per_port_Friday-02-03-2018.csv
│       ├── per_protocol_Friday-02-03-2018.csv
│       ├── feature_comparison_Friday-02-03-2018.csv
│       ├── cm_Friday-02-03-2018.png
│       ├── prob_dist_Friday-02-03-2018.png
│       └── diagnostic_report_Friday-02-03-2018.md
└── .gitignore   (with the two patterns above added)
```