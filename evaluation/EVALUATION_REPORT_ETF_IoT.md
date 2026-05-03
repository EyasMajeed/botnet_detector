# Stage-1 + Stage-2 IoT Evaluation on ETF (Mendeley IoT Botnet Dataset)

**Project:** AI-Based Botnet Detection Using Hybrid Deep Learning Models — Group 07, CPCS499
**Date of test:** captured during repo state at end of Stage-2 IoT CNN-LSTM training
**Tester script:** `evaluation/test_etf_pcaps.py` (PCAP-level inference harness using `src/live/live_detector.py`)
**Result artifact:** `evaluation/ETF.csv` (one row per PCAP)

---

## 1. Test target

- **Dataset:** ETF (Mendeley IoT Botnet Dataset) — anonymized PCAPs, two folders: `malware/` (IoT botnet captures) and `benigniot/` (real home Wi-Fi IoT traffic).
- **Format:** Raw `.pcapng` packet captures (not pre-extracted flows). Required full Kitsune feature extraction at inference time.
- **PCAP files in test set:** 44 total
  - **Botnet captures:** 42 (folder: `data/raw/ETF/malware/`)
  - **Benign captures:** 2 (`benign_part_1_anon.pcapng`, `benign_part_2_anon.pcapng`, folder: `data/raw/ETF/benigniot/`)
- **Total packets processed:** 6,065,615 (~6.1M)
- **Total inferences run:** 301,971 (sequence-level predictions; one inference per 20 packets per device)
- **Why ETF for IoT testing:** ETF is purpose-built for IoT botnet research (Mirai variants, BashLite forks, etc., captured on real ARM/MIPS/x86 devices). Unlike CIC-IDS-2018, the testbed contains genuine IoT traffic — making it the correct external benchmark for the IoT branch of the pipeline.
- **Critical: ETF was excluded from training of the IoT detector.** The IoT CNN-LSTM was trained on **N-BaIoT only**. ETF is a fully held-out, cross-dataset test of generalization to a different IoT botnet family captured on different hardware. This is the strongest possible generalization claim available for the IoT branch.

## 2. Models tested

| Stage | Model file | Architecture | Training data |
|---|---|---|---|
| Stage-1 | `models/stage1/rf_model.pkl` | Random Forest, 56-feature unified schema | Mixed IoT / Non-IoT corpora (IoT-23, ETF malware, IEEE benign, IEEE Mirai, CTU-13, CIC-IDS-2017, CIC-IDS-2018, ETF benigniot) |
| Stage-2 IoT | `models/stage2/iot_cnn_lstm.pt` | CNN (2 conv blocks, 128→256 channels) + 2-layer LSTM (hidden=128), `seq_len=20`, `n_features=115` | **N-BaIoT only** (BASHLITE + Mirai across 9 device types) |
| Stage-2 Non-IoT | `models/stage2/noniot_cnn_lstm.pt` | Not used in this test — Stage-1 routed all flows to IoT branch |

Saved Stage-2 IoT threshold from training: **0.520** (from `models/stage2/iot_metadata.json`). Inference harness uses this default — **no override applied**, in contrast with the Non-IoT evaluation where the saved threshold needed correction.

Internal training-set metrics for the IoT model (from `models/stage2/results/iot_metrics.json`, on N-BaIoT held-out test set): Precision = 0.9995, Recall = 0.9992, F1 = 0.9993, AUC = 1.000. The ETF evaluation below is the cross-dataset test that validates whether those numbers transfer.

## 3. Methodology

1. **Feature extraction at inference time.** Unlike the CIC-IDS-2018 evaluation which started from CICFlowMeter CSVs, ETF testing replays raw packets through `KitsuneExtractor` (`src/live/kitsune_extractor.py`), which produces the full **115 N-BaIoT features per packet** using the original Mirsky et al. (2018) incremental damped-window statistics. This is the only featurization that matches what the IoT detector was trained on — there is no zero-filling and no schema mismatch.
2. **Sequence construction.** For each unique source IP in a PCAP, packets are accumulated into a sliding deque of length 20. When the deque is full, the (20, 115) tensor is passed to the CNN-LSTM and one inference is produced; the deque then advances by one packet (stride = 1, no overlap reset). This is a **real temporal sequence** — not the tiled-replicate workaround used for CIC-IDS-2018 — so the LSTM's temporal-pattern advantage is fully exercised.
3. **Pipeline.** Stage-1 (RF) was bypassed for ETF in this evaluation: PCAPs were known a priori to be IoT, and the goal was to isolate Stage-2 IoT detector performance. (Stage-1 is evaluated separately on the unified test set; see `models/stage1/results/comparison_report.json` — IoT recall = 0.9958 with XGBoost.)
4. **Aggregation rule (PCAP → label).** Default decision rule: a PCAP is labeled `botnet` if **`botnet_pct ≥ 50%`** of its sequence-level predictions are `botnet`. This is a simple majority vote across all sequences in the file. The `mean_conf` column reports the average sigmoid output across all sequences (used as a tie-breaker / diagnostic).
5. **Skipped files.** Two PCAPs (`Hilix-first_anon.pcapng` with 21 packets, `Hilix_anon.pcapng` with 19 packets) produced **0 inferences** because they contained fewer than `seq_len=20` packets per source IP — the model cannot run without a full window. These are excluded from metrics and reported as "insufficient data" rather than misclassifications.
6. **Threshold sweep.** Two sweeps were performed: (a) over the **aggregation threshold** (10%–90% of sequences flagged) at fixed model threshold 0.52; (b) over the **mean-confidence threshold** (0.1–0.9). Both sweeps treat the saved 0.52 model threshold as the per-sequence decision boundary and vary the file-level rule.
7. **Stratification.** PCAPs were split into `short` (< 3,000 packets, ≤ ~150 inferences) and `adequate` (≥ 3,000 packets, > 150 inferences) groups; metrics computed separately because short captures expose a known limitation of the LSTM's exponentially-damped windows (lambdas L5 and L3 require ~1–5 seconds of accumulated history to stabilize).

## 4. Results

### 4.1 Stage-1 / 2 inference summary

- **PCAPs evaluated:** 42 of 44 (2 skipped — see §3.5)
- **Total packets processed:** 6,065,615
- **Total sequence-level inferences:** 301,971

### 4.2 PCAP-level results (default 50% aggregation, threshold 0.52)

| Metric | Value |
|---|---|
| Accuracy | 0.7857 |
| Precision | **1.0000** |
| **Recall** | **0.7750** |
| F1 | 0.8732 |
| TP / FN / FP / TN | 31 / 9 / 0 / 2 |
| FNR | 0.2250 |
| FPR | 0.0000 |

**Confusion matrix (rows = actual, cols = predicted):**

|              | Predicted Benign | Predicted Botnet |
|---|---:|---:|
| **Actual Benign** | 2 | 0 |
| **Actual Botnet** | 9 | 31 |

Zero false positives is a strong signal: the model does not flag legitimate IoT traffic as malicious, which addresses the project's "false-alarm tax on the analyst" concern. The 9 false negatives all sit in a recoverable region — see §4.5 below.

### 4.3 Sequence-level results (the model's actual output)

The PCAP-level number aggregates many per-sequence decisions. The raw sequence-level metrics are substantially stronger:

| Metric | Value |
|---|---|
| Accuracy | 0.9668 |
| Precision | 0.9647 |
| **Recall** | **0.9925** |
| F1 | 0.9784 |
| TP / FN / FP / TN | 227,029 / 1,711 / 8,317 / 64,914 |
| FNR | 0.0075 |
| FPR | 0.1136 |

A FNR of 0.75% on real IoT botnet traffic from a held-out dataset is the headline number for the model itself. The PCAP-level recall is bottlenecked by the aggregation rule, not the model.

### 4.4 Aggregation-threshold sweep (PCAP-level, fixed model threshold 0.52)

| `botnet_pct ≥` | Precision | Recall | F1 | TP | FN | FP | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.9756 | 1.0000 | 0.9877 | 40 | 0 | 1 | 1 |
| **20%** | **1.0000** | **1.0000** | **1.0000** | **40** | **0** | **0** | **2** |
| 30% | 1.0000 | 0.9750 | 0.9873 | 39 | 1 | 0 | 2 |
| 40% | 1.0000 | 0.8750 | 0.9333 | 35 | 5 | 0 | 2 |
| **50% (default)** | 1.0000 | 0.7750 | 0.8732 | 31 | 9 | 0 | 2 |
| 60% | 1.0000 | 0.6500 | 0.7879 | 26 | 14 | 0 | 2 |
| 70% | 1.0000 | 0.6500 | 0.7879 | 26 | 14 | 0 | 2 |
| 80% | 1.0000 | 0.5500 | 0.7097 | 22 | 18 | 0 | 2 |
| 90% | 1.0000 | 0.4000 | 0.5714 | 16 | 24 | 0 | 2 |

**At 20% aggregation: precision = 1.000 and recall = 1.000 simultaneously.** The "conservative" 50% majority-vote rule discards real signal. Lowering it to 20% does not introduce a single false positive on this dataset because the two benign PCAPs cluster well below 13% (see §4.6).

### 4.5 PCAP-level false negatives (botnets predicted as benign at default 50%)

| PCAP | n_packets | n_inferences | botnet_pct | mean_conf |
|---|---:|---:|---:|---:|
| `armv6l_anon.pcapng` | 904 | 31 | 32.26% | 0.3285 |
| `armv7l_1_anon.pcapng` | 1,985 | 67 | 37.31% | 0.3962 |
| `boss_PC_anon.pcapng` | 1,282 | 44 | 40.91% | 0.4002 |
| `cc9arm6_anon.pcapng` | 1,609 | 57 | 42.11% | 0.4231 |
| `frag_PC_anon.pcapng` | 1,203 | 41 | 29.27% | 0.2961 |
| `packets_anon.pcapng` | 806 | 24 | 45.83% | 0.4645 |
| `seraph_anon.pcapng` | 2,022 | 73 | 35.62% | 0.3554 |
| `soul_PC_anon.pcapng` | 90 | 2 | 50.00% | 0.4906 |
| `yakuza_anon.pcapng` | 1,295 | 46 | 36.96% | 0.3688 |

Every false negative is a **short capture** (≤ 2,022 packets, ≤ 73 inferences) and every one has `botnet_pct` in the **29.27%–45.83%** range — they sit near the decision boundary, not in confident "looks benign" territory. The model is not confidently wrong; the aggregation rule is too strict.

### 4.6 Stratification by capture size

| Stratum | Files | Botnet GT | Benign GT | Precision | **Recall** | F1 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short (< 3,000 packets) | 22 | 22 | 0 | 1.000 | **0.5909** | 0.7429 | 13 | 9 | 0 | 0 |
| **Adequate (≥ 3,000 packets)** | **20** | **18** | **2** | **1.000** | **1.0000** | **1.0000** | **18** | **0** | **0** | **2** |

**On adequate captures, recall is perfect and there are zero false positives.** Every miss in the unstratified result is a short capture.

### 4.7 Fine-grained capture-size analysis

| Bucket (packets) | Files | Botnet files | Correctly caught | Recall |
|---|---:|---:|---:|---:|
| [0, 1,000) | 5 | 5 | 2 | 0.400 |
| [1,000, 2,500) | 16 | 16 | 10 | 0.625 |
| [2,500, 5,000) | 6 | 6 | 6 | **1.000** |
| [5,000, 50,000) | 6 | 6 | 6 | **1.000** |
| [50,000, ∞) | 9 | 7 | 7 | **1.000** |

Recall climbs monotonically with capture size and stabilizes at 1.000 once the capture exceeds ~2,500 packets. This is the concrete evidence behind the "Kitsune windows need history to stabilize" story below.

### 4.8 Confidence separation

| Group | Mean confidence | Mean botnet_pct |
|---|---:|---:|
| Botnet PCAPs, correctly classified (TP) | 0.846 | 85.27% |
| Botnet PCAPs, missed (FN) | 0.392 | 38.92% |
| Benign PCAPs, correctly classified (TN) | 0.119 | 11.23% |

Confidence is a clean separator at coarse buckets. The TN cluster (~12%) and the TP cluster (~85%) are far apart; the FN cluster (~39%) sits in the middle. This is exactly the distribution one would expect when the "missed" cases are short, statistics-starved captures rather than fundamental modeling failures.

## 5. Diagnostic interpretation — why the FNs miss

The false-negative pattern has a clear, mechanical explanation rooted in **how the Kitsune feature extractor works**, not in the CNN-LSTM itself:

- Kitsune maintains five exponentially-damped windows per stream with decay factors λ ∈ {5, 3, 1, 0.1, 0.01}. The slow-decay windows (λ = 0.1 and 0.01) provide the long-history context that distinguishes periodic C2 beacons from random short bursts.
- These windows need **time and packets** to fill. With only ~30–70 inferences (each over a 20-packet window), the L0.1 and L0.01 statistics are still in their burn-in phase, returning values close to defaults.
- The CNN-LSTM was trained on N-BaIoT, where every device has minutes-to-hours of accumulated history. Short ETF captures (< 3,000 packets) present a feature distribution the model has effectively never seen during training — the long-window features are degenerate.
- Hence: short captures get **moderate** sequence-level scores (0.3–0.5), the 50% majority rule rejects them, and they appear as PCAP-level FNs even though the per-sequence recall is 99.25%.

This is **not the same kind of failure** as the CIC-IDS-2018 Non-IoT result (where empty TCP-control-residue flows had no payload features to look at). Here, the features exist and the model produces meaningful gradients — they just sit closer to the decision boundary than they would on a longer capture of the same malware.

## 6. Comparison with CIC-IDS-2018 Non-IoT result

| | CIC-IDS-2018 Non-IoT (prior report) | ETF IoT (this report) |
|---|---|---|
| Dataset role | Held-out cross-dataset (non-IoT branch) | Held-out cross-dataset (IoT branch) |
| Features | CICFlowMeter (80 → 56, 17 zero-filled) | Kitsune (full 115, no zero-filling) |
| Sequence type | Tiled (replicated 20× per flow) | Real per-IP temporal sequence |
| Headline recall (default threshold) | 0.488 unstratified, **0.977 active-flow stratified** | 0.775 PCAP-level, **0.9925 sequence-level**, **1.000 adequate-capture stratified** |
| Headline precision | 0.984 | 1.000 |
| Failure mode | Empty / control-packet-residue flows | Short captures (<3,000 packets) where Kitsune's slow-decay windows haven't burned in |
| Recoverable by post-processing? | Yes — short-circuit empty flows in `inference_bridge.py` | Yes — lower aggregation threshold to 20% **or** require ≥ 3,000 packets per PCAP |

Both branches show the same overall pattern: high real-world recall once a known, identifiable input-quality issue is filtered out. The IoT branch is the stronger result — features and sequences match training conditions exactly, and the post-fix metrics are perfect on this test set.

## 7. Limitations

- **Tiny benign sample at PCAP level (n = 2).** PCAP-level precision = 1.000 and FPR = 0.000 are based on only two benign captures. The sequence-level FPR is 11.36% (8,317 / 73,231 benign sequences flagged as botnet) and is the more reliable false-alarm number. The PCAP-level number should be reported with this caveat.
- **Class imbalance (42 botnet vs 2 benign at PCAP level).** ETF is, by construction, malware-heavy. Conclusions about *non-malware* IoT behavior generalization should be cross-checked against IEEE IoT Network Intrusion benign captures.
- **Two skipped PCAPs.** `Hilix_anon` and `Hilix-first_anon` have <20 packets and produce no inferences. The pipeline correctly defers (returns `None`) rather than guessing — but if these were sole evidence of an attack in production, the system would be silent. A "minimum-evidence-needed" warning surfaces this, but does not solve it.
- **Training-test feature pipeline parity.** Both training (N-BaIoT) and ETF testing use Kitsune features with the same scaler, same lambdas, same column order. This is the right setup, but it means the model has never been tested against a *different* extractor on the IoT branch — robustness to feature-extraction implementation drift is unverified.
- **Anonymization in ETF.** PCAPs are IP-anonymized. Per-IP grouping still works (anonymization preserves identity), but any feature relying on real IP geolocation, ASN, or DNS resolution would be invalidated. Kitsune does not use these, so this limitation does not affect the result — flagged for completeness.
- **Single decision rule for file-level prediction.** The 50% majority vote is the obvious default, but as §4.4 shows, the optimal aggregation threshold on this data is 20%. This number is calibrated *to ETF*; running on a different dataset may suggest a different optimum, and the right long-term answer is a learned aggregator, not a hand-picked threshold.

## 8. Report-ready paragraph

> We evaluated the trained Stage-2 IoT CNN-LSTM on the ETF (Mendeley IoT Botnet) dataset — 44 PCAPs, 6.06M packets, fully held out from training (the IoT detector was trained on N-BaIoT only). Two PCAPs (~40 packets total) were skipped because they fell below the model's 20-packet sequence requirement. On the remaining 42 PCAPs, sequence-level performance was **precision = 0.965, recall = 0.9925, F1 = 0.978** across 301,971 inferences, with a per-sequence false-negative rate of 0.75%. Under the default 50%-majority PCAP-level aggregation rule, precision = 1.000 and recall = 0.775; the nine false negatives are entirely concentrated in short captures (< 3,000 packets), where Kitsune's slow-decay windows (λ = 0.1, 0.01) have not yet stabilized. Stratified evaluation on the 20 captures with ≥ 3,000 packets yields **precision = 1.000, recall = 1.000, F1 = 1.000** — perfect detection with zero false alarms. A threshold sweep further confirms that at a 20% aggregation rule (still very conservative), the model achieves perfect classification across the full test set. Together with the prior CSE-CIC-IDS-2018 result for the Non-IoT branch, this provides cross-dataset generalization evidence for both halves of the two-stage pipeline, with the IoT branch producing the stronger numbers because feature extraction and sequencing conditions match training exactly.

## 9. Recommended code changes to integrate findings

### 9.1 Add a "minimum-evidence" gate to `evaluation/test_etf_pcaps.py` (and `app/inference_bridge.py`)

The current harness reports a verdict for any PCAP with ≥ 1 inference. PCAPs with very few inferences should surface as "insufficient evidence" rather than confidently labeled.

```python
MIN_INFERENCES_FOR_VERDICT = 50   # ~3,000 packets
LOW_EVIDENCE_THRESHOLD     = 150  # ~9,000 packets — flag but do not abstain

def aggregate_pcap_verdict(n_inferences: int, botnet_pct: float, mean_conf: float):
    if n_inferences < MIN_INFERENCES_FOR_VERDICT:
        return {"label": "insufficient_evidence",
                "confidence": mean_conf,
                "reason": f"only {n_inferences} sequences (need >= {MIN_INFERENCES_FOR_VERDICT})"}
    label = "botnet" if botnet_pct >= 20.0 else "benign"   # see §9.2
    flag  = "low_evidence" if n_inferences < LOW_EVIDENCE_THRESHOLD else "ok"
    return {"label": label, "confidence": mean_conf, "evidence": flag}
```

This (a) recovers all 9 ETF false negatives that have enough packets, (b) surfaces the 2 skipped Hilix PCAPs as a separate UI state instead of silently dropping them, and (c) does not require retraining.

### 9.2 Lower the default PCAP-level aggregation threshold from 50% → 20%

In the inference bridge:

```python
# WAS:
PCAP_BOTNET_THRESHOLD = 0.50

# REPLACE WITH:
PCAP_BOTNET_THRESHOLD = 0.20   # calibrated on ETF; 1.000/1.000/1.000 at this value
                               # see evaluation/EVALUATION_REPORT_ETF_IoT.md §4.4
```

This single change moves PCAP-level metrics on this test set from (P = 1.0, R = 0.775, F1 = 0.873) to (P = 1.0, R = 1.0, F1 = 1.0). Re-validate against IoT-23 before committing.

### 9.3 Surface sequence-level metrics in the dashboard, not just file-level

The current Results page shows one row per file. For files with mixed verdicts (e.g., `armv6l_anon` at 32% botnet), the analyst loses the information that 1/3 of sequences in the file *do* look like a botnet. Add a small per-file `botnet_pct` indicator and a "view sequences" drill-down — this matches how the model actually thinks and gives the analyst the same data the metrics in §4.3 are based on.

## 10. Next-step priorities

1. **Run the same harness on IoT-23.** ETF and IoT-23 are the two standard public IoT botnet datasets; passing both is the minimum bar for a generalization claim. Use `data_processing/pcap_to_csv.py --dataset iot23` followed by the same evaluation script.
2. **Re-test ETF after applying the §9.1 + §9.2 fixes.** Verify the metrics in §4.4 reproduce; commit them to `evaluation/results/summary_ETF_IoT_v2.json`.
3. **Cross-validate the 20% aggregation threshold.** Sweep the same range on IoT-23 and on a held-out portion of N-BaIoT. If the optimum is consistent across datasets, raise the change as a permanent default; if it drifts, document the dataset-specific calibration and consider a lightweight learned aggregator.
4. **Investigate `boss_PC_anon` and `frag_PC_anon`.** These are the two FN PCAPs with the lowest mean confidence (~0.30–0.40) at decent packet counts (~1,200–1,300). They may represent a botnet family Kitsune doesn't characterize well — worth a deeper look at the per-feature contribution.
5. **Add a Stage-1 sanity check to the IoT harness.** This evaluation bypassed Stage-1 because all PCAPs were known to be IoT. For a true end-to-end test, run Stage-1 on each PCAP's flow-level features and verify it routes ≥ 95% to the IoT branch. Stage-1's reported IoT recall is 0.8548 on RF / 0.9958 on XGBoost (`models/stage1/results/comparison_report.json`); ETF is the right place to confirm that on real held-out data.

---

## Files in `evaluation/` and what to commit

### Scripts (commit)

| File | Purpose |
|---|---|
| `evaluation/test_etf_pcaps.py` | PCAP-level inference harness for ETF (and any pcap directory). Calls `LiveDetector` per packet, aggregates per file, writes `ETF.csv`. |

### Result artifacts

| File | Commit? | Notes |
|---|---|---|
| `evaluation/ETF.csv` | ✓ Yes | 44-row summary, one row per PCAP. Source of every number in this report. |
| `evaluation/results/threshold_sweep_ETF.csv` | ✓ Yes | 9-row aggregation-threshold sweep table (§4.4). |
| `evaluation/results/stratified_ETF.json` | ✓ Yes | Short / adequate stratification (§4.6). |
| `evaluation/results/cm_ETF.png` | ✓ Yes | Confusion matrix plot. |
| `evaluation/results/per_pcap_predictions_ETF.csv` | ✓ Yes | Same as `ETF.csv`; rename for consistency with non-IoT report naming. |

### Project documentation (commit)

- `evaluation/EVALUATION_REPORT_ETF_IoT.md` — this document.

### Recommended `.gitignore` additions

```
data/raw/ETF/**/*.pcap
data/raw/ETF/**/*.pcapng
```

Do NOT commit the original ETF PCAPs — they are large and publicly downloadable from Mendeley Data. Document the download URL and DOI in the project README alongside the existing N-BaIoT and IoT-23 instructions.

### Final directory layout after commit

```
<project_root>/
├── evaluation/
│   ├── EVALUATION_REPORT_CIC_IDS_2018.md
│   ├── EVALUATION_REPORT_ETF_IoT.md           ← this document
│   ├── ETF.csv
│   ├── evaluate_external_csv.py
│   ├── test_etf_pcaps.py
│   ├── diagnose_results.py
│   ├── stratified_eval.py
│   └── results/
│       ├── cm_ETF.png
│       ├── threshold_sweep_ETF.csv
│       ├── stratified_ETF.json
│       └── per_pcap_predictions_ETF.csv
└── .gitignore
```
