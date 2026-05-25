# Test Harness — Recommendation Report

_Generated_: `2026-05-17T18:11:00.086763Z`

## Headline

- 25 PASS / 8 FAIL / 0 ERROR / 1 SKIP, 34 total.

## Failures, ordered by severity

### HIGH

- **M-EF — Empty-flow regression**  _(theme: Scaler defect (documented))_
    - expected: avg confidence < 0.05
    - actual:   avg confidence 0.3188 on empty flows is no longer near zero — scaler may have been re-fit (good) OR something else changed.
    - so what:  Empty flows produce a constant near-zero probability — Non-IoT scaler fitted on already-normalised data.
- **M-WS — Window starvation**  _(theme: Window starvation)_
    - expected: F1 drop ≤ 0.05 between balanced and starved
    - actual:   F1 drops 62.7 pp under unique-IP starvation: 0.6273 → 0.0; recall drops 85.0 pp (0.85 → 0.0). The LSTM has not seen 1-real + 19-padded sequences at training time.
    - so what:  Stage-2 Non-IoT detector pads to seq_len=20 with zeros; unique-per-flow src_ip distributions never appeared in training and recall collapses.
- **RT-1 — Slow beaconing botnet (multi-session)**  _(theme: Detection blind spot — slow beaconing)_
    - expected: detection rate ≥ 50%
    - actual:   only 0/25 (0%) flagged botnet on 25 sessions — model does not recognise the slow-beacon shape even after _idle=120s fix
    - so what:  Beacons across multiple sessions from one src_ip should produce a complete LSTM window after the project raised _idle to 120 s. If this still fails, the model has not learned to flag short flows even given a real window.
- **RT-1b — Slow-C2 evasion (period > idle)**  _(theme: Detection blind spot — slow C2 evasion)_
    - expected: detection rate ≥ 30%
    - actual:   only 0/25 (0%) flagged botnet — slow-C2 evasion at period > idle is effective. Detection requires cross-session temporal aggregation per src_ip; the LSTM's per-flow input window cannot see across flows.
    - so what:  Sessions spaced LONGER than _idle (e.g. 5 minutes) cannot be detected with per-flow LSTM input alone. Cross-flow temporal aggregation per src_ip is required.
- **M-ADV — Scaler-aware adversarial perturbation**  _(theme: Adversarial brittleness)_
    - expected: max routing-flip rate ≤ 30% at any tested ε
    - actual:   max flip rate 100% — Stage-1 is brittle to scaler-aware feature-space perturbations.
    - so what:  Stage-1 routing flips at ε=0.01 in scaled space — the scaler.json on disk is enough to weaponise this.

### MEDIUM

- **M-T1 — Threshold sweep (synthetic)**  _(theme: ML calibration)_
    - expected: F1 ≥ 0.80 at some threshold; precision and recall both ≥ 0.80
    - actual:   best F1 at τ=0.1 only P=0.2828, R=0.205; below 0.80
    - so what:  operating threshold should be picked from a sweep, not hard-coded to whatever the training script wrote.
- **M-CAL — Confidence calibration**  _(theme: Confidence calibration)_
    - expected: ECE ≤ 0.1
    - actual:   ECE = 0.4572 exceeds 0.1; model confidence is not well-calibrated.
    - so what:  ECE > 0.10 means the probabilities the GUI shows users do not reflect the real per-bin positive rate.
- **ST-2b — End-to-end throughput**  _(theme: Throughput — end-to-end with XAI)_
    - expected: end-to-end ≥ 200 pps
    - actual:   end-to-end only 69 pps (ingest is fine at 34526 pps, but flush of 10000 flows took 145.5s = 69 flow/s). The bottleneck is flow finalisation: Stage-2 inference + per-flow XAI.
    - so what:  End-to-end pipeline must keep up with the rate flows complete. Per-flow Integrated Gradients XAI is expensive (~30 ms/flow CPU). Common fix: gate XAI on label=='botnet' or run XAI lazily on GUI request only.

## Failures by theme

- Scaler defect (documented): 1 failure(s)
- Window starvation: 1 failure(s)
- Detection blind spot — slow beaconing: 1 failure(s)
- Detection blind spot — slow C2 evasion: 1 failure(s)
- Adversarial brittleness: 1 failure(s)
- ML calibration: 1 failure(s)
- Confidence calibration: 1 failure(s)
- Throughput — end-to-end with XAI: 1 failure(s)

## Skipped tests

- S1-06: only 1 result(s) from 200 packets (200 processed, 0 per-packet errors). Insufficient data — fix replay first.

## Performance hotspots

**Slowest 5 tests:**
- M-ADV Scaler-aware adversarial perturbation: 375.5s
- SC-K Kitsune key explosion: 152.5s
- ST-2b End-to-end throughput: 147.5s
- ST-1 Large PCAP throughput: 133.3s
- SE-A SYN flood: 74.9s

**Top 5 by peak RSS:**
- ST-1 Large PCAP throughput: 2063 MB
- SC-K Kitsune key explosion: 578 MB
- ST-4 Concurrent uploads: 487 MB
- ST-2b End-to-end throughput: 456 MB
- ST-3 Long-duration monitoring: 448 MB

## Files for downstream review

- `reports/summary.json`     – machine-readable, full payload
- `reports/summary.md`       – tabular overview by phase
- `reports/failures.json`    – just the failures, sorted
- `reports/performance.csv`  – per-test duration + CPU/RSS peaks
- `reports/memory_profile.csv`– per-test memory sampling info
- `logs/<test_id>/`          – stdout, stderr, run.log, result.json
- `artifacts/<test_id>/`     – PCAPs, CSVs, sweep tables, resources.csv