# Test Harness — Run Summary

_Generated_: `2026-05-17T18:11:00.085639Z`

## Totals

- **Tests run**: 34
- **PASS**: 25
- **FAIL**: 8
- **ERROR**: 0
- **SKIPPED**: 1

## Phase A

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| M-00 | Schema drift | CRITICAL | PASS | 1.3s | 270 MB | both lists have 56 features in same order |
| F-02 | Claimed-huge PCAP header | HIGH | PASS | 0.8s | 188 MB | detected as PCAP without parsing body: {'is_valid': True, 'format': 'FileFormat.PCAP', ... |
| F-07 | UTF-16 CSV | HIGH | PASS | 0.8s | 187 MB | validator flagged or rejected: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', '... |
| P-02 | Malformed Ethernet frames | HIGH | PASS | 1.8s | 380 MB | 0/50 packets processed; 0 per-packet errors logged |
| P-03 | VLAN-tagged frames | HIGH | PASS | 4.8s | 390 MB | 200/200 (100.0%) processed |
| P-04 | IPv6-only flows | HIGH | PASS | 2.3s | 386 MB | 30 IPv6 flow(s) produced results |
| P-07 | Backwards timestamps | HIGH | PASS | 9.2s | 391 MB | 500 flows produced; no anomalies |
| F-01 | Truncated PCAPNG | MEDIUM | PASS | 0.8s | 200 MB | validator handled cleanly: {'is_valid': True, 'format': 'FileFormat.PCAPNG', 'error': '... |
| F-03 | Oversized CSV | MEDIUM | PASS | 3.3s | 200 MB | detected: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', 'error': '', 'size_mb'... |
| F-04 | Header-only CSV | MEDIUM | PASS | 0.8s | 186 MB | validator did not crash: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', 'error'... |
| F-08 | Parser fuzzing | MEDIUM | PASS | 0.8s | 186 MB | 32/32 fuzz inputs handled cleanly |
| F-09 | Zip bomb disguised as pcap | MEDIUM | PASS | 0.8s | 200 MB | correctly not classified as PCAP: {'is_valid': False, 'format': 'FileFormat.UNKNOWN', '... |
| F-05 | PCAP renamed .csv | LOW | PASS | 0.8s | 186 MB | magic-byte detection wins over extension: {'is_valid': True, 'format': 'FileFormat.PCAP... |

## Phase B

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| M-EF | Empty-flow regression | HIGH | FAIL | 7.4s | 427 MB | avg confidence 0.3188 on empty flows is no longer near zero — scaler may have been re-f... |
| M-WS | Window starvation | HIGH | FAIL | 30.2s | 442 MB | F1 drops 62.7 pp under unique-IP starvation: 0.6273 → 0.0; recall drops 85.0 pp (0.85 →... |
| M-T1 | Threshold sweep (synthetic) | MEDIUM | FAIL | 16.5s | 437 MB | best F1 at τ=0.1 only P=0.2828, R=0.205; below 0.80 |
| M-CAL | Confidence calibration | MEDIUM | FAIL | 42.7s | 432 MB | ECE = 0.4572 exceeds 0.1; model confidence is not well-calibrated. |

## Phase C

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| RT-1 | Slow beaconing botnet (multi-session) | HIGH | FAIL | 2.3s | 385 MB | only 0/25 (0%) flagged botnet on 25 sessions — model does not recognise the slow-beacon... |
| RT-1b | Slow-C2 evasion (period > idle) | HIGH | FAIL | 2.3s | 386 MB | only 0/25 (0%) flagged botnet — slow-C2 evasion at period > idle is effective. Detectio... |
| M-ADV | Scaler-aware adversarial perturbation | HIGH | FAIL | 375.5s | 375 MB | max flip rate 100% — Stage-1 is brittle to scaler-aware feature-space perturbations. |
| S1-06 | MAC-OUI spoofing override | CRITICAL | SKIPPED | 1.8s | 387 MB | only 1 result(s) from 200 packets (200 processed, 0 per-packet errors). Insufficient da... |
| SE-A | SYN flood | HIGH | PASS | 74.9s | 415 MB | scorer.score=7, trigger_sniff=True; replay processed 5000 packets |
| SC-K | Kitsune key explosion | HIGH | PASS | 152.5s | 578 MB | hh=10000, hphp=10000, h=1 — bounded under cap=10000 after 12000 unique destinations (ev... |
| X-A1 | XAI sanity battery | HIGH | PASS | 13.5s | 381 MB | all stability metrics within target: {'repeat_stability_rate': 1.0, 'perturb_stability_... |
| P-05 | GRE tunnel handling | MEDIUM | PASS | 3.6s | 386 MB | 100/100 packets processed |

## Phase D

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| ST-2b | End-to-end throughput | MEDIUM | FAIL | 147.5s | 456 MB | end-to-end only 69 pps (ingest is fine at 34526 pps, but flush of 10000 flows took 145.... |
| G-IT | PcapInferenceThread | HIGH | PASS | 2.1s | 407 MB | done hit; n_results=10 |
| G-DS | DetectionStore behaviour | MEDIUM | PASS | 5.6s | 254 MB | size=50000 (cap=50000); relabel: changed_at_high=7143, changed_at_zero=50000 |
| G-AF | Alert flood | MEDIUM | PASS | 9.7s | 269 MB | 100000 rows inserted in 8.91s; flows_changed fired 20 times; store retained 50000 flows |
| G-RU | Repeated upload cycles | MEDIUM | PASS | 24.1s | 385 MB | RSS slope 0.04 MB/iter post-warmup (samples: [383.4, 384.0, 384.0, 384.0, 384.3, 384.3,... |
| ST-1 | Large PCAP throughput | MEDIUM | PASS | 133.3s | 2063 MB | 201000 packets in 41.9s (4792 pps) |
| ST-2a | Ingestion throughput (process_packet only) | MEDIUM | PASS | 2.3s | 390 MB | 10000 packets in 0.24s = 41073 pps (0.024 ms/call) |
| ST-3 | Long-duration monitoring | MEDIUM | PASS | 33.0s | 448 MB | 16 iterations, 33600 packets processed |
| ST-4 | Concurrent uploads | MEDIUM | PASS | 6.4s | 487 MB | 12 runs in 5.41s, avg 1.80s/run |
