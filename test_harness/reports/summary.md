# Test Harness — Run Summary

_Generated_: `2026-05-10T17:21:37.275384Z`

## Totals

- **Tests run**: 34
- **PASS**: 25
- **FAIL**: 8
- **ERROR**: 0
- **SKIPPED**: 1

## Phase A

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| M-00 | Schema drift | CRITICAL | PASS | 1.3s | 337 MB | both lists have 56 features in same order |
| F-02 | Claimed-huge PCAP header | HIGH | PASS | 1.0s | 260 MB | detected as PCAP without parsing body: {'is_valid': True, 'format': 'FileFormat.PCAP', ... |
| F-07 | UTF-16 CSV | HIGH | PASS | 1.0s | 260 MB | validator flagged or rejected: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', '... |
| P-02 | Malformed Ethernet frames | HIGH | PASS | 2.0s | 464 MB | 0/50 packets processed; 0 per-packet errors logged |
| P-03 | VLAN-tagged frames | HIGH | PASS | 4.8s | 482 MB | 200/200 (100.0%) processed |
| P-04 | IPv6-only flows | HIGH | PASS | 2.5s | 478 MB | 30 IPv6 flow(s) produced results |
| P-07 | Backwards timestamps | HIGH | PASS | 8.9s | 483 MB | 500 flows produced; no anomalies |
| F-01 | Truncated PCAPNG | MEDIUM | PASS | 1.0s | 261 MB | validator handled cleanly: {'is_valid': True, 'format': 'FileFormat.PCAPNG', 'error': '... |
| F-03 | Oversized CSV | MEDIUM | PASS | 2.8s | 260 MB | detected: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', 'error': '', 'size_mb'... |
| F-04 | Header-only CSV | MEDIUM | PASS | 1.0s | 260 MB | validator did not crash: {'is_valid': True, 'format': 'FileFormat.CSV_GENERIC', 'error'... |
| F-08 | Parser fuzzing | MEDIUM | PASS | 1.0s | 260 MB | 32/32 fuzz inputs handled cleanly |
| F-09 | Zip bomb disguised as pcap | MEDIUM | PASS | 1.0s | 260 MB | correctly not classified as PCAP: {'is_valid': False, 'format': 'FileFormat.UNKNOWN', '... |
| F-05 | PCAP renamed .csv | LOW | PASS | 1.0s | 260 MB | magic-byte detection wins over extension: {'is_valid': True, 'format': 'FileFormat.PCAP... |

## Phase B

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| M-EF | Empty-flow regression | HIGH | FAIL | 6.9s | 502 MB | avg confidence 0.3188 on empty flows is no longer near zero — scaler may have been re-f... |
| M-WS | Window starvation | HIGH | FAIL | 27.5s | 510 MB | F1 drops 62.7 pp under unique-IP starvation: 0.6273 → 0.0; recall drops 85.0 pp (0.85 →... |
| M-T1 | Threshold sweep (synthetic) | MEDIUM | FAIL | 15.5s | 509 MB | best F1 at τ=0.1 only P=0.2828, R=0.205; below 0.80 |
| M-CAL | Confidence calibration | MEDIUM | FAIL | 38.4s | 509 MB | ECE = 0.4572 exceeds 0.1; model confidence is not well-calibrated. |

## Phase C

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| RT-1 | Slow beaconing botnet (multi-session) | HIGH | FAIL | 2.4s | 477 MB | only 0/25 (0%) flagged botnet on 25 sessions — model does not recognise the slow-beacon... |
| RT-1b | Slow-C2 evasion (period > idle) | HIGH | FAIL | 2.3s | 478 MB | only 0/25 (0%) flagged botnet — slow-C2 evasion at period > idle is effective. Detectio... |
| M-ADV | Scaler-aware adversarial perturbation | HIGH | FAIL | 409.4s | 449 MB | max flip rate 100% — Stage-1 is brittle to scaler-aware feature-space perturbations. |
| S1-06 | MAC-OUI spoofing override | CRITICAL | SKIPPED | 2.3s | 468 MB | only 1 result(s) from 200 packets (200 processed, 0 per-packet errors). Insufficient da... |
| SE-A | SYN flood | HIGH | PASS | 69.0s | 508 MB | scorer.score=7, trigger_sniff=True; replay processed 5000 packets |
| SC-K | Kitsune key explosion | HIGH | PASS | 145.1s | 671 MB | hh=10000, hphp=10000, h=1 — bounded under cap=10000 after 12000 unique destinations (ev... |
| X-A1 | XAI sanity battery | HIGH | PASS | 11.9s | 453 MB | all stability metrics within target: {'repeat_stability_rate': 1.0, 'perturb_stability_... |
| P-05 | GRE tunnel handling | MEDIUM | PASS | 3.6s | 479 MB | 100/100 packets processed |

## Phase D

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| ST-2b | End-to-end throughput | MEDIUM | FAIL | 155.8s | 552 MB | end-to-end only 65 pps (ingest is fine at 39346 pps, but flush of 10000 flows took 153.... |
| G-IT | PcapInferenceThread | HIGH | PASS | 2.3s | 490 MB | done hit; n_results=10 |
| G-DS | DetectionStore behaviour | MEDIUM | PASS | 4.7s | 325 MB | size=50000 (cap=50000); relabel: changed_at_high=7143, changed_at_zero=50000 |
| G-AF | Alert flood | MEDIUM | PASS | 8.0s | 340 MB | 100000 rows inserted in 6.82s; flows_changed fired 20 times; store retained 50000 flows |
| G-RU | Repeated upload cycles | MEDIUM | PASS | 25.4s | 457 MB | RSS slope 0.00 MB/iter post-warmup (samples: [456.2, 456.8, 456.8, 456.8, 456.8, 456.8,... |
| ST-1 | Large PCAP throughput | MEDIUM | PASS | 110.7s | 2187 MB | 201000 packets in 35.9s (5606 pps) |
| ST-2a | Ingestion throughput (process_packet only) | MEDIUM | PASS | 2.6s | 481 MB | 10000 packets in 0.25s = 39821 pps (0.025 ms/call) |
| ST-3 | Long-duration monitoring | MEDIUM | PASS | 32.3s | 574 MB | 15 iterations, 31500 packets processed |
| ST-4 | Concurrent uploads | MEDIUM | PASS | 8.2s | 564 MB | 12 runs in 6.96s, avg 2.32s/run |
