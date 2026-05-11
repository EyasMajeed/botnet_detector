# Test Harness — Run Summary

_Generated_: `2026-05-10T12:28:24.591694Z`

## Totals

- **Tests run**: 16
- **PASS**: 7
- **FAIL**: 9
- **ERROR**: 0
- **SKIPPED**: 0

## Phase A

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| P-04 | IPv6-only flows | HIGH | FAIL | 11.7s | 412 MB | All IPv6 packets silently dropped: seen=180, results=0. Confirms the IPv6 blind spot. |
| P-03 | VLAN-tagged frames | HIGH | FAIL | 4.1s | 380 MB | only 0/200 (0.0%) processed - VLAN handling likely missing |
| M-00 | Schema drift | CRITICAL | PASS | 0.4s | 9 MB | both lists have 56 features in same order |
| P-07 | Backwards timestamps | HIGH | PASS | 6.8s | 410 MB | 97 flows produced; no anomalies |
| F-08 | Parser fuzzing | MEDIUM | PASS | 0.6s | 4 MB | 32/32 fuzz inputs handled cleanly |

## Phase B

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| M-WS | Window starvation | HIGH | FAIL | 42.6s | 1820 MB | F1 drops 24.0 pp under unique-IP starvation: 0.91 → 0.67; recall drops 31.0 pp. |
| M-CAL | Confidence calibration | MEDIUM | FAIL | 19.4s | 1400 MB | ECE = 0.176 exceeds 0.10; model confidence is not well-calibrated. |
| M-EF | Empty-flow regression | HIGH | PASS | 8.2s | 1320 MB | avg confidence 0.0070 on empty flows — matches documented ≈0.0068. |

## Phase C

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| S1-06 | MAC-OUI spoofing | CRITICAL | FAIL | 3.9s | 1180 MB | 100% of spoofed-MAC flows routed to IoT branch without behavioural verification. |
| SC-K | Kitsune key explosion | HIGH | FAIL | 28.1s | 2100 MB | Kitsune state unbounded: hh=8000, hphp=8000, h=1 (cap=5000). No LRU/TTL in extractor. |
| RT-1 | Slow beaconing botnet | HIGH | FAIL | 5.6s | 1290 MB | botnet rate only 6.7%; slow beacons defeat 30-s flow idle. |
| M-ADV | Scaler-aware FGSM | HIGH | FAIL | 33.7s | 1560 MB | max flip rate 78% — Stage-1 is brittle to scaler-aware perturbations. |
| X-A1 | XAI sanity battery | HIGH | PASS | 27.3s | 1610 MB | all stability metrics within target |

## Phase D

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| ST-2 | Packet flood throughput | MEDIUM | FAIL | 14.5s | 1380 MB | 10000 packets in 14.20s = 704 pps |
| G-DS | DetectionStore behaviour | MEDIUM | PASS | 1.2s | 290 MB | size=50000 (cap=50000); apply_threshold ok=True |
| G-RU | Repeated upload cycles | MEDIUM | PASS | 41.0s | 1460 MB | RSS slope 1.30 MB/iter post-warmup |
