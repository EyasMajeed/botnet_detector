# Test Harness — Recommendation Report

_Generated_: `2026-05-10T12:28:24.592677Z`

## Headline

- 7 PASS / 9 FAIL / 0 ERROR / 0 SKIP, 16 total.

## Failures, ordered by severity

### CRITICAL

- **S1-06 — MAC-OUI spoofing**  _(theme: Stage-1 routing exploit)_
    - expected: iot routing rate < 50% on workstation-cadence flows
    - actual:   100% of spoofed-MAC flows routed to IoT branch without behavioural verification.
    - so what:  OUI override is bypass-able via MAC spoofing — workstation traffic gets routed to IoT branch with no behavioural gate.

### HIGH

- **P-04 — IPv6-only flows**  _(theme: IPv6 blind spot)_
    - expected: IPv6 flows produce >=1 result OR explicit log of unsupported
    - actual:   All IPv6 packets silently dropped: seen=180, results=0. Confirms the IPv6 blind spot.
    - so what:  live_detector and PCAP path silently drop IPv6, matching the documented `if IP not in pkt: return` bug.
- **P-03 — VLAN-tagged frames**  _(theme: Packet handling)_
    - expected: ≥80% of VLAN packets reach process_packet
    - actual:   only 0/200 (0.0%) processed - VLAN handling likely missing
    - so what:  VLAN-tagged traffic dropped by IP-only path.
- **M-WS — Window starvation**  _(theme: Window starvation)_
    - expected: F1 drop ≤ 0.05 between balanced and starved
    - actual:   F1 drops 24.0 pp under unique-IP starvation: 0.91 → 0.67; recall drops 31.0 pp.
    - so what:  Stage-2 Non-IoT detector pads to seq_len=20 with zeros; unique-per-flow src_ip distributions never appeared in training and recall collapses.
- **SC-K — Kitsune key explosion**  _(theme: Memory exhaustion)_
    - expected: hh and hphp key counts ≤ 5000 via LRU/TTL
    - actual:   Kitsune state unbounded: hh=8000, hphp=8000, h=1 (cap=5000). No LRU/TTL in extractor.
    - so what:  KitsuneExtractor _hh / _hphp dictionaries grow without bound — long-running live capture leaks state continuously.
- **RT-1 — Slow beaconing botnet**  _(theme: Detection blind spot — slow beaconing)_
    - expected: botnet detection rate ≥ 50%
    - actual:   botnet rate only 6.7%; slow beacons defeat 30-s flow idle.
    - so what:  Beacons spaced > FlowAggregator._idle (30 s) become separate one-row flows; LSTM has no temporal context across them.
- **M-ADV — Scaler-aware FGSM**  _(theme: Adversarial brittleness)_
    - expected: max routing-flip rate ≤ 30% at any tested ε
    - actual:   max flip rate 78% — Stage-1 is brittle to scaler-aware perturbations.
    - so what:  Stage-1 routing flips at ε=0.01 in scaled space — the scaler.json on disk is enough to weaponise this.

### MEDIUM

- **M-CAL — Confidence calibration**  _(theme: Confidence calibration)_
    - expected: ECE ≤ 0.10
    - actual:   ECE = 0.176 exceeds 0.10; model confidence is not well-calibrated.
    - so what:  ECE > 0.10 means the probabilities the GUI shows users do not reflect the real per-bin positive rate.
- **ST-2 — Packet flood throughput**  _(theme: Throughput)_
    - expected: ≥ 1000 pps
    - actual:   10000 packets in 14.20s = 704 pps
    - so what:  process_packet hot path falls below 1 kpps under burst.

## Failures by theme

- Stage-1 routing exploit: 1 failure(s)
- IPv6 blind spot: 1 failure(s)
- Packet handling: 1 failure(s)
- Window starvation: 1 failure(s)
- Memory exhaustion: 1 failure(s)
- Detection blind spot — slow beaconing: 1 failure(s)
- Adversarial brittleness: 1 failure(s)
- Confidence calibration: 1 failure(s)
- Throughput: 1 failure(s)

## Performance hotspots

**Slowest 5 tests:**
- M-WS Window starvation: 42.6s
- G-RU Repeated upload cycles: 41.0s
- M-ADV Scaler-aware FGSM: 33.7s
- SC-K Kitsune key explosion: 28.1s
- X-A1 XAI sanity battery: 27.3s

**Top 5 by peak RSS:**
- SC-K Kitsune key explosion: 2100 MB
- M-WS Window starvation: 1820 MB
- X-A1 XAI sanity battery: 1610 MB
- M-ADV Scaler-aware FGSM: 1560 MB
- G-RU Repeated upload cycles: 1460 MB

## Files for downstream review

- `reports/summary.json`     – machine-readable, full payload
- `reports/summary.md`       – tabular overview by phase
- `reports/failures.json`    – just the failures, sorted
- `reports/performance.csv`  – per-test duration + CPU/RSS peaks
- `reports/memory_profile.csv`– per-test memory sampling info
- `logs/<test_id>/`          – stdout, stderr, run.log, result.json
- `artifacts/<test_id>/`     – PCAPs, CSVs, sweep tables, resources.csv