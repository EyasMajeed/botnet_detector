# Test Harness — Recommendation Report

_Generated_: `2026-05-10T12:27:24.350995Z`

## Headline

- 9 PASS / 0 FAIL / 0 ERROR / 4 SKIP, 13 total.

All non-skipped tests passed. See `summary.md` for full results and `performance.csv` for resource profiles.
## Skipped tests

- P-02: skipped via --skip-models
- P-03: skipped via --skip-models
- P-04: skipped via --skip-models
- P-07: skipped via --skip-models

## Performance hotspots

**Slowest 5 tests:**
- F-03 Oversized CSV: 7.3s
- F-08 Parser fuzzing: 0.5s
- M-00 Schema drift: 0.3s
- F-02 Claimed-huge PCAP header: 0.3s
- F-01 Truncated PCAPNG: 0.3s

**Top 5 by peak RSS:**
- F-03 Oversized CSV: 79 MB
- M-00 Schema drift: 9 MB
- F-02 Claimed-huge PCAP header: 6 MB
- F-07 UTF-16 CSV: 5 MB
- F-01 Truncated PCAPNG: 5 MB

## Files for downstream review

- `reports/summary.json`     – machine-readable, full payload
- `reports/summary.md`       – tabular overview by phase
- `reports/failures.json`    – just the failures, sorted
- `reports/performance.csv`  – per-test duration + CPU/RSS peaks
- `reports/memory_profile.csv`– per-test memory sampling info
- `logs/<test_id>/`          – stdout, stderr, run.log, result.json
- `artifacts/<test_id>/`     – PCAPs, CSVs, sweep tables, resources.csv