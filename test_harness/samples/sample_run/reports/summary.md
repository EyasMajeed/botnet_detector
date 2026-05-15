# Test Harness — Run Summary

_Generated_: `2026-05-10T12:27:24.350382Z`

## Totals

- **Tests run**: 13
- **PASS**: 9
- **FAIL**: 0
- **ERROR**: 0
- **SKIPPED**: 4

## Phase A

| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |
|----|------|----------|---------|----------|----------|-------|
| P-02 | Malformed Ethernet | HIGH | SKIPPED | 0.0s | - | skipped via --skip-models |
| P-03 | VLAN-tagged frames | HIGH | SKIPPED | 0.0s | - | skipped via --skip-models |
| P-04 | IPv6-only flows | HIGH | SKIPPED | 0.0s | - | skipped via --skip-models |
| P-07 | Backwards timestamps | HIGH | SKIPPED | 0.0s | - | skipped via --skip-models |
| M-00 | Schema drift | CRITICAL | PASS | 0.3s | 9 MB | both lists have 56 features in same order |
| F-02 | Claimed-huge PCAP header | HIGH | PASS | 0.3s | 6 MB | detected as PCAP without parsing body: {'is_valid': True, 'format': 'PCAP', 'error': ''... |
| F-07 | UTF-16 CSV | HIGH | PASS | 0.3s | 5 MB | validator flagged or rejected: {'is_valid': False, 'format': 'UNKNOWN', 'error': 'unrec... |
| F-01 | Truncated PCAPNG | MEDIUM | PASS | 0.3s | 5 MB | validator handled cleanly: {'is_valid': True, 'format': 'PCAPNG', 'error': '', 'size_mb... |
| F-03 | Oversized CSV | MEDIUM | PASS | 7.3s | 79 MB | detected: {'is_valid': True, 'format': 'GENERIC_CSV', 'error': '', 'size_mb': 50.000163... |
| F-04 | Header-only CSV | MEDIUM | PASS | 0.3s | 1 MB | validator did not crash: {'is_valid': True, 'format': 'GENERIC_CSV', 'error': '', 'size... |
| F-08 | Parser fuzzing | MEDIUM | PASS | 0.5s | 1 MB | 32/32 fuzz inputs handled cleanly |
| F-09 | Zip bomb disguised as pcap | MEDIUM | PASS | 0.3s | 3 MB | correctly not classified as PCAP: {'is_valid': True, 'format': 'GENERIC_CSV', 'error': ... |
| F-05 | PCAP renamed .csv | LOW | PASS | 0.3s | 4 MB | magic-byte detection wins over extension: {'is_valid': True, 'format': 'PCAP', 'error':... |
