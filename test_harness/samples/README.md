# Sample run

`sample_run/reports/` was captured by running:

```bash
BOTNET_PROJECT_ROOT=/path/to/minimal_stub \
    python -m test_harness.orchestrator --phase A --skip-models --skip-qt
```

Against a minimal stand-in repo with a correct `monitoring.S1_FEATURES`
and a magic-byte-aware `app/file_handler.load_file`. The run executes
the 9 tests of Phase A that don't need the trained models. The 4
PCAP-replay tests are recorded as SKIPPED because of `--skip-models`.

When you point the harness at the real Group-07 repository you'll see:

- M-00 — should still PASS (schema unchanged in the project)
- F-* — generally PASS unless `app.file_handler` is changed
- P-04 — expected FAIL: IPv6 packets are silently dropped
- P-07 — expected FAIL or PASS depending on whether the aggregator
  guards against negative IATs
- M-EF — expected PASS (i.e. confirms the documented near-zero output
  on empty flows; the test PASSES when the bug is reproducible — we
  flip it once the bug is fixed)
- M-WS — expected FAIL: window starvation cuts recall under unique-IP
  flows
- S1-06 — expected FAIL if MAC OUI override is active without behavioural
  gating
- SC-K — expected FAIL: KitsuneExtractor has no LRU eviction

The sample reports below are the PASS-only baseline; the FAIL versions
will look the same shape but with populated `failures.json` and a much
longer `recommendation_report.md`.
