# Test Harness — Hybrid AI-Based Botnet Detection (Group 07, CPCS499)

Automated validation framework for the project repository. Generates
malicious / malformed / synthetic traffic, replays it through the
project's own modules, captures crashes, measures performance, and
produces machine-readable reports for downstream analysis.

The harness is designed to be dropped into the project repo at
`<repo_root>/test_harness/` and invoked from there. It can also live
elsewhere if you set `BOTNET_PROJECT_ROOT`.

```
test_harness/
├── orchestrator.py          # main entrypoint
├── configs/registry.py      # test catalogue (single source of truth)
├── generators/              # PCAP and CSV generators
├── parsers/                 # tests targeting app/file_handler.load_file
├── pcap_tests/              # tests that replay PCAPs through BotnetMonitor
├── ml_tests/                # threshold sweeps, calibration, drift, FGSM, etc.
├── xai_tests/               # Integrated Gradients sanity battery
├── gui_tests/               # DetectionStore / inference_worker / alert flood
├── stress_tests/            # large PCAP, packet flood, long-run, concurrent
├── reports/                 # generators of summary.* / failures.* / etc.
├── utils/                   # isolation, logging, paths, resource_monitor
├── outputs/run_<ts>/        # per-run output directory (auto-created)
├── logs/<test_id>/          # stdout, stderr, run.log, result.json
└── artifacts/<test_id>/     # PCAPs, CSVs, sweeps, resource time-series
```

## 1. Installation

```bash
# In your project repo:
git clone <this harness> test_harness
cd test_harness
pip install -r requirements.txt
```

Minimum required: `scapy`, `psutil`. Everything else degrades gracefully —
tests that need an absent dependency are recorded as `SKIPPED` with a
clear reason instead of failing.

## 2. Quick start

```bash
# From the project root:
python -m test_harness.orchestrator --list             # list every test in the registry
python -m test_harness.orchestrator --phase A          # run one phase
python -m test_harness.orchestrator --severity CRITICAL # only critical-severity tests
python -m test_harness.orchestrator --test-ids P-04 M-EF M-WS
python -m test_harness.orchestrator --quick            # smoke-run with reduced kwargs

# Skip whole capability buckets:
python -m test_harness.orchestrator --skip-models      # parser/GUI tests only
python -m test_harness.orchestrator --skip-qt          # no PyQt6 needed
python -m test_harness.orchestrator --skip-scapy       # parser-safety tests only

# Run everything, default settings:
python -m test_harness.orchestrator
```

The orchestrator returns exit code `0` on a clean run, `1` when one or
more tests fail / error, `2` when no tests matched the filter.

## 3. Phases

| Phase | What it covers |
|------:|----------------|
| **A** | Core stability — parser safety, packet handling, schema integrity, IPv6/VLAN/timestamp robustness. |
| **B** | ML validation — threshold sweep, empty-flow regression, calibration, window starvation. |
| **C** | Adversarial & security — GRE, MAC-OUI spoofing, SYN flood, Kitsune key explosion, slow beacons, FGSM, XAI sanity. |
| **D** | GUI / long-run / stress — DetectionStore, alert flood, inference thread, repeated upload, large PCAP, packet flood, long-run loop, concurrent upload. |

## 4. Output layout

Every run creates `outputs/run_<timestamp>/` with this structure:

```
outputs/run_20260510_141022/
└── reports/
    ├── summary.json             # full machine-readable record
    ├── summary.md               # tabular per-phase overview
    ├── failures.json            # just the failures, severity-sorted
    ├── performance.csv          # per-test duration, peak CPU/RSS/GPU
    ├── memory_profile.csv       # per-test memory rows (links to time-series)
    └── recommendation_report.md # final master report
```

Per-test logs and artifacts live alongside the harness, NOT inside the
run directory:

```
logs/<test_id>/
├── stdout.txt
├── stderr.txt
├── run.log
├── _payload.json        # the kwargs/target dispatched to the subprocess
├── _target_result.json  # raw return from the test's run()
└── result.json          # final structured verdict written by harness

artifacts/<test_id>/
├── *.pcap / *.csv       # generated input files
├── threshold_sweep.csv  # if applicable
├── reliability.csv      # if applicable
└── resources.csv        # CPU/RSS/GPU time-series at 4 Hz
```

## 5. Test result schema

Every test's `run()` returns a dict of this shape:

```json
{
  "test_id":  "P-04",
  "name":     "IPv6-only flows",
  "severity": "HIGH",
  "verdict":  "FAIL",
  "expected": "IPv6 flows produce >=1 result OR explicit log of unsupported",
  "actual":   "All IPv6 packets silently dropped: seen=180, results=0.",
  "pcap":     "/.../artifacts/P-04/ipv6_only.pcap",
  "raw":      { ... }
}
```

The orchestrator wraps that in:

```json
{
  "test_id":  "P-04",
  "spec":     { "phase": "A", "timeout_sec": 120, ... },
  "result":   { ...the dict above... },
  "isolated": {
    "return_code":  0,
    "timed_out":    false,
    "crashed":      false,
    "duration_sec": 11.7,
    "stdout_path":  "logs/P-04/stdout.txt",
    "stderr_path":  "logs/P-04/stderr.txt",
    "resources":    { "peak_cpu_pct": 81, "peak_rss_mb": 1430, ... }
  }
}
```

## 6. How isolation works

`utils.isolation.run_isolated` spawns a fresh Python interpreter per
test via `utils/runner_subprocess.py`, with a hard wall-clock timeout
and process-group kill on overrun. The orchestrator can therefore
survive any segfault, OOM, or infinite loop in a target. Resource
monitoring runs in a thread inside the orchestrator, polling the
child's PID at 4 Hz.

## 7. Adding a new test

1. Pick a phase/folder (e.g. `pcap_tests/`).
2. Create `t_my_test.py` exporting `run(**kwargs) -> dict`. The dict
   must follow the schema in §5.
3. Append a `TestSpec(...)` to `configs/registry.py`. Choose a unique
   `test_id`, set `phase`, `severity`, `timeout_sec`, `kwargs`, and
   the capability flags (`requires_models`, `requires_scapy`, etc.).
4. Optional: add an entry to `INSIGHTS` in `reports/generate.py` so
   the recommendation report can theme the failure.

That is the whole contract. Generators in `generators/` and helpers in
`utils/` are reusable and should be preferred over copy-paste.

## 8. CI integration

A minimal recipe that runs on every PR:

```yaml
# .github/workflows/harness.yml
- run: pip install -r test_harness/requirements.txt
- run: python -m test_harness.orchestrator --phase A --skip-qt
- run: python -m test_harness.orchestrator --severity CRITICAL --skip-qt
- uses: actions/upload-artifact@v4
  with:
    name: harness-reports
    path: test_harness/outputs/run_*/reports/
```

The orchestrator's non-zero exit code on any FAIL/ERROR latches the
build red. For full nightly runs drop the filters.

## 9. Logging-format reference

`run.log` lines (file logger inside each test):

```
14:22:31 INFO    [orchestrator] Run directory: outputs/run_20260510_141022
14:22:32 INFO    [orchestrator] Capabilities: scapy=True qt=True models_ok=True
14:22:33 INFO    [orchestrator] [1/29] M-00 — Schema drift (timeout 30s)
14:22:33 INFO    [orchestrator]     -> PASS in 1.4s
```

`stdout.txt` and `stderr.txt` capture the subprocess's raw streams.

`_payload.json` (input to runner_subprocess):

```json
{ "test_id": "P-04", "target": "test_harness.pcap_tests.t_ipv6_only:run",
  "kwargs": {"n_flows": 30}, "result_path": "logs/P-04/_target_result.json" }
```

## 10. Determinism

`utils.determinism.seed_everything()` is called at the top of every
subprocess. Seeds are set for `random`, `numpy`, `torch` (CPU + CUDA).
Deterministic torch algorithms are enabled with `warn_only=True` so
expensive cuDNN flags don't blow up long-running stress tests.

## 11. Troubleshooting

- **"No tests matched the filter"** → run `--list` and check IDs.
- **Every test SKIPPED with "missing models: ..."** → either the model
  files really aren't on disk, or `BOTNET_PROJECT_ROOT` isn't set right.
- **Tests time out on CI** → bump `timeout_sec` in `configs/registry.py`,
  or use `--quick` on the smoke run and full settings on nightly.
- **GUI tests SKIP under headless CI** → set `QT_QPA_PLATFORM=offscreen`
  before running. The harness sets it on its own children, but if you
  invoke a test directly you'll need it too.
