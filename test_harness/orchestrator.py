"""
orchestrator.py — Main entrypoint of the test harness.

Usage:
    python -m test_harness.orchestrator [options]

Options:
    --phase {A,B,C,D}             Run only one phase
    --severity {CRITICAL,HIGH,MEDIUM,LOW}
                                  Run only tests at this severity
    --test-ids ID [ID ...]        Run only the listed test IDs
    --skip-models                 Skip tests that need model files on disk
    --skip-scapy                  Skip tests that need scapy
    --skip-qt                     Skip tests that need PyQt6
    --skip-xai                    Skip tests that need src.xai
    --no-monitor                  Disable resource monitoring
    --gpu-monitor                 Enable GPU sampling
    --quick                       Smaller kwargs for quick smoke run
    --run-dir DIR                 Override run output dir (default: outputs/run_<ts>)

The orchestrator is the only component that should ever be run by hand.
Every test is executed in a fresh subprocess via utils.isolation, so
nothing a target does — segfault, OOM, infinite loop — can hurt the
parent. After all tests run, reports/ is regenerated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

# ── Bootstrap ─────────────────────────────────────────────────────────────
# Ensure the project ROOT (parent of the test_harness/ package) is on
# sys.path so that `import test_harness` works regardless of how Python
# was invoked (python -m test_harness.orchestrator, python orchestrator.py,
# or any other mechanism).  We derive the root from *this file's* location,
# which is always reliable.
_HARNESS_PACKAGE_DIR = Path(__file__).resolve().parent   # …/test_harness/
_REPO_ROOT           = _HARNESS_PACKAGE_DIR.parent       # …/botnet_detector-main/
for _p in [str(_REPO_ROOT), str(_HARNESS_PACKAGE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ──────────────────────────────────────────────────────────────────────────

from test_harness.configs.registry import TestSpec, TESTS
from test_harness.reports.generate import write_all
from test_harness.utils.isolation import IsolatedRun, run_isolated
from test_harness.utils.logging_setup import make_logger
from test_harness.utils.paths import HARNESS_ROOT, OUTPUTS_DIR
from test_harness.utils.project_imports import models_state, soft_import


# ── Filtering ──────────────────────────────────────────────────────────────

def _filter(specs: Iterable[TestSpec], args: argparse.Namespace) -> list[TestSpec]:
    out = list(specs)
    if args.phase:
        out = [s for s in out if s.phase.upper() == args.phase.upper()]
    if args.severity:
        out = [s for s in out if s.severity.upper() == args.severity.upper()]
    if args.test_ids:
        ids = {x.upper() for x in args.test_ids}
        out = [s for s in out if s.test_id.upper() in ids]
    return out


def _capability_skip(spec: TestSpec, caps: dict, args: argparse.Namespace) -> str | None:
    """Decide whether to skip BEFORE spawning a subprocess. Returns reason or None."""
    if args.skip_models and spec.requires_models:
        return "skipped via --skip-models"
    if args.skip_scapy and spec.requires_scapy:
        return "skipped via --skip-scapy"
    if args.skip_qt and spec.requires_qt:
        return "skipped via --skip-qt"
    if args.skip_xai and spec.requires_xai:
        return "skipped via --skip-xai"
    if spec.requires_models and not caps["models_ok"]:
        return f"missing models: {caps['missing_models']}"
    if spec.requires_scapy and not caps["scapy_ok"]:
        return "scapy not installed"
    if spec.requires_qt and not caps["qt_ok"]:
        return "PyQt6 not installed"
    return None


def _capabilities() -> dict:
    ms = models_state()
    missing_models = [k for k, v in ms.items() if not v]
    scapy, _ = soft_import("scapy.all")
    qt, _    = soft_import("PyQt6.QtCore")
    return {"models_ok": all(ms.values()),
            "missing_models": missing_models,
            "scapy_ok": scapy is not None,
            "qt_ok": qt is not None,
            "models": ms}


# ── Result construction ────────────────────────────────────────────────────

def _entry_from(spec: TestSpec, iso: IsolatedRun) -> dict:
    """Compose the orchestrator's per-test record from the IsolatedRun."""
    target_result = iso.target_result or {}
    if iso.timed_out:
        result = {
            "test_id":  spec.test_id,
            "name":     spec.name,
            "severity": spec.severity,
            "verdict":  "ERROR",
            "expected": "completes within timeout",
            "actual":   f"timed out after {iso.timeout_sec:.0f}s",
        }
    elif iso.crashed:
        # The runner_subprocess writes a result file even on exception,
        # but a fatal C-level segfault would leave it empty.
        if target_result.get("_traceback"):
            result = {
                "test_id":  spec.test_id,
                "name":     spec.name,
                "severity": spec.severity,
                "verdict":  "ERROR",
                "expected": "test runs without raising",
                "actual":   (f"raised {target_result.get('_exception')}: "
                             f"{target_result.get('_message','')}"),
                "traceback": target_result.get("_traceback"),
            }
        else:
            result = {
                "test_id":  spec.test_id,
                "name":     spec.name,
                "severity": spec.severity,
                "verdict":  "ERROR",
                "expected": "subprocess exits 0",
                "actual":   f"subprocess crashed (rc={iso.return_code})",
            }
    elif "verdict" in target_result:
        result = target_result
    else:
        # Test ran but didn't return the standard schema.
        result = {
            "test_id":  spec.test_id,
            "name":     spec.name,
            "severity": spec.severity,
            "verdict":  "ERROR",
            "expected": "test returns standard verdict dict",
            "actual":   f"non-conforming return: {sorted(target_result.keys())[:8]}",
        }
    return {
        "test_id":  spec.test_id,
        "spec":     asdict(spec),
        "result":   result,
        "isolated": {
            "return_code": iso.return_code,
            "timed_out":   iso.timed_out,
            "crashed":     iso.crashed,
            "duration_sec": iso.duration_sec,
            "stdout_path": iso.stdout_path,
            "stderr_path": iso.stderr_path,
            "resources":   iso.resources,
        },
    }


# ── Run loop ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else (
        OUTPUTS_DIR / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger, dirs = make_logger("ORCHESTRATOR")
    logger.info("Run directory: %s", run_dir)
    logger.info("Harness root:  %s", HARNESS_ROOT)

    caps = _capabilities()
    logger.info("Capabilities: scapy=%s qt=%s models_ok=%s missing=%s",
                caps["scapy_ok"], caps["qt_ok"], caps["models_ok"],
                caps["missing_models"])

    selected = _filter(TESTS, args)
    if not selected:
        logger.error("No tests matched the filter.")
        return 2

    if args.quick:
        selected = _quick_kwargs(selected)

    logger.info("Running %d test(s).", len(selected))
    entries: list[dict] = []
    for i, spec in enumerate(selected, 1):
        skip_reason = _capability_skip(spec, caps, args)
        if skip_reason is not None:
            entries.append({
                "test_id":  spec.test_id,
                "spec":     asdict(spec),
                "result": {
                    "test_id":  spec.test_id, "name": spec.name,
                    "severity": spec.severity, "verdict": "SKIPPED",
                    "expected": "(test would have run)", "actual": skip_reason,
                },
                "isolated": {"return_code": 0, "timed_out": False,
                             "crashed": False, "duration_sec": 0.0,
                             "stdout_path": None, "stderr_path": None,
                             "resources": None},
            })
            logger.info("[%d/%d] %s SKIPPED — %s",
                        i, len(selected), spec.test_id, skip_reason)
            continue

        logger.info("[%d/%d] %s — %s (timeout %.0fs)",
                    i, len(selected), spec.test_id, spec.name, spec.timeout_sec)
        iso = run_isolated(
            test_id=spec.test_id, target=spec.target, kwargs=spec.kwargs,
            timeout_sec=spec.timeout_sec,
            monitor_resources=(spec.monitor_resources and not args.no_monitor),
            monitor_gpu=args.gpu_monitor,
        )
        entry = _entry_from(spec, iso)
        entries.append(entry)
        v = entry["result"].get("verdict", "?")
        logger.info("    -> %s in %.1fs", v, iso.duration_sec)

    paths = write_all(run_dir, entries)
    # Mirror the reports into the canonical reports/ for convenience.
    canonical = HARNESS_ROOT / "reports"
    for p in paths.values():
        try:
            shutil.copy2(p, canonical / p.name)
        except Exception:
            pass

    logger.info("All reports written to %s", run_dir / "reports")
    counts = _summary_counts(entries)
    logger.info("Result: %s", counts)

    # Exit non-zero when something failed so CI can latch.
    n_problems = counts.get("FAIL", 0) + counts.get("ERROR", 0)
    return 1 if n_problems > 0 else 0


def _summary_counts(entries: list[dict]) -> dict[str, int]:
    out = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}
    for e in entries:
        v = (e.get("result") or {}).get("verdict", "ERROR")
        out[v] = out.get(v, 0) + 1
    return out


def _quick_kwargs(specs: list[TestSpec]) -> list[TestSpec]:
    """Return a shallow copy of specs with reduced-cost kwargs for smoke tests."""
    out: list[TestSpec] = []
    for s in specs:
        kw = dict(s.kwargs)
        # Heuristic: shrink anything that looks like an N parameter.
        for k in list(kw.keys()):
            if k in ("n_packets", "n_iters", "n_rows", "n_flows",
                     "n_botnet", "n_benign", "n_each", "n_destinations",
                     "n_samples", "n_threads", "n_iterations"):
                try:
                    kw[k] = max(int(kw[k]) // 4, 8)
                except Exception:
                    pass
            if k == "duration_sec":
                kw[k] = min(float(kw[k]), 5.0)
        out.append(TestSpec(
            test_id=s.test_id, name=s.name, target=s.target, phase=s.phase,
            severity=s.severity, timeout_sec=max(s.timeout_sec / 2, 30.0),
            kwargs=kw, requires_models=s.requires_models,
            requires_scapy=s.requires_scapy, requires_xai=s.requires_xai,
            requires_qt=s.requires_qt,
            monitor_resources=s.monitor_resources, notes=s.notes))
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="test_harness.orchestrator",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=list("ABCD"))
    p.add_argument("--severity",
                   choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    p.add_argument("--test-ids", nargs="+", default=None)
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-scapy",  action="store_true")
    p.add_argument("--skip-qt",     action="store_true")
    p.add_argument("--skip-xai",    action="store_true")
    p.add_argument("--no-monitor",  action="store_true")
    p.add_argument("--gpu-monitor", action="store_true")
    p.add_argument("--quick",       action="store_true")
    p.add_argument("--run-dir",     default=None)
    p.add_argument("--list",        action="store_true",
                   help="List the test registry and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.list:
        for s in TESTS:
            print(f"  {s.test_id:6}  phase={s.phase}  severity={s.severity:<8}  "
                  f"timeout={s.timeout_sec:.0f}s  {s.name}")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
