"""
reports/generate.py — Produce all post-run artifacts.

Inputs: a list of dicts, each with the schema:

    {
        "test_id": str,
        "spec":    {test_id, name, phase, severity, target, ...},
        "result":  {test_id, name, severity, verdict, expected, actual, raw, ...}
                   ─ what the test's run() returned, OR
                   ─ a synthesised dict when the test ERRORed before
                     returning anything.
        "isolated":{
            "return_code":   int,
            "timed_out":     bool,
            "crashed":       bool,
            "duration_sec":  float,
            "stdout_path":   str,
            "stderr_path":   str,
            "resources":     {peak_cpu_pct, peak_rss_mb, ...} | None,
        }
    }

Outputs (all under <run_dir>/reports/):
    summary.json
    summary.md
    failures.json
    performance.csv
    memory_profile.csv
    recommendation_report.md
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
VERDICT_ORDER  = {"FAIL": 0, "ERROR": 1, "SKIPPED": 2, "PASS": 3}


def write_all(run_dir: Path, entries: list[dict]) -> dict[str, Path]:
    run_dir = Path(run_dir)
    rdir = run_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json":   rdir / "summary.json",
        "summary_md":     rdir / "summary.md",
        "failures_json":  rdir / "failures.json",
        "performance_csv": rdir / "performance.csv",
        "memory_csv":     rdir / "memory_profile.csv",
        "recommendation": rdir / "recommendation_report.md",
    }
    _write_summary_json(entries,   paths["summary_json"])
    _write_summary_md(entries,     paths["summary_md"])
    _write_failures_json(entries,  paths["failures_json"])
    _write_performance_csv(entries, paths["performance_csv"])
    _write_memory_csv(entries,     paths["memory_csv"])
    _write_recommendations(entries, paths["recommendation"])
    return paths


# ── Helpers ────────────────────────────────────────────────────────────────

def _verdict(e: dict) -> str:
    r = e.get("result") or {}
    return str(r.get("verdict") or "ERROR").upper()


def _severity(e: dict) -> str:
    return str(((e.get("result") or {}).get("severity")
                or (e.get("spec") or {}).get("severity")
                or "MEDIUM")).upper()


def _name(e: dict) -> str:
    return ((e.get("result") or {}).get("name")
            or (e.get("spec") or {}).get("name")
            or e.get("test_id", "?"))


def _summary_counts(entries: list[dict]) -> dict[str, int]:
    out = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}
    for e in entries:
        out[_verdict(e)] = out.get(_verdict(e), 0) + 1
    return out


# ── Writers ────────────────────────────────────────────────────────────────

def _write_summary_json(entries: list[dict], path: Path) -> None:
    counts = _summary_counts(entries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "totals": counts,
            "n_tests": len(entries),
            "entries": entries,
        }, f, indent=2, default=str)


def _write_summary_md(entries: list[dict], path: Path) -> None:
    counts = _summary_counts(entries)
    lines: list[str] = []
    lines.append("# Test Harness — Run Summary")
    lines.append("")
    lines.append(f"_Generated_: `{dt.datetime.utcnow().isoformat()}Z`")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- **Tests run**: {len(entries)}")
    for k in ("PASS", "FAIL", "ERROR", "SKIPPED"):
        lines.append(f"- **{k}**: {counts.get(k, 0)}")
    lines.append("")

    # Group by phase, then verdict.
    by_phase: dict[str, list[dict]] = {}
    for e in entries:
        ph = (e.get("spec") or {}).get("phase", "?")
        by_phase.setdefault(ph, []).append(e)

    for phase in sorted(by_phase):
        lines.append(f"## Phase {phase}")
        lines.append("")
        lines.append("| ID | Name | Severity | Verdict | Duration | Peak RSS | Notes |")
        lines.append("|----|------|----------|---------|----------|----------|-------|")
        rows = sorted(by_phase[phase],
                      key=lambda e: (VERDICT_ORDER.get(_verdict(e), 9),
                                     SEVERITY_ORDER.get(_severity(e), 9)))
        for e in rows:
            iso = e.get("isolated") or {}
            res = e.get("result") or {}
            notes = res.get("actual", "")
            if isinstance(notes, str) and len(notes) > 90:
                notes = notes[:87] + "..."
            res_dict = iso.get("resources") or {}
            rss = res_dict.get("peak_rss_mb")
            lines.append("| {id} | {name} | {sev} | {ver} | {dur:.1f}s | {rss} | {notes} |".format(
                id=e.get("test_id", "?"),
                name=_name(e).replace("|", "/"),
                sev=_severity(e),
                ver=_verdict(e),
                dur=float(iso.get("duration_sec", 0.0) or 0.0),
                rss=(f"{rss:.0f} MB" if rss else "-"),
                notes=str(notes).replace("|", "/").replace("\n", " ")))
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_failures_json(entries: list[dict], path: Path) -> None:
    fails = [e for e in entries if _verdict(e) in ("FAIL", "ERROR")]
    fails.sort(key=lambda e: (SEVERITY_ORDER.get(_severity(e), 9),
                              VERDICT_ORDER.get(_verdict(e), 9)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_failures": len(fails), "entries": fails},
                  f, indent=2, default=str)


def _write_performance_csv(entries: list[dict], path: Path) -> None:
    fields = ["test_id", "name", "phase", "severity", "verdict",
              "duration_sec", "peak_cpu_pct", "avg_cpu_pct",
              "peak_rss_mb", "avg_rss_mb", "peak_gpu_mb",
              "timed_out", "return_code"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            iso = e.get("isolated") or {}
            res_dict = iso.get("resources") or {}
            spec = e.get("spec") or {}
            w.writerow({
                "test_id": e.get("test_id"),
                "name":    _name(e),
                "phase":   spec.get("phase", ""),
                "severity": _severity(e),
                "verdict": _verdict(e),
                "duration_sec": iso.get("duration_sec"),
                "peak_cpu_pct": res_dict.get("peak_cpu_pct"),
                "avg_cpu_pct":  res_dict.get("avg_cpu_pct"),
                "peak_rss_mb":  res_dict.get("peak_rss_mb"),
                "avg_rss_mb":   res_dict.get("avg_rss_mb"),
                "peak_gpu_mb":  res_dict.get("peak_gpu_mb"),
                "timed_out":    iso.get("timed_out", False),
                "return_code":  iso.get("return_code"),
            })


def _write_memory_csv(entries: list[dict], path: Path) -> None:
    """One row per test with key memory stats. Detailed time-series sit in
    artifacts/<test_id>/resources.csv."""
    fields = ["test_id", "phase", "verdict",
              "peak_rss_mb", "avg_rss_mb", "samples",
              "duration_sec", "resources_csv"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            iso = e.get("isolated") or {}
            res_dict = iso.get("resources") or {}
            spec = e.get("spec") or {}
            w.writerow({
                "test_id":     e.get("test_id"),
                "phase":       spec.get("phase", ""),
                "verdict":     _verdict(e),
                "peak_rss_mb": res_dict.get("peak_rss_mb"),
                "avg_rss_mb":  res_dict.get("avg_rss_mb"),
                "samples":     res_dict.get("samples"),
                "duration_sec": res_dict.get("duration_sec"),
                "resources_csv": res_dict.get("csv_path"),
            })


# ── Recommendation report ─────────────────────────────────────────────────

# Mapping from test_id to the architectural / ML insight that test surfaces.
# Used to group failures into themes in the recommendation report.
INSIGHTS = {
    "M-00":  ("Schema integrity",
              "S1_FEATURES list drift between project and harness — every "
              "downstream ML test result is invalid until reconciled."),
    "F-01":  ("Parser robustness", "PCAPNG truncation handling."),
    "F-02":  ("Parser robustness",
              "claimed-huge snaplen — naive parsers OOM on a few-byte file."),
    "F-03":  ("Parser robustness", "oversized CSV handling."),
    "F-07":  ("Parser robustness",
              "non-UTF-8 CSVs silently flowing into Stage-1 with mis-decoded columns."),
    "F-08":  ("Parser robustness",
              "fuzz inputs raising exceptions inside file_handler — should "
              "always return is_valid=False, never raise."),
    "F-09":  ("Parser robustness",
              "magic-byte vs extension trust — zip bomb wrongly classified as PCAP."),
    "P-02":  ("Packet handling", "malformed Ethernet frames crashing replay."),
    "P-03":  ("Packet handling",
              "VLAN-tagged traffic dropped by IP-only path."),
    "P-04":  ("IPv6 blind spot",
              "live_detector and PCAP path silently drop IPv6, "
              "matching the documented `if IP not in pkt: return` bug."),
    "P-05":  ("Tunnelled traffic",
              "GRE-encapsulated inner IP not unwrapped."),
    "P-07":  ("Aggregator robustness",
              "negative IAT or NaN propagation when timestamps go backwards."),
    "S1-06": ("Stage-1 routing exploit",
              "OUI override is bypass-able via MAC spoofing — workstation "
              "traffic gets routed to IoT branch with no behavioural gate."),
    "SE-A":  ("Suspicion scorer wiring",
              "Suspicion scorer fires correctly on a synthetic SYN-flood "
              "flow, but it is NOT wired into the file-upload path; only "
              "the live path benefits from it."),
    "SC-K":  ("Memory exhaustion",
              "KitsuneExtractor _hh / _hphp dictionaries grow without "
              "bound — long-running live capture leaks state continuously."),
    "RT-1":  ("Detection blind spot — slow beaconing",
              "Beacons across multiple sessions from one src_ip should "
              "produce a complete LSTM window after the project raised "
              "_idle to 120 s. If this still fails, the model has not "
              "learned to flag short flows even given a real window."),
    "RT-1b": ("Detection blind spot — slow C2 evasion",
              "Sessions spaced LONGER than _idle (e.g. 5 minutes) cannot "
              "be detected with per-flow LSTM input alone. Cross-flow "
              "temporal aggregation per src_ip is required."),
    "M-T1":  ("ML calibration",
              "operating threshold should be picked from a sweep, not "
              "hard-coded to whatever the training script wrote."),
    "M-EF":  ("Scaler defect (documented)",
              "Empty flows produce a constant near-zero probability — "
              "Non-IoT scaler fitted on already-normalised data."),
    "M-WS":  ("Window starvation",
              "Stage-2 Non-IoT detector pads to seq_len=20 with zeros; "
              "unique-per-flow src_ip distributions never appeared in "
              "training and recall collapses."),
    "M-CAL": ("Confidence calibration",
              "ECE > 0.10 means the probabilities the GUI shows users do "
              "not reflect the real per-bin positive rate."),
    "M-ADV": ("Adversarial brittleness",
              "Stage-1 routing flips at ε=0.01 in scaled space — the "
              "scaler.json on disk is enough to weaponise this."),
    "X-A1":  ("XAI sanity",
              "Integrated Gradients attributions unstable under repeat or "
              "tiny perturbation, OR sign-incoherent on positive predictions."),
    "G-DS":  ("DetectionStore correctness",
              "MAX_FLOWS cap or apply_threshold relabel broken."),
    "G-AF":  ("UI under load",
              "alert flood floods the changed signal — UI repaints fire "
              "once per row instead of once per debounce window."),
    "G-IT":  ("Worker thread",
              "PcapInferenceThread does not finish or surfaces an error."),
    "G-RU":  ("Memory leak",
              "RSS grows monotonically across repeated uploads."),
    "ST-1":  ("Throughput",
              "200k-packet PCAP exceeds the time budget."),
    "ST-2a": ("Throughput — ingestion hot path",
              "process_packet must keep up with line-rate packet arrival. "
              "If this fails the bottleneck is in flow-key construction, "
              "Kitsune state updates, or per-packet logging."),
    "ST-2b": ("Throughput — end-to-end with XAI",
              "End-to-end pipeline must keep up with the rate flows "
              "complete. Per-flow Integrated Gradients XAI is expensive "
              "(~30 ms/flow CPU). Common fix: gate XAI on label=='botnet' "
              "or run XAI lazily on GUI request only."),
    "ST-3":  ("Long-run stability",
              "errors accumulate over a sustained replay loop."),
    "ST-4":  ("Concurrency",
              "inference_bridge is not safe to call from multiple threads."),
}


def _write_recommendations(entries: list[dict], path: Path) -> None:
    counts = _summary_counts(entries)
    fails = [e for e in entries if _verdict(e) in ("FAIL", "ERROR")]

    # Sort failures by (severity, verdict).
    fails.sort(key=lambda e: (SEVERITY_ORDER.get(_severity(e), 9),
                              VERDICT_ORDER.get(_verdict(e), 9)))

    lines: list[str] = []
    lines.append("# Test Harness — Recommendation Report")
    lines.append("")
    lines.append(f"_Generated_: `{dt.datetime.utcnow().isoformat()}Z`")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- {counts.get('PASS',0)} PASS / {counts.get('FAIL',0)} FAIL / "
                 f"{counts.get('ERROR',0)} ERROR / {counts.get('SKIPPED',0)} SKIP, "
                 f"{len(entries)} total.")
    lines.append("")

    if not fails:
        lines.append("All non-skipped tests passed. See `summary.md` for full results "
                     "and `performance.csv` for resource profiles.")
    else:
        lines.append("## Failures, ordered by severity")
        lines.append("")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            bucket = [e for e in fails if _severity(e) == sev]
            if not bucket:
                continue
            lines.append(f"### {sev}")
            lines.append("")
            for e in bucket:
                tid = e.get("test_id", "?")
                name = _name(e)
                res  = e.get("result") or {}
                exp  = res.get("expected", "")
                act  = res.get("actual", "")
                theme, insight = INSIGHTS.get(tid, ("Other", ""))
                lines.append(f"- **{tid} — {name}**  _(theme: {theme})_")
                lines.append(f"    - expected: {exp}")
                lines.append(f"    - actual:   {act}")
                if insight:
                    lines.append(f"    - so what:  {insight}")
            lines.append("")

        # Theme roll-up — how many failures landed under each insight bucket
        themes: dict[str, int] = {}
        for e in fails:
            theme = INSIGHTS.get(e.get("test_id", ""), ("Other", ""))[0]
            themes[theme] = themes.get(theme, 0) + 1
        lines.append("## Failures by theme")
        lines.append("")
        for theme, n in sorted(themes.items(), key=lambda x: -x[1]):
            lines.append(f"- {theme}: {n} failure(s)")
        lines.append("")

    # Skipped tests — reasons matter for interpreting the run
    skipped = [e for e in entries if _verdict(e) == "SKIPPED"]
    if skipped:
        lines.append("## Skipped tests")
        lines.append("")
        for e in skipped:
            res = e.get("result") or {}
            lines.append(f"- {e.get('test_id')}: {res.get('actual', 'no reason given')}")
        lines.append("")

    # Performance hotspots
    timing = sorted(
        ((e, float((e.get("isolated") or {}).get("duration_sec") or 0.0))
         for e in entries),
        key=lambda x: -x[1])[:5]
    rss_top = sorted(
        ((e, float(((e.get("isolated") or {}).get("resources") or {}).get("peak_rss_mb") or 0.0))
         for e in entries),
        key=lambda x: -x[1])[:5]

    lines.append("## Performance hotspots")
    lines.append("")
    lines.append("**Slowest 5 tests:**")
    for e, d in timing:
        lines.append(f"- {e.get('test_id')} {_name(e)}: {d:.1f}s")
    lines.append("")
    lines.append("**Top 5 by peak RSS:**")
    for e, rss in rss_top:
        lines.append(f"- {e.get('test_id')} {_name(e)}: {rss:.0f} MB")
    lines.append("")

    lines.append("## Files for downstream review")
    lines.append("")
    lines.append("- `reports/summary.json`     – machine-readable, full payload")
    lines.append("- `reports/summary.md`       – tabular overview by phase")
    lines.append("- `reports/failures.json`    – just the failures, sorted")
    lines.append("- `reports/performance.csv`  – per-test duration + CPU/RSS peaks")
    lines.append("- `reports/memory_profile.csv`– per-test memory sampling info")
    lines.append("- `logs/<test_id>/`          – stdout, stderr, run.log, result.json")
    lines.append("- `artifacts/<test_id>/`     – PCAPs, CSVs, sweep tables, resources.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
