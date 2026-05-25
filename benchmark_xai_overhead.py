"""
benchmark_xai_overhead.py — Standalone XAI A/B latency benchmark.

Measures the per-flow cost of the XAI explainer by replaying a fixed
synthetic packet workload through BotnetMonitor twice — once with XAI
enabled and once with it disabled via BOTNET_DISABLE_XAI=1 — and printing
a side-by-side comparison.

Why this exists (and not just ST-2b):
    The harness's ST-2b uses ONE src_ip for all 10,000 packets, so
    BotnetMonitor's 30-second per-src-ip XAI rate limit
    (XAI_MIN_INTERVAL_SEC, see monitoring.py) suppresses explain_flow()
    on every flow but the first. That makes ST-2b basically insensitive
    to XAI cost. This script rotates src_ips per flow, so every
    finalised flow is XAI-eligible and the cost difference is direct.

Why standalone (not a new test file under test_harness/):
    The test harness is the project's evaluation artifact for the
    chapter — we don't want to modify it for ad-hoc experiments.
    This script lives at the project root next to verify_xai.py and
    test_pcap_xai_pipeline.py, follows the same standalone-script
    pattern, and is fully self-contained.

Prerequisites:
    1. monitoring.py must honour BOTNET_DISABLE_XAI (already patched).
    2. The Stage-1 and Stage-2 model files in models/ must be readable.
    3. Run from the project root so `import monitoring` resolves.

Usage (Windows):
    python benchmark_xai_overhead.py
    python benchmark_xai_overhead.py --n-packets 4000 --packets-per-flow 25

Usage (macOS / Linux):
    python3 benchmark_xai_overhead.py
    python3 benchmark_xai_overhead.py --n-packets 4000 --packets-per-flow 25

Expected output (numbers vary by hardware):
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  XAI OVERHEAD BENCHMARK — A/B                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Metric                       XAI ON       XAI OFF     Delta     ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║  Bundle init (s)              2.34          0.00       -2.34     ║
    ║  Ingestion (pps)              34,200        34,180     ~0        ║
    ║  Flush time (s)               13.45          1.02       -12.43   ║
    ║  Per-flow latency (ms)        134.5          10.2       -124.3   ║
    ║  Flows/sec                    7.4            98.0        +90.6   ║
    ║  End-to-end pps               148            1,420       +1,272  ║
    ╚══════════════════════════════════════════════════════════════════╝

A JSON copy of all measurements is written to:
    benchmark_xai_overhead_<timestamp>.json
in the directory the script was launched from, so you can paste
the numbers straight into the evaluation chapter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Make sure `import monitoring` resolves whether the user launched this
# from the project root or from elsewhere with a CWD override.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════════════
# Inner runner — what runs inside each isolated child process.
# ════════════════════════════════════════════════════════════════════════════

def _inner_run(n_packets: int, packets_per_flow: int) -> dict:
    """
    Build a fresh BotnetMonitor, ingest N packets across N/PPF distinct
    src_ips, flush every flow, and return timing measurements as a dict.

    Reads BOTNET_DISABLE_XAI from the environment via monitoring.py's own
    construction-time check — we don't decide here, we just measure what
    happens in this process's instantiation.
    """
    # Re-import in case a prior invocation in the same Python session
    # cached an older module state. The CLI path uses subprocess so this
    # is belt-and-braces; the in-process fallback path (--no-subprocess)
    # relies on it.
    if "monitoring" in sys.modules:
        importlib.reload(sys.modules["monitoring"])
    monitoring = importlib.import_module("monitoring")

    # Measure BotnetMonitor construction cost separately — this is where
    # the ExplainerBundle init happens (~1–3s with XAI on, ~0s with it off
    # because the bundle import is short-circuited in monitoring.py).
    t_init = time.monotonic()
    bm = monitoring.BotnetMonitor()
    init_dur = time.monotonic() - t_init

    # Sanity: confirm the kill-switch took effect when it was supposed to.
    env_off = os.environ.get("BOTNET_DISABLE_XAI", "").strip().lower() in (
        "1", "true", "yes", "on")
    monitor_xai_enabled = getattr(bm, "_xai_enabled", True)
    consistent = (not env_off) == monitor_xai_enabled
    if env_off and monitor_xai_enabled:
        # The env var was set but monitoring.py didn't honour it. Bail
        # before we waste time on a meaningless run.
        return {"_error": "BOTNET_DISABLE_XAI=1 was set but "
                          "monitor._xai_enabled is still True. "
                          "Apply the env-var patch to monitoring.py.",
                "env_off": env_off,
                "monitor_xai_enabled": monitor_xai_enabled}

    base_ts = time.time()
    dst_ip  = "10.0.0.1"

    # Phase 1: ingest with rotating src_ips so every closed flow is
    # XAI-eligible (no rate-limit collisions).
    t0 = time.monotonic()
    for i in range(n_packets):
        flow_idx = i // packets_per_flow
        # 10.A.B.C — supports >16M distinct flows; we use up to ~10k.
        src_ip = (f"10.{(flow_idx >> 16) & 0xFF}."
                  f"{(flow_idx >> 8) & 0xFF}."
                  f"{flow_idx & 0xFF}")
        bm.process_packet(base_ts + i * 1e-4, src_ip, dst_ip,
                          40000 + (i % 20000), 53, 17, 100, 64, "", 0)
    ingest_dur = time.monotonic() - t0

    # Phase 2: force-flush every open flow. Same trick the harness uses.
    flushed = []
    flush_dur = 0.0
    flush_error = None
    try:
        if hasattr(bm, "aggregator"):
            bm.aggregator._idle = -1e9
        if hasattr(bm, "flush_idle_flows"):
            t1 = time.monotonic()
            flushed = bm.flush_idle_flows() or []
            flush_dur = time.monotonic() - t1
    except Exception as e:                                    # noqa: BLE001
        flush_error = f"{type(e).__name__}: {e}"

    flushed_count = len(flushed)
    n_botnet      = sum(1 for r in flushed
                        if getattr(r, "label", "") == "botnet")

    # Per-flow XAI cost estimate: when XAI is ON, this includes the
    # explain_flow() time amortised over botnet flows. When XAI is OFF,
    # this is Stage-2-only flush cost. The DIFFERENCE is the metric.
    if n_botnet > 0:
        ms_per_botnet = (flush_dur * 1000.0) / n_botnet
    else:
        # No botnet labels means XAI never fired anyway — use total flush.
        # Note this in the output so the comparison row is interpretable.
        ms_per_botnet = ((flush_dur * 1000.0) / flushed_count
                         if flushed_count else 0.0)

    return {
        "env_off":             env_off,
        "monitor_xai_enabled": monitor_xai_enabled,
        "consistent":          consistent,
        "init_dur_sec":        round(init_dur, 3),
        "ingest_dur_sec":      round(ingest_dur, 3),
        "ingest_pps":          round(n_packets / max(ingest_dur, 1e-6), 1),
        "flush_dur_sec":       round(flush_dur, 3),
        "flush_error":         flush_error,
        "flushed_flows":       flushed_count,
        "n_botnet":            n_botnet,
        "flow_per_sec":        round(flushed_count / max(flush_dur, 1e-6), 1)
                                if flush_dur > 0 else 0.0,
        "e2e_pps":             round(n_packets
                                     / max(ingest_dur + flush_dur, 1e-6), 1),
        "ms_per_botnet_flow":  round(ms_per_botnet, 2),
        "ms_per_flow_overall": round((flush_dur * 1000.0) / flushed_count, 2)
                                if flushed_count else 0.0,
        "n_packets":           n_packets,
        "packets_per_flow":    packets_per_flow,
    }


# ════════════════════════════════════════════════════════════════════════════
# Outer runner — spawns each measurement in an isolated subprocess so the
# bundle init cost is real (a fresh process = a cold cache) and the env var
# is honored at the language-runtime boundary.
# ════════════════════════════════════════════════════════════════════════════

def _run_in_subprocess(label: str, env_off: bool,
                       n_packets: int, packets_per_flow: int) -> dict:
    """Spawn `python -c <recipe>` with the right env, return the dict."""
    # Build a tiny Python program that re-imports this module and calls
    # _inner_run. Using `-c` avoids any "where do we put the temp file"
    # questions, and `os.path.dirname(__file__)` is rock-solid here.
    recipe = (
        f"import sys, json, os; "
        f"sys.path.insert(0, {repr(str(PROJECT_ROOT))}); "
        f"from benchmark_xai_overhead import _inner_run; "
        f"r = _inner_run({n_packets}, {packets_per_flow}); "
        f"print('__RESULT_JSON__' + json.dumps(r))"
    )

    env = os.environ.copy()
    # Same thread pinning the test harness uses — avoid blowing up the
    # host with parallel BLAS during the burst.
    env["KMP_DUPLICATE_LIB_OK"] = env.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["OMP_NUM_THREADS"]      = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"]      = env.get("MKL_NUM_THREADS", "1")
    if env_off:
        env["BOTNET_DISABLE_XAI"] = "1"
    else:
        # Belt-and-braces: scrub any inherited value from the parent shell.
        env.pop("BOTNET_DISABLE_XAI", None)

    print(f"\n→ Running [{label}] in a fresh subprocess "
          f"(BOTNET_DISABLE_XAI={'1' if env_off else 'unset'})...")
    t_wall = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-u", "-c", recipe],
        env=env, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    wall_dur = time.monotonic() - t_wall

    if proc.returncode != 0:
        print(f"  ✗ Subprocess exited {proc.returncode}.")
        print(f"  stderr (last 30 lines):")
        for line in proc.stderr.splitlines()[-30:]:
            print("    " + line)
        return {"_error": f"subprocess rc={proc.returncode}",
                "_stderr": proc.stderr[-2000:],
                "_wall_dur_sec": round(wall_dur, 3)}

    # Pull the result line out of the stdout — the subprocess may print
    # other diagnostics from monitoring.py's logger and we want to keep
    # only the JSON line.
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT_JSON__"):
            try:
                result = json.loads(line[len("__RESULT_JSON__"):])
                break
            except json.JSONDecodeError as e:
                print(f"  ✗ Result JSON malformed: {e}")
    if result is None:
        return {"_error": "no __RESULT_JSON__ line in subprocess stdout",
                "_stdout_tail": proc.stdout[-2000:]}

    result["_wall_dur_sec"] = round(wall_dur, 3)
    print(f"  ✓ Done in {wall_dur:.1f}s wall clock.")
    return result


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _print_comparison(on_run: dict, off_run: dict) -> None:
    """Print a side-by-side table to stdout. Plain ASCII for Windows."""
    if "_error" in on_run:
        print(f"\n!! XAI-ON run failed: {on_run['_error']}")
        return
    if "_error" in off_run:
        print(f"\n!! XAI-OFF run failed: {off_run['_error']}")
        return

    rows = [
        ("Bundle init time (s)",     on_run["init_dur_sec"],
                                     off_run["init_dur_sec"]),
        ("Ingestion (pps)",          on_run["ingest_pps"],
                                     off_run["ingest_pps"]),
        ("Flush wall time (s)",      on_run["flush_dur_sec"],
                                     off_run["flush_dur_sec"]),
        ("Flushed flows",            on_run["flushed_flows"],
                                     off_run["flushed_flows"]),
        ("Botnet flows",             on_run["n_botnet"],
                                     off_run["n_botnet"]),
        ("Per-flow flush latency (ms)", on_run["ms_per_flow_overall"],
                                        off_run["ms_per_flow_overall"]),
        ("Per-botnet-flow lat (ms)", on_run["ms_per_botnet_flow"],
                                    off_run["ms_per_botnet_flow"]),
        ("Flows / sec",              on_run["flow_per_sec"],
                                     off_run["flow_per_sec"]),
        ("End-to-end pps",           on_run["e2e_pps"],
                                     off_run["e2e_pps"]),
        ("Subprocess wall (s)",      on_run["_wall_dur_sec"],
                                     off_run["_wall_dur_sec"]),
    ]

    print()
    print("=" * 76)
    print("                  XAI OVERHEAD BENCHMARK — A/B RESULTS")
    print("=" * 76)
    print(f"  {'Metric':<32} {'XAI ON':>14} {'XAI OFF':>14} {'Delta':>12}")
    print(f"  {'-'*32} {'-'*14:>14} {'-'*14:>14} {'-'*12:>12}")
    for label, on_v, off_v in rows:
        try:
            delta = float(on_v) - float(off_v)
            delta_s = f"{delta:+,.2f}"
        except Exception:
            delta_s = "n/a"
        on_s  = _fmt_int(on_v)  if isinstance(on_v,  int) else f"{on_v:,.2f}"
        off_s = _fmt_int(off_v) if isinstance(off_v, int) else f"{off_v:,.2f}"
        print(f"  {label:<32} {on_s:>14} {off_s:>14} {delta_s:>12}")
    print("=" * 76)

    # Headline interpretation — the per-botnet-flow XAI cost.
    xai_per_flow_ms = (on_run["ms_per_botnet_flow"]
                       - off_run["ms_per_botnet_flow"])
    print(f"\n  Estimated per-flow XAI cost: {xai_per_flow_ms:.1f} ms")
    print(f"  (= ms_per_botnet_flow[ON] - ms_per_botnet_flow[OFF])")
    print(f"\n  Throughput delta: {on_run['e2e_pps']:.0f} → "
          f"{off_run['e2e_pps']:.0f} pps "
          f"({off_run['e2e_pps'] / max(on_run['e2e_pps'], 1):.1f}x speedup "
          f"with XAI off)")
    print()


def _write_json(on_run: dict, off_run: dict, args: argparse.Namespace) -> Path:
    """Persist both runs + metadata so chapter authors can copy verbatim."""
    out = {
        "schema":            "benchmark_xai_overhead/v1",
        "generated_at":      dt.datetime.now().isoformat(timespec="seconds"),
        "python":            sys.version.split()[0],
        "platform":          platform.platform(),
        "config": {
            "n_packets":       args.n_packets,
            "packets_per_flow": args.packets_per_flow,
        },
        "xai_on":  on_run,
        "xai_off": off_run,
        "summary": {
            "per_flow_xai_cost_ms": (
                round(on_run.get("ms_per_botnet_flow", 0.0)
                      - off_run.get("ms_per_botnet_flow", 0.0), 2)
                if ("_error" not in on_run and "_error" not in off_run)
                else None
            ),
            "bundle_init_cost_sec": (
                round(on_run.get("init_dur_sec", 0.0)
                      - off_run.get("init_dur_sec", 0.0), 3)
                if ("_error" not in on_run and "_error" not in off_run)
                else None
            ),
            "throughput_speedup_factor": (
                round(off_run.get("e2e_pps", 0.0)
                      / max(on_run.get("e2e_pps", 1.0), 1.0), 2)
                if ("_error" not in on_run and "_error" not in off_run)
                else None
            ),
        },
    }
    ts   = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PROJECT_ROOT / f"benchmark_xai_overhead_{ts}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="XAI overhead A/B latency benchmark (standalone — "
                    "no test_harness modifications).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--n-packets",        type=int, default=2_000,
                   help="Total packets ingested per run (default 2000).")
    p.add_argument("--packets-per-flow", type=int, default=20,
                   help="Packets per unique src_ip (default 20). "
                        "n_packets / packets_per_flow = number of flows "
                        "finalised at flush time.")
    p.add_argument("--skip-on", action="store_true",
                   help="Skip the XAI-ON run (e.g. if you already ran it).")
    p.add_argument("--skip-off", action="store_true",
                   help="Skip the XAI-OFF run.")
    args = p.parse_args(argv)

    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Python:            {sys.version.split()[0]}")
    print(f"Packets / run:     {args.n_packets:,}")
    print(f"Packets / flow:    {args.packets_per_flow}")
    print(f"=> distinct flows: {args.n_packets // args.packets_per_flow}")

    on_run, off_run = {}, {}

    if not args.skip_on:
        on_run = _run_in_subprocess("XAI ON", env_off=False,
                                    n_packets=args.n_packets,
                                    packets_per_flow=args.packets_per_flow)
    if not args.skip_off:
        off_run = _run_in_subprocess("XAI OFF", env_off=True,
                                     n_packets=args.n_packets,
                                     packets_per_flow=args.packets_per_flow)

    if on_run and off_run:
        _print_comparison(on_run, off_run)
        out_path = _write_json(on_run, off_run, args)
        print(f"  Full results written to:")
        print(f"    {out_path}\n")

    # Non-zero exit if either run errored, so a calling script can catch it.
    if on_run.get("_error") or off_run.get("_error"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
