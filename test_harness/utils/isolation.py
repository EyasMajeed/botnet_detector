"""
isolation.py — Subprocess isolation for potentially destructive tests.

The orchestrator never invokes a target function directly. Instead it
spawns a fresh Python interpreter that runs the target inside the
runner_subprocess.py entrypoint. This guarantees:

    - the orchestrator survives any segfault or OOM in the target
    - per-test timeouts are real (kill the child, not the parent)
    - resource usage is monitored via the child PID
    - stdout/stderr are captured to logs/<test_id>/{stdout,stderr}.txt
    - the child's last-write JSON result file is read back

The target is identified as "module:function" plus a JSON-serialisable
kwargs dict. The function must return a JSON-serialisable dict.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .paths import HARNESS_ROOT, PROJECT_ROOT, for_test
from .resource_monitor import ResourceMonitor, ResourceSummary


@dataclass
class IsolatedRun:
    test_id:        str
    target:         str   # "module.path:function"
    kwargs:         dict
    timeout_sec:    float
    return_code:    int = 0
    timed_out:      bool = False
    crashed:        bool = False
    duration_sec:   float = 0.0
    target_result:  dict = field(default_factory=dict)
    stdout_path:    Optional[str] = None
    stderr_path:    Optional[str] = None
    resources:      Optional[dict] = None
    error:          Optional[str] = None


def run_isolated(
    test_id: str,
    target:  str,
    kwargs:  dict | None = None,
    timeout_sec: float = 60.0,
    monitor_resources: bool = True,
    monitor_gpu: bool = False,
    extra_env: dict | None = None,
) -> IsolatedRun:
    """
    Run `target` (e.g. "test_harness.pcap_tests.t_ipv6_only:run") in a
    fresh Python subprocess, with a hard timeout and resource monitoring.
    """
    kwargs = dict(kwargs or {})
    dirs   = for_test(test_id)
    payload_path  = dirs["logs"] / "_payload.json"
    result_path   = dirs["logs"] / "_target_result.json"
    stdout_path   = dirs["logs"] / "stdout.txt"
    stderr_path   = dirs["logs"] / "stderr.txt"
    resources_csv = dirs["artifacts"] / "resources.csv"

    # Cleanup stale outputs so a previous run doesn't pollute this one.
    for p in (payload_path, result_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    payload = {
        "test_id":     test_id,
        "target":      target,
        "kwargs":      kwargs,
        "result_path": str(result_path),
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    runner_script = HARNESS_ROOT / "utils" / "runner_subprocess.py"

    env = os.environ.copy()
    # Pin torch/numpy threads so stress tests don't blow up the host.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Ensure both the harness and the project are importable.
    pp = [str(HARNESS_ROOT.parent), str(HARNESS_ROOT), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["BOTNET_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-u", str(runner_script), str(payload_path)]

    fout = open(stdout_path, "w", encoding="utf-8", errors="replace")
    ferr = open(stderr_path, "w", encoding="utf-8", errors="replace")

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd, stdout=fout, stderr=ferr, env=env,
            cwd=str(PROJECT_ROOT),
            start_new_session=(os.name != "nt"),
        )
    except Exception as e:
        fout.close(); ferr.close()
        return IsolatedRun(
            test_id=test_id, target=target, kwargs=kwargs,
            timeout_sec=timeout_sec, return_code=-1, crashed=True,
            duration_sec=0.0, error=f"spawn failed: {e!r}",
            stdout_path=str(stdout_path), stderr_path=str(stderr_path),
        )

    rm: Optional[ResourceMonitor] = None
    rc: int = -1
    timed_out = False
    crashed   = False
    try:
        if monitor_resources:
            rm = ResourceMonitor(proc.pid, csv_path=resources_csv,
                                 interval=0.25, gpu=monitor_gpu)
            rm.start()
        try:
            rc = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                rc = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rc = -9
    finally:
        if rm is not None:
            rm.stop()
        fout.close(); ferr.close()

    duration = time.monotonic() - t0
    crashed  = (rc not in (0, None)) and not timed_out

    target_result: dict = {}
    if result_path.exists():
        try:
            target_result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            target_result = {"_warning": "result_path was not valid JSON"}

    res_summary: ResourceSummary | None = rm.summary if rm else None

    return IsolatedRun(
        test_id=test_id, target=target, kwargs=kwargs,
        timeout_sec=timeout_sec, return_code=int(rc),
        timed_out=timed_out, crashed=crashed,
        duration_sec=round(duration, 3),
        target_result=target_result,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        resources=res_summary.as_dict() if res_summary else None,
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and its descendants. Cross-platform best effort."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=5)
        except Exception:
            try: proc.kill()
            except Exception: pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try: proc.terminate()
        except Exception: pass
    time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try: proc.kill()
        except Exception: pass
