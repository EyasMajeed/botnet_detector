"""
resource_monitor.py — Sampling resource monitor.

Spawns a background thread that polls a target PID for CPU%, RSS, and
optional GPU memory, writes the time series to artifacts/<test_id>/resources.csv,
and reports peak/average summaries.

psutil is required. GPU sampling is best-effort via nvidia-smi or pynvml.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ResourceSample:
    t:        float
    cpu_pct:  float
    rss_mb:   float
    gpu_mem_mb: Optional[float] = None


@dataclass
class ResourceSummary:
    samples:        int   = 0
    duration_sec:   float = 0.0
    peak_cpu_pct:   float = 0.0
    avg_cpu_pct:    float = 0.0
    peak_rss_mb:    float = 0.0
    avg_rss_mb:     float = 0.0
    peak_gpu_mb:    Optional[float] = None
    csv_path:       Optional[str]   = None
    series:         list[ResourceSample] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "samples":      self.samples,
            "duration_sec": round(self.duration_sec, 3),
            "peak_cpu_pct": round(self.peak_cpu_pct, 1),
            "avg_cpu_pct":  round(self.avg_cpu_pct, 1),
            "peak_rss_mb":  round(self.peak_rss_mb, 1),
            "avg_rss_mb":   round(self.avg_rss_mb, 1),
            "peak_gpu_mb":  round(self.peak_gpu_mb, 1) if self.peak_gpu_mb else None,
            "csv_path":     self.csv_path,
        }


class ResourceMonitor:
    """
    Usage:
        with ResourceMonitor(pid, csv_path=...) as rm:
            ...           # subprocess runs
        summary = rm.summary
    """

    def __init__(self, pid: int, csv_path: Optional[Path] = None,
                 interval: float = 0.25, gpu: bool = False) -> None:
        self.pid       = pid
        self.csv_path  = Path(csv_path) if csv_path else None
        self.interval  = float(interval)
        self.gpu       = bool(gpu)
        self._stop     = threading.Event()
        self._thread:  Optional[threading.Thread] = None
        self._t0:      float = 0.0
        self.summary   = ResourceSummary()

    # ── Context-manager interface ──────────────────────────────────────────
    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._finalise()

    # ── Inner loop ─────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            # No psutil → just record one synthetic zero sample.
            self.summary.series.append(ResourceSample(0.0, 0.0, 0.0, None))
            return

        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return

        # Prime cpu_percent so the first reading isn't zero.
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                if not proc.is_running():
                    break
                cpu = proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                break

            gpu_mb: Optional[float] = None
            if self.gpu:
                gpu_mb = _read_gpu_mb(self.pid)

            t = time.monotonic() - self._t0
            self.summary.series.append(ResourceSample(t, cpu, rss, gpu_mb))
            time.sleep(self.interval)

    def _finalise(self) -> None:
        s = self.summary.series
        if not s:
            return
        self.summary.samples      = len(s)
        self.summary.duration_sec = s[-1].t
        self.summary.peak_cpu_pct = max(x.cpu_pct for x in s)
        self.summary.avg_cpu_pct  = sum(x.cpu_pct for x in s) / len(s)
        self.summary.peak_rss_mb  = max(x.rss_mb for x in s)
        self.summary.avg_rss_mb   = sum(x.rss_mb for x in s) / len(s)
        gpu_vals = [x.gpu_mem_mb for x in s if x.gpu_mem_mb is not None]
        if gpu_vals:
            self.summary.peak_gpu_mb = max(gpu_vals)

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_sec", "cpu_pct", "rss_mb", "gpu_mem_mb"])
                for x in s:
                    w.writerow([f"{x.t:.3f}", f"{x.cpu_pct:.1f}",
                                f"{x.rss_mb:.1f}",
                                "" if x.gpu_mem_mb is None else f"{x.gpu_mem_mb:.1f}"])
            self.summary.csv_path = str(self.csv_path)


# ── GPU sampling — best effort, never fatal ─────────────────────────────────

def _read_gpu_mb(pid: int) -> Optional[float]:
    """
    Try pynvml first, then nvidia-smi. Returns None if no GPU or nothing
    matches our PID.
    """
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if int(p.pid) == int(pid):
                    return float(p.usedGpuMemory) / (1024 * 1024)
        return 0.0
    except Exception:
        pass

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.check_output(
            [smi, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            timeout=2.0, text=True,
        )
        for line in out.strip().splitlines():
            try:
                spid, mem = line.split(",")
                if int(spid.strip()) == int(pid):
                    return float(mem.strip())
            except Exception:
                continue
    except Exception:
        return None
    return 0.0
