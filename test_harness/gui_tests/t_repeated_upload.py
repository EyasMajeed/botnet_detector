"""
G-RU Repeated upload cycles.

Loads the same small CSV through inference_bridge.run_file_inference
N times in the same process, sampling RSS before and after, and reports
the slope. A persistent leak shows up as monotonic RSS growth across
iterations beyond the first warmup.

Why a CSV (not a PCAP):
    inference_bridge.run_file_inference is the SYNC, CSV-only path used
    by UploadPage._run_csv_async. PCAP files in the GUI go through the
    ASYNC PcapInferenceThread instead — that's what G-IT covers. Sending
    a PCAP here raises ValueError("Unsupported file format ...").
"""

from __future__ import annotations

import gc
import os

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import, ensure_on_path


TEST_ID = "G-RU"


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run(n_iters: int = 10, n_botnet: int = 50, n_noniot: int = 100) -> dict:
    ensure_on_path()
    fh, err = soft_import("file_handler")
    if fh is None:
        fh, err = soft_import("app.file_handler")
        if fh is None:
            return _skip(f"file_handler import: {err}")

    bridge, err = soft_import("inference_bridge")
    if bridge is None:
        bridge, err = soft_import("app.inference_bridge")
        if bridge is None:
            return _skip(f"inference_bridge import: {err}")

    # Synthetic mixed CSV — bridge supports CSV (Stage-1 + Stage-2 NonIoT path).
    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(
        dirs["artifacts"] / "loop.csv",
        n_iot=0, n_noniot=n_noniot, n_botnet=n_botnet,
        unique_src_ips=20,
    )
    info = fh.load_file(str(csv_path))
    if not info.is_valid:
        return _skip(f"file_handler refused: {info.error}")

    samples: list[float] = []
    errors: list[str]   = []
    n_results: list[int] = []
    for i in range(n_iters):
        try:
            res = bridge.run_file_inference(info)
            n_results.append(len(res) if res is not None else 0)
        except Exception as e:                                # noqa: BLE001
            errors.append(f"iter {i}: {type(e).__name__}: {e}")
        gc.collect()
        samples.append(_rss_mb())

    if errors:
        return {"test_id": TEST_ID, "name": "Repeated upload cycles",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": "no exceptions across iterations; bounded RSS growth",
                "actual":   f"{len(errors)} iterations raised; first: {errors[0]}",
                "raw":      {"errors": errors[:3], "samples": samples,
                             "n_results_per_iter": n_results}}

    if len(samples) < 4:
        verdict, actual = "PASS", f"only {len(samples)} samples"
        slope = 0.0
    else:
        # Linear regression over post-warmup tail (skip first 2 iters).
        x = list(range(len(samples)))[2:]
        y = samples[2:]
        n = len(x)
        mx = sum(x) / n; my = sum(y) / n
        slope = (sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
                 / max(sum((xi - mx) ** 2 for xi in x), 1e-9))
        if slope > 5.0:
            verdict = "FAIL"
            actual  = (f"RSS grows ~{slope:.1f} MB/iter post-warmup — "
                       "likely leak in run_file_inference / models.")
        else:
            verdict = "PASS"
            actual  = (f"RSS slope {slope:.2f} MB/iter post-warmup "
                       f"(samples: {[round(s, 1) for s in samples]})")
    return {"test_id": TEST_ID, "name": "Repeated upload cycles",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": "RSS slope ≤ 5 MB/iter",
            "actual":   actual,
            "raw":      {"samples_mb":         [round(s, 1) for s in samples],
                         "slope_mb_per_iter":  round(slope, 3),
                         "n_results_per_iter": n_results}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "Repeated upload cycles",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "bounded RSS over repeated runs",
            "actual":   reason}
