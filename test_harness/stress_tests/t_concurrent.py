"""
ST-4 Concurrent-upload simulation.

The GUI doesn't support concurrent uploads, but if the orchestrator
ever ships a batch UI it'd run multiple inference jobs in parallel. We
fire run_file_inference (the CSV path) from N threads simultaneously
to verify the bridge is thread-safe — or surface that it isn't.

Same CSV-vs-PCAP rationale as G-RU: run_file_inference accepts only CSV.
PCAP concurrency would need parallel PcapInferenceThread instances,
which is GUI-side and not what this test targets.
"""

from __future__ import annotations

import threading
import time

from test_harness.generators.flow_csv_gen import synthetic_csv
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import, ensure_on_path


TEST_ID = "ST-4"


def run(n_threads: int = 4, n_iters: int = 3,
        n_botnet: int = 30, n_noniot: int = 60) -> dict:
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

    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(
        dirs["artifacts"] / "concurrent.csv",
        n_iot=0, n_noniot=n_noniot, n_botnet=n_botnet,
        unique_src_ips=20,
    )
    info = fh.load_file(str(csv_path))
    if not info.is_valid:
        return _skip(f"file_handler refused: {info.error}")

    errors: list[str]    = []
    durations: list[float] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        for it in range(n_iters):
            t0 = time.monotonic()
            try:
                bridge.run_file_inference(info)
            except Exception as e:                            # noqa: BLE001
                with lock:
                    errors.append(f"t{idx}.{it}: {type(e).__name__}: {e}")
            with lock:
                durations.append(time.monotonic() - t0)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    t0 = time.monotonic()
    for t in threads: t.start()
    for t in threads: t.join(timeout=120)
    wall = time.monotonic() - t0

    if errors:
        return {"test_id": TEST_ID, "name": "Concurrent uploads",
                "severity": "MEDIUM", "verdict": "FAIL",
                "expected": f"{n_threads}×{n_iters} uploads complete cleanly",
                "actual":   f"{len(errors)} errors; first: {errors[0]}",
                "raw":      {"errors_sample": errors[:5],
                             "wall_sec": round(wall, 2)}}
    return {"test_id": TEST_ID, "name": "Concurrent uploads",
            "severity": "MEDIUM", "verdict": "PASS",
            "expected": f"{n_threads}×{n_iters} uploads complete cleanly",
            "actual":   (f"{len(durations)} runs in {wall:.2f}s, "
                         f"avg {sum(durations) / max(len(durations), 1):.2f}s/run"),
            "raw":      {"wall_sec":    round(wall, 2),
                         "avg_run_sec": round(sum(durations)
                                              / max(len(durations), 1), 3),
                         "n_runs":      len(durations)}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "Concurrent uploads",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "concurrent uploads complete cleanly",
            "actual":   reason}
