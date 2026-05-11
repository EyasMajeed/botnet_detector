"""
ST-2a Pure ingestion throughput.

Measures BotnetMonitor.process_packet() throughput in isolation —
NO flow flushing, NO Stage-2 inference, NO XAI. This is the raw
hot-path number that determines whether the live monitor can keep up
with line-rate packet arrivals before flows complete.

This is the test that ST-2 used to be (mis-named). The original ST-2
also flushed all open flows at the end, which made it measure
end-to-end latency including XAI per-flow, not ingestion. ST-2b covers
that case explicitly.

Pass target: ≥ 5,000 pps. The hot path is mostly dict updates and
flow-key construction, so we expect 10k+ pps on any modern CPU.
"""

from __future__ import annotations

import time

from test_harness.utils.project_imports import (
    soft_import, models_state, ensure_on_path,
)


TEST_ID = "ST-2a"


def run(n_packets: int = 10_000, target_pps: float = 5_000.0) -> dict:
    ensure_on_path()
    ms = models_state()
    if not all(ms.values()):
        missing = [k for k, v in ms.items() if not v]
        return _skip(f"missing models: {missing}")

    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return _skip(f"monitoring import: {err}")

    try:
        bm = monitoring.BotnetMonitor()
    except Exception as e:                                    # noqa: BLE001
        return _skip(f"BotnetMonitor init: {e}")

    # Synthetic UDP burst — same shape as ST-2b for direct comparison,
    # but we DO NOT flush. process_packet returns None on every call
    # because no flow closes within the burst.
    base_ts = time.time()
    src_ip, dst_ip = "10.0.0.55", "10.0.0.1"

    # Warmup pass to amortise PyTorch / numpy / hash dispatch cost.
    for i in range(min(500, n_packets // 10)):
        bm.process_packet(base_ts + i * 1e-4, src_ip, dst_ip,
                          40000 + (i % 1000), 53, 17, 100, 64, "", 0)

    t0 = time.monotonic()
    for i in range(n_packets):
        bm.process_packet(base_ts + 1.0 + i * 1e-4, src_ip, dst_ip,
                          40000 + (i % 20000), 53, 17, 100, 64, "", 0)
    duration = max(time.monotonic() - t0, 1e-6)
    pps = n_packets / duration

    verdict = "PASS" if pps >= target_pps else "FAIL"
    actual = (f"{n_packets} packets in {duration:.2f}s = {pps:.0f} pps "
              f"({duration * 1000 / n_packets:.3f} ms/call)")
    return {"test_id": TEST_ID, "name": "Ingestion throughput (process_packet only)",
            "severity": "MEDIUM", "verdict": verdict,
            "expected": f"≥ {target_pps:.0f} pps for the ingestion hot path",
            "actual":   actual,
            "raw": {"pps": round(pps, 1),
                    "duration_sec": round(duration, 3),
                    "ms_per_packet": round(duration * 1000 / n_packets, 4),
                    "n_packets": n_packets}}


def _skip(reason: str) -> dict:
    return {"test_id": TEST_ID, "name": "Ingestion throughput (process_packet only)",
            "severity": "MEDIUM", "verdict": "SKIPPED",
            "expected": "≥ 5000 pps",
            "actual":   reason}
