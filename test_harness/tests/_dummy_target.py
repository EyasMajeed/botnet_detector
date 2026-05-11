"""
Dummy targets used by the harness self-tests. Lives inside the package
so the subprocess runner can import it by dotted name.
"""
from __future__ import annotations

import time


def hi(name: str = "world") -> dict:
    return {
        "test_id":  "SELF-001",
        "name":     "hello",
        "severity": "LOW",
        "verdict":  "PASS",
        "expected": "_",
        "actual":   name,
    }


def loop() -> dict:
    while True:
        time.sleep(0.5)


def boom() -> dict:
    raise RuntimeError("intentional boom for harness self-test")
