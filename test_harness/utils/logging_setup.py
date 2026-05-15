"""
logging_setup.py — per-test logging plumbing.

Each test gets:
    logs/<test_id>/stdout.txt
    logs/<test_id>/stderr.txt
    logs/<test_id>/run.log     (structured python logger, level INFO)
    logs/<test_id>/result.json (final, machine-readable)
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .paths import for_test


def make_logger(test_id: str) -> tuple[logging.Logger, dict[str, Path]]:
    """
    Return (logger, dirs). Logger writes to logs/<test_id>/run.log AND stdout.
    dirs is a dict: {"logs": ..., "artifacts": ...}
    """
    dirs = for_test(test_id)
    log_path = dirs["logs"] / "run.log"

    logger = logging.getLogger(f"harness.{test_id}")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if someone calls this twice with the same id.
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, dirs


@contextmanager
def capture_streams(test_id: str) -> Iterator[dict[str, Path]]:
    """
    Capture stdout/stderr to logs/<test_id>/{stdout,stderr}.txt while the
    block runs. The original streams are restored on exit. Useful when
    running tests in-process; subprocess tests get capture for free via
    Popen pipes.
    """
    dirs = for_test(test_id)
    out_path = dirs["logs"] / "stdout.txt"
    err_path = dirs["logs"] / "stderr.txt"
    saved_out, saved_err = sys.stdout, sys.stderr
    fout = open(out_path, "w", encoding="utf-8", errors="replace")
    ferr = open(err_path, "w", encoding="utf-8", errors="replace")
    try:
        sys.stdout = _Tee(saved_out, fout)
        sys.stderr = _Tee(saved_err, ferr)
        yield dirs
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err
        fout.close()
        ferr.close()


class _Tee:
    """Duplicate writes to two streams. Used by capture_streams."""
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # for libs that introspect
        return False


def write_result(test_id: str, payload: dict[str, Any]) -> Path:
    """Write logs/<test_id>/result.json. Returns the path."""
    dirs = for_test(test_id)
    p = dirs["logs"] / "result.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)
