"""
assertions.py — Domain-specific assertion helpers.

These are NOT pytest assertions. They return (passed: bool, detail: str)
so the harness can record a structured pass/fail with a human-readable
explanation in the result.json file.
"""

from __future__ import annotations

from typing import Any, Iterable


def expect(cond: bool, detail: str) -> tuple[bool, str]:
    return bool(cond), detail


def expect_recall_at_least(recall: float, threshold: float) -> tuple[bool, str]:
    ok = recall >= threshold
    return ok, f"recall={recall:.4f} target>={threshold:.4f}"


def expect_precision_at_least(p: float, threshold: float) -> tuple[bool, str]:
    ok = p >= threshold
    return ok, f"precision={p:.4f} target>={threshold:.4f}"


def expect_no_crash(rc: int, timed_out: bool) -> tuple[bool, str]:
    if timed_out:
        return False, f"timed out (rc={rc})"
    if rc != 0:
        return False, f"non-zero exit ({rc})"
    return True, "exited cleanly"


def expect_keys_present(d: dict, keys: Iterable[str]) -> tuple[bool, str]:
    missing = [k for k in keys if k not in d]
    if missing:
        return False, f"missing keys: {missing}"
    return True, f"all {len(list(keys))} keys present"


def expect_finite(name: str, value: Any) -> tuple[bool, str]:
    """value must be a finite float."""
    try:
        f = float(value)
    except Exception:
        return False, f"{name} is not a number ({value!r})"
    import math
    if math.isnan(f) or math.isinf(f):
        return False, f"{name} is non-finite ({f!r})"
    return True, f"{name} = {f} (finite)"


def expect_in_range(name: str, value: float,
                    lo: float, hi: float) -> tuple[bool, str]:
    ok = lo <= value <= hi
    return ok, f"{name}={value:g} target in [{lo:g}, {hi:g}]"


def expect_artifact_exists(path) -> tuple[bool, str]:
    from pathlib import Path
    p = Path(str(path))
    return (p.exists(), f"artifact {p}: " + ("exists" if p.exists() else "missing"))


def combine(*checks: tuple[bool, str]) -> tuple[bool, list[str]]:
    """Collapse multiple checks into a single (all_pass, details) tuple."""
    all_ok = all(c[0] for c in checks)
    return all_ok, [f"{'PASS' if c[0] else 'FAIL'}: {c[1]}" for c in checks]
