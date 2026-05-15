"""
Parser safety tests targeting app/file_handler.load_file.

Each test produces an exotic input, calls load_file, and asserts the
expected verdict. The framework executes each in a subprocess so a
parser bug taking down the whole test process is contained.
"""

from __future__ import annotations

from pathlib import Path

from test_harness.generators import file_corruption as fc
from test_harness.generators.pcap_gen import (
    fake_magic_pcap, malformed_eth_pcap, pcapng_truncated,
)
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import


def _load(path: str):
    # Mirror inference_bridge: app/ is on sys.path so `import file_handler`
    # works; fall back to `app.file_handler` if not.
    from test_harness.utils.project_imports import ensure_on_path
    ensure_on_path()
    fh, err = soft_import("file_handler")
    if fh is None:
        fh, err2 = soft_import("app.file_handler")
        if fh is None:
            return None, f"file_handler: {err} | app.file_handler: {err2}"
    try:
        info = fh.load_file(path)
    except Exception as e:                                    # noqa: BLE001
        return {"_exception": f"{type(e).__name__}: {e}"}, None
    return {
        "is_valid":   bool(getattr(info, "is_valid", False)),
        "format":     str(getattr(info, "format", None)),
        "error":      str(getattr(info, "error", "") or ""),
        "size_mb":    float(getattr(info, "size_mb", 0.0) or 0.0),
        "row_count":  getattr(info, "row_count", None),
    }, None


# ── F-01 PCAPNG header then EOF ──────────────────────────────────────────
def t_pcapng_truncated() -> dict:
    TEST_ID = "F-01"
    dirs = for_test(TEST_ID)
    p = pcapng_truncated(dirs["artifacts"] / "trunc.pcapng")
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "Truncated PCAPNG", "MEDIUM", skip)
    # Detector detects PCAPNG by magic, so is_valid=True is acceptable
    # AS LONG AS no exception was raised (downstream parser is supposed
    # to handle the truncation, not the validator).
    if info.get("_exception"):
        return _fail(TEST_ID, "Truncated PCAPNG", "HIGH",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo without raising",
                     str(p), info)
    return _pass(TEST_ID, "Truncated PCAPNG", "MEDIUM",
                 f"validator handled cleanly: {info}", str(p), info)


# ── F-02 Claimed-huge PCAP (snaplen=0xFFFFFFFF) ──────────────────────────
def t_pcap_huge_claimed() -> dict:
    TEST_ID = "F-02"
    dirs = for_test(TEST_ID)
    p = fc.claimed_huge_pcap_header(dirs["artifacts"] / "huge_claim.pcap")
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "Claimed-huge PCAP header", "HIGH", skip)
    if info.get("_exception"):
        return _fail(TEST_ID, "Claimed-huge PCAP header", "HIGH",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo without raising", str(p), info)
    return _pass(TEST_ID, "Claimed-huge PCAP header", "HIGH",
                 f"detected as PCAP without parsing body: {info}", str(p), info)


# ── F-03 Oversized CSV (just under 500 MB cap, structurally valid) ───────
def t_csv_oversized() -> dict:
    TEST_ID = "F-03"
    dirs = for_test(TEST_ID)
    # 50 MB is enough to verify behaviour without exhausting CI disks.
    p = fc.oversized_csv(dirs["artifacts"] / "huge.csv", target_mb=50)
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "Oversized CSV", "MEDIUM", skip)
    if info.get("_exception"):
        return _fail(TEST_ID, "Oversized CSV", "MEDIUM",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo without raising", str(p), info)
    return _pass(TEST_ID, "Oversized CSV", "MEDIUM",
                 f"detected: {info}", str(p), info)


# ── F-04 Header-only CSV ─────────────────────────────────────────────────
def t_csv_header_only() -> dict:
    TEST_ID = "F-04"
    dirs = for_test(TEST_ID)
    p = fc.header_only_csv(dirs["artifacts"] / "header_only.csv")
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "Header-only CSV", "MEDIUM", skip)
    if info.get("_exception"):
        return _fail(TEST_ID, "Header-only CSV", "MEDIUM",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo, possibly is_valid=False",
                     str(p), info)
    return _pass(TEST_ID, "Header-only CSV", "MEDIUM",
                 f"validator did not crash: {info}", str(p), info)


# ── F-05 Mismatched extension (PCAP renamed .csv) ────────────────────────
def t_pcap_renamed_csv() -> dict:
    TEST_ID = "F-05"
    dirs = for_test(TEST_ID)
    src = fake_magic_pcap(dirs["artifacts"] / "src.pcap")
    p   = dirs["artifacts"] / "src_renamed.csv"
    p.write_bytes(src.read_bytes())
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "PCAP renamed .csv", "LOW", skip)
    fmt = (info or {}).get("format", "")
    if "PCAP" in fmt:
        return _pass(TEST_ID, "PCAP renamed .csv", "LOW",
                     f"magic-byte detection wins over extension: {info}",
                     str(p), info)
    return _fail(TEST_ID, "PCAP renamed .csv", "LOW",
                 f"format=={fmt}, extension trumped magic", "format == PCAP",
                 str(p), info)


# ── F-07 UTF-16 BOM CSV ──────────────────────────────────────────────────
def t_csv_utf16() -> dict:
    TEST_ID = "F-07"
    dirs = for_test(TEST_ID)
    p = fc.utf16_bom_csv(dirs["artifacts"] / "utf16.csv")
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "UTF-16 CSV", "HIGH", skip)
    # The pass condition is "detected and rejected with a clear error" OR
    # "detected and warned". Silently classified as a known schema is a fail.
    fmt = (info or {}).get("format", "")
    is_valid = (info or {}).get("is_valid", False)
    if info.get("_exception"):
        return _fail(TEST_ID, "UTF-16 CSV", "HIGH",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo without raising",
                     str(p), info)
    if is_valid and "GENERIC" not in fmt and "UNIFIED" not in fmt:
        return _fail(TEST_ID, "UTF-16 CSV", "HIGH",
                     f"silently accepted as {fmt}; columns unreliable",
                     "rejected or flagged with a warning", str(p), info)
    return _pass(TEST_ID, "UTF-16 CSV", "HIGH",
                 f"validator flagged or rejected: {info}", str(p), info)


# ── F-08 Random-bytes fuzz ───────────────────────────────────────────────
def t_random_bytes_fuzz(n_iters: int = 32) -> dict:
    TEST_ID = "F-08"
    dirs = for_test(TEST_ID)
    crashes = []
    for i in range(n_iters):
        p = fc.fuzz_random_bytes(dirs["artifacts"] / f"fuzz_{i:03d}.bin",
                                 size_bytes=4096, seed=i)
        info, skip = _load(str(p))
        if skip:
            return _skip(TEST_ID, "Parser fuzzing", "MEDIUM", skip)
        if info.get("_exception"):
            crashes.append({"seed": i, "exception": info["_exception"],
                            "file": str(p)})
    if crashes:
        return _fail(TEST_ID, "Parser fuzzing", "MEDIUM",
                     f"{len(crashes)}/{n_iters} fuzz inputs raised exceptions",
                     "0 raised exceptions; all return is_valid=False cleanly",
                     None, {"crashes_sample": crashes[:5]})
    return _pass(TEST_ID, "Parser fuzzing", "MEDIUM",
                 f"{n_iters}/{n_iters} fuzz inputs handled cleanly",
                 None, {"n_iters": n_iters})


# ── F-09 Zip bomb disguised as pcap ──────────────────────────────────────
def t_zip_bomb() -> dict:
    TEST_ID = "F-09"
    dirs = for_test(TEST_ID)
    p = fc.zip_bomb_disguised_as_pcap(dirs["artifacts"] / "bomb.pcap")
    info, skip = _load(str(p))
    if skip:
        return _skip(TEST_ID, "Zip bomb disguised as pcap", "MEDIUM", skip)
    if info.get("_exception"):
        return _fail(TEST_ID, "Zip bomb disguised as pcap", "MEDIUM",
                     f"validator raised: {info['_exception']}",
                     "validator returns FileInfo without raising",
                     str(p), info)
    fmt = (info or {}).get("format", "")
    if "PCAP" in fmt:
        return _fail(TEST_ID, "Zip bomb disguised as pcap", "MEDIUM",
                     f"identified as PCAP despite ZIP magic: {info}",
                     "format != PCAP / explicit rejection", str(p), info)
    return _pass(TEST_ID, "Zip bomb disguised as pcap", "MEDIUM",
                 f"correctly not classified as PCAP: {info}", str(p), info)


# ── Helpers ──────────────────────────────────────────────────────────────

def _pass(test_id: str, name: str, severity: str, actual: str,
          artifact: str | None, raw) -> dict:
    return {"test_id": test_id, "name": name, "severity": severity,
            "verdict": "PASS", "expected": "no crash; correct verdict",
            "actual": actual,
            "pcap": artifact, "raw": raw}


def _fail(test_id: str, name: str, severity: str, actual: str,
          expected: str, artifact: str | None, raw) -> dict:
    return {"test_id": test_id, "name": name, "severity": severity,
            "verdict": "FAIL", "expected": expected, "actual": actual,
            "pcap": artifact, "raw": raw}


def _skip(test_id: str, name: str, severity: str, why: str) -> dict:
    return {"test_id": test_id, "name": name, "severity": severity,
            "verdict": "SKIPPED", "expected": "n/a", "actual": why,
            "pcap": None, "raw": None}


# ── Single entrypoint to run all parser tests in one go. ─────────────────
def run_all() -> dict:
    out = []
    for fn in [t_pcapng_truncated, t_pcap_huge_claimed, t_csv_oversized,
               t_csv_header_only, t_pcap_renamed_csv, t_csv_utf16,
               t_random_bytes_fuzz, t_zip_bomb]:
        try:
            out.append(fn())
        except Exception as e:                                # noqa: BLE001
            out.append({"test_id": fn.__name__, "verdict": "ERROR",
                        "actual": f"{type(e).__name__}: {e}"})
    return {"results": out, "n": len(out)}
