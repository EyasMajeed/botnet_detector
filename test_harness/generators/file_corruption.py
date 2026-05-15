"""
file_corruption.py — Generic file-level corruption helpers.

Used by parser_safety tests to construct malicious inputs without needing
scapy. Everything is plain stdlib.
"""

from __future__ import annotations

import os
import random
import struct
import zlib
from pathlib import Path


def oversized_csv(out: Path, target_mb: int = 50,
                  cols: int = 56, seed: int = 0) -> Path:
    """Write a CSV with a header and as many junk rows as needed to hit
    ~target_mb. We cap individual cell sizes so the file is structurally
    valid CSV, just large."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    header = ",".join(f"col{i}" for i in range(cols)) + "\n"
    target = target_mb * 1024 * 1024
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        written = len(header)
        while written < target:
            row = ",".join(str(rng.randint(0, 99999)) for _ in range(cols)) + "\n"
            f.write(row)
            written += len(row)
    return out


def long_line_csv(out: Path, line_bytes: int = 5_000_000) -> Path:
    """Single 5-MB row — designed to break naive CSV parsing."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"col1,col2,col3\n")
        f.write(b"a" * line_bytes + b",b,c\n")
    return out


def zip_bomb_disguised_as_pcap(out: Path,
                               compression_ratio: int = 1000) -> Path:
    """
    A small file whose magic bytes are NOT a real PCAP — but whose
    extension is .pcap and whose deflate body inflates ~`compression_ratio`x.

    Tests that file_handler rejects via magic-byte check rather than
    extension trust, AND that downstream parsers don't try to inflate.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # 10 MB of zeros compresses to a few hundred bytes
    body = zlib.compress(b"\x00" * (compression_ratio * 1024), 9)
    with open(out, "wb") as f:
        # Generic ZIP magic, NOT pcap magic.
        f.write(b"PK\x03\x04" + b"\x00" * 8 + body)
    return out


def fuzz_random_bytes(out: Path, size_bytes: int = 4096, seed: int = 0) -> Path:
    """Pseudo-random bytes (deterministic per seed) for parser fuzzing."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(out, "wb") as f:
        f.write(bytes(rng.randint(0, 255) for _ in range(size_bytes)))
    return out


def utf16_bom_csv(out: Path, n_rows: int = 100, cols: int = 56) -> Path:
    """CSV in UTF-16-LE with BOM — should not be silently accepted as UTF-8."""
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"col{i}" for i in range(cols)) + "\n"
    body = "\n".join(",".join("0" for _ in range(cols)) for _ in range(n_rows))
    text = header + body
    with open(out, "wb") as f:
        f.write(b"\xff\xfe")            # UTF-16-LE BOM
        f.write(text.encode("utf-16-le"))
    return out


def header_only_csv(out: Path, cols: int = 56) -> Path:
    """Valid header, zero rows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(f"col{i}" for i in range(cols)) + "\n")
    return out


def claimed_huge_pcap_header(out: Path) -> Path:
    """
    libpcap header that claims snaplen=0xFFFFFFFF, then a few packets
    with sane lengths. A naive parser that allocates `snaplen` bytes
    per record before reading will OOM.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        # global header: magic, ver_major, ver_minor, thiszone, sigfigs,
        # snaplen, linktype
        f.write(struct.pack("<IHHIIII",
                            0xA1B2C3D4, 2, 4, 0, 0, 0xFFFFFFFF, 1))
        # 5 plausible 64-byte records
        for i in range(5):
            f.write(struct.pack("<IIII", 1_700_000_000 + i, 0, 64, 64))
            f.write(b"\x00" * 64)
    return out
