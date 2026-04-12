"""
file_handler.py  —  File Validation & Format Detection
=======================================================
Validates uploaded traffic files and detects their format without
parsing the full file. Returns a structured FileInfo dataclass that
the UploadPage displays and the inference_bridge receives.

Supported formats:
    PCAP / PCAPng   — detected via magic bytes
    CSV (various)   — CICFlowMeter, CTU-13 binetflow, UNSW-NB15,
                      unified schema, or generic flow CSV
    NetFlow / IPFIX — nfdump binary OR nfdump ASCII CSV export

What this module does NOT do:
    - Parse full packet data
    - Extract features
    - Run inference
    Those are handled by your teammates' preprocessing pipeline via
    inference_bridge.py.

Usage:
    from app.file_handler import load_file, FileInfo, FileFormat

    info = load_file("/path/to/capture.pcap")
    if info.is_valid:
        print(info.format, info.row_count, info.size_mb)
    else:
        print(info.error)
"""

from __future__ import annotations

import csv
import os
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 500
SAMPLE_ROWS      = 5        # rows to peek at for CSV sub-format detection
CSV_PEEK_BYTES   = 8192     # bytes to read for header sniffing


# ══════════════════════════════════════════════════════════════════════════════
# Format enum
# ══════════════════════════════════════════════════════════════════════════════

class FileFormat(Enum):
    PCAP          = auto()   # classic libpcap  (.pcap)
    PCAPNG        = auto()   # next-gen capture (.pcapng)
    CSV_CICFLOW   = auto()   # CICFlowMeter output
    CSV_CTU13     = auto()   # CTU-13 binetflow / Argus
    CSV_UNSW      = auto()   # UNSW-NB15 standard CSV
    CSV_UNIFIED   = auto()   # your team's 56-feature unified schema
    CSV_GENERIC   = auto()   # CSV with flow-like columns, unknown schema
    NETFLOW_BIN   = auto()   # nfdump binary  (.nfcapd / .nfdump)
    NETFLOW_CSV   = auto()   # nfdump -o csv ASCII export
    UNKNOWN       = auto()


# ══════════════════════════════════════════════════════════════════════════════
# FileInfo dataclass  — the single object passed around the app
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileInfo:
    # ── Basic ──────────────────────────────────────────────────────────────
    path:        str          = ""
    filename:    str          = ""
    extension:   str          = ""
    size_bytes:  int          = 0

    # ── Detection results ──────────────────────────────────────────────────
    format:      FileFormat   = FileFormat.UNKNOWN
    format_label: str         = "Unknown"    # human-readable, e.g. "PCAP (libpcap)"
    is_valid:    bool         = False
    error:       str          = ""
    warnings:    list[str]    = field(default_factory=list)

    # ── Quick stats (from peeking, not full parse) ─────────────────────────
    row_count:   Optional[int] = None   # CSV: estimated rows; PCAP: None
    col_count:   Optional[int] = None   # CSV: column count
    columns:     list[str]    = field(default_factory=list)
    sample_rows: list[dict]   = field(default_factory=list)  # first SAMPLE_ROWS rows

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def size_label(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 ** 2:
            return f"{self.size_bytes / 1024:.1f} KB"
        if self.size_bytes < 1024 ** 3:
            return f"{self.size_bytes / (1024**2):.1f} MB"
        return f"{self.size_bytes / (1024**3):.2f} GB"

    @property
    def icon(self) -> str:
        icons = {
            FileFormat.PCAP:        "📦",
            FileFormat.PCAPNG:      "📦",
            FileFormat.CSV_CICFLOW: "📊",
            FileFormat.CSV_CTU13:   "📊",
            FileFormat.CSV_UNSW:    "📊",
            FileFormat.CSV_UNIFIED: "📊",
            FileFormat.CSV_GENERIC: "📄",
            FileFormat.NETFLOW_BIN: "🌐",
            FileFormat.NETFLOW_CSV: "🌐",
            FileFormat.UNKNOWN:     "❓",
        }
        return icons.get(self.format, "📄")

    @property
    def ready_for_inference(self) -> bool:
        """True when the inference bridge can accept this file."""
        return self.is_valid and self.format != FileFormat.UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_file(path: str) -> FileInfo:
    """
    Main entry point. Validates and detects the format of a traffic file.

    Args:
        path: Absolute path to the uploaded file.

    Returns:
        FileInfo with all fields populated.
    """
    p = Path(path)
    info = FileInfo(
        path      = str(p),
        filename  = p.name,
        extension = p.suffix.lower(),
    )

    # ── 1. Existence check ─────────────────────────────────────────────────
    if not p.exists():
        info.error = f"File not found: {path}"
        return info

    if not p.is_file():
        info.error = "Path is not a regular file."
        return info

    info.size_bytes = p.stat().st_size

    # ── 2. Size check ──────────────────────────────────────────────────────
    if info.size_bytes == 0:
        info.error = "File is empty."
        return info

    if info.size_mb > MAX_FILE_SIZE_MB:
        info.error = (
            f"File is too large ({info.size_label}). "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )
        return info

    # ── 3. Format detection ────────────────────────────────────────────────
    try:
        _detect_format(info)
    except Exception as e:
        info.error = f"Format detection failed: {e}"
        return info

    if info.format == FileFormat.UNKNOWN:
        info.error = (
            f"Unrecognised file format (extension: '{info.extension}'). "
            "Supported: .pcap, .pcapng, .csv, .txt, .log, .nfcapd, .nfdump"
        )
        return info

    info.is_valid = True
    return info


# ══════════════════════════════════════════════════════════════════════════════
# Format detection internals
# ══════════════════════════════════════════════════════════════════════════════

# PCAP magic bytes
_PCAP_MAGIC_LE   = b'\xd4\xc3\xb2\xa1'   # little-endian
_PCAP_MAGIC_BE   = b'\xa1\xb2\xc3\xd4'   # big-endian
_PCAP_MAGIC_NS   = b'\x4d\x3c\xb2\xa1'   # nanosecond timestamps
_PCAPNG_MAGIC    = b'\x0a\x0d\x0d\x0a'   # Section Header Block

# NetFlow nfdump binary magic
_NFCAPD_MAGIC    = b'nfca'


def _detect_format(info: FileInfo) -> None:
    """Dispatch to the appropriate detector based on extension + magic bytes."""
    ext = info.extension

    # ── Try binary magic first (works regardless of extension) ────────────
    magic = _read_magic(info.path, 12)

    if magic[:4] in (_PCAP_MAGIC_LE, _PCAP_MAGIC_BE, _PCAP_MAGIC_NS):
        info.format       = FileFormat.PCAP
        info.format_label = "PCAP (libpcap)"
        return

    if magic[:4] == _PCAPNG_MAGIC:
        info.format       = FileFormat.PCAPNG
        info.format_label = "PCAPng (next-gen capture)"
        return

    if magic[:4] == _NFCAPD_MAGIC:
        info.format       = FileFormat.NETFLOW_BIN
        info.format_label = "NetFlow binary (nfdump)"
        return

    # ── Extension-guided text detection ────────────────────────────────────
    if ext in ('.pcap',):
        # Extension says PCAP but magic didn't match → corrupted
        info.format = FileFormat.UNKNOWN
        info.error  = "File has .pcap extension but invalid magic bytes — may be corrupted."
        return

    if ext in ('.pcapng',):
        info.format = FileFormat.UNKNOWN
        info.error  = "File has .pcapng extension but invalid magic bytes — may be corrupted."
        return

    if ext in ('.nfcapd', '.nfdump'):
        # Could be a CSV export from nfdump (ASCII)
        _detect_netflow_csv(info)
        return

    if ext in ('.csv', '.txt', '.log', '.binetflow', '.biargus', ''):
        _detect_csv_subformat(info)
        return

    # No match
    info.format = FileFormat.UNKNOWN


def _read_magic(path: str, n: int = 12) -> bytes:
    try:
        with open(path, 'rb') as f:
            return f.read(n)
    except OSError:
        return b''


# ── CSV sub-format detection ──────────────────────────────────────────────────

# Signature column sets for each known CSV schema
_CSV_SIGNATURES: dict[FileFormat, set[str]] = {
    FileFormat.CSV_UNIFIED: {
        # your team's 56-feature schema (config.py)
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "bytes_per_second_window", "periodicity_score",
    },
    FileFormat.CSV_CICFLOW: {
        # CICFlowMeter standard output
        "flow duration", "total fwd packets", "flow bytes/s",
        "fwd packet length mean",
    },
    FileFormat.CSV_CTU13: {
        # CTU-13 binetflow / Argus
        "starttime", "dur", "proto", "srcaddr", "sport",
        "dir", "dstaddr", "dport", "state", "totpkts", "totbytes",
    },
    FileFormat.CSV_UNSW: {
        # UNSW-NB15 standard CSV
        "dur", "spkts", "dpkts", "sbytes", "dbytes",
        "proto", "state", "attack_cat",
    },
    FileFormat.NETFLOW_CSV: {
        # nfdump -o csv ASCII export
        "ts", "te", "td", "sa", "da", "sp", "dp",
        "pr", "ipkt", "ibyt",
    },
}

# Minimum fraction of signature columns that must match
_MATCH_THRESHOLD = 0.6


def _detect_csv_subformat(info: FileInfo) -> None:
    """
    Read the CSV header + a few rows, match against known schemas,
    and populate info.columns, info.row_count, info.sample_rows.
    """
    try:
        with open(info.path, 'r', encoding='utf-8', errors='replace') as f:
            peek = f.read(CSV_PEEK_BYTES)
    except OSError as e:
        info.error = f"Could not read file: {e}"
        return

    if not peek.strip():
        info.error = "File appears to be empty or unreadable as text."
        return

    # Detect delimiter
    try:
        dialect = csv.Sniffer().sniff(peek, delimiters=',\t|; ')
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ','

    # Parse header + sample rows
    try:
        with open(info.path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            raw_cols = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                if i >= SAMPLE_ROWS:
                    break
                rows.append(dict(row))
    except Exception as e:
        info.error = f"CSV parse error: {e}"
        return

    if not raw_cols:
        info.error = "CSV file has no header row."
        return

    info.columns    = [c.strip() for c in raw_cols]
    info.col_count  = len(info.columns)
    info.sample_rows = rows

    # Estimate total row count without reading the whole file
    info.row_count = _estimate_row_count(info.path, delimiter)

    # Match columns against known schemas
    cols_lower = {c.strip().lower() for c in raw_cols}
    best_format = FileFormat.CSV_GENERIC
    best_score  = 0.0

    for fmt, sig in _CSV_SIGNATURES.items():
        matches = len(sig & cols_lower)
        score   = matches / len(sig)
        if score > best_score and score >= _MATCH_THRESHOLD:
            best_score  = score
            best_format = fmt

    # Label descriptions
    labels = {
        FileFormat.CSV_UNIFIED:  "CSV — Unified 56-feature schema (your pipeline)",
        FileFormat.CSV_CICFLOW:  "CSV — CICFlowMeter output",
        FileFormat.CSV_CTU13:    "CSV — CTU-13 / Argus binetflow",
        FileFormat.CSV_UNSW:     "CSV — UNSW-NB15 standard",
        FileFormat.NETFLOW_CSV:  "CSV — nfdump NetFlow export",
        FileFormat.CSV_GENERIC:  "CSV — Generic flow CSV (schema unknown)",
    }

    info.format       = best_format
    info.format_label = labels[best_format]

    # Warnings for generic CSV
    if best_format == FileFormat.CSV_GENERIC:
        info.warnings.append(
            "Schema not recognised. Your teammates' preprocessing pipeline "
            "will attempt to handle it via column-mapping heuristics."
        )

    # Warn if very few rows
    if info.row_count is not None and info.row_count < 10:
        info.warnings.append(
            f"Only {info.row_count} data rows found — file may be a sample or truncated."
        )


def _detect_netflow_csv(info: FileInfo) -> None:
    """Handle .nfcapd/.nfdump files that might be ASCII CSV exports."""
    # Try parsing as CSV
    info_copy = FileInfo(path=info.path, filename=info.filename,
                         extension='.csv', size_bytes=info.size_bytes)
    _detect_csv_subformat(info_copy)

    if info_copy.format == FileFormat.NETFLOW_CSV:
        info.format       = FileFormat.NETFLOW_CSV
        info.format_label = "NetFlow CSV (nfdump ASCII export)"
        info.columns      = info_copy.columns
        info.col_count    = info_copy.col_count
        info.row_count    = info_copy.row_count
        info.sample_rows  = info_copy.sample_rows
    else:
        # Unrecognised binary
        info.format       = FileFormat.UNKNOWN
        info.error        = (
            "File has a NetFlow extension but could not be parsed as "
            "nfdump binary or ASCII CSV. "
            "Export from nfdump with: nfdump -r <file> -o csv > output.csv"
        )


# ── Row count estimator ────────────────────────────────────────────────────────

def _estimate_row_count(path: str, delimiter: str = ',') -> Optional[int]:
    """
    Estimate row count by sampling the first 64 KB and extrapolating.
    Fast even for 500 MB files.
    """
    SAMPLE_SIZE = 64 * 1024
    try:
        file_size = os.path.getsize(path)
        with open(path, 'rb') as f:
            sample = f.read(SAMPLE_SIZE)

        if not sample:
            return None

        newlines_in_sample = sample.count(b'\n')
        if newlines_in_sample == 0:
            return 1

        avg_row_bytes = len(sample) / newlines_in_sample
        estimated     = int(file_size / avg_row_bytes)

        # Subtract header row and clamp to 0
        return max(0, estimated - 1)

    except OSError:
        return None
