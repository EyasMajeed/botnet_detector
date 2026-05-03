"""
inference_worker.py — Async PCAP inference worker.

QThread that wraps inference_bridge._run_pcap_inference but emits progress
signals during processing instead of blocking the main thread.

Same main-thread torch.load caveat as monitor_bridge.py: BotnetMonitor must
be CONSTRUCTED on the main thread (via ensure_monitor()) before start() is
called. The worker only does scapy I/O and inference forward passes.
"""

from __future__ import annotations

# Force torch single-threaded BEFORE the monitoring import pulls torch in.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")

import sys
import time
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Project root on sys.path so `from monitoring import …` works regardless
# of the cwd the GUI was launched from.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Heavy imports are deferred to first use — keeps GUI startup fast and
# allows the GUI to start even when monitoring.py / scapy are missing
# (the worker will fail gracefully through ensure_monitor's return value).
_BotnetMonitor = None
_scapy_syms    = None     # tuple (rdpcap, IP, TCP, UDP, ICMP, Ether)


def _lazy_imports() -> tuple[bool, str]:
    """Pull in monitoring.BotnetMonitor + scapy. Returns (ok, error_message)."""
    global _BotnetMonitor, _scapy_syms
    if _BotnetMonitor is not None and _scapy_syms is not None:
        return True, ""
    try:
        from monitoring import BotnetMonitor as _BM
        _BotnetMonitor = _BM
    except Exception as e:
        return False, f"monitoring import failed: {e!r}"
    try:
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Ether
        _scapy_syms = (rdpcap, IP, TCP, UDP, ICMP, Ether)
    except Exception as e:
        return False, f"scapy import failed: {e!r}"
    return True, ""


# Emit progress every N packets. 500 strikes a balance — enough updates
# that the bar feels live, few enough that signal queuing doesn't dominate.
PROGRESS_EVERY = 500


# ═════════════════════════════════════════════════════════════════════════════
# PcapInferenceThread
# ═════════════════════════════════════════════════════════════════════════════

class PcapInferenceThread(QThread):
    """
    Async PCAP inference thread.

    Lifecycle:
        __init__()        — store config, no I/O.
        ensure_monitor()  — construct BotnetMonitor on the CALLING thread.
                            On macOS this MUST be the main thread (torch.load
                            segfaults from worker threads when PyQt6 has
                            initialised libomp first). Returns True/False.
        start()           — begin the PCAP replay loop.
        cancel()          — request cancellation; thread exits within
                            ~PROGRESS_EVERY packets.

    Signals:
        progress(int, int, str)  — (current, total, status_text). When total
                                   is 0 the progress bar is rendered as
                                   indeterminate (e.g. during PCAP read).
        error(str)               — fatal error or cancellation reason.
        done(list)               — list of result dicts in the same shape
                                   as inference_bridge.run_file_inference.
    """

    progress = pyqtSignal(int, int, str)
    error    = pyqtSignal(str)
    done     = pyqtSignal(list)

    def __init__(self, pcap_path: str, parent=None):
        super().__init__(parent)
        self.pcap_path = pcap_path
        self._monitor: Optional[Any] = None
        self._init_error: Optional[str] = None
        self._cancelled = False

    # ── Main-thread construction ───────────────────────────────────────────
    def ensure_monitor(self) -> bool:
        """Idempotent. Returns True on success, False with self._init_error set."""
        if self._monitor is not None:
            return True
        if self._init_error is not None:
            return False
        ok, msg = _lazy_imports()
        if not ok:
            self._init_error = msg
            return False
        try:
            self._monitor = _BotnetMonitor()
            return True
        except FileNotFoundError as e:
            self._init_error = f"Missing model artifact: {e}"
        except Exception as e:
            self._init_error = f"BotnetMonitor init failed: {e!r}"
        return False

    # ── Public control ─────────────────────────────────────────────────────
    def cancel(self) -> None:
        """Request cancellation. The run loop checks this every PROGRESS_EVERY pkts."""
        self._cancelled = True

    # ── QThread entry ──────────────────────────────────────────────────────
    def run(self) -> None:
        if self._monitor is None:
            self.error.emit(
                self._init_error
                or "BotnetMonitor not initialised — call ensure_monitor() first."
            )
            return
        if _scapy_syms is None:
            self.error.emit("scapy not available")
            return
        rdpcap, IP, TCP, UDP, ICMP, Ether = _scapy_syms

        # ── Phase 1: read PCAP into memory ────────────────────────────────
        try:
            self.progress.emit(0, 0, "Reading PCAP file…")
            packets = rdpcap(self.pcap_path)
        except Exception as e:
            self.error.emit(f"PCAP read failed: {e!r}")
            return

        total = len(packets)
        if total == 0:
            self.error.emit("PCAP file contains no packets.")
            return

        # ── Phase 2: per-packet processing through BotnetMonitor ──────────
        detection_results = []
        t0 = time.perf_counter()

        for i, pkt in enumerate(packets):
            if self._cancelled:
                self.error.emit("Cancelled by user")
                return

            if IP not in pkt:
                # Still bump progress so the bar never appears stuck on
                # PCAPs full of non-IP traffic (ARP / loopback / etc).
                if (i + 1) % PROGRESS_EVERY == 0:
                    self.progress.emit(
                        i + 1, total,
                        f"Processing {i+1:,} / {total:,} packets · "
                        f"{len(detection_results)} flows completed",
                    )
                continue

            ip   = pkt[IP]
            ts   = float(pkt.time)
            mac  = pkt[Ether].src if Ether in pkt else ip.src
            plen = len(pkt)
            try:
                if TCP in pkt:
                    tcp = pkt[TCP]
                    r = self._monitor.process_packet(
                        ts, ip.src, ip.dst, int(tcp.sport), int(tcp.dport),
                        6, plen, int(ip.ttl), mac, int(tcp.flags),
                    )
                elif UDP in pkt:
                    udp = pkt[UDP]
                    r = self._monitor.process_packet(
                        ts, ip.src, ip.dst, int(udp.sport), int(udp.dport),
                        17, plen, int(ip.ttl), mac, 0,
                    )
                elif ICMP in pkt:
                    r = self._monitor.process_packet(
                        ts, ip.src, ip.dst, 0, 0,
                        1, plen, int(ip.ttl), mac, 0,
                    )
                else:
                    r = None
            except Exception:
                # One bad packet must not kill the entire job.
                r = None
            if r is not None:
                detection_results.append(r)

            if (i + 1) % PROGRESS_EVERY == 0:
                self.progress.emit(
                    i + 1, total,
                    f"Processing {i+1:,} / {total:,} packets · "
                    f"{len(detection_results)} flows completed",
                )

        # ── Phase 3: force-flush any still-open flows ─────────────────────
        self.progress.emit(total, total, "Finalising — flushing remaining flows…")
        try:
            # _idle = -1e9 makes (now - last_seen) > _idle always true, so
            # every open flow is exported and inferenced regardless of
            # historical PCAP timestamps. See inference_bridge for context.
            self._monitor.aggregator._idle = -1e9
            detection_results.extend(self._monitor.flush_idle_flows())
        except Exception as e:
            self.error.emit(f"final flush error: {e!r}")
            return

        # ── Phase 4: convert DetectionResult → result dicts ───────────────
        try:
            from inference_bridge import _detection_results_to_dicts
            out = _detection_results_to_dicts(detection_results, t0)
        except Exception as e:
            self.error.emit(f"result conversion failed: {e!r}")
            return

        self.done.emit(out)