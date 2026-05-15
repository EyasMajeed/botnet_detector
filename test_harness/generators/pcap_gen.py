"""
pcap_gen.py — Scapy-based PCAP generators.

Every function returns the path to the produced PCAP. Scapy is the only
hard dependency; everything is plain stdlib. If scapy is unavailable a
clear ImportError is raised — the harness records the test as SKIPPED.

Generators implemented (one function each):
    benign_tcp_pcap            simple back-and-forth TCP exchanges
    benign_iot_pcap            MQTT-port (1883) telemetry pattern
    ipv6_only_pcap             pure IPv6 TCP traffic
    vlan_tagged_pcap           802.1Q VLAN-encapsulated frames
    gre_tunneled_pcap          GRE-encapsulated inner IP
    syn_flood_pcap             SYN-no-ACK at high pps
    slow_beacon_pcap           single small packet every N seconds
    burst_pcap                 short, very high-pps burst
    timestamp_reversal_pcap    half the packets have backwards timestamps
    cardinality_explosion_pcap thousands of unique dst IPs from one src
    malformed_eth_pcap         zero-length / truncated frames
    truncated_pcap             first N bytes of an existing PCAP
    fake_magic_pcap            valid magic header, garbage body
    pcapng_truncated           PCAPNG SHB then EOF
    spoofed_oui_pcap           IoT-vendor OUI with desktop-grade traffic
"""

from __future__ import annotations

import math
import os
import random
import struct
from pathlib import Path
from typing import Optional

# ── Defer scapy import to function bodies so importing this file is cheap.
def _scapy():
    try:
        from scapy.all import (                              # type: ignore
            Ether, IP, IPv6, TCP, UDP, Raw, Dot1Q, GRE, wrpcap, RandIP,
        )
        return dict(Ether=Ether, IP=IP, IPv6=IPv6, TCP=TCP, UDP=UDP,
                    Raw=Raw, Dot1Q=Dot1Q, GRE=GRE, wrpcap=wrpcap,
                    RandIP=RandIP)
    except ImportError as e:
        raise ImportError(
            "scapy is required for pcap_gen. Install with `pip install scapy`."
        ) from e


# ── Public helpers ──────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def benign_tcp_pcap(out: Path, n_flows: int = 50, pkts_per_flow: int = 8,
                    seed: int = 0) -> Path:
    """Simple two-way HTTP-like exchanges."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for i in range(n_flows):
        sip  = f"10.0.0.{rng.randint(2, 254)}"
        dip  = f"93.184.216.{rng.randint(2, 254)}"
        sp   = rng.randint(40000, 60000)
        dp   = 80
        for k in range(pkts_per_flow):
            t += rng.uniform(0.001, 0.05)
            payload = b"GET /x HTTP/1.1\r\n\r\n" if k == 0 else b"x" * rng.randint(40, 600)
            up   = (s["IP"](src=sip, dst=dip)/s["TCP"](sport=sp, dport=dp,
                                                       flags="PA", seq=k))/s["Raw"](load=payload)
            down = (s["IP"](src=dip, dst=sip)/s["TCP"](sport=dp, dport=sp,
                                                       flags="PA", seq=k))/s["Raw"](load=b"resp" + payload[:40])
            up.time   = t
            down.time = t + rng.uniform(0.002, 0.02)
            pkts.extend([up, down])
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def benign_iot_pcap(out: Path, n_devices: int = 5, pkts_per_dev: int = 80,
                    seed: int = 0) -> Path:
    """MQTT-style IoT telemetry on port 1883."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    broker = "10.0.0.1"
    for i in range(n_devices):
        sip = f"10.0.0.{20 + i}"
        sp  = rng.randint(50000, 60000)
        for k in range(pkts_per_dev):
            t += rng.uniform(0.5, 2.0)   # slow telemetry
            up   = s["Ether"]()/s["IP"](src=sip, dst=broker)/s["TCP"](sport=sp, dport=1883, flags="PA")/s["Raw"](load=b"\x30\x10topic/sensorvalue=" + str(rng.randint(0, 100)).encode())
            ack  = s["Ether"]()/s["IP"](src=broker, dst=sip)/s["TCP"](sport=1883, dport=sp, flags="A")
            up.time, ack.time = t, t + rng.uniform(0.002, 0.01)
            pkts.extend([up, ack])
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def ipv6_only_pcap(out: Path, n_flows: int = 30, pkts_per_flow: int = 6,
                   seed: int = 0) -> Path:
    """All flows are IPv6. Used to confirm the pipeline either handles
    or explicitly skips v6 (current code drops it silently)."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for i in range(n_flows):
        sip = f"2001:db8::{rng.randint(2, 65534):x}"
        dip = f"2001:db8:1::{rng.randint(2, 65534):x}"
        sp  = rng.randint(40000, 60000)
        dp  = rng.choice([80, 443, 8080])
        for k in range(pkts_per_flow):
            t += rng.uniform(0.001, 0.05)
            up = s["IPv6"](src=sip, dst=dip)/s["TCP"](sport=sp, dport=dp, flags="PA")/s["Raw"](load=b"v6-payload" + bytes([k]))
            up.time = t
            pkts.append(up)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def vlan_tagged_pcap(out: Path, n_packets: int = 200, vlan_id: int = 100,
                     seed: int = 0) -> Path:
    """802.1Q VLAN-tagged frames carrying ordinary TCP."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_packets):
        sip = f"10.{vlan_id}.{rng.randint(0,255)}.{rng.randint(2,254)}"
        dip = f"10.{vlan_id}.{rng.randint(0,255)}.{rng.randint(2,254)}"
        eth = s["Ether"]()/s["Dot1Q"](vlan=vlan_id)/s["IP"](src=sip, dst=dip)/s["TCP"](sport=rng.randint(40000,60000), dport=80, flags="PA")/s["Raw"](load=b"vlan-payload")
        eth.time = t + k * 0.001
        pkts.append(eth)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def gre_tunneled_pcap(out: Path, n_packets: int = 100, seed: int = 0) -> Path:
    """GRE-encapsulated inner TCP — tests tunnel handling."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_packets):
        outer_src = f"203.0.113.{rng.randint(2,254)}"
        outer_dst = f"198.51.100.{rng.randint(2,254)}"
        inner_src = f"10.99.0.{rng.randint(2,254)}"
        inner_dst = f"10.99.1.{rng.randint(2,254)}"
        pkt = (s["IP"](src=outer_src, dst=outer_dst, proto=47)
               / s["GRE"]()
               / s["IP"](src=inner_src, dst=inner_dst)
               / s["TCP"](sport=rng.randint(40000,60000), dport=80, flags="PA")
               / s["Raw"](load=b"inner"))
        pkt.time = t + k * 0.001
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def syn_flood_pcap(out: Path, n_packets: int = 5000, src: str = "10.0.0.66",
                   dst: str = "10.0.0.1", seed: int = 0) -> Path:
    """High-pps SYN-no-ACK from a single source. Targets suspicion scorer."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_packets):
        sp = rng.randint(40000, 60000)
        dp = rng.choice([22, 23, 80, 443, 2323, 31337])
        pkt = s["IP"](src=src, dst=dst)/s["TCP"](sport=sp, dport=dp, flags="S", seq=k)
        pkt.time = t + k * 0.0001    # 10 kpps
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def slow_beacon_pcap(out: Path, n_beacons: int = 60, period_sec: float = 35.0,
                     src: str = "10.0.0.7", dst: str = "203.0.113.4",
                     port: int = 8443) -> Path:
    """Single small packet every period_sec — defeats 30 s flow idle."""
    s = _scapy()
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_beacons):
        pkt = s["IP"](src=src, dst=dst)/s["TCP"](sport=44321, dport=port, flags="PA")/s["Raw"](load=b"hb" + bytes([k & 0xFF]))
        pkt.time = t + k * period_sec
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def multi_session_beacon_pcap(out: Path,
                              n_sessions: int = 25,
                              pkts_per_session: int = 4,
                              session_period_sec: float = 35.0,
                              intra_session_gap: float = 0.5,
                              src: str = "10.0.0.7",
                              dst: str = "203.0.113.4",
                              port: int = 8443) -> Path:
    """
    Slow-C2-shaped traffic across N distinct sessions.

    Each session opens a fresh ephemeral src_port, sends a few packets
    `intra_session_gap` apart (so each session DOES end as a real flow
    — fitting inside the FlowAggregator idle timeout), then waits
    `session_period_sec` before the next session begins.

    Use this to validate the slow-beacon detection AFTER the project
    raised _idle to 120 s: with `intra_session_gap=0.5` and 4 packets,
    each session is ~2 s long and closes cleanly; the LSTM then sees
    n_sessions completed flows for one src_ip — a real input window.

    Set `session_period_sec` > project _idle to test cross-session
    aggregation (the harder threat model — see RT-1b).
    """
    s = _scapy()
    pkts = []
    t = 1_700_000_000.0
    for ses in range(n_sessions):
        sport = 40000 + ses
        ses_t0 = t + ses * session_period_sec
        for k in range(pkts_per_session):
            pkt = s["IP"](src=src, dst=dst) / \
                  s["TCP"](sport=sport, dport=port, flags="PA") / \
                  s["Raw"](load=b"hb" + bytes([k & 0xFF]))
            pkt.time = ses_t0 + k * intra_session_gap
            pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def burst_pcap(out: Path, n_packets: int = 10_000, src: str = "10.0.0.55",
               dst: str = "10.0.0.1", seed: int = 0) -> Path:
    """A short, intense burst — ~10k packets/sec."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_packets):
        pkt = s["IP"](src=src, dst=dst)/s["UDP"](sport=rng.randint(40000,60000), dport=53)/s["Raw"](load=b"q" * rng.randint(20, 80))
        pkt.time = t + k * 1e-4
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def timestamp_reversal_pcap(out: Path, n_packets: int = 500,
                            seed: int = 0) -> Path:
    """Half the packets have backwards timestamps — tests aggregator robustness."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    base = 1_700_000_000.0
    for k in range(n_packets):
        sip = f"10.0.0.{rng.randint(2,254)}"
        pkt = s["IP"](src=sip, dst="10.0.0.1")/s["TCP"](sport=rng.randint(40000,60000),
                                                         dport=80, flags="PA")/s["Raw"](load=b"x")
        # Every 5th packet is back-dated by 60 seconds.
        pkt.time = base + k * 0.001 - (60.0 if k % 5 == 0 else 0.0)
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def cardinality_explosion_pcap(out: Path, n_destinations: int = 8000,
                               seed: int = 0) -> Path:
    """One src, thousands of dst IPs — stresses Kitsune state."""
    s = _scapy()
    rng = random.Random(seed)
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_destinations):
        d_octet1 = (k >> 8) & 0xFF
        d_octet2 = k & 0xFF
        dip = f"172.16.{d_octet1}.{d_octet2 if d_octet2 != 0 else 1}"
        pkt = s["IP"](src="10.0.0.99", dst=dip)/s["TCP"](sport=rng.randint(40000,60000),
                                                          dport=rng.choice([22,23,80,443,8080]),
                                                          flags="S", seq=k)
        pkt.time = t + k * 0.0005
        pkts.append(pkt)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out


def malformed_eth_pcap(out: Path, n_packets: int = 50) -> Path:
    """
    Frames with invalid lengths / partial Ethernet headers. Built by
    writing the libpcap container by hand so we can include illegal
    record sizes.
    """
    _ensure_dir(out)
    with open(out, "wb") as f:
        # libpcap global header: magic + version + thiszone + sigfigs + snaplen + linktype
        f.write(struct.pack("<IHHIIII",
                            0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))   # LINKTYPE_ETHERNET
        for i in range(n_packets):
            ts_sec  = 1_700_000_000 + i
            ts_usec = (i * 137) % 1_000_000
            # Half of the records claim length 6 (smaller than even an
            # Ethernet header), the other half claim 0xFFFF and the body
            # is truncated to a few bytes — both should be skipped.
            if i % 2 == 0:
                incl = orig = 6
                body = b"\x00" * 6
            else:
                incl = 4
                orig = 0xFFFF
                body = b"\xff\xff\xff\xff"
            f.write(struct.pack("<IIII", ts_sec, ts_usec, incl, orig))
            f.write(body)
    return out


def truncated_pcap(src_pcap: Path, out: Path, keep_bytes: int = 1024) -> Path:
    """Copy first `keep_bytes` of a real PCAP — simulates partial download."""
    _ensure_dir(out)
    with open(src_pcap, "rb") as fi, open(out, "wb") as fo:
        fo.write(fi.read(keep_bytes))
    return out


def fake_magic_pcap(out: Path, magic: bytes = b"\xd4\xc3\xb2\xa1",
                    body_size: int = 1024) -> Path:
    """Valid magic, garbage body."""
    _ensure_dir(out)
    with open(out, "wb") as f:
        f.write(magic + os.urandom(20) + os.urandom(body_size))
    return out


def pcapng_truncated(out: Path) -> Path:
    """PCAPNG Section Header Block then EOF."""
    _ensure_dir(out)
    # Just write a minimal SHB that pcapng readers will accept then a 12-byte
    # cut-off in the middle of an Interface Description Block.
    shb = struct.pack("<IIIHHQ",
                      0x0A0D0D0A,        # block type SHB
                      0x1C,              # block total length (28)
                      0x1A2B3C4D,        # byte order magic
                      1, 0,              # major.minor version
                      0xFFFFFFFFFFFFFFFF) # section length unspecified
    shb += struct.pack("<I", 0x1C)       # block total length trailer
    with open(out, "wb") as f:
        f.write(shb)
        # 12 bytes of partial IDB block, deliberately truncated:
        f.write(b"\x01\x00\x00\x00\x14\x00\x00\x00\x01\x00\x00")
    return out


def spoofed_oui_pcap(out: Path, n_packets: int = 200,
                     spoofed_mac: str = "d8:a0:1d:11:22:33") -> Path:
    """
    Desktop-grade C2-like traffic from a MAC whose OUI matches an IoT
    vendor (Espressif d8:a0:1d). Tests Stage-1 OUI-override exploitability.
    """
    s = _scapy()
    pkts = []
    t = 1_700_000_000.0
    for k in range(n_packets):
        eth = s["Ether"](src=spoofed_mac)/s["IP"](src="10.0.0.77", dst="203.0.113.5")/s["TCP"](sport=44321, dport=8443, flags="PA")/s["Raw"](load=b"c2" + bytes([k & 0xFF]))
        eth.time = t + k * 0.5    # workstation cadence
        pkts.append(eth)
    _ensure_dir(out)
    s["wrpcap"](str(out), pkts)
    return out
