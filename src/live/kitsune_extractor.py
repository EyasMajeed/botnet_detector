"""
════════════════════════════════════════════════════════════════════════
 kitsune_extractor.py  —  Real-time N-BaIoT Feature Extractor
 Group 07 | CPCS499
════════════════════════════════════════════════════════════════════════
 Implements the Kitsune incremental statistics engine that produces
 the exact 115 features used to train the N-BaIoT CNN-LSTM model.

 Reference:
   Mirsky et al. (2018) "Kitsune: An Ensemble of Autoencoders for
   Online Network Intrusion Detection", NDSS Symposium.

   Meidan et al. (2018) "N-BaIoT: Network-based Detection of IoT
   Botnet Attacks Using Deep Autoencoders", IEEE Pervasive Computing.

 HOW IT WORKS:
   For each incoming packet, the extractor maintains 5 "stream" types,
   each keyed on a different tuple of packet fields:

     MI    keyed on (src_MAC, src_IP)   — MAC+IP source stats
     H     keyed on (src_IP,)           — IP source stats
     HH    keyed on (src_IP, dst_IP)    — channel stats
     HH_jit keyed on (src_IP, dst_IP)  — channel jitter stats
     HpHp  keyed on (src_IP:src_port, dst_IP:dst_port) — socket stats

   Each stream maintains 5 exponentially-damped windows (lambdas):
     L5=5, L3=3, L1=1, L0.1=0.1, L0.01=0.01
   (These are decay factors, not seconds. Higher lambda = faster decay
    = shorter effective history. L5 ≈ 100ms, L0.01 ≈ 1min.)

   Each window tracks an IncStat (1D) or IncStatCov (2D) object that
   updates incrementally on each new packet using the damped formula:
     new_weight = old_weight * w + 1
     new_mean   = old_mean   * w + value * (1-w)    (approximate)
   where w = e^(-lambda * elapsed_time)

   Per-packet output: 115 float values in the exact column order
   matching N-BaIoT training data.

 USAGE:
   from src.live.kitsune_extractor import KitsuneExtractor
   ext = KitsuneExtractor()

   # Call once per packet:
   features = ext.update(
       timestamp  = pkt_time,       # float, epoch seconds
       src_mac    = "aa:bb:cc:...", # string
       src_ip     = "192.168.1.5",
       dst_ip     = "8.8.8.8",
       src_port   = 54321,
       dst_port   = 443,
       pkt_len    = 84,             # total packet length in bytes
       protocol   = "TCP"
   )
   # features is a numpy array of shape (115,) ready for the CNN-LSTM
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import math
import numpy as np
from collections import defaultdict


# ── Exact column order matching N-BaIoT training data ─────────────────
FEATURE_NAMES: list[str] = [
    # MI  (src_MAC+src_IP → 5 windows × 3 stats = 15)
    "MI_dir_L5_weight",  "MI_dir_L5_mean",  "MI_dir_L5_variance",
    "MI_dir_L3_weight",  "MI_dir_L3_mean",  "MI_dir_L3_variance",
    "MI_dir_L1_weight",  "MI_dir_L1_mean",  "MI_dir_L1_variance",
    "MI_dir_L0.1_weight","MI_dir_L0.1_mean","MI_dir_L0.1_variance",
    "MI_dir_L0.01_weight","MI_dir_L0.01_mean","MI_dir_L0.01_variance",
    # H   (src_IP → 5 windows × 3 stats = 15)
    "H_L5_weight",  "H_L5_mean",  "H_L5_variance",
    "H_L3_weight",  "H_L3_mean",  "H_L3_variance",
    "H_L1_weight",  "H_L1_mean",  "H_L1_variance",
    "H_L0.1_weight","H_L0.1_mean","H_L0.1_variance",
    "H_L0.01_weight","H_L0.01_mean","H_L0.01_variance",
    # HH  (src_IP→dst_IP → 5 windows × 7 stats = 35)
    "HH_L5_weight",  "HH_L5_mean",  "HH_L5_std",  "HH_L5_magnitude",  "HH_L5_radius",  "HH_L5_covariance",  "HH_L5_pcc",
    "HH_L3_weight",  "HH_L3_mean",  "HH_L3_std",  "HH_L3_magnitude",  "HH_L3_radius",  "HH_L3_covariance",  "HH_L3_pcc",
    "HH_L1_weight",  "HH_L1_mean",  "HH_L1_std",  "HH_L1_magnitude",  "HH_L1_radius",  "HH_L1_covariance",  "HH_L1_pcc",
    "HH_L0.1_weight","HH_L0.1_mean","HH_L0.1_std","HH_L0.1_magnitude","HH_L0.1_radius","HH_L0.1_covariance","HH_L0.1_pcc",
    "HH_L0.01_weight","HH_L0.01_mean","HH_L0.01_std","HH_L0.01_magnitude","HH_L0.01_radius","HH_L0.01_covariance","HH_L0.01_pcc",
    # HH_jit (src_IP→dst_IP jitter → 5 windows × 3 stats = 15)
    "HH_jit_L5_weight",  "HH_jit_L5_mean",  "HH_jit_L5_variance",
    "HH_jit_L3_weight",  "HH_jit_L3_mean",  "HH_jit_L3_variance",
    "HH_jit_L1_weight",  "HH_jit_L1_mean",  "HH_jit_L1_variance",
    "HH_jit_L0.1_weight","HH_jit_L0.1_mean","HH_jit_L0.1_variance",
    "HH_jit_L0.01_weight","HH_jit_L0.01_mean","HH_jit_L0.01_variance",
    # HpHp (src_ip:port→dst_ip:port → 5 windows × 7 stats = 35)
    "HpHp_L5_weight",  "HpHp_L5_mean",  "HpHp_L5_std",  "HpHp_L5_magnitude",  "HpHp_L5_radius",  "HpHp_L5_covariance",  "HpHp_L5_pcc",
    "HpHp_L3_weight",  "HpHp_L3_mean",  "HpHp_L3_std",  "HpHp_L3_magnitude",  "HpHp_L3_radius",  "HpHp_L3_covariance",  "HpHp_L3_pcc",
    "HpHp_L1_weight",  "HpHp_L1_mean",  "HpHp_L1_std",  "HpHp_L1_magnitude",  "HpHp_L1_radius",  "HpHp_L1_covariance",  "HpHp_L1_pcc",
    "HpHp_L0.1_weight","HpHp_L0.1_mean","HpHp_L0.1_std","HpHp_L0.1_magnitude","HpHp_L0.1_radius","HpHp_L0.1_covariance","HpHp_L0.1_pcc",
    "HpHp_L0.01_weight","HpHp_L0.01_mean","HpHp_L0.01_std","HpHp_L0.01_magnitude","HpHp_L0.01_radius","HpHp_L0.01_covariance","HpHp_L0.01_pcc",
]
assert len(FEATURE_NAMES) == 115, f"Expected 115 features, got {len(FEATURE_NAMES)}"

# Decay lambdas — same values used in N-BaIoT paper
LAMBDAS = [5.0, 3.0, 1.0, 0.1, 0.01]


# ════════════════════════════════════════════════════════════════════════
# INCREMENTAL STATISTICS — 1D (weight, mean, variance)
# ════════════════════════════════════════════════════════════════════════

class IncStat1D:
    """
    Maintains exponentially-damped weight, mean, and variance for a
    single stream of scalar values (packet sizes or inter-arrival times).

    Update formula (Welford-style with exponential decay):
      w    = e^(-lambda * dt)   (decay factor)
      weight_new = weight * w + 1
      mean_new   = (mean * weight * w + value) / weight_new
      var_new    = (var + (mean-value)^2 * weight*w/(weight_new)) * ...
    """
    __slots__ = ("lam", "weight", "mean", "var", "last_time")

    def __init__(self, lam: float):
        self.lam       = lam
        self.weight    = 0.0
        self.mean      = 0.0
        self.var       = 0.0
        self.last_time = 0.0

    def update(self, value: float, t: float) -> None:
        if self.weight == 0.0:
            self.weight    = 1.0
            self.mean      = value
            self.var       = 0.0
            self.last_time = t
            return

        dt = t - self.last_time
        if dt < 0:
            dt = 0.0
        self.last_time = t

        w = math.exp(-self.lam * dt)
        old_w    = self.weight * w
        new_w    = old_w + 1.0
        old_mean = self.mean
        new_mean = (old_mean * old_w + value) / new_w
        # Incremental variance update
        self.var = (self.var + (old_mean - value) ** 2 * old_w / new_w) * w
        self.weight = new_w
        self.mean   = new_mean

    def get(self) -> tuple[float, float, float]:
        """Returns (weight, mean, variance)."""
        return self.weight, self.mean, self.var


# ════════════════════════════════════════════════════════════════════════
# INCREMENTAL STATISTICS — 2D (adds magnitude, radius, covariance, pcc)
# ════════════════════════════════════════════════════════════════════════

class IncStat2D:
    """
    Maintains two correlated IncStat1D streams and computes joint
    statistics: magnitude, radius, covariance, and PCC.

    Used for HH, HH_jit, HpHp stream types where two values per packet
    are tracked (packet size and inter-arrival time).
    """
    __slots__ = ("s1", "s2", "lam", "cov_sum", "weight", "last_time")

    def __init__(self, lam: float):
        self.lam       = lam
        self.s1        = IncStat1D(lam)
        self.s2        = IncStat1D(lam)
        self.cov_sum   = 0.0   # running covariance numerator
        self.weight    = 0.0
        self.last_time = 0.0

    def update(self, v1: float, v2: float, t: float) -> None:
        dt = max(t - self.last_time, 0.0) if self.weight > 0 else 0.0
        self.last_time = t

        # Decay existing covariance
        if self.weight > 0:
            w = math.exp(-self.lam * dt)
            self.cov_sum *= w
            self.weight  *= w
        self.weight += 1.0

        # Update marginals
        old_m1 = self.s1.mean
        old_m2 = self.s2.mean
        self.s1.update(v1, t)
        self.s2.update(v2, t)

        # Covariance increment
        self.cov_sum += (v1 - old_m1) * (v2 - old_m2)

    def get(self) -> tuple[float, float, float, float, float, float, float]:
        """
        Returns (weight, mean1, std1, magnitude, radius, covariance, pcc).
        magnitude = sqrt(mean1^2 + mean2^2)
        radius    = sqrt(var1 + var2)
        covariance = cov_sum / weight  (approx)
        pcc        = covariance / (std1 * std2)
        """
        w1, m1, v1 = self.s1.get()
        w2, m2, v2 = self.s2.get()
        std1 = math.sqrt(max(v1, 0.0))
        std2 = math.sqrt(max(v2, 0.0))

        weight    = w1                   # use first stream weight
        mean      = m1
        std       = std1
        magnitude = math.sqrt(m1*m1 + m2*m2)
        radius    = math.sqrt(max(v1, 0.0) + max(v2, 0.0))
        cov       = self.cov_sum / max(self.weight, 1.0)
        denom     = std1 * std2
        pcc       = cov / denom if denom > 1e-10 else 0.0
        pcc       = max(-1.0, min(1.0, pcc))   # clip to [-1, 1]

        return weight, mean, std, magnitude, radius, cov, pcc


# ════════════════════════════════════════════════════════════════════════
# KITSUNE FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════════════════

class KitsuneExtractor:
    """
    Stateful, per-packet feature extractor that produces the 115 N-BaIoT
    features in the exact column order used for training.

    One extractor instance should be maintained for the lifetime of the
    capture session — it accumulates stream history across all packets.

    Memory: O(unique_streams × 5_lambdas) — grows with unique IP pairs.
    For a typical home/office network (~50 devices) this is negligible.
    """

    def __init__(self):
        # Each key maps to a list of 5 stat objects (one per lambda)
        # MI, H: list of IncStat1D
        # HH, HH_jit, HpHp: list of IncStat2D
        self._mi    : dict[str, list[IncStat1D]]  = defaultdict(
            lambda: [IncStat1D(l) for l in LAMBDAS])
        self._h     : dict[str, list[IncStat1D]]  = defaultdict(
            lambda: [IncStat1D(l) for l in LAMBDAS])
        self._hh    : dict[str, list[IncStat2D]]  = defaultdict(
            lambda: [IncStat2D(l) for l in LAMBDAS])
        self._hhjit : dict[str, list[IncStat2D]]  = defaultdict(
            lambda: [IncStat2D(l) for l in LAMBDAS])
        self._hphp  : dict[str, list[IncStat2D]]  = defaultdict(
            lambda: [IncStat2D(l) for l in LAMBDAS])

        # Last packet time per HH key (for jitter computation)
        self._last_hh_time: dict[str, float] = {}

    def update(self,
               timestamp : float,
               src_mac   : str,
               src_ip    : str,
               dst_ip    : str,
               src_port  : int,
               dst_port  : int,
               pkt_len   : int,
               protocol  : str = "TCP") -> np.ndarray:
        """
        Process one packet and return a (115,) float32 feature vector.

        Parameters
        ----------
        timestamp : packet capture time (epoch seconds, float)
        src_mac   : source MAC address string (e.g. "aa:bb:cc:dd:ee:ff")
                    If not available, pass src_ip as fallback.
        src_ip    : source IP string
        dst_ip    : destination IP string
        src_port  : source port (int), 0 if not applicable
        dst_port  : destination port (int), 0 if not applicable
        pkt_len   : total packet length in bytes
        protocol  : "TCP", "UDP", "ICMP", etc.
        """
        t   = float(timestamp)
        sz  = float(pkt_len)

        # ── Stream keys ───────────────────────────────────────────────
        mi_key    = f"{src_mac}_{src_ip}"
        h_key     = src_ip
        hh_key    = f"{src_ip}_{dst_ip}"
        hphp_key  = f"{src_ip}:{src_port}_{dst_ip}:{dst_port}"

        # ── Jitter: inter-arrival time for this channel ────────────────
        if hh_key in self._last_hh_time:
            jitter = t - self._last_hh_time[hh_key]
        else:
            jitter = 0.0
        self._last_hh_time[hh_key] = t

        # ── Update all streams ─────────────────────────────────────────
        # MI: packet size
        for stat in self._mi[mi_key]:
            stat.update(sz, t)

        # H: packet size
        for stat in self._h[h_key]:
            stat.update(sz, t)

        # HH: (packet size, packet size) — 2D over the channel
        for stat in self._hh[hh_key]:
            stat.update(sz, sz, t)

        # HH_jit: (jitter, jitter) — 2D jitter stats over the channel
        for stat in self._hhjit[hh_key]:
            stat.update(jitter, jitter, t)

        # HpHp: (packet size, packet size) — 2D over the socket
        for stat in self._hphp[hphp_key]:
            stat.update(sz, sz, t)

        # ── Assemble 115 features ─────────────────────────────────────
        vec: list[float] = []

        # MI: 5 windows × (weight, mean, variance)
        for stat in self._mi[mi_key]:
            w, m, v = stat.get()
            vec.extend([w, m, v])

        # H: 5 windows × (weight, mean, variance)
        for stat in self._h[h_key]:
            w, m, v = stat.get()
            vec.extend([w, m, v])

        # HH: 5 windows × (weight, mean, std, magnitude, radius, cov, pcc)
        for stat in self._hh[hh_key]:
            vec.extend(stat.get())

        # HH_jit: 5 windows × (weight, mean, variance only — 3 stats)
        for stat in self._hhjit[hh_key]:
            w, m, std, mag, rad, cov, pcc = stat.get()
            vec.extend([w, m, std * std])   # variance = std^2

        # HpHp: 5 windows × (weight, mean, std, magnitude, radius, cov, pcc)
        for stat in self._hphp[hphp_key]:
            vec.extend(stat.get())

        assert len(vec) == 115, f"Feature count error: {len(vec)}"
        return np.array(vec, dtype=np.float32)

    def reset(self) -> None:
        """Clear all stream history (call when starting a new capture)."""
        self._mi.clear()
        self._h.clear()
        self._hh.clear()
        self._hhjit.clear()
        self._hphp.clear()
        self._last_hh_time.clear()

    @property
    def n_streams(self) -> int:
        """Total number of unique streams currently tracked."""
        return (len(self._mi) + len(self._h) +
                len(self._hh) + len(self._hhjit) + len(self._hphp))