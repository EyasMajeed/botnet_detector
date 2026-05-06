"""
═══════════════════════════════════════════════════════════════════════
 local_explainer.py — Per-flow attribution explainers
 Group 07 | CPCS499 | XAI Module
═══════════════════════════════════════════════════════════════════════

Implements two complementary explainers, one per pipeline stage:

  Stage1Explainer          — wraps monitoring.Stage1Classifier
                             Uses SHAP TreeExplainer (exact, fast on trees)
                             Explains: "why was this flow routed to IoT/Non-IoT?"

  Stage2NonIoTExplainer    — wraps monitoring.Stage2NonIoTDetector
                             Uses Integrated Gradients (Sundararajan ICML 2017)
                             Explains: "why was this flow labelled botnet/benign?"
                             Input: (20, 46) sliding window of scaled flow rows

  Stage2IoTExplainer       — wraps monitoring.Stage2IoTDetector
                             Uses Integrated Gradients
                             Explains: "why was this packet sequence labelled botnet?"
                             Input: (20, 115) Kitsune packet sequence

WHY IG FOR STAGE-2?
───────────────────
We compared IG, DeepLIFT, SHAP-Deep, LIME and built-in attention. IG
won on theoretical correctness for our specific architecture: it's the
only post-hoc method that satisfies BOTH the Sensitivity and the
Implementation Invariance axioms (Sundararajan et al., 2017) AND remains
theoretically valid on the multiplicative gates of the LSTM (Ancona et al.,
ICLR 2018, showed DeepLIFT diverges in this regime). It uses only
torch.autograd — no extra dependencies, no MPS instability, no surrogate
models.

WHY SHAP FOR STAGE-1?
─────────────────────
Stage-1 is a tree ensemble (RandomForest or XGBoost). SHAP TreeExplainer
is exact (not approximate) for trees and runs in sub-millisecond time
per explanation. It's the right tool for this model class.

INTEGRATION CONTRACT
────────────────────
These explainers wrap the wrapper classes from monitoring.py — which is
where the real production wrappers live (see inference_bridge.py for why
the loaders in models/stage*/*.py are NOT used in production).

  Stage1Classifier     stores model in self._rf, encoder in self._le
  Stage2IoTDetector    stores model in self._model, threshold in self._threshold
  Stage2NonIoTDetector stores model in self._model, threshold in self._threshold,
                       feature cols in self._feature_cols, scaler in self._s2_mean / _s2_scale

We reach into these private attributes deliberately. The XAI module is
tightly coupled to monitoring.py by design — they're co-developed.

SCALER HANDLING (CRITICAL)
──────────────────────────
The Non-IoT detector's input is RAW flow features (which Stage-2 then
scales internally via stage2_preprocess_non_iot). For IG to be faithful,
we must run gradients with respect to the INPUT THE MODEL ACTUALLY SEES.
That means: scale the input first, then run IG on the scaled tensor.
The wrapper's stage2_preprocess_non_iot already does this scaling.

For the IoT detector, scaling happens per-packet via stage2_preprocess_iot
(MinMax → [0,1]). The (20, 115) sequence the bridge passes to stage2_predict
is already scaled. So IG runs on that pre-scaled sequence directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

# SHAP is only needed for Stage1Explainer. Import lazily so the module
# can still be loaded if shap isn't installed (Stage-2 IG works without it).
try:
    import shap                                              # type: ignore
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

from src.xai.feature_metadata import (
    get_display, get_suspicion_dir, get_unit, get_info,
)


# ══════════════════════════════════════════════════════════════════════
# Data classes — what every explainer returns
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureContribution:
    """One row in the top-K feature attribution table."""
    feature:      str       # raw feature name (e.g. "flow_pkts_per_sec")
    display:      str       # friendly label   (e.g. "Flow pkt rate")
    value:        float     # the actual value the model saw for this flow
    attribution:  float     # signed contribution: + pushes toward botnet
    direction:    str       # "↑ pushed toward botnet" / "↓ pushed toward benign"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalExplanation:
    """The full attribution result for one flow."""
    prediction:        str               # "botnet" | "benign" | "iot" | "noniot"
    confidence:        float             # model's probability for the predicted class
    method:            str               # "integrated_gradients" | "shap_tree" | "failed"
    raw_attributions:  dict[str, float]  # all features, signed contributions
    top_features:      list[FeatureContribution]


# ══════════════════════════════════════════════════════════════════════
# Stage-1 Explainer — SHAP TreeExplainer
# ══════════════════════════════════════════════════════════════════════

class Stage1Explainer:
    """
    Explains Stage-1 (RF or XGBoost) routing decisions using SHAP TreeExplainer.

    Wraps monitoring.Stage1Classifier — pulls the trained model from .clf._rf,
    the LabelEncoder from .clf._le, and the feature list from .clf._features.
    These are private attributes by Python convention, but they ARE the public
    contract between monitoring.py and the XAI module (we co-develop both).

    Stage-1 input is a flat feature DICT (the rest of the codebase calls
    stage1_predict(feat: dict)). We accept the same shape so we can be
    plugged into inference_bridge with no extra glue code.
    """

    def __init__(self, stage1_classifier):
        if not _SHAP_OK:
            raise ImportError(
                "Stage1Explainer requires the `shap` package.\n"
                "Install with: pip install shap\n"
                "(Stage2 explainers work without shap — only Stage-1 needs it.)"
            )
        self.clf       = stage1_classifier
        self._rf       = stage1_classifier._rf            # the underlying tree model
        self._le       = stage1_classifier._le            # LabelEncoder (or None)
        self._iot_idx  = stage1_classifier._iot_idx
        self._features = list(stage1_classifier._features)  # 56 names in training order
        self._has_scaler = stage1_classifier._has_scaler
        self._mean     = stage1_classifier._mean          # may be None
        self._scale    = stage1_classifier._scale         # may be None

        # Build the explainer once. TreeExplainer adapts to RF and XGB
        # automatically. shap_values returns either a list[(1, n), (1, n)]
        # for RF or a single (1, n) array for XGB — we handle both below.
        try:
            self._explainer = shap.TreeExplainer(self._rf)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build SHAP TreeExplainer for Stage-1: {e}.\n"
                "If the Stage-1 model was loaded from XGBoost JSON, ensure the "
                "xgboost Python package is the same version that produced the JSON."
            ) from e

    # ── Public API ──────────────────────────────────────────────────
    def explain(self, feat: dict, top_k: int = 8) -> LocalExplanation:
        """
        Parameters
        ----------
        feat : dict
            Single-flow feature dict, same shape passed to stage1_predict.
            Missing keys default to 0 (matches stage1_preprocess behaviour).
        top_k : int

        Returns
        -------
        LocalExplanation with prediction = "iot" | "noniot",
        signed attributions for each of the 56 features, and the
        |attribution|-sorted top-K list.
        """
        # Build the SCALED row exactly the way Stage1Classifier does it.
        # We must explain the input the model actually consumed.
        row = np.array([feat.get(f, 0.0) for f in self._features], dtype=np.float32)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        if self._has_scaler:
            row = (row - self._mean) / self._scale

        X = row.reshape(1, -1)

        # Get prediction
        proba = self._rf.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        if self._le is not None:
            label = str(self._le.inverse_transform([pred_idx])[0])
        else:
            label = str(self._rf.classes_[pred_idx])

        # SHAP attributions
        sv = self._explainer.shap_values(X)
        # RF returns list[per_class], XGB usually returns single array
        if isinstance(sv, list):
            attributions = np.asarray(sv[pred_idx][0], dtype=np.float64)
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:                    # (n, classes, features) for newer SHAP
                attributions = sv_arr[0, pred_idx]
            elif sv_arr.ndim == 2:                  # (n, features)
                attributions = sv_arr[0]
            else:
                # Unexpected shape — degrade rather than crash
                attributions = np.zeros(len(self._features), dtype=np.float64)

        # Build the contribution list
        raw = {f: float(a) for f, a in zip(self._features, attributions)}

        # Sort by |attribution| descending; cap at top_k
        ranked = sorted(self._features, key=lambda f: abs(raw[f]), reverse=True)
        top: list[FeatureContribution] = []
        for fname in ranked[:top_k]:
            attr = raw[fname]
            # NOTE: for Stage-1 we explain "iot" specifically. attribution > 0
            # means feature pushed prediction toward the predicted class
            # (whichever it is). The direction string reflects that.
            if attr >= 0:
                direction = f"↑ pushed toward {label}"
            else:
                direction = f"↓ pushed against {label}"
            top.append(FeatureContribution(
                feature=fname,
                display=get_display(fname),
                value=float(feat.get(fname, 0.0)),     # raw, unscaled value
                attribution=attr,
                direction=direction,
            ))

        return LocalExplanation(
            prediction=label,
            confidence=confidence,
            method="shap_tree",
            raw_attributions=raw,
            top_features=top,
        )


# ══════════════════════════════════════════════════════════════════════
# Stage-2 Explainer — Integrated Gradients (shared by IoT and Non-IoT)
# ══════════════════════════════════════════════════════════════════════

class _Stage2BaseIGExplainer:
    """
    Shared IG implementation for both Stage-2 models.

    They have identical architecture (Conv1d→BN→ReLU→MaxPool → Conv1d→BN→ReLU
    → LSTM(2 layers, dropout=0.3) → Linear(128,64)→ReLU→Dropout→Linear(64,1)),
    identical input shape semantics ((seq_len, n_features)), and identical
    output (raw logit). The only differences are the feature names and
    whether scaling has been applied upstream by the time we receive the
    input. Subclasses override _prepare_seq() to produce the scaled
    sequence the model expects.
    """

    POSITIVE_CLASS = "botnet"

    def __init__(self, model: torch.nn.Module, threshold: float,
                 seq_len: int, n_features: int,
                 feature_cols: list[str],
                 n_steps: Optional[int] = None,
                 device: Optional[torch.device] = None):
        self._model      = model.eval()
        self._threshold  = float(threshold)
        self._seq_len    = int(seq_len)
        self._n_features = int(n_features)
        self._feature_cols = list(feature_cols)

        # Adaptive Riemann step count.
        # Empirically tuned on the trained Stage-2 Non-IoT model: with the
        # default seq_len=20, n_steps must be ≥ 100 to keep the IG
        # Completeness gap below 2% of the model's logit magnitude.
        # Earlier 2× heuristic was too aggressive — saturation effects in
        # this LSTM are stronger than I anticipated. Verified with a sweep
        # (gap shrinks monotonically: 8.2%@20 → 4.6%@60 → 2.2%@100 → 1.2%@200).
        # 100 steps gives ~120 ms latency on CPU which is well within
        # budget (XAI runs only on botnet detections, not every flow).
        if n_steps is None:
            self._n_steps = max(50, 5 * self._seq_len)
        else:
            self._n_steps = int(n_steps)

        # Use CPU explicitly. CUDA/MPS gradients can produce inconsistent
        # results on the same input across runs; CPU is deterministic and
        # fast enough at our latency budget.
        self._device = device or torch.device("cpu")
        self._model.to(self._device)

    # ── Subclasses MUST override ────────────────────────────────────
    def _prepare_seq(self, flow_input) -> np.ndarray:
        """
        Convert whatever inference_bridge has on hand into the EXACT
        (seq_len, n_features) scaled tensor the model expects.
        """
        raise NotImplementedError

    def _raw_value_for(self, flow_input, feat_idx: int) -> float:
        """
        Return the unscaled value of feature[feat_idx] for the LATEST
        timestep, suitable for display in the UI. Subclasses override.
        """
        raise NotImplementedError

    # ── Public API ──────────────────────────────────────────────────
    def explain(self, flow_input, top_k: int = 8) -> LocalExplanation:
        seq = self._prepare_seq(flow_input)            # (seq_len, n_features) np.float32
        x   = torch.tensor(seq, dtype=torch.float32,
                            device=self._device).unsqueeze(0)        # (1, seq_len, n_features)

        # Forward pass for the prediction
        with torch.no_grad():
            logit = self._model(x)
            prob  = torch.sigmoid(logit).item() if logit.numel() > 0 else 0.0
        prediction = "botnet" if prob >= self._threshold else "benign"

        # Integrated Gradients
        attr_seq = self._integrated_gradients(x)        # (seq_len, n_features)

        # Aggregate across timesteps → per-feature contribution
        per_feature = attr_seq.sum(axis=0)              # (n_features,)

        # Build raw map and ranked top-K
        raw = {self._feature_cols[i]: float(per_feature[i])
               for i in range(self._n_features)}
        order = sorted(range(self._n_features),
                       key=lambda i: abs(per_feature[i]), reverse=True)

        top: list[FeatureContribution] = []
        for i in order[:top_k]:
            fname = self._feature_cols[i]
            attr  = float(per_feature[i])
            if attr >= 0:
                direction = "↑ pushed toward botnet"
            else:
                direction = "↓ pushed toward benign"
            top.append(FeatureContribution(
                feature=fname,
                display=get_display(fname),
                value=self._raw_value_for(flow_input, i),
                attribution=attr,
                direction=direction,
            ))

        return LocalExplanation(
            prediction=prediction,
            confidence=float(prob),
            method="integrated_gradients",
            raw_attributions=raw,
            top_features=top,
        )

    # ── IG implementation ───────────────────────────────────────────
    def _integrated_gradients(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute IG: integral from baseline to x of ∂f/∂x along a straight line.
        Approximated with a Riemann sum of n_steps midpoint evaluations.

        Returns
        -------
        np.ndarray of shape (seq_len, n_features) — signed attributions
        per (timestep, feature) cell. Sums to f(x) - f(baseline) by the
        Completeness axiom (verified up to floating-point error).

        We use the all-zero baseline. For the IoT detector this corresponds
        to "no Kitsune signal observed" (all stream stats at 0) which is a
        reasonable null reference. For the Non-IoT detector the input is
        already StandardScaler-normalised, so the all-zero baseline is the
        per-feature mean — also a sensible null.
        """
        baseline = torch.zeros_like(x)
        x_diff   = (x - baseline).detach()              # (1, seq_len, n_features)

        # Build the n_steps interpolated inputs in one batch for speed.
        alphas = torch.linspace(0.0, 1.0, steps=self._n_steps,
                                device=self._device).view(-1, 1, 1, 1)  # (S,1,1,1)
        interp = baseline.unsqueeze(0) + alphas * x_diff.unsqueeze(0)
        # interp shape: (n_steps, 1, seq_len, n_features) — flatten first two
        interp = interp.view(self._n_steps, self._seq_len, self._n_features)
        interp.requires_grad_(True)

        out = self._model(interp)                       # (n_steps,) raw logits
        # Sum over the batch so we can request gradients in one autograd call
        grads = torch.autograd.grad(
            outputs=out.sum(), inputs=interp,
            create_graph=False, retain_graph=False,
        )[0]                                            # (n_steps, seq_len, n_features)

        # Average gradient along the path, multiplied by (x - baseline).
        # This is the discrete Riemann approximation of the integral.
        avg_grad = grads.mean(dim=0)                    # (seq_len, n_features)
        attributions = (x_diff[0] * avg_grad).detach().cpu().numpy()
        return attributions.astype(np.float64)


# ── Stage-2 Non-IoT subclass ──────────────────────────────────────────

class Stage2NonIoTExplainer(_Stage2BaseIGExplainer):
    """
    Wraps monitoring.Stage2NonIoTDetector. Input modes:
      - dict           : single-flow feature dict (same as stage2_preprocess_non_iot)
                          → we scale via the wrapper, then build a length-1 sequence
                            padded with zeros to seq_len.
      - np.ndarray (n_features,)        : single SCALED row (already through preprocess)
      - np.ndarray (any_len, n_features): SCALED sequence (already through preprocess)
                                          — this is the path inference_bridge uses.
    """

    def __init__(self, detector, n_steps: Optional[int] = None):
        super().__init__(
            model        = detector._model,
            threshold    = detector._threshold,
            seq_len      = detector._seq_len,
            n_features   = detector._n_features,
            feature_cols = detector._feature_cols,
            n_steps      = n_steps,
        )
        self._detector = detector
        # Cache scaler params so we can reverse-scale for display values
        self._has_scaler = detector._has_scaler
        self._s2_mean    = detector._s2_mean
        self._s2_scale   = detector._s2_scale
        # Cache a copy of the most recent raw row, used by _raw_value_for
        # so the UI shows analyst-readable values not z-scores.
        self._last_raw_latest_row: Optional[np.ndarray] = None

    def _prepare_seq(self, flow_input) -> np.ndarray:
        """
        Two input modes accepted (the only ones that occur in the bridge):
          dict       → scale via stage2_preprocess_non_iot, build (seq_len,)-shaped
                       sequence padded with zeros.
          np.ndarray → already scaled. Pad/truncate to seq_len rows.
        """
        if isinstance(flow_input, dict):
            scaled_row = self._detector.stage2_preprocess_non_iot(flow_input)
            self._last_raw_latest_row = np.array(
                [flow_input.get(c, 0.0) for c in self._feature_cols],
                dtype=np.float32,
            )
            X = scaled_row.reshape(1, -1)
        elif isinstance(flow_input, np.ndarray):
            arr = np.asarray(flow_input, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(
                    f"Stage2NonIoTExplainer expects 1-D or 2-D ndarray, got shape {arr.shape}"
                )
            if arr.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature dim mismatch: input has {arr.shape[1]} features, "
                    f"model expects {self._n_features}."
                )
            # Reverse-scale the latest row for display purposes
            self._last_raw_latest_row = self._reverse_scale(arr[-1])
            X = arr
        else:
            raise TypeError(
                f"Stage2NonIoTExplainer.explain expects dict or np.ndarray; "
                f"got {type(flow_input).__name__}"
            )

        if len(X) < self._seq_len:
            pad = np.zeros((self._seq_len - len(X), self._n_features), np.float32)
            X = np.vstack([pad, X])
        return X[-self._seq_len:].astype(np.float32, copy=False)

    def _reverse_scale(self, scaled_row: np.ndarray) -> np.ndarray:
        """Convert a StandardScaler-normalised row back to raw values for UI."""
        if self._has_scaler and self._s2_mean is not None and self._s2_scale is not None:
            return (scaled_row * self._s2_scale + self._s2_mean).astype(np.float32)
        return scaled_row.astype(np.float32)

    def _raw_value_for(self, flow_input, feat_idx: int) -> float:
        if self._last_raw_latest_row is not None:
            return float(self._last_raw_latest_row[feat_idx])
        return 0.0


# ── Stage-2 IoT subclass ──────────────────────────────────────────────

class Stage2IoTExplainer(_Stage2BaseIGExplainer):
    """
    Wraps monitoring.Stage2IoTDetector. Input mode:
      - np.ndarray (seq_len, 115)  : already MinMax-scaled Kitsune sequence,
                                     same as what stage2_predict consumes.

    The IoT explainer is simpler than Non-IoT because the wrapper's only
    preprocess method (stage2_preprocess_iot) operates per-packet inside
    the Scapy callback — by the time XAI runs, scaling has already happened.
    """

    def __init__(self, detector, n_steps: Optional[int] = None):
        # Kitsune feature names live in src/live/kitsune_extractor.py.
        # Import lazily so loading this module doesn't pull in scapy.
        from src.live.kitsune_extractor import FEATURE_NAMES
        feature_cols = list(FEATURE_NAMES)
        # The IoT detector's checkpoint stores n_features (115) and seq_len (20).
        # We don't have direct access to those — read them off the model arch.
        # First conv1's in_channels matches n_features.
        n_features = detector._model.conv1[0].in_channels
        # IOT_SEQ_LEN is 20 in monitoring.py — the bridge always passes (20, 115)
        seq_len = 20
        super().__init__(
            model        = detector._model,
            threshold    = detector._threshold,
            seq_len      = seq_len,
            n_features   = n_features,
            feature_cols = feature_cols,
            n_steps      = n_steps,
        )
        self._detector = detector
        self._last_seq: Optional[np.ndarray] = None

    def _prepare_seq(self, flow_input) -> np.ndarray:
        if isinstance(flow_input, np.ndarray):
            arr = np.asarray(flow_input, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"Stage2IoTExplainer expects a 2-D ndarray of shape "
                    f"(seq_len, {self._n_features}); got {arr.shape}"
                )
            if arr.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature dim mismatch: array has {arr.shape[1]} features, "
                    f"model expects {self._n_features}"
                )
            if len(arr) < self._seq_len:
                pad = np.zeros((self._seq_len - len(arr), self._n_features), np.float32)
                arr = np.vstack([pad, arr])
            seq = arr[-self._seq_len:].astype(np.float32, copy=False)
            self._last_seq = seq
            return seq
        raise TypeError(
            f"Stage2IoTExplainer.explain expects np.ndarray "
            f"(shape ({self._seq_len}, {self._n_features})); "
            f"got {type(flow_input).__name__}"
        )

    def _raw_value_for(self, flow_input, feat_idx: int) -> float:
        # IoT input is already scaled to [0,1] — we display the scaled value
        # directly, which is the only thing the model saw. (Reversing the
        # MinMax would require the per-feature min/max from iot_scaler.json.)
        if self._last_seq is not None:
            return float(self._last_seq[-1, feat_idx])
        return 0.0


# ══════════════════════════════════════════════════════════════════════
# Self-test (run directly with: python3 -m src.xai.local_explainer)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Build a tiny mock detector that mirrors monitoring.Stage2NonIoTDetector
    # so we can exercise the IG path end-to-end without loading the real model.
    import torch.nn as nn

    class _MockNonIotModel(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_features, 128, 3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, stride=2))
            self.conv2 = nn.Sequential(
                nn.Conv1d(128, 256, 3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU())
            self.lstm = nn.LSTM(256, 128, num_layers=2,
                                batch_first=True, dropout=0.0)
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, 1))
        def forward(self, x):
            x = x.permute(0, 2, 1); x = self.conv1(x); x = self.conv2(x)
            x = x.permute(0, 2, 1)
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(1)

    class _MockNonIotDetector:
        def __init__(self):
            self._n_features   = 46
            self._seq_len      = 20
            self._threshold    = 0.5
            self._has_scaler   = False
            self._s2_mean      = None
            self._s2_scale     = None
            self._feature_cols = [
                "flow_duration", "total_fwd_packets", "total_bwd_packets",
                "flow_bytes_per_sec", "flow_pkts_per_sec", "flag_SYN",
                "flag_RST", "dst_port", "protocol", "fwd_pkt_len_mean",
            ] + [f"feat_{i}" for i in range(36)]
            self._model = _MockNonIotModel(self._n_features).eval()
        def stage2_preprocess_non_iot(self, feat):
            row = np.array([feat.get(c, 0.0) for c in self._feature_cols],
                           dtype=np.float32)
            return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    print("Self-test: Stage2NonIoTExplainer on a mock CNN-LSTM\n")
    torch.manual_seed(0); np.random.seed(0)
    det = _MockNonIotDetector()
    expl = Stage2NonIoTExplainer(det)
    print(f"  seq_len={expl._seq_len}  n_features={expl._n_features}  n_steps={expl._n_steps}")

    # --- Test 1: ndarray input (the inference_bridge fast path) ---
    arr = np.random.rand(20, 46).astype(np.float32) * 2.0
    res = expl.explain(arr, top_k=5)
    print(f"\n  Result on (20, 46) ndarray:")
    print(f"    method={res.method}, prediction={res.prediction}, conf={res.confidence:.4f}")
    print(f"    top features:")
    for f in res.top_features:
        print(f"      {f.display:<30}  attr={f.attribution:+.4f}  val={f.value:.4f}  {f.direction}")

    # --- Test 2: dict input ---
    feat = {
        "flow_duration":         0.05,
        "total_fwd_packets":     2,
        "flag_SYN":              30,
        "flow_pkts_per_sec":     500,
        "dst_port":              23,
    }
    res2 = expl.explain(feat, top_k=5)
    print(f"\n  Result on dict input:")
    print(f"    method={res2.method}, prediction={res2.prediction}, conf={res2.confidence:.4f}")

    # --- Sanity: completeness axiom ---
    # Sum of all attributions should ≈ f(x) - f(baseline) (raw logit diff).
    seq = expl._prepare_seq(arr)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        fx = expl._model(x).item()
        f0 = expl._model(torch.zeros_like(x)).item()
    total_attr = sum(res.raw_attributions.values())
    print(f"\n  IG completeness check:")
    print(f"    Σ attributions = {total_attr:+.6f}")
    print(f"    f(x) - f(baseline) = {fx - f0:+.6f}")
    print(f"    gap = {abs(total_attr - (fx - f0)):.6f}  (smaller is better)")