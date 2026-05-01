"""
local_explainer.py — Per-Flow Attribution for Stage-1 and Stage-2 Models
=========================================================================
Group 07 | CPCS499 | XAI Module

Purpose
-------
Given a single network flow (or a sequence of flows), explain WHY the
model classified it the way it did. Two explainer classes are exposed:

    Stage1Explainer     — wraps Stage-1 (RF / XGBoost). Uses SHAP TreeExplainer
                          (exact, fast, no sampling required for tree models).

    Stage2Explainer     — wraps the CNN-LSTM. Uses Integrated Gradients
                          (Sundararajan et al., 2017) — a gradient-based
                          attribution method that is:
                              · axiomatically grounded (completeness)
                              · model-internal (works on any nn.Module)
                              · ~50ms per flow on CPU
                              · stable for sequence models

Why Integrated Gradients (not SHAP) for the CNN-LSTM
----------------------------------------------------
SHAP DeepExplainer has known numerical issues with LSTMs in PyTorch
(particularly on MPS / CUDA). KernelExplainer is too slow for online
inference. IG sidesteps both: it only needs forward+backward passes
through the existing model, with no surrogate.

Output Schema
-------------
Both explainers return a `LocalExplanation` dataclass containing:
    prediction       : str            "benign"|"botnet" or "iot"|"noniot"
    confidence       : float          0–1, model probability
    top_features     : list[FeatureContribution]   sorted by |attribution|
    raw_attributions : dict[str, float]   feature → signed contribution
    method           : str            "integrated_gradients" | "shap_tree"

Typical Use
-----------
    from src.xai.local_explainer import Stage2Explainer
    expl = Stage2Explainer(stage2_detector)
    result = expl.explain(flow_df, top_k=8)
    print(result.top_features)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

from models.stage1.classifier import ALL_FEATURES
import numpy as np
import pandas as pd
import torch

# Optional dep — used only by Stage1Explainer
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.xai.feature_metadata import get_display, get_suspicion_dir


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureContribution:
    """One feature's signed contribution to the prediction."""
    feature:     str       # raw column name
    display:     str       # human-readable name
    value:       float     # observed value of the feature in this flow
    attribution: float     # signed contribution toward the predicted class
    direction:   str       # "↑ pushed toward botnet" / "↓ pushed toward benign"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalExplanation:
    """Container returned by every explainer."""
    prediction:       str
    confidence:       float
    top_features:     list[FeatureContribution] = field(default_factory=list)
    raw_attributions: dict[str, float]          = field(default_factory=dict)
    method:           str                       = ""

    def to_dict(self) -> dict:
        return {
            "prediction":       self.prediction,
            "confidence":       float(self.confidence),
            "method":           self.method,
            "top_features":     [f.to_dict() for f in self.top_features],
            "raw_attributions": {k: float(v) for k, v in self.raw_attributions.items()},
        }


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _direction_label(attribution: float, predicted_class: str,
                     positive_class: str) -> str:
    """
    Convert a signed attribution into a human-readable direction string.
    Positive attribution = pushed toward `positive_class` (e.g. "botnet").
    """
    if abs(attribution) < 1e-9:
        return "neutral"
    pushed_toward_positive = attribution > 0
    if pushed_toward_positive:
        return f"↑ pushed toward {positive_class}"
    else:
        # attribution < 0 → pushed toward the OTHER class
        other = "benign" if positive_class == "botnet" else (
                "noniot" if positive_class == "iot" else "other")
        return f"↓ pushed toward {other}"


def _rank_top_k(attributions: np.ndarray,
                feature_names: list[str],
                feature_values: np.ndarray,
                predicted_class: str,
                positive_class: str,
                top_k: int) -> list[FeatureContribution]:
    """Sort features by |attribution| descending and wrap as FeatureContribution."""
    order = np.argsort(np.abs(attributions))[::-1][:top_k]
    out = []
    for idx in order:
        name = feature_names[idx]
        attr = float(attributions[idx])
        # If the model predicted the negative class, flip sign so that
        # "+attribution" always means "pushed toward the predicted class".
        oriented_attr = attr if predicted_class == positive_class else -attr
        out.append(FeatureContribution(
            feature     = name,
            display     = get_display(name),
            value       = float(feature_values[idx]),
            attribution = oriented_attr,
            direction   = _direction_label(oriented_attr, predicted_class, positive_class),
        ))
    return out


# ══════════════════════════════════════════════════════════════════════
# STAGE-1 EXPLAINER  (RF / XGBoost — SHAP TreeExplainer)
# ══════════════════════════════════════════════════════════════════════

class Stage1Explainer:
    """
    Wraps a Stage1Classifier (sklearn RF or XGBoost) with SHAP TreeExplainer.

    SHAP TreeExplainer is exact for tree ensembles — no background sampling
    needed, no numerical approximation. Output: per-feature signed
    contributions in log-odds space (for XGB) or probability space (for RF).
    """

    POSITIVE_CLASS = "iot"   # we orient explanations toward "iot" by default

    def __init__(self, stage1_classifier):
        if not HAS_SHAP:
            raise ImportError(
                "shap is required for Stage1Explainer. Install with:\n"
                "  pip install shap        (Windows)\n"
                "  pip3 install shap       (macOS)"
            )
        self.clf = stage1_classifier
        self._model = stage1_classifier.model
        self._le    = stage1_classifier.label_encoder

        # Build TreeExplainer once — it's cheap to construct, expensive to call.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.TreeExplainer(self._model)

        # Feature ordering must match the classifier's training order.
        from models.stage1.classifier import ALL_FEATURES
        self.feature_cols = list(ALL_FEATURES)

    # ── Public API ──────────────────────────────────────────────────
    def explain(self, df: pd.DataFrame, top_k: int = 8) -> LocalExplanation:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Single-row DataFrame containing the 56 unified features.
        top_k : int
            Number of top contributing features to return.

        Returns
        -------
        LocalExplanation
        """
        if len(df) != 1:
            raise ValueError(
                f"Stage1Explainer.explain() expects a single-row DataFrame; "
                f"got {len(df)} rows."
            )

        # Align to the canonical feature order
        X = self.clf._align(df)              # shape (1, 56)
        prediction, confidence = self.clf.predict(df)

        # SHAP returns either an array (binary) or list-of-arrays (multiclass).
        # For a 2-class classifier, shap_values has shape (1, n_features) for
        # XGBoost, or list[(1, n_features), (1, n_features)] for RF.
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            # RF case: take SHAP values for the predicted class
            pred_idx = int(self._le.transform([prediction])[0])
            attributions = sv[pred_idx][0]
        else:
            # XGBoost case: single array. Sign already aligned with positive class.
            attributions = sv[0]

        top = _rank_top_k(
            attributions   = attributions,
            feature_names  = self.feature_cols,
            feature_values = X[0],
            predicted_class= prediction,
            positive_class = self.POSITIVE_CLASS,
            top_k          = top_k,
        )

        return LocalExplanation(
            prediction       = prediction,
            confidence       = float(confidence),
            top_features     = top,
            raw_attributions = dict(zip(self.feature_cols, attributions.tolist())),
            method           = "shap_tree",
        )


# ══════════════════════════════════════════════════════════════════════
# STAGE-2 EXPLAINER  (CNN-LSTM — Integrated Gradients)
# ══════════════════════════════════════════════════════════════════════

class Stage2Explainer:
    """
    Wraps a Stage2Detector (CNN-LSTM) with Integrated Gradients.

    Integrated Gradients (IG) computes the path integral of gradients
    from a baseline (zeros) to the actual input:

        IG_i(x) = (x_i - x'_i) * ∫_{α=0}^{1} ∂f(x' + α(x-x')) / ∂x_i  dα

    Approximated via Riemann sum with N steps (default 50). For sequence
    inputs, attributions are computed per-(timestep, feature), then summed
    across timesteps to produce a per-feature score.

    Why baseline = zeros: All input features are MinMax/StandardScaler
    normalized, so 0 represents "no information / minimum observed value".
    For a deeper analysis, you can later swap in a class-conditional baseline.
    """

    POSITIVE_CLASS = "botnet"
    N_STEPS_DEFAULT = 50    # IG Riemann steps. 50 is the canonical value.

    def __init__(self, stage2_detector, n_steps: int = N_STEPS_DEFAULT):
        self.det = stage2_detector
        self.model = stage2_detector.model
        self.seq_len = stage2_detector.seq_len
        self.n_features = stage2_detector.n_features
        self.threshold = stage2_detector.threshold
        self.feature_cols = list(stage2_detector.feature_cols)
        self.n_steps = n_steps

        # Same device the model lives on
        self.device = next(self.model.parameters()).device
        self.model.eval()

    # ── Public API ──────────────────────────────────────────────────
    def explain(self, df: pd.DataFrame, top_k: int = 8) -> LocalExplanation:
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least seq_len rows (will pad with zeros if shorter).
            Columns must include the model's expected feature_cols.
        top_k : int

        Returns
        -------
        LocalExplanation
        """
        # 1) Build the sequence tensor exactly as Stage2Detector.predict does
        X = self.det._align(df)                       # (n_rows, n_features)
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), self.n_features), np.float32)
            X = np.vstack([pad, X])
        seq = X[-self.seq_len:]                       # (seq_len, n_features)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        # x shape: (1, seq_len, n_features)

        # 2) Forward pass for the prediction
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        prediction = "botnet" if prob >= self.threshold else "benign"

        # 3) Compute Integrated Gradients
        attributions_seq = self._integrated_gradients(x)   # (seq_len, n_features)

        # 4) Aggregate across timesteps → per-feature contributions
        per_feature = attributions_seq.sum(axis=0)         # (n_features,)

        # 5) Use the LATEST timestep's values as "the" feature values shown to user.
        #    This is the most relevant flow being explained.
        feature_values = seq[-1]                            # (n_features,)

        top = _rank_top_k(
            attributions   = per_feature,
            feature_names  = self.feature_cols,
            feature_values = feature_values,
            predicted_class= prediction,
            positive_class = self.POSITIVE_CLASS,
            top_k          = top_k,
        )

        return LocalExplanation(
            prediction       = prediction,
            confidence       = float(prob),
            top_features     = top,
            raw_attributions = dict(zip(self.feature_cols, per_feature.tolist())),
            method           = "integrated_gradients",
        )

    # ── Internal: the IG algorithm ─────────────────────────────────
    def _integrated_gradients(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute IG attributions for input x against a zero baseline.

        Parameters
        ----------
        x : torch.Tensor of shape (1, seq_len, n_features)

        Returns
        -------
        attributions : np.ndarray of shape (seq_len, n_features)
            Per-(timestep, feature) signed contribution toward the
            "botnet" logit (the model's raw output before sigmoid).
        """
        baseline = torch.zeros_like(x)
        # Riemann mid-point alphas (more stable than left/right rule)
        alphas = torch.linspace(1.0 / (2 * self.n_steps),
                                1.0 - 1.0 / (2 * self.n_steps),
                                self.n_steps,
                                device=self.device)

        accumulated_grads = torch.zeros_like(x)
        for alpha in alphas:
            interp = baseline + alpha * (x - baseline)
            interp.requires_grad_(True)

            # Forward — model returns raw logit; we explain that directly.
            logit = self.model(interp)
            # Sum is fine because batch size is 1.
            grads = torch.autograd.grad(
                outputs=logit.sum(),
                inputs=interp,
                retain_graph=False,
                create_graph=False,
            )[0]
            accumulated_grads = accumulated_grads + grads.detach()

        # Average gradient over the path × (x - baseline)
        avg_grads = accumulated_grads / self.n_steps
        attributions = (x - baseline) * avg_grads
        # Squeeze the batch dim
        return attributions.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Build a minimal dummy CNN-LSTM and run IG on random input,
    # to verify the algorithm is wired correctly.
    import torch.nn as nn

    print("Self-test: Integrated Gradients on a dummy CNN-LSTM\n")

    SEQ_LEN = 20

    class _Dummy(nn.Module):
        def __init__(self, n_feat: int):
            super().__init__()
            self.conv = nn.Conv1d(n_feat, 32, 3, padding=1)
            self.lstm = nn.LSTM(32, 16, batch_first=True)
            self.head = nn.Linear(16, 1)
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv(x))
            x = x.permute(0, 2, 1)
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(1)

    class _DetectorMock:
        def __init__(self):
            from src.ingestion.preprocess_from_pcap_csvs import ALL_FEATURES
            cols = [c for c in ALL_FEATURES if c not in
                    {"class_label", "device_type", "src_ip", "timestamp", "ttl_mean"}]
            self.feature_cols = cols
            self.n_features = len(cols)
            self.model = _Dummy(self.n_features).eval()
            self.seq_len = SEQ_LEN
            self.threshold = 0.5
        def _align(self, df):
            arr = np.zeros((len(df), self.n_features), dtype=np.float32)
            for i, c in enumerate(self.feature_cols):
                if c in df.columns:
                    arr[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.).values
            return arr

    det = _DetectorMock()
    df = pd.DataFrame(
        np.random.rand(SEQ_LEN, det.n_features).astype(np.float32),
        columns=det.feature_cols,
    )
    expl = Stage2Explainer(det, n_steps=20)
    result = expl.explain(df, top_k=5)
    print(f"  Prediction: {result.prediction}  ({result.confidence:.4f})")
    print(f"  Method: {result.method}")
    print(f"  Top-{len(result.top_features)} features:")
    for f in result.top_features:
        print(f"    {f.display:30s} attr={f.attribution:+.4f}  val={f.value:.4f}  {f.direction}")
    print("\n  Sanity check (IG completeness axiom):")
    total = sum(result.raw_attributions.values())
    print(f"    Σ attributions ≈ f(x) - f(baseline)  →  {total:+.4f}")
    print("    (Should be roughly equal to the model's raw logit on x.)")
