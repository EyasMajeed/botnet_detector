"""
src/xai/__init__.py — Public API of the XAI module
====================================================
Group 07 | CPCS499

This is the ONLY file the rest of the system (inference_bridge.py,
the desktop GUI, reports, etc.) needs to import from.

Public objects
--------------
    Stage1Explainer       : per-flow attribution for Stage-1 (RF/XGBoost)
    Stage2Explainer       : per-flow attribution for Stage-2 (CNN-LSTM)
    LocalExplanation      : dataclass returned by Stage{1,2}Explainer.explain()
    HumanExplanation      : final analyst-facing dataclass
    explain_flow()        : end-to-end convenience function (recommended)
    build_human_explanation()  : low-level: LocalExplanation → HumanExplanation

Recommended usage from inference_bridge.py
------------------------------------------
    from src.xai import explain_flow, ExplainerBundle

    bundle = ExplainerBundle(stage1_clf, iot_detector, noniot_detector)

    # ... after running Stage-1 + Stage-2 ...
    result = explain_flow(
        flow_df       = current_window_df,
        stage1_label  = "iot",
        stage2_label  = "botnet",
        bundle        = bundle,
    )
    # result is dict with keys: prediction, confidence, top_features,
    # summary, severity, reasons, recommendations
"""

from __future__ import annotations

from dataclasses import dataclass

from src.xai.local_explainer import (
    Stage1Explainer,
    Stage2Explainer,
    LocalExplanation,
    FeatureContribution,
)
from src.xai.explanation_engine import (
    HumanExplanation,
    build_human_explanation,
)
from src.xai.feature_metadata import (
    FEATURE_META,
    get_display,
    get_description,
    get_category,
)


__all__ = [
    "Stage1Explainer",
    "Stage2Explainer",
    "LocalExplanation",
    "FeatureContribution",
    "HumanExplanation",
    "build_human_explanation",
    "FEATURE_META",
    "get_display",
    "get_description",
    "get_category",
    "ExplainerBundle",
    "explain_flow",
]


# ══════════════════════════════════════════════════════════════════════
# Bundle: lazy-loads explainers once and reuses them across calls
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExplainerBundle:
    """
    Holds the three explainer instances + makes them lazy-instantiated.
    Construct ONCE at app startup; reuse for every detected flow.
    """
    stage1_classifier: object = None    # Stage1Classifier instance (or None)
    iot_detector:      object = None    # Stage2Detector for IoT
    noniot_detector:   object = None    # Stage2Detector for non-IoT

    def __post_init__(self):
        self._stage1_expl = None
        self._iot_expl    = None
        self._noniot_expl = None

    def stage1(self) -> Stage1Explainer | None:
        if self._stage1_expl is None and self.stage1_classifier is not None:
            self._stage1_expl = Stage1Explainer(self.stage1_classifier)
        return self._stage1_expl

    def stage2(self, device_type: str) -> Stage2Explainer | None:
        if device_type == "iot":
            if self._iot_expl is None and self.iot_detector is not None:
                self._iot_expl = Stage2Explainer(self.iot_detector)
            return self._iot_expl
        else:
            if self._noniot_expl is None and self.noniot_detector is not None:
                self._noniot_expl = Stage2Explainer(self.noniot_detector)
            return self._noniot_expl


# ══════════════════════════════════════════════════════════════════════
# End-to-end convenience function
# ══════════════════════════════════════════════════════════════════════

def explain_flow(flow_df,
                 stage1_label: str,
                 stage2_label: str,
                 bundle: ExplainerBundle,
                 top_k: int = 8) -> dict:
    """
    Produce a full explanation for one flow / window.

    Parameters
    ----------
    flow_df : pd.DataFrame
        The window of flow records that the Stage-2 model was given.
        For Stage-1 (single-flow), only the LAST row is used.
    stage1_label : str   — "iot" or "noniot"  (already predicted upstream)
    stage2_label : str   — "benign" or "botnet"  (already predicted upstream)
    bundle       : ExplainerBundle   — initialised at app startup
    top_k        : int   — number of features to surface

    Returns
    -------
    dict ready to be written to JSON / shown in the GUI / saved to the
    EXPLAINABILITY DB table from your ERD. Keys:
        stage          : "1" | "2"
        prediction     : str
        confidence     : float
        method         : str
        top_features   : list[dict]   (display, value, attribution, direction)
        summary        : str          (human-readable)
        pattern        : str
        severity       : str
        reasons        : list[str]
        recommendations: list[str]

    Notes
    -----
    For benign Stage-2 predictions we still produce an explanation — the
    analyst may want to see why the model is confident, especially for
    audit purposes. The recommendations list will simply be empty.
    """
    s2_expl = bundle.stage2(stage1_label)
    if s2_expl is None:
        raise ValueError(
            f"No Stage-2 detector loaded for device_type={stage1_label!r}. "
            "Initialise ExplainerBundle with both iot_detector and noniot_detector."
        )

    local: LocalExplanation = s2_expl.explain(flow_df, top_k=top_k)
    # Sanity: explainer's prediction should match the upstream Stage-2 label.
    # If they differ (rare race condition), we trust the upstream label and
    # use the local explanation only for attributions.
    if local.prediction != stage2_label:
        local.prediction = stage2_label

    human: HumanExplanation = build_human_explanation(local)

    return {
        "stage":          "2",
        "device_type":    stage1_label,
        "prediction":     local.prediction,
        "confidence":     float(local.confidence),
        "method":         local.method,
        "top_features":   [f.to_dict() for f in local.top_features],
        "summary":        human.summary,
        "pattern":        human.pattern,
        "severity":       human.severity,
        "reasons":        human.reasons,
        "recommendations":human.recommendations,
    }
