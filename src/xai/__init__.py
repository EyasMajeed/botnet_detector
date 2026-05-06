"""
═══════════════════════════════════════════════════════════════════════
 src/xai — Explainable AI module for the AI-Based Botnet Detection system
 Group 07 | CPCS499
═══════════════════════════════════════════════════════════════════════

PUBLIC API
──────────
  ExplainerBundle(stage1_classifier=None,
                  iot_detector=None,
                  noniot_detector=None)
      Holds the three explainers. Construct ONCE at app startup, after
      the monitoring.py wrappers are loaded, then pass the bundle to
      explain_flow() per detection.

  explain_flow(flow_input, *, stage1_label, stage2_label,
               bundle, top_k=8)
      One-call entrypoint for the inference bridge. Returns a dict ready
      for the GUI / DB / report module. On any runtime failure it returns
      a degraded-but-valid dict (method="failed") instead of raising —
      this keeps the live monitor and the upload page from crashing
      mid-detection.

USAGE EXAMPLE  (see app/inference_bridge.py for real integration)
─────────────────────────────────────────────────────────────────
    from src.xai import ExplainerBundle, explain_flow

    # At app startup, after monitoring.py wrappers are loaded:
    bundle = ExplainerBundle(
        stage1_classifier = stage1_clf,        # monitoring.Stage1Classifier
        iot_detector      = iot_det,           # monitoring.Stage2IoTDetector
        noniot_detector   = noniot_det,        # monitoring.Stage2NonIoTDetector
    )

    # Per detected botnet flow:
    result = explain_flow(
        seq_array,                             # np.ndarray (20, n_features)
        stage1_label = "noniot",
        stage2_label = "botnet",
        bundle       = bundle,
        top_k        = 8,
    )
    # result is a plain dict ready to attach to the existing flow record:
    #   result["top_features"]    → list of {feature, display, value, attribution, direction}
    #   result["summary"]         → one-sentence English summary
    #   result["reasons"]         → 2–4 bullet-point reasons
    #   result["recommendations"] → list of next-step actions
    #   result["pattern"]         → behaviour tag (PORT_SCAN, DDOS, ...)
    #   result["severity"]        → "low" | "medium" | "high" | "critical"
"""

from __future__ import annotations

from typing import Optional, Union

# Public dataclasses re-exported for convenience
from src.xai.local_explainer import (
    Stage1Explainer,
    Stage2IoTExplainer,
    Stage2NonIoTExplainer,
    LocalExplanation,
    FeatureContribution,
)
from src.xai.explanation_engine import (
    HumanExplanation,
    build_human_explanation,
    PATTERN_THRESHOLD,
    LIVE_CONSTANT_FEATURES,
)


# ══════════════════════════════════════════════════════════════════════
# ExplainerBundle
# ══════════════════════════════════════════════════════════════════════

class ExplainerBundle:
    """
    Holds the three explainers initialised at app startup.

    All three slots are optional:
      · If stage1_classifier is None, Stage-1 explanations are unavailable
        (Stage-2 still works — Stage-1 is just routing, not the consequential
         decision).
      · If iot_detector / noniot_detector are None, that branch's
        explanations are unavailable. explain_flow() raises ValueError
        if asked to explain a branch with no loaded detector.

    The constructor is lazy: explainers are built only for the slots
    that received a non-None argument. This means you can incrementally
    enable Stage-1, then Stage-2 IoT, then Stage-2 Non-IoT as the rest
    of the pipeline comes online.
    """

    def __init__(self,
                 stage1_classifier=None,
                 iot_detector=None,
                 noniot_detector=None,
                 n_steps: Optional[int] = None):
        self._s1 = None
        self._s2_iot = None
        self._s2_noniot = None

        # Stage-1 explainer — only build if SHAP is available AND a model
        # was supplied. Build failure (e.g. shap not installed) is logged
        # and skipped, not propagated, so the bundle remains usable for
        # Stage-2 even without Stage-1.
        if stage1_classifier is not None:
            try:
                self._s1 = Stage1Explainer(stage1_classifier)
            except Exception as e:
                import sys
                print(f"[xai] Stage1Explainer disabled: {type(e).__name__}: {e}",
                      file=sys.stderr)
                self._s1 = None

        if iot_detector is not None:
            self._s2_iot = Stage2IoTExplainer(iot_detector, n_steps=n_steps)

        if noniot_detector is not None:
            self._s2_noniot = Stage2NonIoTExplainer(noniot_detector, n_steps=n_steps)

    # ── Accessors ──────────────────────────────────────────────────
    def stage1(self) -> Optional[Stage1Explainer]:
        return self._s1

    def stage2(self, device_type: str):
        """Return the Stage-2 explainer for a given Stage-1 device label."""
        if device_type == "iot":
            return self._s2_iot
        if device_type == "noniot":
            return self._s2_noniot
        return None


# ══════════════════════════════════════════════════════════════════════
# explain_flow — the one function inference_bridge calls per detection
# ══════════════════════════════════════════════════════════════════════

def explain_flow(flow_input,
                 *,
                 stage1_label: str,
                 stage2_label: str,
                 bundle: ExplainerBundle,
                 top_k: int = 8) -> dict:
    """
    Produce a full Stage-2 explanation for one flow / window.

    Parameters
    ----------
    flow_input : Union[np.ndarray, dict]
        The same input that was fed to stage2_predict for this flow.
        For Non-IoT (CSV mode in inference_bridge): the bridge has
            seq = np.stack(list(noniot_bufs[key]))   # shape (≤20, 46)
        and passes that. We accept the same shape directly so the bridge
        doesn't have to repack into a DataFrame.
        For IoT (live PCAP mode): the bridge has the (20, 115) Kitsune
        sliding window as a numpy array — pass that.
        A flow dict is also accepted for the Non-IoT branch as a fallback
        (e.g. for run_inference's single-flow path).

    stage1_label : str
        "iot" or "noniot" — the device type Stage-1 routed this flow to.
        Selects which Stage-2 explainer the bundle dispatches to.

    stage2_label : str
        The Stage-2 prediction the bridge already obtained from
        stage2_predict. Passed in so the explanation matches the upstream
        prediction exactly even in the rare case where IG's reconstructed
        prediction differs (e.g. due to floating-point noise near threshold).

    bundle : ExplainerBundle

    top_k : int
        Number of features to surface in top_features. Default 8.

    Returns
    -------
    dict
        A flat, JSON-serialisable dict ready to:
          · attach to the existing flow record dict in inference_bridge
          · render in the GUI's HBar widget (via top_features)
          · write to the EXPLAINABILITY DB table (per the ER diagram)
          · include in CSV/PDF exports

    Keys:
        device_type        : "iot" | "noniot"
        prediction         : "botnet" | "benign"
        confidence         : float (Stage-2 sigmoid output)
        method             : "integrated_gradients" | "shap_tree" | "failed"
        top_features       : list of {feature, display, value, attribution, direction}
        feature_importance : dict[feature → magnitude]   ← for HBar widget
        summary            : str    (one-sentence summary)
        pattern            : str    ("PORT_SCAN" | ... | "GENERIC")
        severity           : str    ("low" | "medium" | "high" | "critical")
        reasons            : list[str]
        recommendations    : list[str]

    Failure handling
    ----------------
    If the explainer or the rule engine raises (gradient computation NaN,
    MPS quirk, schema mismatch, unexpected input shape, etc.) we catch
    the exception and return a degraded-but-valid dict with method="failed".
    Configuration errors (no detector loaded for the requested branch) are
    raised as ValueError so they're caught at startup, not silently swallowed
    per-flow.
    """
    s2_expl = bundle.stage2(stage1_label)
    if s2_expl is None:
        # Configuration error — surface this loudly.
        raise ValueError(
            f"No Stage-2 detector loaded for device_type={stage1_label!r}. "
            "Initialise ExplainerBundle with iot_detector and/or noniot_detector."
        )

    # ── Run the explainer (catch runtime failures) ─────────────────────
    try:
        local: LocalExplanation = s2_expl.explain(flow_input, top_k=top_k)
    except Exception as e:
        import sys, traceback
        print(f"[xai] Stage-2 explainer ({stage1_label}) failed: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return _failed_explanation(stage1_label, stage2_label,
                                   reason=f"{type(e).__name__}: {e}")

    # Sanity: explainer's prediction should match the upstream Stage-2 label.
    # If they differ, we trust the upstream label and use the explanation
    # only for its attributions.
    if local.prediction != stage2_label:
        local.prediction = stage2_label

    # ── Run the rule engine (catch runtime failures) ──────────────────
    try:
        human: HumanExplanation = build_human_explanation(local)
    except Exception as e:
        import sys, traceback
        print(f"[xai] explanation_engine failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Still return the attributions so the GUI's bar chart works.
        return _result_dict(stage1_label, local,
                            summary  = "Rule engine failed; raw feature attributions only.",
                            pattern  = "GENERIC",
                            severity = "low",
                            reasons  = [],
                            recommendations = [])

    return _result_dict(stage1_label, local,
                        summary  = human.summary,
                        pattern  = human.pattern,
                        severity = human.severity,
                        reasons  = human.reasons,
                        recommendations = human.recommendations)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _result_dict(stage1_label: str,
                 local: LocalExplanation,
                 *,
                 summary: str, pattern: str, severity: str,
                 reasons: list, recommendations: list) -> dict:
    """Build the standard success-path result dict."""
    return {
        "device_type":        stage1_label,
        "prediction":         local.prediction,
        "confidence":         float(local.confidence),
        "method":             local.method,
        "top_features":       [f.to_dict() for f in local.top_features],
        # Magnitude-only dict for the GUI's HBar widget (which expects
        # positive values keyed by feature name and normalises by max).
        "feature_importance": {
            f.display: abs(f.attribution) for f in local.top_features
        },
        "summary":            summary,
        "pattern":            pattern,
        "severity":           severity,
        "reasons":            reasons,
        "recommendations":    recommendations,
    }


def _failed_explanation(stage1_label: str, stage2_label: str, reason: str) -> dict:
    """
    Standardised degraded payload returned by explain_flow when the
    explainer raises. Shape matches the success payload exactly so
    downstream consumers (GUI, DB writer, report generator) don't need
    a separate code path for failures.
    """
    return {
        "device_type":        stage1_label,
        "prediction":         stage2_label,
        "confidence":         0.0,
        "method":             "failed",
        "top_features":       [],
        "feature_importance": {},
        "summary":            "Explanation unavailable (XAI module error).",
        "pattern":            "UNKNOWN",
        "severity":           "low",
        "reasons":            [f"XAI runtime error: {reason}"],
        "recommendations":    [
            "Inspect the application log for the full traceback.",
            "If this happens repeatedly, file an issue with the flow's feature vector.",
        ],
    }


__all__ = [
    "ExplainerBundle",
    "explain_flow",
    "Stage1Explainer",
    "Stage2IoTExplainer",
    "Stage2NonIoTExplainer",
    "LocalExplanation",
    "FeatureContribution",
    "HumanExplanation",
    "build_human_explanation",
    "PATTERN_THRESHOLD",
    "LIVE_CONSTANT_FEATURES",
]
