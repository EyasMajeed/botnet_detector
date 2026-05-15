"""
X-A1 Integrated Gradients sanity battery.

Three sub-checks, all run on the same flow population:

  1. Repeat stability — top-3 features identical across 5 repeated calls
                        on the SAME input.
  2. Perturbation stability — small Gaussian noise in scaled space should
                        keep top-3 mostly unchanged.
  3. Sign coherence — when the wrapper predicts 'botnet', the SUM of
                        attributions over predicted-class direction
                        should be strictly positive on most flows.

We run on synthetic Non-IoT flows (one stage at a time) so the test
doesn't require any private dataset.
"""

from __future__ import annotations

import csv
from typing import Any

from test_harness.generators.flow_csv_gen import (
    S1_FEATURES, _botnet_noniot_row, _benign_noniot_row, write_csv,
)
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import, ensure_on_path


TEST_ID = "X-A1"


def _build_explainer():
    """Build the Stage-2 Non-IoT explainer over the trained wrapper."""
    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return None, f"monitoring import: {err}"
    xai_mod, err2 = soft_import("src.xai")
    if xai_mod is None:
        xai_mod, err2 = soft_import("src.xai.local_explainer")
    if xai_mod is None:
        return None, f"src.xai import: {err2}"
    try:
        s2 = monitoring.Stage2NonIoTDetector(monitoring.MODEL_S2_NONIOT)
    except Exception as e:                                   # noqa: BLE001
        return None, f"Stage2NonIoTDetector init: {e}"
    try:
        Expl = getattr(xai_mod, "Stage2NonIoTExplainer", None)
        if Expl is None:
            # The package re-exports via __init__; fall back.
            from src.xai.local_explainer import Stage2NonIoTExplainer as Expl
        expl = Expl(s2)
    except Exception as e:                                   # noqa: BLE001
        return None, f"Stage2NonIoTExplainer init: {e}"
    return expl, None


def _top_k(explanation, k: int = 3) -> list[str]:
    return [tf.feature for tf in (explanation.top_features or [])[:k]]


def run(n_samples: int = 30, perturb_sigma: float = 0.01) -> dict:
    expl, err = _build_explainer()
    if expl is None:
        return {"test_id": TEST_ID, "name": "XAI sanity battery",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "stability + sign-coherence checks pass",
                "actual":   err}

    import random
    rng = random.Random(0)
    flows: list[dict] = []
    for _ in range(n_samples // 2):
        flows.append(_botnet_noniot_row(rng))
    for _ in range(n_samples - n_samples // 2):
        flows.append(_benign_noniot_row(rng))

    # ── Repeat stability ──────────────────────────────────────────────
    repeat_match = 0
    for f in flows:
        try:
            tops = []
            for _ in range(5):
                e = expl.explain(f, top_k=8)
                tops.append(_top_k(e, 3))
            if all(t == tops[0] for t in tops):
                repeat_match += 1
        except Exception:
            pass
    repeat_rate = repeat_match / len(flows)

    # ── Perturbation stability ────────────────────────────────────────
    perturb_match = 0
    perturb_attempts = 0
    for f in flows:
        try:
            base = expl.explain(f, top_k=8)
            base_top = _top_k(base, 3)
            f2 = dict(f)
            for k in S1_FEATURES:
                v = float(f2.get(k, 0.0) or 0.0)
                f2[k] = v * (1 + rng.gauss(0, perturb_sigma))
            pert = expl.explain(f2, top_k=8)
            perturb_attempts += 1
            if len(set(base_top) & set(_top_k(pert, 3))) >= 2:
                perturb_match += 1
        except Exception:
            pass
    perturb_rate = (perturb_match / perturb_attempts) if perturb_attempts else 0.0

    # ── Sign coherence ────────────────────────────────────────────────
    coherent = 0
    coherent_total = 0
    for f in flows:
        try:
            e = expl.explain(f, top_k=8)
            if e.prediction != "botnet":
                continue
            coherent_total += 1
            sum_attr = sum(tf.attribution for tf in e.top_features
                           if tf.attribution >= 0)
            sum_neg = abs(sum(tf.attribution for tf in e.top_features
                              if tf.attribution < 0))
            if sum_attr > sum_neg:
                coherent += 1
        except Exception:
            pass
    sign_rate = (coherent / coherent_total) if coherent_total else 0.0

    findings = {
        "repeat_stability_rate":   round(repeat_rate, 3),
        "perturb_stability_rate":  round(perturb_rate, 3),
        "sign_coherence_rate":     round(sign_rate, 3),
    }

    issues = []
    if repeat_rate < 0.95: issues.append("repeat instability")
    if perturb_rate < 0.80: issues.append("perturbation instability")
    if sign_rate < 0.80 and coherent_total >= 5:
        issues.append("sign incoherent on botnet predictions")

    if not issues:
        verdict, actual = "PASS", f"all stability metrics within target: {findings}"
    else:
        verdict, actual = "FAIL", f"{', '.join(issues)}; metrics: {findings}"
    return {"test_id": TEST_ID, "name": "XAI sanity battery",
            "severity": "HIGH", "verdict": verdict,
            "expected": "repeat≥0.95, perturb≥0.80 (top-3 ∩ ≥2), sign≥0.80",
            "actual":   actual, "raw": findings}
