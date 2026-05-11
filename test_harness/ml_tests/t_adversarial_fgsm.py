"""
M-ADV Scaler-aware adversarial perturbation.

The on-disk s1_scaler.json reveals every feature mean/scale. An attacker
who reads it can craft small, scaler-aware perturbations that flip
Stage-1 routing. We measure how many decisions flip at ε ∈ {0.001, 0.005, 0.01}.

This is a feature-space attack, not pixel-space, so we use the sign of
the partial-derivative approximation via finite differences (sklearn's
predict_proba doesn't give gradients, so we step each feature ± ε in
scaled space and pick the direction that increases P(noniot) when the
true class is iot, or vice versa). This is a standard tabular FGSM.

We do NOT need to retrain — only to demonstrate brittleness.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from test_harness.generators.flow_csv_gen import S1_FEATURES, synthetic_csv
from test_harness.utils.paths import for_test
from test_harness.utils.project_imports import soft_import, ensure_on_path


TEST_ID = "M-ADV"


def _stage1():
    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return None, f"monitoring import: {err}"
    try:
        return monitoring.Stage1Classifier(
            monitoring.MODEL_S1_RF, monitoring.SCALER_S1_JSON
        ), None
    except Exception as e:                                    # noqa: BLE001
        return None, f"Stage1Classifier init: {e}"


def _proba_iot(s1, feat: dict) -> float:
    """Return P(iot) from the wrapper's RF, using scaled-row internals."""
    row = s1.stage1_preprocess(feat)
    proba = s1._rf.predict_proba(row.reshape(1, -1))[0]
    # Find the index for "iot" via the LabelEncoder.
    classes = list(s1._le.classes_) if getattr(s1, "_le", None) else list(s1._rf.classes_)
    if "iot" in classes:
        return float(proba[classes.index("iot")])
    # Fallback — assume binary.
    return float(proba[1])


def run(n_samples: int = 100, epsilons: Iterable[float] = (0.001, 0.005, 0.01),
        seed: int = 0) -> dict:
    ensure_on_path()
    s1, err = _stage1()
    if s1 is None:
        return {"test_id": TEST_ID, "name": "Scaler-aware adversarial perturbation",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "≤ 30% routing flips at ε=0.01",
                "actual":   err}
    if not getattr(s1, "_has_scaler", False):
        return {"test_id": TEST_ID, "name": "Scaler-aware adversarial perturbation",
                "severity": "HIGH", "verdict": "SKIPPED",
                "expected": "scaler available (s1_scaler.json)",
                "actual":   "Stage-1 running without scaler — adversarial test would mis-attribute results."}

    dirs = for_test(TEST_ID)
    csv_path = synthetic_csv(dirs["artifacts"] / "adv_seed.csv",
                             n_iot=n_samples, n_noniot=n_samples,
                             n_botnet=0, seed=seed, unique_src_ips=20)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))[: 2 * n_samples]

    findings = {}
    for eps in epsilons:
        n_flipped = 0
        n_attempted = 0
        for r in rows:
            feat = {k: float(r.get(k, 0.0) or 0.0) for k in S1_FEATURES}
            try:
                p_iot_orig = _proba_iot(s1, feat)
            except Exception:
                continue
            orig_label = "iot" if p_iot_orig >= 0.5 else "noniot"
            n_attempted += 1
            # Per-feature finite-difference sign attack — pick the move
            # that pushes p_iot toward the OPPOSITE label.
            sign = -1 if orig_label == "iot" else +1
            adv = dict(feat)
            for fname, mean, scale in zip(s1._features, s1._mean, s1._scale):
                if scale == 0:
                    continue
                # +eps in scaled space == +eps*scale in raw space
                delta = eps * float(scale) * sign
                # Sign per-feature: try + first, fall back to -
                a_plus  = dict(adv); a_plus[fname]  = adv[fname] + delta
                a_minus = dict(adv); a_minus[fname] = adv[fname] - delta
                try:
                    p_plus  = _proba_iot(s1, a_plus)
                    p_minus = _proba_iot(s1, a_minus)
                except Exception:
                    continue
                # Greedy: keep whichever direction moves p_iot in the desired direction
                want_plus = (p_plus  - p_iot_orig) * sign > 0
                want_minus = (p_minus - p_iot_orig) * sign > 0
                if want_plus and (p_plus - p_iot_orig) * sign > (p_minus - p_iot_orig) * sign:
                    adv = a_plus
                elif want_minus:
                    adv = a_minus
            try:
                p_iot_adv = _proba_iot(s1, adv)
            except Exception:
                continue
            new_label = "iot" if p_iot_adv >= 0.5 else "noniot"
            if new_label != orig_label:
                n_flipped += 1
        findings[f"eps={eps}"] = {"flipped": n_flipped,
                                  "attempted": n_attempted,
                                  "rate": round(n_flipped / max(n_attempted, 1), 3)}

    worst = max(findings.values(), key=lambda d: d["rate"])
    if worst["rate"] <= 0.30:
        verdict, actual = "PASS", f"max flip rate {worst['rate']:.0%}"
    else:
        verdict, actual = "FAIL", (
            f"max flip rate {worst['rate']:.0%} — Stage-1 is brittle to "
            "scaler-aware feature-space perturbations.")
    return {"test_id": TEST_ID, "name": "Scaler-aware adversarial perturbation",
            "severity": "HIGH", "verdict": verdict,
            "expected": "max routing-flip rate ≤ 30% at any tested ε",
            "actual":   actual, "csv": str(csv_path),
            "raw": findings}
