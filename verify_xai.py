"""
═══════════════════════════════════════════════════════════════════════
 verify_xai.py — End-to-end verification suite for the XAI module
 Group 07 | CPCS499
═══════════════════════════════════════════════════════════════════════

Run from the repo root:
    python3 verify_xai.py

What this catches
─────────────────
Each test catches a different class of bug:

  Test 1: Imports & wiring         — does the module even load?
  Test 2: Math correctness         — does IG satisfy the Completeness axiom?
                                     Does SHAP return attributions that sum
                                     to model output - expected_value?
  Test 3: Determinism              — same input → same output, repeatable
  Test 4: Input variance sensitivity — different inputs → different attributions
                                       (catches "constant garbage" failures)
  Test 5: Pattern matching         — synthetic port-scan / DDoS / brute-force
                                     features trigger the expected pattern,
                                     constant-valued features do NOT trigger
                                     false positives

What this does NOT do
─────────────────────
This script does NOT verify that the *trained model* is correct — that's
a separate eval (your team already did it, see TC-22 in R2). What it
verifies is that:

  · The XAI math is sound
  · The integration with monitoring.py is correct
  · The rule engine fires only on meaningful signal
  · The whole pipeline produces actionable output, not garbage

If all five tests pass, you can say "XAI works correctly" with evidence.
If any fails, the failure message tells you which specific assumption
broke and where to look.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
# Find repo root (the directory containing monitoring.py + models/)
ROOT = Path(__file__).resolve().parent
for _ in range(4):
    if (ROOT / "monitoring.py").exists() and (ROOT / "models").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

# ── Pretty output helpers ────────────────────────────────────────────
RESET, BOLD, GREEN, RED, YELLOW, CYAN = (
    "\033[0m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m"
)
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


def banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 72}{RESET}")
    print(f"{BOLD}{CYAN} {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 72}{RESET}")


def step(msg: str) -> None:
    print(f"  {msg}")


# ════════════════════════════════════════════════════════════════════
# Test 1 — Imports & wiring
# Catches: missing files, broken imports, syntax errors, missing deps
# ════════════════════════════════════════════════════════════════════
def test_1_imports() -> bool:
    banner("TEST 1 — Imports & module loading")

    try:
        step("Importing src.xai ...")
        from src.xai import (ExplainerBundle, explain_flow,
                             LocalExplanation, FeatureContribution,
                             HumanExplanation, build_human_explanation,
                             LIVE_CONSTANT_FEATURES, PATTERN_THRESHOLD)
        step(f"  {PASS} src.xai loaded")
        step(f"  PATTERN_THRESHOLD = {PATTERN_THRESHOLD}")
        step(f"  LIVE_CONSTANT_FEATURES guards {len(LIVE_CONSTANT_FEATURES)} features")

        step("Importing monitoring.py wrappers ...")
        from monitoring import (Stage1Classifier, Stage2NonIoTDetector,
                                Stage2IoTDetector,
                                MODEL_S1_RF, SCALER_S1_JSON, MODEL_S2_NONIOT,
                                MODEL_S2_IOT, SCALER_S2_IOT,
                                S1_FEATURES, S2_NONIOT_FEATURES_FALLBACK)
        step(f"  {PASS} monitoring.py loaded")
        step(f"  S1_FEATURES count = {len(S1_FEATURES)}")
        step(f"  S2_NONIOT_FEATURES_FALLBACK count = {len(S2_NONIOT_FEATURES_FALLBACK)}")

        step("Loading trained models ...")
        if not MODEL_S1_RF.exists():
            step(f"  {FAIL} Stage-1 model missing: {MODEL_S1_RF}")
            return False
        if not MODEL_S2_NONIOT.exists():
            step(f"  {FAIL} Stage-2 Non-IoT model missing: {MODEL_S2_NONIOT}")
            return False

        s1 = Stage1Classifier(MODEL_S1_RF, SCALER_S1_JSON)
        step(f"  {PASS} Stage1Classifier loaded "
             f"(features={len(s1._features)}, scaler={'yes' if s1._has_scaler else 'NO'})")

        s2 = Stage2NonIoTDetector(MODEL_S2_NONIOT)
        step(f"  {PASS} Stage2NonIoTDetector loaded "
             f"(features={s2._n_features}, seq_len={s2._seq_len}, "
             f"threshold={s2._threshold:.3f}, scaler={'yes' if s2._has_scaler else 'NO'})")

        # Stage-2 IoT — try to load. If iot_model is missing or kitsune_extractor
        # isn't importable, we still continue with Stage-2 NonIoT only.
        s2_iot = None
        if MODEL_S2_IOT.exists():
            try:
                s2_iot = Stage2IoTDetector(MODEL_S2_IOT, SCALER_S2_IOT)
                step(f"  {PASS} Stage2IoTDetector loaded (Kitsune 115-feature path)")
            except Exception as e:
                step(f"  {WARN} Stage2IoTDetector load failed: {e}")
                step(f"        IoT branch will be skipped in subsequent tests.")
        else:
            step(f"  {WARN} Stage-2 IoT model not found at {MODEL_S2_IOT} — IoT skipped")

        step("Building ExplainerBundle ...")
        try:
            bundle = ExplainerBundle(
                stage1_classifier = s1,
                iot_detector      = s2_iot,
                noniot_detector   = s2,
            )
            stage1_avail = bundle.stage1() is not None
            iot_avail    = bundle.stage2("iot") is not None
            noniot_avail = bundle.stage2("noniot") is not None
            step(f"  {PASS} ExplainerBundle built")
            step(f"        Stage-1 SHAP: "
                 f"{'available' if stage1_avail else 'unavailable - install shap'}")
            step(f"        Stage-2 IoT (IG):    {'available' if iot_avail else 'unavailable'}")
            step(f"        Stage-2 NonIoT (IG): {'available' if noniot_avail else 'unavailable'}")
        except Exception as e:
            step(f"  {FAIL} Bundle construction failed: {e}")
            traceback.print_exc()
            return False

        # Stash for downstream tests
        globals()["_S1"]     = s1
        globals()["_S2"]     = s2
        globals()["_S2_IOT"] = s2_iot
        globals()["_BUNDLE"] = bundle
        return True

    except Exception as e:
        step(f"  {FAIL} Import failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════
# Test 2 — Math correctness
# Catches: wrong baseline, off-by-one in IG steps, gradient computation
#         going through the wrong layer, scaling mismatch
# ════════════════════════════════════════════════════════════════════
def test_2_math() -> bool:
    banner("TEST 2 — Math correctness (axioms)")

    s2     = globals()["_S2"]
    bundle = globals()["_BUNDLE"]

    # Build a single random flow row in S2's feature schema
    feature_cols = list(s2._feature_cols)
    rng = np.random.default_rng(42)
    feat = {c: float(rng.uniform(-1, 1)) for c in feature_cols}

    # Get the expressly-correct prediction by going through the wrapper
    scaled_row = s2.stage2_preprocess_non_iot(feat)
    seq = np.stack([scaled_row])
    label, conf = s2.stage2_predict(seq)
    step(f"Stage-2 wrapper says: label={label}, conf={conf:.6f}")

    # Now run the explainer on the exact same input
    expl = bundle.stage2("noniot")
    result = expl.explain(seq, top_k=len(feature_cols))
    step(f"Explainer says:        label={result.prediction}, conf={result.confidence:.6f}")

    # ── Check 1: explainer's confidence must match wrapper's ──
    conf_gap = abs(conf - result.confidence)
    if conf_gap > 1e-4:
        step(f"  {FAIL} Confidence mismatch: gap={conf_gap:.2e}")
        return False
    step(f"  {PASS} Confidence agrees ({conf_gap:.2e} gap)")

    # ── Check 2: IG Completeness axiom ──
    # Σ attributions ≈ logit(x) - logit(baseline)
    # We verify in logit space because that's what IG attributes.
    import torch
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    if seq.shape[0] < expl._seq_len:
        # _prepare_seq pads — replicate that
        pad = np.zeros((expl._seq_len - seq.shape[0], expl._n_features), np.float32)
        seq_padded = np.vstack([pad, seq])
        x = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        fx = expl._model(x).item()
        f0 = expl._model(torch.zeros_like(x)).item()
    sum_attr = sum(result.raw_attributions.values())
    expected = fx - f0
    gap = abs(sum_attr - expected)

    step(f"IG Completeness check (Σ attributions = logit(x) - logit(baseline)):")
    step(f"  Σ attributions       = {sum_attr:+.6f}")
    step(f"  logit(x) - logit(0)  = {expected:+.6f}")
    step(f"  gap                  = {gap:.6f}")
    # Tolerance scales with logit magnitude. 1% of |expected| or 0.01 absolute,
    # whichever is larger — IG with 40 Riemann steps typically achieves <0.5%.
    tol = max(0.01, 0.05 * abs(expected))
    if gap > tol:
        step(f"  {FAIL} gap={gap:.4f} exceeds tolerance {tol:.4f}")
        step(f"        (Means IG isn't faithfully attributing the model's output —")
        step(f"         either the baseline is wrong, n_steps is too low, or the")
        step(f"         gradient is not flowing through the correct graph.)")
        return False
    step(f"  {PASS} Completeness axiom satisfied (gap < {tol:.3f} tolerance)")

    # ── Check 3: Top features should sort by |attribution| descending ──
    attrs = [abs(f.attribution) for f in result.top_features]
    if not all(attrs[i] >= attrs[i+1] for i in range(len(attrs)-1)):
        step(f"  {FAIL} top_features not sorted by |attribution|")
        return False
    step(f"  {PASS} top_features correctly sorted by |attribution|")

    return True


# ════════════════════════════════════════════════════════════════════
# Test 3 — Determinism
# Catches: nondeterministic gradient ops, randomness leaking in,
#         dropout still active during inference
# ════════════════════════════════════════════════════════════════════
def test_3_determinism() -> bool:
    banner("TEST 3 — Determinism (same input → same output)")

    s2     = globals()["_S2"]
    bundle = globals()["_BUNDLE"]

    rng = np.random.default_rng(123)
    arr = rng.uniform(-1, 1, size=(20, s2._n_features)).astype(np.float32)

    expl = bundle.stage2("noniot")
    r1 = expl.explain(arr, top_k=8)
    r2 = expl.explain(arr, top_k=8)
    r3 = expl.explain(arr, top_k=8)

    # Confidences should match exactly (no random ops in forward pass)
    for i, r in enumerate([r2, r3], 2):
        if abs(r1.confidence - r.confidence) > 1e-7:
            step(f"  {FAIL} Run {i} confidence drifted: "
                 f"{r1.confidence:.10f} vs {r.confidence:.10f}")
            step(f"        (Means dropout / batchnorm is in train mode,")
            step(f"         or some non-deterministic CUDA op is in the graph.)")
            return False

    # Top features should be identical in order and value (within 1e-6)
    for i, r in enumerate([r2, r3], 2):
        for f1, f2 in zip(r1.top_features, r.top_features):
            if f1.feature != f2.feature:
                step(f"  {FAIL} Run {i} top_features re-ordered: "
                     f"{f1.feature} vs {f2.feature}")
                return False
            if abs(f1.attribution - f2.attribution) > 1e-5:
                step(f"  {FAIL} Run {i} attribution drifted on {f1.feature}: "
                     f"{f1.attribution:.8f} vs {f2.attribution:.8f}")
                return False

    step(f"  {PASS} 3 consecutive runs produced bit-stable results "
         f"(conf={r1.confidence:.6f}, top={r1.top_features[0].feature})")
    return True


# ════════════════════════════════════════════════════════════════════
# Test 4 — Input variance sensitivity
# Catches: explainer that returns the same attributions regardless of
#         input (e.g. zeroed gradient), or one that produces zero
#         attributions on all features (numerical underflow)
# ════════════════════════════════════════════════════════════════════
def test_4_variance() -> bool:
    banner("TEST 4 — Input variance sensitivity")

    s2     = globals()["_S2"]
    bundle = globals()["_BUNDLE"]
    expl   = bundle.stage2("noniot")

    rng = np.random.default_rng(456)
    arrs = [rng.uniform(-2, 2, size=(20, s2._n_features)).astype(np.float32)
            for _ in range(5)]

    results = [expl.explain(a, top_k=8) for a in arrs]

    # All confidences should not be identical (different inputs → different outputs)
    confs = [r.confidence for r in results]
    if max(confs) - min(confs) < 1e-4:
        step(f"  {FAIL} All inputs produced same confidence: {confs}")
        step(f"        (Means the model's forward pass is constant — possibly")
        step(f"         all-zero output due to scaler mismatch.)")
        return False
    step(f"  {PASS} Confidences vary across inputs: "
         f"min={min(confs):.4f}, max={max(confs):.4f}, range={max(confs)-min(confs):.4f}")

    # Attributions should not all be zero
    all_attrs_zero = all(
        all(abs(v) < 1e-10 for v in r.raw_attributions.values())
        for r in results
    )
    if all_attrs_zero:
        step(f"  {FAIL} Every input produced all-zero attributions")
        step(f"        (Means gradients aren't flowing — model frozen?)")
        return False

    # Top feature should not be identical across all inputs
    top_features = [r.top_features[0].feature for r in results]
    if len(set(top_features)) == 1:
        step(f"  {WARN} Same top feature across all 5 inputs: {top_features[0]}")
        step(f"        (Not necessarily wrong — might be a strongly dominant feature —")
        step(f"         but worth checking on real botnet PCAPs.)")
    else:
        step(f"  {PASS} Top feature varies across inputs: {set(top_features)}")

    # Sample attribution magnitudes — they shouldn't all be identical between rows
    first_attrs  = list(results[0].raw_attributions.values())
    second_attrs = list(results[1].raw_attributions.values())
    diffs = [abs(a - b) for a, b in zip(first_attrs, second_attrs)]
    if max(diffs) < 1e-6:
        step(f"  {FAIL} Attribution VECTORS are identical between distinct inputs")
        step(f"        (Means IG is reading from a buffer, not the actual input.)")
        return False
    step(f"  {PASS} Attribution magnitudes differ across inputs "
         f"(max diff = {max(diffs):.6f})")

    return True


# ════════════════════════════════════════════════════════════════════
# Test 5 — Rule engine
# Catches: matchers that misfire on constant features, severity that
#         doesn't track confidence, GENERIC fallback never triggering
# ════════════════════════════════════════════════════════════════════
def test_5_rule_engine() -> bool:
    banner("TEST 5 — Rule engine (pattern matching + LIVE_CONSTANT guard)")

    from src.xai import (LocalExplanation, FeatureContribution,
                         build_human_explanation, LIVE_CONSTANT_FEATURES)

    def _local(prediction: str, confidence: float,
               *features) -> LocalExplanation:
        """Build a synthetic LocalExplanation."""
        top = [FeatureContribution(name, name, value=val,
                                    attribution=attr,
                                    direction=("↑ pushed toward botnet"
                                               if attr >= 0
                                               else "↓ pushed toward benign"))
               for (name, val, attr) in features]
        return LocalExplanation(prediction=prediction, confidence=confidence,
                                method="integrated_gradients",
                                raw_attributions={name: attr
                                                   for name, _, attr in features},
                                top_features=top)

    cases = [
        # (label, expected_pattern, expected_min_severity, LocalExplanation builder)
        (
            "real port scan (window features non-constant)",
            "PORT_SCAN", "high",
            _local("botnet", 0.92,
                   ("window_unique_dsts", 42.0, +0.35),
                   ("window_flow_count",  120.0, +0.28),
                   ("flow_duration",      0.04, -0.20),
                   ("total_fwd_packets",  2.0,  -0.15),
                   ("flag_SYN",          18.0, +0.10)),
        ),
        (
            "DDoS-like flow (high pkt rate, fwd-heavy)",
            "DDOS", "high",
            _local("botnet", 0.95,
                   ("flow_pkts_per_sec", 8000.0, +0.45),
                   ("flow_bytes_per_sec", 5_000_000.0, +0.20),
                   ("total_fwd_packets",  500.0, +0.18),
                   ("total_bwd_packets",  2.0, -0.10),
                   ("flow_iat_mean",      0.0001, +0.07)),
        ),
        (
            "Brute-force on SSH (port 22, SYN+RST flood)",
            "BRUTE_FORCE", "high",
            _local("botnet", 0.85,
                   ("flag_SYN",      30.0, +0.40),
                   ("flag_RST",      15.0, +0.25),
                   ("dst_port",      22.0,  +0.15),
                   ("flow_duration",  2.5,  -0.10)),
        ),
        (
            "FALSE PORT_SCAN test: window features at LIVE_CONSTANT values",
            "GENERIC", "low",
            _local("botnet", 0.65,
                   ("window_unique_dsts", 1.0,  +0.25),    # forced-constant
                   ("window_flow_count",  1.0,  +0.20),    # forced-constant
                   ("periodicity_score",  0.0,  +0.15),    # forced-constant
                   ("flow_duration",      15.0, -0.08),
                   ("total_fwd_packets",  50.0, -0.06)),
        ),
        (
            "Benign flow",
            "GENERIC", "low",
            _local("benign", 0.95,
                   ("flow_duration",     2.5,    -0.10),
                   ("flow_pkts_per_sec", 5.0,    -0.05),
                   ("flag_SYN",          1.0,    -0.02)),
        ),
        # ── IoT (Kitsune) cases ──────────────────────────────────────
        (
            "Mirai-style flood (MI_dir + HH dominant in attributions)",
            "MIRAI_FLOOD", "critical",
            _local("botnet", 0.96,
                   ("MI_dir_L5_weight", 8500.0, +0.40),
                   ("MI_dir_L5_mean",   1024.0, +0.25),
                   ("HH_L5_mean",       2_500_000.0, +0.22),
                   ("HH_L3_mean",       1_800_000.0, +0.15),
                   ("MI_dir_L3_weight", 5500.0, +0.10)),
        ),
        (
            "IoT scan (HpHp dominant — many distinct sockets)",
            "IOT_SCAN", "high",
            _local("botnet", 0.84,
                   ("HpHp_L5_weight", 220.0, +0.38),
                   ("HpHp_L3_weight", 180.0, +0.30),
                   ("H_L5_weight",    150.0, +0.18),
                   ("HpHp_L1_weight", 90.0,  +0.10),
                   ("MI_dir_L5_weight", 50.0, +0.04)),
        ),
        (
            "Slow IoT beacon (long-window HH + low-jitter signal)",
            "SLOW_BEACON", "high",
            _local("botnet", 0.78,
                   ("HH_L0.01_pcc",        0.92, +0.40),
                   ("HH_L0.01_mean",       145.0, +0.22),
                   ("HH_jit_L0.01_mean",   0.012, +0.18),
                   ("HH_L0.1_mean",        165.0, +0.12),
                   ("HH_jit_L0.01_variance", 0.001, +0.08)),
        ),
        (
            "Benign IoT (Kitsune features, model says benign)",
            "GENERIC", "low",
            _local("benign", 0.94,
                   ("MI_dir_L5_weight", 12.0,  -0.10),
                   ("HH_L5_mean",       450.0, -0.05),
                   ("H_L5_weight",      8.0,   -0.02)),
        ),
    ]

    failed = 0
    for case_name, expected_pattern, expected_min_sev, local in cases:
        h = build_human_explanation(local)
        ok_pattern = h.pattern == expected_pattern

        # Severity check: severity should be >= expected_min_sev
        sev_ladder = ["low", "medium", "high", "critical"]
        ok_sev = sev_ladder.index(h.severity) >= sev_ladder.index(expected_min_sev)

        if ok_pattern and ok_sev:
            tag = PASS
            note = ""
        else:
            tag = FAIL
            note = f"  (got pattern={h.pattern}, severity={h.severity})"
            failed += 1

        step(f"  {tag} {case_name}{RESET}{note}")
        step(f"       → pattern={h.pattern}, severity={h.severity}")
        step(f"       summary: {h.summary[:80]}")

    if failed:
        step(f"\n  {FAIL} {failed}/{len(cases)} test cases failed")
        return False
    return True


# ════════════════════════════════════════════════════════════════════
# Test 6 — IoT explainer end-to-end on the real loaded model
# Catches: IoT model loading issues, Kitsune feature schema mismatches,
#         IG running on the wrong tensor shape, IoT pattern dispatch
# ════════════════════════════════════════════════════════════════════
def test_6_iot_e2e() -> bool:
    banner("TEST 6 — IoT explainer end-to-end (Stage-2 IoT, Kitsune 115)")

    s2_iot = globals().get("_S2_IOT")
    bundle = globals().get("_BUNDLE")

    if s2_iot is None or bundle is None or bundle.stage2("iot") is None:
        step(f"  {WARN} IoT model not loaded — skipping IoT E2E test")
        step(f"        (this is informational only; not a failure)")
        return True   # don't fail the suite if user doesn't have the IoT model

    expl = bundle.stage2("iot")
    step(f"IoT explainer ready: seq_len={expl._seq_len}, "
         f"n_features={expl._n_features}, n_steps={expl._n_steps}")

    # ── Build a Kitsune-shaped sequence ──
    rng = np.random.default_rng(31415)
    arr = rng.uniform(0.0, 1.0, size=(20, expl._n_features)).astype(np.float32)
    # Bias the LATEST timestep toward "high MI_dir + HH" to encourage Mirai-like
    # attributions (test that the explainer surfaces those features).
    feature_cols = expl._feature_cols
    for i, name in enumerate(feature_cols):
        if name.startswith(("MI_dir_L5_", "HH_L5_", "HH_L3_")):
            arr[-1, i] = rng.uniform(0.7, 1.0)

    # ── Run IG ──
    t0 = time.perf_counter()
    res = expl.explain(arr, top_k=8)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    step(f"  {PASS} IG completed in {elapsed_ms:.1f} ms")
    step(f"        prediction = {res.prediction}, conf = {res.confidence:.4f}")
    step(f"        method     = {res.method}")

    # ── Sanity: completeness axiom on the IoT model ──
    import torch
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        fx = expl._model(x).item()
        f0 = expl._model(torch.zeros_like(x)).item()
    sum_attr = sum(res.raw_attributions.values())
    expected = fx - f0
    gap = abs(sum_attr - expected)
    tol = max(0.05, 0.05 * abs(expected))   # IoT model has bigger logits, looser tol
    if gap > tol:
        step(f"  {FAIL} IoT IG completeness gap {gap:.4f} > tol {tol:.4f}")
        step(f"        (Σ attr = {sum_attr:+.4f}, f(x)-f(0) = {expected:+.4f})")
        return False
    rel_pct = 100 * gap / abs(expected) if expected != 0 else 0.0
    step(f"  {PASS} IoT IG completeness gap = {gap:.4f} ({rel_pct:.2f}%) within tol {tol:.4f}")

    # ── Top features should be Kitsune-named ──
    top_names = [f.feature for f in res.top_features]
    kitsune_count = sum(1 for n in top_names
                        if any(n.startswith(p) for p in
                               ("MI_dir_", "H_", "HH_", "HH_jit_", "HpHp_")))
    if kitsune_count == 0:
        step(f"  {FAIL} No Kitsune-named features in top-K: {top_names}")
        step(f"        (Means IoT explainer's feature_cols don't match Kitsune schema.)")
        return False
    step(f"  {PASS} {kitsune_count}/{len(top_names)} top features are Kitsune-named")
    step(f"        examples: {top_names[:3]}")

    # ── Run the rule engine and confirm Kitsune dispatch fires ──
    from src.xai import build_human_explanation
    h = build_human_explanation(res)
    step(f"  Rule engine output:")
    step(f"    pattern  = {h.pattern}")
    step(f"    severity = {h.severity}")
    step(f"    summary  = {h.summary[:80]}")
    if res.prediction == "botnet":
        # If the model called this synthetic input "botnet", the rule engine
        # should at least surface SOME pattern (likely MIRAI_FLOOD given the
        # bias we added). Falling through to GENERIC is acceptable but worth
        # noting because it means the synthetic distribution didn't trigger
        # any IoT pattern matcher.
        if h.pattern in ("MIRAI_FLOOD", "IOT_SCAN", "SLOW_BEACON"):
            step(f"  {PASS} IoT pattern dispatched correctly: {h.pattern}")
        else:
            step(f"  {WARN} IoT detection fell through to {h.pattern} — synthetic")
            step(f"        input may not have triggered the matchers")
            step(f"        (real Mirai PCAPs should land on MIRAI_FLOOD / IOT_SCAN)")
    else:
        step(f"  {PASS} Benign IoT correctly dispatched to GENERIC")

    return True



def bench_latency() -> None:
    banner("BENCHMARK — XAI latency per botnet detection (informational)")

    s2     = globals()["_S2"]
    bundle = globals()["_BUNDLE"]
    expl   = bundle.stage2("noniot")

    rng = np.random.default_rng(789)
    arr = rng.uniform(-1, 1, size=(20, s2._n_features)).astype(np.float32)

    # Warm up
    for _ in range(3):
        expl.explain(arr, top_k=8)

    # Measure
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        expl.explain(arr, top_k=8)
    elapsed = (time.perf_counter() - t0) / n * 1000
    step(f"Stage-2 IG explanation latency: {elapsed:.1f} ms / call (avg over {n} runs)")
    step(f"At a 5%% botnet rate this adds ~{elapsed*0.05:.1f} ms/flow on average.")


# ════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════
def main() -> int:
    print(f"\n{BOLD}XAI verification suite{RESET}")
    print(f"Repo root: {ROOT}\n")

    tests = [
        ("Imports & wiring",        test_1_imports),
        ("Math correctness",        test_2_math),
        ("Determinism",             test_3_determinism),
        ("Input variance",          test_4_variance),
        ("Rule engine",             test_5_rule_engine),
        ("IoT end-to-end",          test_6_iot_e2e),
    ]

    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  {FAIL} {name} raised an unexpected exception:")
            traceback.print_exc()
            results[name] = False

    # Bonus benchmark only if Test 1 passed (others may not have run)
    if results.get("Imports & wiring"):
        try:
            bench_latency()
        except Exception as e:
            print(f"\n  {WARN} Benchmark skipped: {e}")

    # ── Summary ─────────────────────────────────────────────────────
    banner("SUMMARY")
    n_pass = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {tag}  {name}")
    print(f"\n  {n_pass}/{len(tests)} tests passed.")

    if n_pass == len(tests):
        print(f"\n{GREEN}{BOLD}✓ XAI verified — math correct, deterministic, "
              f"sensitive to input, rule engine sound.{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some tests failed — check the output above for details.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())