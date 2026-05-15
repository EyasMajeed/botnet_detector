"""
End-to-end test for the PCAP-upload XAI fix.

Simulates the full pipeline:
  monitoring.DetectionResult (with numpy-typed XAI)
    -> inference_bridge._detection_results_to_dicts
    -> upload result dict
    -> _on_upload_done's field map (replicated here)
    -> DetectionStore.add_upload_batch
    -> DetectionStore.save() round-trip (JSON safety test)
    -> report_generator.generate_pdf_report

Asserts that XAI, alerted, and suspicion_score survive every step and that
the produced PDF contains a non-empty per-flow detail section.
"""
import sys
import json
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
APP  = HERE / "app"
if not APP.is_dir():
    sys.exit(f"Could not find app/ at {APP}.")
sys.path.insert(0, str(APP))

from detection_store    import DetectionStore, DetectionFlow, apply_threshold
from inference_bridge   import (_detection_results_to_dicts,
                                _coerce_xai_to_json_safe,
                                get_xai_status)
from report_generator   import generate_pdf_report


# ── Minimal stand-in for monitoring.DetectionResult so we don't need to
#    import the real monitoring.py (which loads torch + heavy artifacts). ─
@dataclass
class FakeDetectionResult:
    flow_id:         str
    src_ip:          str
    dst_ip:          str
    device_type:     str
    label:           str
    s1_confidence:   float
    s2_confidence:   float
    suspicion_score: float
    latency_ms:      float
    alerted:         bool
    timestamp:       float = field(default_factory=time.time)
    s1_method:       str   = "ml"
    xai:             Optional[dict] = None


# ── A realistic XAI payload, BUT seeded with numpy scalars + an ndarray
#    to verify the JSON-safety coercion. monitoring.py's real XAI module
#    casts most values, but defensive-coding-wise we need to handle stragglers. ─
def make_dirty_xai(pattern: str):
    """An XAI dict with numpy types sprinkled in, like a real IG/SHAP output."""
    return {
        "device_type": "noniot",
        "prediction":  "botnet",
        "confidence":  np.float32(0.97),                    # numpy scalar
        "method":      "integrated_gradients",
        "pattern":     pattern,
        "severity":    "high",
        "summary":     f"Likely {pattern.lower().replace('_',' ')}.",
        "top_features": [
            {"feature": "flow_pkts_per_sec", "display": "Flow pkt rate",
             "value": np.float64(482.0),                    # numpy scalar
             "attribution": np.float32(0.42),
             "direction": "↑ pushed toward botnet"},
            {"feature": "flag_SYN", "display": "SYN flag count",
             "value": np.int64(32),                          # numpy int
             "attribution": np.float32(0.38),
             "direction": "↑ pushed toward botnet"},
        ],
        "feature_importance": {
            "Flow pkt rate":   np.float32(0.42),
            "SYN flag count":  np.float32(0.38),
        },
        "reasons": ["High packet rate to single dst", "SMTP scanning pattern"],
        "recommendations": [
            f"Block source IP — pattern={pattern}.",
            "Inspect destination IP for known C&C IoCs.",
        ],
        # And an ndarray buried in there for good measure
        "_raw_attributions_sample": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Build synthetic DetectionResults like BotnetMonitor would.
# ─────────────────────────────────────────────────────────────────────────
results = [
    FakeDetectionResult(
        flow_id="147.32.84.165:2330<->216.32.181.178:25/6",
        src_ip="147.32.84.165", dst_ip="216.32.181.178",
        device_type="noniot", label="botnet",
        s1_confidence=0.85, s2_confidence=0.997,
        suspicion_score=3.0, latency_ms=42.3, alerted=True,
        xai=make_dirty_xai("SPAM_SMTP"),
    ),
    FakeDetectionResult(
        flow_id="147.32.84.165:33210<->185.220.101.5:23/6",
        src_ip="147.32.84.165", dst_ip="185.220.101.5",
        device_type="noniot", label="botnet",
        s1_confidence=0.82, s2_confidence=0.99,
        suspicion_score=2.5, latency_ms=38.1, alerted=True,
        xai=make_dirty_xai("PORT_SCAN"),
    ),
    FakeDetectionResult(
        flow_id="147.32.84.165:1234<->8.8.8.8:53/17",
        src_ip="147.32.84.165", dst_ip="8.8.8.8",
        device_type="noniot", label="benign",
        s1_confidence=0.88, s2_confidence=0.0002,
        suspicion_score=0.0, latency_ms=12.4, alerted=False,
        xai=None,
    ),
]

# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Convert via the FIXED _detection_results_to_dicts.
#          Verify all the previously-dropped fields are now present.
# ─────────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
dicts = _detection_results_to_dicts(results, t0)
assert len(dicts) == 3

REQUIRED_KEYS = {"row", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "device_type", "label", "confidence", "stage1_conf",
                 "suspicion", "alerted", "timestamp", "s1_method", "xai",
                 "latency_ms"}
for d in dicts:
    missing = REQUIRED_KEYS - set(d.keys())
    assert not missing, f"Missing keys in result dict: {missing}"
print(f"OK All {len(REQUIRED_KEYS)} required keys present on every result dict")

# Botnet flows: xai populated, alerted True, suspicion > 0
bot = [d for d in dicts if d["label"] == "botnet"]
assert len(bot) == 2
for d in bot:
    assert d["xai"] is not None,          f"botnet flow has xai=None: {d}"
    assert d["alerted"] is True,           f"botnet flow has alerted=False: {d}"
    assert d["suspicion"] > 0,             f"botnet flow has suspicion<=0: {d}"
print(f"OK Botnet flows: xai populated, alerted=True, suspicion>0")

# Benign flow: xai is None, alerted False, suspicion 0
ben = [d for d in dicts if d["label"] == "benign"][0]
assert ben["xai"] is None
assert ben["alerted"] is False
assert ben["suspicion"] == 0.0
print(f"OK Benign flow: xai=None, alerted=False, suspicion=0")

# Port parsing from flow_id worked
assert bot[0]["src_port"] == 2330 and bot[0]["dst_port"] == 25
assert bot[0]["protocol"] == "TCP"
assert ben["src_port"] == 1234 and ben["protocol"] == "UDP"
print(f"OK flow_id parsing produces correct ports/protocol")

# ─────────────────────────────────────────────────────────────────────────
# Step 3 — JSON safety: the XAI dicts we built contain numpy types.
#          The coercion in _detection_results_to_dicts must have stripped them.
# ─────────────────────────────────────────────────────────────────────────
for d in bot:
    try:
        json.dumps(d["xai"])
    except TypeError as e:
        raise AssertionError(
            f"XAI dict not JSON-serialisable after coercion: {e}"
        ) from e
print(f"OK XAI dicts JSON-serialisable (numpy types coerced to primitives)")

# Spot-check that the problematic numpy types (float32, int64) are gone.
# np.float64 is a Python float subclass and JSON handles it natively, so it's
# fine if those pass through unchanged — they don't break anything.
def _has_bad_numpy_type(value):
    """Returns True if value is a numpy scalar that json.dumps can't handle."""
    import numpy as _np
    return isinstance(value, (_np.floating, _np.integer)) and not isinstance(value, (float, int))

for d in bot:
    for tf in d["xai"]["top_features"]:
        assert not _has_bad_numpy_type(tf["value"]), \
            f"top_features[value] still a problematic numpy scalar: {type(tf['value'])}"
        assert not _has_bad_numpy_type(tf["attribution"]), \
            f"top_features[attribution] still a problematic numpy scalar: {type(tf['attribution'])}"
    for v in d["xai"]["feature_importance"].values():
        assert not _has_bad_numpy_type(v), \
            f"feature_importance value still a problematic numpy scalar: {type(v)}"
sample_attr = bot[0]["xai"]["top_features"][0]["attribution"]
sample_conf = bot[0]["xai"]["confidence"]
print(f"OK Problematic numpy scalars (float32 / int64) coerced "
      f"(sample attr={sample_attr}, conf={sample_conf:.3f})")

# Coerce-on-None returns None
assert _coerce_xai_to_json_safe(None) is None
print(f"OK _coerce_xai_to_json_safe(None) returns None")

# ─────────────────────────────────────────────────────────────────────────
# Step 4 — Replicate _on_upload_done's field map and push to a DetectionStore.
#          The store JSON-dumps on save; if anything is still numpy, this crashes.
# ─────────────────────────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
store = DetectionStore(tmp / "store.json")

flows = []
for r in dicts:
    flows.append(DetectionFlow(
        src_ip        = str(r.get("src_ip", "") or ""),
        dst_ip        = str(r.get("dst_ip", "") or ""),
        src_port      = int(r.get("src_port", 0) or 0),
        dst_port      = int(r.get("dst_port", 0) or 0),
        protocol      = str(r.get("protocol", "—")),
        label         = apply_threshold(r["label"], r["confidence"], 0.5),
        confidence    = float(r.get("confidence", 0.0)),
        device_type   = str(r.get("device_type", "noniot")),
        s1_confidence = float(r.get("stage1_conf", 0.0)),
        suspicion     = float(r.get("suspicion", 0.0)),
        latency_ms    = float(r.get("latency_ms", 0.0)),
        alerted       = bool(r.get("alerted", False)),
        timestamp     = float(r.get("timestamp", 0.0)) or time.time(),
        xai           = r.get("xai"),
    ))

rid = store.add_upload_batch(flows, "botnet-capture-20110816-donbot.pcap")
print(f"OK add_upload_batch succeeded: {rid}")

# Verify the persisted JSON round-trips
store_round_trip = DetectionStore(tmp / "store.json")
persisted_flows = store_round_trip.flows_for_report(rid)
assert len(persisted_flows) == 3
# XAI must survive the JSON round-trip
bot_persisted = [f for f in persisted_flows if f.label == "botnet"]
assert len(bot_persisted) == 2
for f in bot_persisted:
    assert f.xai is not None, "XAI was dropped during persistence!"
    assert f.alerted is True
    assert f.suspicion > 0
print(f"OK Persisted JSON round-trips with XAI + alerted + suspicion intact")

# ─────────────────────────────────────────────────────────────────────────
# Step 5 — Generate a PDF. It should contain the per-flow XAI section.
# ─────────────────────────────────────────────────────────────────────────
out_pdf = tmp / "report.pdf"
generate_pdf_report(persisted_flows, out_path=out_pdf, report_meta={
    "report_id":    rid,
    "source":       "upload",
    "filename":     "botnet-capture-20110816-donbot.pcap",
    "duration_sec": 0,
})
assert out_pdf.exists()
size = out_pdf.stat().st_size
# A PDF with 2 fully-detailed botnet flows + cover + summary should be > 5KB.
# An empty / minimal PDF is around 2-3KB. We use 5KB as the floor.
assert size > 5000, f"PDF suspiciously small ({size:,} bytes) — XAI section probably missing"
print(f"OK PDF generated: {size:,} bytes  ({out_pdf})")

# Check the PDF content actually mentions the patterns from our XAI.
# reportlab compresses content streams, so plain binary search on the PDF
# bytes doesn't work. Use pdftotext if it's installed (it ships with
# poppler-utils on most macOS/Linux setups); skip the assertion otherwise
# and rely on the file-size + round-trip checks above to prove XAI is rendered.
import shutil, subprocess
if shutil.which("pdftotext"):
    try:
        text = subprocess.check_output(["pdftotext", str(out_pdf), "-"],
                                       stderr=subprocess.DEVNULL).decode("utf-8", "ignore")
        patterns_found = [
            tok for tok in ("SPAM_SMTP", "PORT_SCAN", "Block source IP",
                            "Flow pkt rate", "SYN flag count")
            if tok in text
        ]
        assert len(patterns_found) >= 3, \
            f"PDF text missing XAI tokens. Found only: {patterns_found}"
        print(f"OK PDF text contains XAI tokens: {patterns_found}")
    except subprocess.CalledProcessError as e:
        print(f"WARN pdftotext failed ({e}); skipping text-content check")
else:
    print("WARN pdftotext not installed; skipping text-content check "
          "(install poppler-utils to enable)")

# Print the diagnostic
print(f"OK get_xai_status() = {get_xai_status()}")

print()
print("==================================================")
print("All assertions passed.")
print(f"Test PDF: {out_pdf}")
print("==================================================")
