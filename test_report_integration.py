"""
Integration test for the fixed report-generation flow.

Run from the repo root:
    python3 test_report_integration.py   (macOS)
    python  test_report_integration.py   (Windows)

The script resolves app/ relative to itself, so it works no matter where
you launch it from as long as it lives next to the app/ folder.
"""
import sys
import time
import tempfile
from pathlib import Path

# Make app/ importable, regardless of where the script is run from.
HERE = Path(__file__).resolve().parent
APP  = HERE / "app"
if not APP.is_dir():
    sys.exit(f"Could not find app/ at {APP}. "
             f"Place this script in the repo root (next to the app/ folder).")
sys.path.insert(0, str(APP))

from detection_store import DetectionStore, DetectionFlow
from report_generator import generate_pdf_report


# ── Build a store with two upload reports + one live session ─────────────
tmp = Path(tempfile.mkdtemp())
store = DetectionStore(tmp / "store.json")

def make_xai(pattern: str, severity: str):
    return {
        "device_type": "noniot",
        "prediction":  "botnet",
        "confidence":  0.91,
        "method":      "integrated_gradients",
        "pattern":     pattern,
        "severity":    severity,
        "summary":     f"Likely {pattern.replace('_',' ').lower()} behaviour.",
        "top_features": [
            {"feature": "flow_pkts_per_sec", "display": "Flow pkt rate",
             "value": 482.0, "attribution": 0.42, "direction": ""},
            {"feature": "flag_SYN",          "display": "SYN flag count",
             "value": 32.0,  "attribution": 0.38, "direction": ""},
        ],
        "feature_importance": {"Flow pkt rate": 0.42, "SYN flag count": 0.38},
        "recommendations": [f"Block source IP — pattern={pattern}."],
        "reasons": [],
    }

upload_a = [
    DetectionFlow(src_ip="10.0.0.5", dst_ip="91.108.4.15", src_port=51234,
                  dst_port=4444, protocol="TCP", label="botnet",
                  confidence=0.94, device_type="noniot",
                  xai=make_xai("PORT_SCAN", "high")),
    DetectionFlow(src_ip="10.0.0.5", dst_ip="91.108.4.15", src_port=51235,
                  dst_port=4444, protocol="TCP", label="botnet",
                  confidence=0.89, device_type="noniot",
                  xai=make_xai("PORT_SCAN", "high")),
    DetectionFlow(src_ip="10.0.0.10", dst_ip="8.8.8.8", src_port=54321,
                  dst_port=53, protocol="UDP", label="benign",
                  confidence=0.97, device_type="noniot", xai=None),
]
rid_a = store.add_upload_batch(upload_a, "capture_A.pcap")

upload_b = [
    DetectionFlow(src_ip="192.168.50.30", dst_ip="185.220.101.5",
                  src_port=33210, dst_port=23, protocol="TCP", label="botnet",
                  confidence=0.88, device_type="iot",
                  xai=make_xai("MIRAI_FLOOD", "critical")),
    DetectionFlow(src_ip="192.168.50.30", dst_ip="1.1.1.1",
                  src_port=33211, dst_port=53, protocol="UDP", label="benign",
                  confidence=0.91, device_type="iot", xai=None),
]
rid_b = store.add_upload_batch(upload_b, "iot_capture.pcap")

rid_live = store.start_live_session()
time.sleep(0.25)
store.add_live_flow(DetectionFlow(
    src_ip="10.0.0.99", dst_ip="203.0.113.5", src_port=12345, dst_port=80,
    protocol="TCP", label="botnet", confidence=0.97,
    device_type="noniot", xai=make_xai("DDOS", "critical")))
store.add_live_flow(DetectionFlow(
    src_ip="10.0.0.99", dst_ip="8.8.4.4", src_port=22222, dst_port=53,
    protocol="UDP", label="benign", confidence=0.92, device_type="noniot"))

# ── Assertions ───────────────────────────────────────────────────────────
returned_rid = store.end_live_session()
assert returned_rid == rid_live, \
    f"end_live_session should return the closed rid, got {returned_rid!r}"
print(f"OK end_live_session() returned {returned_rid}")

again = store.end_live_session()
assert again is None
print(f"OK end_live_session() idempotent (second call -> None)")

live_report = next(r for r in store.reports if r.report_id == rid_live)
assert live_report.duration_sec > 0
print(f"OK live report duration_sec = {live_report.duration_sec}s")

fa = store.flows_for_report(rid_a)
fb = store.flows_for_report(rid_b)
fl = store.flows_for_report(rid_live)
assert len(fa) == 3 and all(f.report_id == rid_a for f in fa)
assert len(fb) == 2 and all(f.report_id == rid_b for f in fb)
assert len(fl) == 2 and all(f.report_id == rid_live for f in fl)
print(f"OK flows_for_report isolates: {rid_a}=3, {rid_b}=2, {rid_live}=2")

out_dir = Path(tempfile.mkdtemp())

pdf_a = out_dir / f"{rid_a}.pdf"
generate_pdf_report(fa, out_path=pdf_a, report_meta={
    "report_id": rid_a, "source": "upload",
    "filename":  "capture_A.pcap", "duration_sec": 0,
})
assert pdf_a.exists() and pdf_a.stat().st_size > 2000
print(f"OK Upload-report PDF ({rid_a}): {pdf_a.stat().st_size:,} bytes")

pdf_l = out_dir / f"{rid_live}.pdf"
generate_pdf_report(fl, out_path=pdf_l, report_meta={
    "report_id": rid_live, "source": "live",
    "filename":  "<live capture>", "duration_sec": live_report.duration_sec,
})
assert pdf_l.exists() and pdf_l.stat().st_size > 2000
print(f"OK Live-report PDF ({rid_live}): {pdf_l.stat().st_size:,} bytes")

pdf_all = out_dir / "all.pdf"
all_flows = list(store.flows)
n_reports = len({f.report_id for f in all_flows if f.report_id})
generate_pdf_report(all_flows, out_path=pdf_all, report_meta={
    "report_id": "ALL",
    "source":    f"mixed ({n_reports} reports)",
    "filename":  "All flows in store",
})
assert pdf_all.exists() and pdf_all.stat().st_size > 2000
print(f"OK All-reports combined PDF: {pdf_all.stat().st_size:,} bytes "
      f"(n_reports={n_reports})")

empty = store.flows_for_report("RPT-DOES-NOT-EXIST")
assert empty == []
print(f"OK Missing rid -> empty flow list (no exception)")

rid_live_2 = store.start_live_session()
assert rid_live_2 != rid_live
print(f"OK Stop->Start creates new report: {rid_live} -> {rid_live_2}")
store.end_live_session()

print()
print("==================================================")
print("All assertions passed.")
print(f"PDFs written to {out_dir}")
for p in sorted(out_dir.glob("*.pdf")):
    print(f"   - {p.name}  ({p.stat().st_size:,} bytes)")
print("==================================================")