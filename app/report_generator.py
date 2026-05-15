"""
═══════════════════════════════════════════════════════════════════════
 report_generator.py — PDF security report generator
 Group 07 | CPCS499
═══════════════════════════════════════════════════════════════════════

Produces a multi-page PDF report from a list of DetectionFlow objects.
The output is structured like a real SOC analyst would expect:

  Page 1   — Cover / executive summary
                · Report ID, generation time, source (live/upload/mixed)
                · Total flows, botnet count, IoT vs Non-IoT split, avg conf
                · "Top 5 most-attributed features across botnet flows"
                  table — aggregated XAI insight, not per-flow

  Page 2   — Pattern breakdown
                · How many flows fell into each pattern bucket
                  (PORT_SCAN, DDOS, MIRAI_FLOOD, ..., GENERIC, no-XAI)
                · Severity distribution (low/medium/high/critical)

  Page 3+  — Per-flow detail (botnet flows only, paginated)
                · For each flow:
                    src→dst, port, protocol, device type, confidence
                    pattern + severity tag (coloured)
                    one-sentence summary
                    top-5 features as a small attribution bar table
                    top recommendation

Design constraints
──────────────────
  · No matplotlib — keeps the dependency surface to just `reportlab`.
    All "charts" are reportlab Tables with coloured cells, which look
    fine in PDF and are perfectly clear.
  · Standalone — no Qt imports. Importable for unit testing.
  · Graceful degradation — if reportlab isn't installed, raise a clear
    ImportError telling the user how to install it.
  · If a flow has no XAI (older flows, benign, or XAI was disabled),
    we still render the flow row but skip the XAI details. The report
    never crashes on missing data.

Public API
──────────
    generate_pdf_report(flows, out_path, *, report_meta=None) -> Path
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ── reportlab — fail loudly if not installed ─────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    _RL_OK = True
except ImportError:
    _RL_OK = False


# ════════════════════════════════════════════════════════════════════
# Color palette — matches the Qt theme so the PDF feels like the GUI
# ════════════════════════════════════════════════════════════════════
# Tuned to read well on white paper (some of the GUI's neon colours
# look bad on white, so we adjust where needed).

_BG_HDR     = "#1A2233"     # dark navy header
_BG_SUBHDR  = "#2A3A5A"     # row separator
_BG_PANEL   = "#F4F6FA"     # light grey panel
_BG_ROW_ALT = "#FAFBFD"     # zebra rows

_TXT_PRI    = "#1A2233"     # main text
_TXT_SEC    = "#5C6680"     # secondary
_TXT_LIGHT  = "#FFFFFF"     # text on dark headers
_TXT_LINK   = "#1F6FEB"     # accent

_OK         = "#16A34A"     # green — benign, low severity
_WARN       = "#D97706"     # amber — medium severity
_ERR        = "#DC2626"     # red — high / critical / botnet
_INFO       = "#2563EB"     # blue — IoT branch

# Severity → colour
_SEV_COLOR = {
    "critical": _ERR,
    "high":     _ERR,
    "medium":   _WARN,
    "low":      _OK,
    "":         _TXT_SEC,
}

# Pattern → friendly display name
_PATTERN_NAMES = {
    "PORT_SCAN":    "Port scan",
    "DDOS":         "DDoS / flood",
    "C2_BEACON":    "C2 beacon",
    "DNS_TUNNEL":   "DNS tunnel",
    "BRUTE_FORCE":  "Brute force",
    "MIRAI_FLOOD":  "Mirai-style flood",
    "IOT_SCAN":     "IoT scan / propagation",
    "SLOW_BEACON":  "Slow IoT beacon",
    "GENERIC":      "Anomalous (generic)",
    "UNKNOWN":      "Unknown",
}


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════

def generate_pdf_report(flows:        Sequence[Any],
                        out_path:     Path | str,
                        *,
                        report_meta:  Optional[Dict[str, Any]] = None,
                        max_botnet_detail_rows: int = 50) -> Path:
    """
    Generate a security-report PDF from a sequence of DetectionFlow objects.

    Parameters
    ----------
    flows : Sequence[DetectionFlow]
        The same flows the CSV export uses. Order is preserved for
        consistency.
    out_path : Path | str
        Where to write the PDF. Parent directory will be created if missing.
    report_meta : dict, optional
        Optional extra metadata to surface on the cover page:
          {"report_id": str, "source": str, "filename": str,
           "duration_sec": float, "generated_by": str}
        Any keys we don't recognise are quietly ignored.
    max_botnet_detail_rows : int
        Cap on how many botnet flows get the full per-flow detail
        treatment. Beyond this we render a "...and N more" footer.
        Prevents the PDF from blowing up on a 10,000-flow capture.

    Returns
    -------
    Path of the generated PDF.

    Raises
    ------
    ImportError if reportlab isn't installed.
    """
    if not _RL_OK:
        raise ImportError(
            "PDF export requires reportlab.\n"
            "Install with:\n"
            "    pip install reportlab\n"
            "(macOS users: pip3 install reportlab)"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build the document skeleton ──────────────────────────────────
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm,  bottomMargin=18*mm,
        title="Botnet Detection Report",
        author="AI-Based Botnet Detection (Group 07, CPCS499)",
    )
    styles = _make_styles()

    story: list = []

    # ── Section 1: Cover / executive summary ─────────────────────────
    story.extend(_build_cover(flows, report_meta or {}, styles))
    story.append(PageBreak())

    # ── Section 2: Pattern + severity breakdown ──────────────────────
    story.extend(_build_pattern_breakdown(flows, styles))
    story.append(PageBreak())

    # ── Section 3: Per-flow detail (botnet only) ─────────────────────
    story.extend(_build_flow_details(flows, styles, cap=max_botnet_detail_rows))

    # ── Build with footer hook ───────────────────────────────────────
    doc.build(story, onFirstPage=_footer_renderer, onLaterPages=_footer_renderer)
    return out_path


# ════════════════════════════════════════════════════════════════════
# Section builders
# ════════════════════════════════════════════════════════════════════

def _build_cover(flows: Sequence[Any], meta: Dict[str, Any],
                 styles: Dict[str, ParagraphStyle]) -> list:
    """Page 1 — title, key stats, top-5 globally-attributed features."""
    parts: list = []

    # Title
    parts.append(Paragraph("Botnet Detection Report", styles["Title"]))
    parts.append(Spacer(1, 4*mm))
    parts.append(Paragraph(
        "AI-Based Botnet Detection · Two-Stage Hybrid CNN-LSTM · Group 07",
        styles["Subtitle"]))
    parts.append(Spacer(1, 8*mm))

    # Metadata block
    meta_rows = [
        ["Generated",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Report ID",     str(meta.get("report_id", "—"))],
        ["Source",        str(meta.get("source", _infer_source(flows)))],
    ]
    if meta.get("filename"):
        meta_rows.append(["Filename", str(meta["filename"])])
    if meta.get("duration_sec"):
        meta_rows.append(["Duration",
                          f"{float(meta['duration_sec']):.1f} seconds"])
    meta_rows.append(["Total flows analysed", str(len(flows))])

    meta_tbl = Table(meta_rows, colWidths=[55*mm, 110*mm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor(_BG_PANEL)),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.HexColor(_TXT_PRI)),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("LINEBELOW",   (0, 0), (-1, -2), 0.4, colors.HexColor("#E5E7EB")),
    ]))
    parts.append(meta_tbl)
    parts.append(Spacer(1, 8*mm))

    # ── Stat cards row ──────────────────────────────────────────────
    parts.append(Paragraph("Detection summary", styles["H2"]))
    parts.append(Spacer(1, 3*mm))

    stats = _compute_stats(flows)
    stat_data = [
        [
            _stat_cell("Botnet flows", str(stats["n_botnet"]),
                       sub=f"of {stats['n_total']} total", colour=_ERR),
            _stat_cell("IoT flows", str(stats["n_iot"]),
                       sub=f"({stats['n_iot_botnet']} botnet)", colour=_INFO),
            _stat_cell("Non-IoT flows", str(stats["n_noniot"]),
                       sub=f"({stats['n_noniot_botnet']} botnet)", colour=_INFO),
            _stat_cell("Avg confidence",
                       f"{stats['avg_botnet_conf']*100:.1f}%",
                       sub="across botnet flows" if stats["n_botnet"] else "—",
                       colour=_OK),
        ]
    ]
    stat_tbl = Table(stat_data, colWidths=[42*mm]*4)
    stat_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    parts.append(stat_tbl)
    parts.append(Spacer(1, 8*mm))

    # ── Top-attributed features across botnet flows (XAI insight) ──
    parts.append(Paragraph(
        "Top 5 most-attributed features across botnet detections",
        styles["H2"]))
    parts.append(Paragraph(
        "Aggregated from per-flow Integrated Gradients (Stage-2) and "
        "SHAP (Stage-1) attributions. Higher value = feature contributed "
        "more strongly to the &quot;botnet&quot; verdict, summed across all "
        "explained botnet flows.",
        styles["Caption"]))
    parts.append(Spacer(1, 3*mm))

    top_features = _aggregate_top_features(flows, top_k=5)
    if top_features:
        # Build a table: name | proportional-width bar | numeric value
        max_v = max((v for _, v in top_features), default=1.0) or 1.0
        rows = [["Feature", "Aggregated importance", ""]]
        for name, v in top_features:
            rows.append([name, _bar_cell(v / max_v, total_width_mm=70,
                                           colour=_INFO), f"{v:.3f}"])

        feat_tbl = Table(rows, colWidths=[55*mm, 80*mm, 30*mm])
        feat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_BG_HDR)),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor(_TXT_LIGHT)),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ALIGN",      (1, 1), (1, -1), "LEFT"),
            ("ALIGN",      (2, 1), (2, -1), "RIGHT"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME",   (1, 1), (1, -1), "Courier"),     # mono for the bar
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("LINEBELOW",  (0, 0), (-1, -2), 0.3, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                [colors.white, colors.HexColor(_BG_ROW_ALT)]),
        ]))
        parts.append(feat_tbl)
    else:
        parts.append(Paragraph(
            "<i>No XAI attributions available — either no botnet flows "
            "were detected, or XAI was disabled at capture time.</i>",
            styles["Body"]))

    return parts


def _build_pattern_breakdown(flows: Sequence[Any],
                              styles: Dict[str, ParagraphStyle]) -> list:
    """Page 2 — pattern + severity counts."""
    parts: list = []
    parts.append(Paragraph("Behaviour pattern breakdown", styles["H1"]))
    parts.append(Paragraph(
        "Each detected botnet flow is matched against the rule engine's "
        "pattern library. Patterns are derived from the model's top "
        "feature attributions, not from raw rules — this is XAI-driven "
        "classification, not signature matching.",
        styles["Caption"]))
    parts.append(Spacer(1, 5*mm))

    pat_counts = _count_by_field(flows, "pattern")
    sev_counts = _count_by_field(flows, "severity")

    if not pat_counts:
        parts.append(Paragraph(
            "<i>No XAI explanations available for this report. "
            "Pattern breakdown is empty.</i>", styles["Body"]))
        return parts

    # ── Pattern table ───────────────────────────────────────────────
    parts.append(Paragraph("By pattern", styles["H2"]))
    parts.append(Spacer(1, 2*mm))

    total_explained = sum(pat_counts.values())
    rows = [["Pattern", "Count", "Share", "", "Severity bias"]]
    # Order by count desc, but put GENERIC last so the named patterns
    # are emphasised first.
    items = sorted(pat_counts.items(),
                   key=lambda kv: (kv[0] in ("GENERIC", "UNKNOWN"), -kv[1]))
    max_count = max(pat_counts.values())
    for pat, count in items:
        share_pct = 100 * count / total_explained if total_explained else 0
        # Severity for this pattern (use first flow's severity as a hint
        # — patterns map deterministically so this is reliable).
        sev_for_pat = next(
            (_get_xai(f).get("severity", "—") for f in flows
             if _get_xai(f).get("pattern") == pat),
            "—",
        )
        rows.append([
            _PATTERN_NAMES.get(pat, pat),
            str(count),
            f"{share_pct:.1f}%",
            _bar_cell(count / max_count, total_width_mm=45, colour=_INFO),
            sev_for_pat,
        ])

    pat_tbl = Table(rows, colWidths=[55*mm, 18*mm, 18*mm, 50*mm, 24*mm])
    pat_tbl.setStyle(_section_table_style())
    parts.append(pat_tbl)
    parts.append(Spacer(1, 8*mm))

    # ── Severity table with coloured cells ──────────────────────────
    parts.append(Paragraph("By severity", styles["H2"]))
    parts.append(Spacer(1, 2*mm))

    sev_order = ["critical", "high", "medium", "low"]
    sev_rows = [["Severity", "Count", "Share"]]
    total_sev = sum(sev_counts.values())
    for sev in sev_order:
        if sev not in sev_counts:
            continue
        n = sev_counts[sev]
        share = 100 * n / total_sev if total_sev else 0
        sev_rows.append([sev.capitalize(), str(n), f"{share:.1f}%"])

    sev_tbl = Table(sev_rows, colWidths=[55*mm, 25*mm, 25*mm])
    sev_style = _section_table_style()
    # Colour the severity cells by their actual severity
    for i, sev in enumerate([s for s in sev_order if s in sev_counts], start=1):
        sev_style.add("TEXTCOLOR", (0, i), (0, i),
                      colors.HexColor(_SEV_COLOR.get(sev, _TXT_PRI)))
        sev_style.add("FONTNAME", (0, i), (0, i), "Helvetica-Bold")
    sev_tbl.setStyle(sev_style)
    parts.append(sev_tbl)

    return parts


def _build_flow_details(flows: Sequence[Any],
                         styles: Dict[str, ParagraphStyle],
                         cap: int) -> list:
    """Pages 3+ — per-flow detail for botnet flows only."""
    parts: list = []
    parts.append(Paragraph("Per-flow detection detail", styles["H1"]))
    parts.append(Paragraph(
        "Each entry below is one botnet detection with its model-driven "
        "explanation. Top features are sorted by absolute attribution magnitude "
        "(↑ = pushed the verdict toward botnet, ↓ = pushed toward benign).",
        styles["Caption"]))
    parts.append(Spacer(1, 4*mm))

    botnet_flows = [f for f in flows if getattr(f, "label", "") == "botnet"]
    if not botnet_flows:
        parts.append(Paragraph(
            "<i>No botnet flows were detected in this report.</i>",
            styles["Body"]))
        return parts

    # Cap the number of detail blocks — a 10,000-flow PDF helps no-one.
    overflow = max(0, len(botnet_flows) - cap)
    for i, flow in enumerate(botnet_flows[:cap], 1):
        parts.append(KeepTogether(_render_one_flow_detail(i, flow, styles)))
        parts.append(Spacer(1, 4*mm))

    if overflow:
        parts.append(Spacer(1, 4*mm))
        parts.append(Paragraph(
            f"<i>… and {overflow} more botnet flow(s) not shown for brevity. "
            f"All flows are included in the CSV export.</i>",
            styles["Caption"]))

    return parts


def _render_one_flow_detail(idx: int, flow: Any,
                             styles: Dict[str, ParagraphStyle]) -> list:
    """Render a single flow's detail block."""
    parts: list = []
    xai = _get_xai(flow)

    # ── Header: flow ID + src/dst + label tag ────────────────────────
    src     = getattr(flow, "src_ip",  "—") or "—"
    dst     = getattr(flow, "dst_ip",  "—") or "—"
    sport   = getattr(flow, "src_port", 0) or 0
    dport   = getattr(flow, "dst_port", 0) or 0
    proto   = getattr(flow, "protocol", "—") or "—"
    dev     = getattr(flow, "device_type", "—") or "—"
    conf    = float(getattr(flow, "confidence", 0.0) or 0.0)

    pattern  = xai.get("pattern", "UNKNOWN") if xai else "UNKNOWN"
    severity = xai.get("severity", "low")    if xai else "low"
    summary  = xai.get("summary",  "")       if xai else ""

    title_html = (
        f"<b>#{idx}</b> &nbsp; "
        f"<font face='Courier'>{src}:{sport} → {dst}:{dport}</font> "
        f"&nbsp; <font color='{_TXT_SEC}'>{proto} · {dev}</font>"
    )
    parts.append(Paragraph(title_html, styles["H3"]))

    # Tag row — label / pattern / severity / confidence
    tag_row = [[
        _color_chip("BOTNET", _ERR),
        _color_chip(_PATTERN_NAMES.get(pattern, pattern), _INFO),
        _color_chip(severity.upper(), _SEV_COLOR.get(severity, _TXT_SEC)),
        Paragraph(f"<b>conf:</b> {conf:.3f}", styles["Tag"]),
    ]]
    tag_tbl = Table(tag_row, colWidths=[28*mm, 50*mm, 24*mm, 28*mm])
    tag_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    parts.append(tag_tbl)
    parts.append(Spacer(1, 2*mm))

    # ── Summary text ────────────────────────────────────────────────
    if summary:
        parts.append(Paragraph(_html_safe(summary), styles["Body"]))
        parts.append(Spacer(1, 2*mm))

    # ── Top features bar table ──────────────────────────────────────
    if xai and xai.get("top_features"):
        feats = xai["top_features"][:5]
        # Use feature_importance for the bar magnitude (signed magnitudes
        # are in top_features individually).
        max_v = max((abs(f.get("attribution", 0.0)) for f in feats), default=1.0) or 1.0

        # Header row uses white-on-dark Paragraphs so the labels are
        # visible against the dark header background. Body rows use
        # the regular dark-text Tiny style.
        hdr_style = ParagraphStyle("hdr", fontSize=8, leading=10,
                                    fontName="Helvetica-Bold",
                                    textColor=colors.HexColor(_TXT_LIGHT))
        rows = [[
            Paragraph("Feature",     hdr_style),
            Paragraph("Value",       hdr_style),
            Paragraph("Direction",   hdr_style),
            Paragraph("Attribution", hdr_style),
            "",
        ]]
        for f in feats:
            disp  = f.get("display") or f.get("feature", "?")
            val   = f.get("value", 0.0)
            attr  = f.get("attribution", 0.0)
            arrow = "↑ botnet" if attr >= 0 else "↓ benign"
            arrow_color = _ERR if attr >= 0 else _OK
            bar_color   = _ERR if attr >= 0 else _OK
            rows.append([
                Paragraph(_html_safe(disp), styles["Tiny"]),
                Paragraph(_html_safe(_format_value(val)), styles["Tiny"]),
                Paragraph(f"<font color='{arrow_color}'>{arrow}</font>",
                          styles["Tiny"]),
                _bar_cell(abs(attr) / max_v, total_width_mm=30, colour=bar_color),
                Paragraph(f"{attr:+.4f}", styles["Tiny"]),
            ])

        feat_tbl = Table(rows,
                         colWidths=[55*mm, 22*mm, 22*mm, 35*mm, 22*mm])
        feat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_BG_HDR)),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (1, 1), (-1, -1), "LEFT"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LINEBELOW",  (0, 0), (-1, -2), 0.3, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                [colors.white, colors.HexColor(_BG_ROW_ALT)]),
        ]))
        parts.append(feat_tbl)
        parts.append(Spacer(1, 2*mm))

    # ── Top recommendation ──────────────────────────────────────────
    recs = xai.get("recommendations", []) if xai else []
    if recs:
        parts.append(Paragraph(
            f"<b>Recommended action:</b> {_html_safe(recs[0])}",
            styles["Recommend"]))

    return parts


# ════════════════════════════════════════════════════════════════════
# Helpers — stats, formatting
# ════════════════════════════════════════════════════════════════════

def _get_xai(flow: Any) -> Dict[str, Any]:
    """Safe accessor — returns the xai dict or an empty dict."""
    x = getattr(flow, "xai", None)
    if isinstance(x, dict):
        return x
    return {}


def _infer_source(flows: Sequence[Any]) -> str:
    sources = {getattr(f, "source", "") for f in flows}
    if not sources:
        return "—"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _compute_stats(flows: Sequence[Any]) -> Dict[str, Any]:
    """Top-line counters and averages for the cover page."""
    n_total      = len(flows)
    n_botnet     = sum(1 for f in flows if getattr(f, "label", "") == "botnet")
    n_iot        = sum(1 for f in flows if getattr(f, "device_type", "") == "iot")
    n_noniot     = sum(1 for f in flows if getattr(f, "device_type", "") == "noniot")
    n_iot_botnet = sum(1 for f in flows
                       if getattr(f, "label", "") == "botnet"
                       and getattr(f, "device_type", "") == "iot")
    n_noniot_botnet = sum(1 for f in flows
                          if getattr(f, "label", "") == "botnet"
                          and getattr(f, "device_type", "") == "noniot")
    avg_botnet_conf = (
        sum(float(getattr(f, "confidence", 0.0) or 0.0)
            for f in flows if getattr(f, "label", "") == "botnet")
        / max(n_botnet, 1)
    )
    return {
        "n_total":         n_total,
        "n_botnet":        n_botnet,
        "n_iot":           n_iot,
        "n_noniot":        n_noniot,
        "n_iot_botnet":    n_iot_botnet,
        "n_noniot_botnet": n_noniot_botnet,
        "avg_botnet_conf": avg_botnet_conf,
    }


def _count_by_field(flows: Sequence[Any], field_name: str) -> Dict[str, int]:
    """Count XAI dict[field_name] across all flows that have an XAI dict."""
    counts: Dict[str, int] = {}
    for f in flows:
        x = _get_xai(f)
        if not x:
            continue
        v = x.get(field_name)
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
    return counts


def _aggregate_top_features(flows: Sequence[Any], top_k: int = 5) -> list:
    """
    Aggregate feature_importance dicts across all botnet flows.
    Returns a list of (display_name, summed_importance) tuples, sorted
    descending. This is the dataset-level XAI insight: "across all the
    botnet detections in this report, which features mattered most?"
    """
    bucket: Dict[str, float] = {}
    for f in flows:
        if getattr(f, "label", "") != "botnet":
            continue
        x = _get_xai(f)
        if not x:
            continue
        # Prefer feature_importance (already-magnitudes dict). Fall back
        # to summing |attribution| from top_features.
        fi = x.get("feature_importance")
        if isinstance(fi, dict):
            for k, v in fi.items():
                bucket[k] = bucket.get(k, 0.0) + float(v)
            continue
        for tf in x.get("top_features", []):
            disp = tf.get("display") or tf.get("feature", "?")
            bucket[disp] = bucket.get(disp, 0.0) + abs(float(tf.get("attribution", 0)))

    items = sorted(bucket.items(), key=lambda kv: -kv[1])
    return items[:top_k]


def _bar_string(fraction: float, width: int = 20) -> str:
    """Unicode block bar for monospace columns. Range [0,1]."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _bar_cell(fraction: float, total_width_mm: float = 60,
              colour: str = _INFO) -> Table:
    """
    A real proportional-width filled rectangle bar, drawn as a 2-cell
    sub-table (filled cell + empty cell). Looks much better in a PDF
    than a unicode-block string. Range [0,1].
    """
    fraction = max(0.0, min(1.0, fraction))
    fw = max(0.5, total_width_mm * fraction)   # min 0.5mm so 0% is still visible
    ew = max(0.0, total_width_mm - fw)
    sub = Table([[" ", " "]], colWidths=[fw*mm, ew*mm], rowHeights=[3.5*mm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(colour)),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    return sub


def _format_value(v: Any) -> str:
    """Format a feature value compactly for the per-flow detail table."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def _html_safe(s: Any) -> str:
    """
    Escape a *plain-text data value* so it is safe to drop into a reportlab
    Paragraph (whose text is mini-HTML).

    This must escape `<` and `>` as well as `&` — reportlab's parser treats
    `<...>` as markup, so an unescaped value like "a<b>c" raises a
    ValueError and aborts the whole PDF. Order matters: escape `&` first so
    we don't double-escape the `&` we introduce for `<`/`>`.

    Only use this for untrusted *data* (feature names, summaries,
    recommendations). Do NOT use it on strings where we intentionally embed
    our own <font>/<b> markup.
    """
    s = str(s)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _color_chip(text: str, hex_color: str) -> Table:
    """A tag-like coloured pill for severity / pattern badges."""
    chip = Table([[Paragraph(
        f"<font color='{_TXT_LIGHT}'><b>{text}</b></font>",
        ParagraphStyle("chip", fontSize=8, alignment=TA_CENTER,
                        textColor=colors.HexColor(_TXT_LIGHT)))
    ]])
    chip.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor(hex_color)),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return chip


def _stat_cell(label: str, value: str, sub: str, colour: str) -> Table:
    """A small stat card for the cover page."""
    cell = Table([
        [Paragraph(f"<font color='{_TXT_SEC}' size='8'>{label}</font>",
                   ParagraphStyle("lbl", fontSize=8))],
        [Paragraph(f"<font color='{colour}' size='20'><b>{value}</b></font>",
                   ParagraphStyle("val", fontSize=20))],
        [Paragraph(f"<font color='{_TXT_SEC}' size='7'>{sub}</font>",
                   ParagraphStyle("sub", fontSize=7))],
    ], colWidths=[40*mm])
    cell.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor(_BG_PANEL)),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return cell


def _section_table_style() -> TableStyle:
    """Reusable table style for the pattern/severity tables."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_BG_HDR)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor(_TXT_LIGHT)),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 1), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LINEBELOW",  (0, 0), (-1, -2), 0.3, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.HexColor(_BG_ROW_ALT)]),
    ])


def _make_styles() -> Dict[str, ParagraphStyle]:
    """Build the paragraph styles used across the report."""
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title", parent=base["Title"],
            fontSize=22, leading=26, spaceAfter=2,
            textColor=colors.HexColor(_TXT_PRI), alignment=TA_LEFT),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=10, textColor=colors.HexColor(_TXT_SEC),
            alignment=TA_LEFT),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=16, leading=20, textColor=colors.HexColor(_TXT_PRI),
            spaceBefore=2, spaceAfter=4),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=12, leading=16, textColor=colors.HexColor(_TXT_PRI),
            spaceBefore=2, spaceAfter=2),
        "H3": ParagraphStyle(
            "H3", parent=base["Heading3"],
            fontSize=11, leading=14, textColor=colors.HexColor(_TXT_PRI),
            spaceBefore=0, spaceAfter=2),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.HexColor(_TXT_PRI)),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, leading=11, textColor=colors.HexColor(_TXT_SEC),
            spaceAfter=2),
        "Tag": ParagraphStyle(
            "Tag", parent=base["Normal"],
            fontSize=9, leading=11, textColor=colors.HexColor(_TXT_PRI)),
        "Tiny": ParagraphStyle(
            "Tiny", parent=base["Normal"],
            fontSize=8, leading=10, textColor=colors.HexColor(_TXT_PRI)),
        "Recommend": ParagraphStyle(
            "Recommend", parent=base["Normal"],
            fontSize=9, leading=12, textColor=colors.HexColor(_TXT_PRI),
            backColor=colors.HexColor("#FFF7ED"),
            borderColor=colors.HexColor(_WARN), borderWidth=0.6,
            borderPadding=4, leftIndent=2, rightIndent=2),
    }


def _footer_renderer(canvas, doc):
    """Page footer — page N of M, generation note."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(_TXT_SEC))
    page_num = canvas.getPageNumber()
    canvas.drawRightString(
        doc.pagesize[0] - 18*mm, 10*mm,
        f"Page {page_num}",
    )
    canvas.drawString(
        18*mm, 10*mm,
        "AI-Based Botnet Detection · Group 07 · CPCS499",
    )
    canvas.restoreState()


# ════════════════════════════════════════════════════════════════════
# Self-test — run with `python -m app.report_generator` from repo root
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Build a synthetic report from mocked DetectionFlow data."""
    from dataclasses import dataclass, field
    from typing import Optional as _Opt

    @dataclass
    class _MockFlow:
        src_ip:        str = ""
        dst_ip:        str = ""
        src_port:      int = 0
        dst_port:      int = 0
        protocol:      str = "—"
        label:         str = "benign"
        confidence:    float = 0.0
        device_type:   str = "noniot"
        s1_confidence: float = 0.0
        suspicion:     float = 0.0
        latency_ms:    float = 0.0
        alerted:       bool  = False
        timestamp:     float = 0.0
        source:        str   = "upload"
        source_file:   str   = "test.csv"
        report_id:     str   = "RPT-001"
        xai:           _Opt[dict] = None

    def _xai(pattern, severity, summary, top, recs):
        return {
            "device_type": "noniot",
            "prediction":  "botnet",
            "confidence":  0.9,
            "method":      "integrated_gradients",
            "pattern":     pattern,
            "severity":    severity,
            "summary":     summary,
            "top_features": [
                {"feature": k, "display": d, "value": v,
                 "attribution": a, "direction": ""}
                for k, d, v, a in top
            ],
            "feature_importance": {d: abs(a) for k, d, v, a in top},
            "recommendations": recs,
            "reasons": [],
        }

    flows = [
        # Botnet flows with varied XAI
        _MockFlow(src_ip="192.168.1.5", dst_ip="91.108.4.15",
                  src_port=51234, dst_port=4444, protocol="TCP",
                  label="botnet", confidence=0.94,
                  device_type="noniot", report_id="RPT-001",
                  xai=_xai("PORT_SCAN", "high",
                          "Likely port-scan / reconnaissance behaviour.",
                          [("flow_pkts_per_sec", "Flow pkt rate", 482.0, 0.42),
                           ("flag_SYN", "SYN flag count", 32.0, 0.38),
                           ("flow_duration", "Flow duration", 0.05, -0.21),
                           ("total_fwd_packets", "Forward packets", 2, -0.15),
                           ("dst_port", "Destination port", 4444, 0.12)],
                          ["Block or rate-limit the source IP at the firewall."])),
        _MockFlow(src_ip="10.0.0.10", dst_ip="203.0.113.5",
                  src_port=12345, dst_port=80, protocol="TCP",
                  label="botnet", confidence=0.97,
                  device_type="noniot", report_id="RPT-001",
                  xai=_xai("DDOS", "critical",
                          "Likely volumetric DDoS or flood attack.",
                          [("flow_pkts_per_sec", "Flow pkt rate", 8500.0, 0.55),
                           ("flow_bytes_per_sec", "Flow byte rate", 2.5e6, 0.30),
                           ("total_fwd_packets", "Forward packets", 600, 0.20)],
                          ["Engage upstream rate-limiting / traffic scrubbing immediately."])),
        _MockFlow(src_ip="192.168.50.30", dst_ip="185.220.101.5",
                  src_port=33210, dst_port=23, protocol="TCP",
                  label="botnet", confidence=0.88,
                  device_type="iot", report_id="RPT-001",
                  xai=_xai("MIRAI_FLOOD", "critical",
                          "Likely Mirai-style flood from an IoT device.",
                          [("MI_dir_L5_weight", "MI_dir weight (100ms window)", 8500, 0.45),
                           ("HH_L5_mean", "HH mean (100ms window)", 2.5e6, 0.30),
                           ("MI_dir_L3_weight", "MI_dir weight (500ms window)", 5500, 0.18)],
                          ["Isolate the source IoT device from the network."])),
        # Benign flows (no XAI by design)
        _MockFlow(src_ip="10.0.0.10", dst_ip="8.8.8.8",
                  src_port=54321, dst_port=53, protocol="UDP",
                  label="benign", confidence=0.97, device_type="noniot",
                  report_id="RPT-001", xai=None),
        _MockFlow(src_ip="192.168.1.7", dst_ip="172.217.16.46",
                  src_port=44444, dst_port=443, protocol="TCP",
                  label="benign", confidence=0.99, device_type="noniot",
                  report_id="RPT-001", xai=None),
    ]

    out = generate_pdf_report(
        flows,
        out_path="/tmp/test_botnet_report.pdf",
        report_meta={
            "report_id":   "RPT-001",
            "source":      "upload",
            "filename":    "test_capture.pcap",
            "duration_sec": 12.5,
        },
    )
    print(f"\n✓ PDF generated: {out}")
    print(f"  Size: {out.stat().st_size:,} bytes")