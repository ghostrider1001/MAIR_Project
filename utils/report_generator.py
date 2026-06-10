"""
Per-Stage Restoration Report Card (C7)
=======================================
MAIR+ Contribution C7.

Generates a structured, per-stage breakdown of every pipeline run —
showing which expert ran, quality before/after, rollback events, and
time taken at each stage.

Usage (from scheduler or run_pipeline):
    from utils.report_generator import generate_report
    generate_report(stage_results, input_path, final_output, format="console")

Called automatically when --report flag is set.
"""

import os
import base64
import textwrap
from datetime import datetime

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────

def _quality_label(score):
    if score is None:      return "N/A"
    if score >= 0.85:      return "Excellent"
    if score >= 0.70:      return "Good"
    if score >= 0.50:      return "Moderate"
    return "Poor"


def _fmt(val, decimals=4, suffix=""):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def generate_console_report(
    stage_results:   dict,
    input_path:      str,
    final_output:    str | None,
    total_time_s:    float | None = None,
    invocation_count: int | None  = None,
    memory_bias_applied: bool     = False,
) -> str:
    """
    Build a formatted per-stage console report string.

    Args:
        stage_results     : dict from run_three_stage_scheduler() — keyed by stage name
        input_path        : path to original degraded image
        final_output      : path to final restored image (or None)
        total_time_s      : total pipeline wall-clock time
        invocation_count  : total expert invocations
        memory_bias_applied: whether memory system influenced expert selection

    Returns:
        Formatted string — print it or write it to a file.
    """
    W = 64
    lines = []
    lines.append("=" * W)
    lines.append("  MAIR+ v2 — Per-Stage Restoration Report Card")
    lines.append("=" * W)
    lines.append(f"  Input        : {os.path.basename(input_path)}")
    lines.append(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if memory_bias_applied:
        lines.append("  Memory       : ✓ Memory bias applied (case-based reasoning)")
    lines.append("")

    # ── Per-stage breakdown ─────────────────────────────────
    stage_order = ["compression", "imaging", "scene"]
    lines.append(f"  {'STAGE':<12}  {'EXPERT':<26}  {'PRE':>7}  {'POST':>7}  {'Δ':>7}  {'STATUS'}")
    lines.append("  " + "─" * (W - 2))

    for stage in stage_order:
        r = stage_results.get(stage, {})

        if not r:
            lines.append(f"  {stage.upper():<12}  {'—':<26}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  NOT RUN")
            continue

        if r.get("skipped"):
            lines.append(f"  {stage.upper():<12}  {'—':<26}  {'':>7}  {'':>7}  {'':>7}  SKIPPED")
            continue

        expert      = r.get("best_expert", r.get("best_expert_key", "N/A")) or "N/A"
        pre_score   = r.get("pre_stage_quality")
        post_score  = r.get("best_score")
        rolled_back = r.get("rolled_back", False)
        attempts    = r.get("attempts", 0)
        lpips_val   = r.get("lpips")

        delta = None
        if pre_score is not None and post_score is not None:
            delta = post_score - pre_score

        status = "ROLLBACK ⚠️" if rolled_back else "ACCEPTED ✓"
        if post_score is None:
            status = "FAILED ✗"

        expert_short = (expert[:24] + "..") if len(expert) > 26 else expert
        lines.append(
            f"  {stage.upper():<12}  {expert_short:<26}  "
            f"{_fmt(pre_score, 4):>7}  {_fmt(post_score, 4):>7}  "
            f"{('+' if delta and delta >= 0 else '') + _fmt(delta, 4) if delta is not None else 'N/A':>7}  "
            f"{status}"
        )

        # Sub-details
        if lpips_val is not None:
            lines.append(f"  {'':12}  {'LPIPS perceptual distance':<26}  {'':>7}  {_fmt(lpips_val, 4):>7}")
        if attempts > 0:
            lines.append(f"  {'':12}  {'Expert attempts':<26}  {attempts:>7}")

        # Per-stage re-detection scores (C2)
        redet = r.get("redetection_scores")
        if redet:
            top3 = sorted(redet.items(), key=lambda x: x[1], reverse=True)[:3]
            scores_str = "  ".join(f"{k}:{v:.2f}" for k, v in top3)
            lines.append(f"  {'':12}  Re-detection: {scores_str}")

    # ── Summary ─────────────────────────────────────────────
    lines.append("  " + "─" * (W - 2))

    if final_output:
        lines.append(f"  Final output : {final_output}")
    else:
        lines.append("  Final output : None (all stages rolled back or failed)")

    if total_time_s is not None:
        lines.append(f"  Total time   : {total_time_s}s")
    if invocation_count is not None:
        lines.append(f"  Expert calls : {invocation_count}")

    lines.append("=" * W)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# HTML REPORT (optional — triggered by --report html)
# ─────────────────────────────────────────────────────────────

def _img_to_b64(path: str, max_px: int = 300) -> str | None:
    """Read an image, downscale for embedding, return base64 PNG string."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return None


def generate_html_report(
    stage_results:    dict,
    input_path:       str,
    final_output:     str | None,
    total_time_s:     float | None = None,
    invocation_count: int | None   = None,
    memory_bias_applied: bool      = False,
    output_path:      str | None   = None,
) -> str:
    """
    Build a self-contained HTML report with embedded before/after thumbnails.

    Args:
        output_path : if provided, write HTML to this path.

    Returns:
        HTML string.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Encode images
    orig_b64  = _img_to_b64(input_path)
    final_b64 = _img_to_b64(final_output) if final_output else None

    def _img_tag(b64, alt="image"):
        if b64 is None:
            return f'<div class="no-img">No image</div>'
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:280px;border-radius:6px;">'

    # Build stage rows
    stage_order = ["compression", "imaging", "scene"]
    stage_rows = ""
    for stage in stage_order:
        r = stage_results.get(stage, {})
        if r.get("skipped") or not r:
            status_cls = "skipped"
            status_txt = "SKIPPED"
            expert_txt = "—"
            pre_txt = post_txt = delta_txt = "—"
        elif r.get("rolled_back"):
            status_cls = "rollback"
            status_txt = "ROLLBACK"
            expert_txt = r.get("best_expert", "N/A") or "N/A"
            pre_txt  = _fmt(r.get("pre_stage_quality"), 4)
            post_txt = _fmt(r.get("best_score"), 4)
            pre = r.get("pre_stage_quality"); post = r.get("best_score")
            delta_txt = _fmt(post - pre if pre is not None and post is not None else None, 4)
        else:
            status_cls = "accepted"
            status_txt = "ACCEPTED"
            expert_txt = r.get("best_expert", "N/A") or "N/A"
            pre_txt  = _fmt(r.get("pre_stage_quality"), 4)
            post_txt = _fmt(r.get("best_score"), 4)
            pre = r.get("pre_stage_quality"); post = r.get("best_score")
            delta_txt = _fmt(post - pre if pre is not None and post is not None else None, 4)

        stage_rows += f"""
        <tr>
            <td><strong>{stage.upper()}</strong></td>
            <td>{expert_txt}</td>
            <td>{pre_txt}</td>
            <td>{post_txt}</td>
            <td>{delta_txt}</td>
            <td><span class="badge {status_cls}">{status_txt}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MAIR+ Report — {os.path.basename(input_path)}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #111; color: #ddd; margin: 40px; }}
  h1   {{ color: #7ec8e3; }}
  h2   {{ color: #aaa; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ padding: 10px 14px; text-align: left; border: 1px solid #333; }}
  th {{ background: #222; color: #7ec8e3; }}
  tr:nth-child(even) {{ background: #1a1a1a; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }}
  .accepted {{ background: #1a4a1a; color: #6fe06f; }}
  .rollback {{ background: #4a1a1a; color: #e06f6f; }}
  .skipped  {{ background: #2a2a2a; color: #888; }}
  .img-row  {{ display: flex; gap: 24px; align-items: flex-start; margin: 20px 0; }}
  .img-box  {{ text-align: center; }}
  .img-box p {{ color: #aaa; margin: 6px 0 0; font-size: 0.9em; }}
  .no-img   {{ width: 280px; height: 200px; background: #222; display: flex;
               align-items: center; justify-content: center; color: #555; border-radius: 6px; }}
  .meta     {{ color: #888; font-size: 0.9em; margin: 4px 0; }}
</style>
</head>
<body>
<h1>MAIR+ v2 — Restoration Report Card</h1>
<p class="meta">Generated: {ts}</p>
<p class="meta">Input: <code>{input_path}</code></p>
{"<p class='meta'>⚡ Memory bias applied (case-based reasoning active)</p>" if memory_bias_applied else ""}

<h2>Before / After</h2>
<div class="img-row">
  <div class="img-box">{_img_tag(orig_b64, 'Original')}<p>Original (Degraded)</p></div>
  <div class="img-box">{_img_tag(final_b64, 'Restored')}<p>Restored (Final Output)</p></div>
</div>

<h2>Per-Stage Breakdown</h2>
<table>
  <tr><th>Stage</th><th>Expert</th><th>Pre-Score</th><th>Post-Score</th><th>Δ Score</th><th>Status</th></tr>
  {stage_rows}
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Final Output</td><td><code>{final_output or "None"}</code></td></tr>
  <tr><td>Total Time</td><td>{f"{total_time_s}s" if total_time_s else "N/A"}</td></tr>
  <tr><td>Expert Invocations</td><td>{invocation_count or "N/A"}</td></tr>
</table>
</body>
</html>"""

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[Report] HTML report saved → {output_path}")

    return html


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_report(
    stage_results:       dict,
    input_path:          str,
    final_output:        str | None,
    total_time_s:        float | None = None,
    invocation_count:    int | None   = None,
    memory_bias_applied: bool         = False,
    format:              str          = "console",   # "console" | "html" | "both"
    html_output_path:    str | None   = None,
) -> str:
    """
    Generate a restoration report in the requested format.

    Args:
        format : "console" (default), "html", or "both"

    Returns:
        Console report string (always). HTML is written to disk if format includes 'html'.
    """
    console = generate_console_report(
        stage_results, input_path, final_output,
        total_time_s, invocation_count, memory_bias_applied,
    )

    if format in ("html", "both"):
        if html_output_path is None:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            name = os.path.splitext(os.path.basename(input_path))[0]
            html_output_path = os.path.join(
                "outputs", "reports", f"report_{name}_{ts}.html"
            )
        generate_html_report(
            stage_results, input_path, final_output,
            total_time_s, invocation_count, memory_bias_applied,
            output_path=html_output_path,
        )

    return console
