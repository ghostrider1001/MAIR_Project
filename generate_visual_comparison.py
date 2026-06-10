"""
MAIR+ v2 — Multi-Panel Visual Comparison Generator
=====================================================
Produces a single large comparison image showing all 6 degradation types
(blur, noise, jpeg, lowlight, haze, rain) processed through the MAIR+ pipeline.

For each degradation:
  - Takes a clean reference image
  - Applies the degradation
  - Runs MAIR+ pipeline
  - Shows: [Reference | Degraded | Restored] + SSIM/PSNR scores

Output: outputs/comparison_all6_<timestamp>.png
        - 6 rows (one per degradation type)
        - 3 columns: Reference | Degraded | MAIR+ Restored

Also saves individual comparison images per degradation type.

Usage:
    python generate_visual_comparison.py
    python generate_visual_comparison.py --source path/to/clean_image.png
"""

import os
import sys
import cv2
import argparse
import time
import numpy as np
from datetime import datetime

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_benchmark import (
    apply_motion_blur, apply_gaussian_noise, apply_jpeg_compression,
    apply_lowlight, apply_haze, apply_rain
)
from scheduler.scheduler import run_three_stage_scheduler
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


# ─────────────────────────────────────────────────────────────
# DEGRADATION CONFIG
# ─────────────────────────────────────────────────────────────

DEGRADATION_CONFIGS = [
    {
        "name":   "Motion Blur",
        "key":    "blur",
        "fn":     lambda img: apply_motion_blur(img, kernel_size=25),
        "desc":   "Motion blur (kernel=25px)",
        "color":  (255, 165, 0),   # orange
    },
    {
        "name":   "Gaussian Noise",
        "key":    "noise",
        "fn":     lambda img: apply_gaussian_noise(img, sigma=30.0),
        "desc":   "Gaussian noise (σ=30)",
        "color":  (100, 200, 255),  # light blue
    },
    {
        "name":   "JPEG Compression",
        "key":    "jpeg",
        "fn":     lambda img: apply_jpeg_compression(img, quality=10),
        "desc":   "JPEG artifacts (q=10)",
        "color":  (200, 100, 255),  # purple
    },
    {
        "name":   "Low-Light",
        "key":    "lowlight",
        "fn":     lambda img: apply_lowlight(img, gamma=3.5),
        "desc":   "Low-light (γ=3.5 darkening)",
        "color":  (50, 200, 50),    # green
    },
    {
        "name":   "Atmospheric Haze",
        "key":    "haze",
        "fn":     lambda img: apply_haze(img, beta=1.5, atm_light=0.85),
        "desc":   "Haze (β=1.5, physics scattering)",
        "color":  (100, 180, 180),  # cyan-grey
    },
    {
        "name":   "Rain Streaks",
        "key":    "rain",
        "fn":     lambda img: apply_rain(img, n_streaks=800),
        "desc":   "Rain streaks (800 streaks, −15°)",
        "color":  (150, 220, 255),  # rain blue
    },
]


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def _compute_metrics(ref, test):
    """Compute SSIM and PSNR between two BGR images."""
    try:
        if ref.shape != test.shape:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
        rg = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
        tg = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        s  = round(float(ssim_fn(rg, tg)), 4)
        p  = round(float(psnr_fn(rg, tg)), 2)
        return s, p
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────
# PANEL DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

PANEL_W = 380
PANEL_H = 300
LABEL_H = 75
BORDER  = 4
FONT    = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX


def _make_panel(img: np.ndarray, title: str, subtitle: str,
                ssim: float = None, psnr: float = None,
                border_color: tuple = (80, 80, 80)) -> np.ndarray:
    """
    Create a labelled image panel with title, subtitle, and metrics.
    Returns a (PANEL_H + LABEL_H) × PANEL_W BGR image.
    """
    total_h = PANEL_H + LABEL_H
    panel = np.zeros((total_h, PANEL_W, 3), dtype=np.uint8)

    # Image area: resize to fit, center
    img_resized = cv2.resize(img, (PANEL_W - 2*BORDER, PANEL_H - 2*BORDER),
                              interpolation=cv2.INTER_AREA)
    panel[BORDER:PANEL_H-BORDER, BORDER:PANEL_W-BORDER] = img_resized

    # Border around image
    cv2.rectangle(panel, (0, 0), (PANEL_W-1, PANEL_H-1), border_color, BORDER)

    # Label background
    panel[PANEL_H:, :] = (25, 25, 35)

    # Title
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.65, 1)
    tx = (PANEL_W - tw) // 2
    cv2.putText(panel, title, (tx, PANEL_H + 22), FONT, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

    # Subtitle
    (sw, sh), _ = cv2.getTextSize(subtitle, FONT_SM, 0.42, 1)
    sx = (PANEL_W - sw) // 2
    cv2.putText(panel, subtitle, (sx, PANEL_H + 44), FONT_SM, 0.42, (160, 160, 180), 1, cv2.LINE_AA)

    # Metrics row
    if ssim is not None and psnr is not None:
        metric_str = f"SSIM: {ssim:.4f}   PSNR: {psnr:.2f} dB"
        (mw, mh), _ = cv2.getTextSize(metric_str, FONT_SM, 0.45, 1)
        mx = (PANEL_W - mw) // 2

        # Colour-coded: green if good, yellow if moderate, red if bad
        if ssim >= 0.80:   mc = (80, 220, 80)
        elif ssim >= 0.60: mc = (80, 200, 220)
        elif ssim >= 0.40: mc = (60, 180, 255)
        else:              mc = (60, 80, 220)
        cv2.putText(panel, metric_str, (mx, PANEL_H + 63), FONT_SM, 0.45, mc, 1, cv2.LINE_AA)

    return panel


def _make_row(ref_img, deg_img, res_img, row_label, deg_config,
              deg_ssim, deg_psnr, res_ssim, res_psnr, res_time) -> np.ndarray:
    """Build one row of the comparison grid (Reference | Degraded | Restored)."""
    bc = deg_config["color"]

    ref_panel  = _make_panel(ref_img,  "Reference",
                              "Clean ground truth",
                              border_color=(60, 180, 60))
    deg_panel  = _make_panel(deg_img,  f"Degraded",
                              deg_config["desc"],
                              ssim=deg_ssim, psnr=deg_psnr,
                              border_color=bc)
    res_panel  = _make_panel(res_img,  "MAIR+ Restored",
                              f"Pipeline output  ({res_time:.1f}s)",
                              ssim=res_ssim, psnr=res_psnr,
                              border_color=(60, 80, 220))

    row = np.hstack([ref_panel, deg_panel, res_panel])

    # Left label bar
    bar_w = 60
    bar_h = row.shape[0]
    bar   = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    bar[:] = np.array(bc[::-1], dtype=np.uint8) * 0 + np.array([30, 30, 40], dtype=np.uint8)

    # Vertical text (rotate 90°)
    text_img = np.zeros((bar_w, bar_h, 3), dtype=np.uint8)
    text_img[:] = (30, 30, 40)
    label_chars = row_label.upper()
    (lw, lh), _ = cv2.getTextSize(label_chars, FONT, 0.55, 1)
    lx = (bar_h - lw) // 2
    cv2.putText(text_img, label_chars, (lx, bar_w - 10), FONT, 0.55,
                tuple(int(c) for c in bc), 1, cv2.LINE_AA)
    bar = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return np.hstack([bar, row])


# ─────────────────────────────────────────────────────────────
# MAIN COMPARISON GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_comparison(source_path: str, output_dir: str = "outputs") -> str:
    """
    Generate the full 6-degradation comparison image.

    Args:
        source_path : path to a clean reference image
        output_dir  : where to save results

    Returns:
        Path to the saved comparison grid image.
    """
    print("\n" + "=" * 70)
    print("  MAIR+ v2 — Full 6-Degradation Visual Comparison")
    print("=" * 70)

    # Load and resize source image
    ref_img = cv2.imread(source_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot load: {source_path}")

    h, w = ref_img.shape[:2]
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        ref_img = cv2.resize(ref_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    print(f"  Source: {source_path}  ({ref_img.shape[1]}×{ref_img.shape[0]})")

    # Temp directories
    tmp_dir = os.path.join(output_dir, "comparison_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    rows = []
    benchmark_results = []

    for i, cfg in enumerate(DEGRADATION_CONFIGS):
        key  = cfg["key"]
        name = cfg["name"]
        print(f"\n  [{i+1}/6] Processing: {name}")

        # Apply degradation
        deg_img = cfg["fn"](ref_img.copy())

        # Save degraded image to temp file (pipeline needs a path)
        deg_path = os.path.join(tmp_dir, f"degraded_{key}.png")
        cv2.imwrite(deg_path, deg_img)

        # Compute degraded metrics
        deg_ssim, deg_psnr = _compute_metrics(ref_img, deg_img)
        print(f"      Degraded  SSIM={deg_ssim:.4f}  PSNR={deg_psnr:.2f} dB")

        # Run MAIR+ pipeline
        t0 = time.time()
        try:
            result = run_three_stage_scheduler(deg_path, verbose=False, use_memory=True)
            res_path = result.get("output_path")
        except Exception as e:
            print(f"      [WARNING] Pipeline failed: {e}")
            res_path = None
        elapsed = round(time.time() - t0, 2)

        # Load restored image
        if res_path and os.path.exists(res_path):
            res_img = cv2.imread(res_path)
            if res_img is None:
                res_img = deg_img.copy()
        else:
            print(f"      [WARNING] No output produced, using degraded as fallback")
            res_img = deg_img.copy()

        # Compute restored metrics
        res_ssim, res_psnr = _compute_metrics(ref_img, res_img)
        ssim_gain = round(res_ssim - deg_ssim, 4) if (res_ssim and deg_ssim) else None
        psnr_gain = round(res_psnr - deg_psnr, 2) if (res_psnr and deg_psnr) else None
        print(f"      Restored  SSIM={res_ssim:.4f}  PSNR={res_psnr:.2f} dB  "
              f"ΔSS={ssim_gain:+.4f}  ΔPS={psnr_gain:+.2f}  ({elapsed}s)")

        # Save individual comparison
        ind_path = os.path.join(output_dir, f"compare_{key}.png")
        ind_grid = np.hstack([ref_img, deg_img, res_img])
        cv2.imwrite(ind_path, ind_grid)

        # Build row for the grid
        row = _make_row(ref_img, deg_img, res_img,
                        name, cfg,
                        deg_ssim, deg_psnr,
                        res_ssim, res_psnr, elapsed)
        rows.append(row)

        benchmark_results.append({
            "degradation":  name,
            "key":          key,
            "deg_ssim":     deg_ssim,
            "deg_psnr":     deg_psnr,
            "res_ssim":     res_ssim,
            "res_psnr":     res_psnr,
            "ssim_gain":    ssim_gain,
            "psnr_gain":    psnr_gain,
            "time_s":       elapsed,
        })

    # ── Build header ─────────────────────────────────────────
    row_w = rows[0].shape[1]
    header_h = 90
    header = np.zeros((header_h, row_w, 3), dtype=np.uint8)
    header[:] = (20, 20, 30)

    title1 = "MAIR+ v2 - Multi-Agent Intelligent Image Restoration"
    title2 = "6-Degradation Visual Comparison  |  Reference vs Degraded vs MAIR+ Restored"
    (w1, h1), _ = cv2.getTextSize(title1, FONT, 0.80, 2)
    cv2.putText(header, title1, ((row_w - w1) // 2, 35), FONT, 0.80, (255, 220, 80), 2, cv2.LINE_AA)
    (w2, h2), _ = cv2.getTextSize(title2, FONT_SM, 0.50, 1)
    cv2.putText(header, title2, ((row_w - w2) // 2, 60), FONT_SM, 0.50, (180, 180, 200), 1, cv2.LINE_AA)

    ts_str = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   Source: {os.path.basename(source_path)}"
    (tw, _), _ = cv2.getTextSize(ts_str, FONT_SM, 0.42, 1)
    cv2.putText(header, ts_str, ((row_w - tw) // 2, 80), FONT_SM, 0.42, (120, 120, 140), 1, cv2.LINE_AA)

    # ── Build footer with benchmark summary ──────────────────
    footer_h = 120
    footer = np.zeros((footer_h, row_w, 3), dtype=np.uint8)
    footer[:] = (20, 20, 30)

    # Table header
    cv2.putText(footer, "BENCHMARK SUMMARY", (20, 22), FONT, 0.55, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(footer, "Degradation Type     Baseline SSIM    MAIR+ SSIM    SSIM Gain    PSNR Gain",
                (20, 45), FONT_SM, 0.40, (160, 160, 180), 1, cv2.LINE_AA)
    cv2.line(footer, (20, 50), (row_w - 20, 50), (80, 80, 100), 1)

    y = 65
    col_positions = [20, 200, 370, 510, 660]
    for r in benchmark_results:
        gain  = r["ssim_gain"] or 0
        color = (80, 220, 80) if gain > 0 else (80, 80, 220)
        cv2.putText(footer, r["name"][:20], (col_positions[0], y), FONT_SM, 0.40, (200, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(footer, f"{r['deg_ssim']:.4f}", (col_positions[1], y), FONT_SM, 0.40, (160, 160, 200), 1, cv2.LINE_AA)
        cv2.putText(footer, f"{r['res_ssim']:.4f}", (col_positions[2], y), FONT_SM, 0.40, (160, 160, 200), 1, cv2.LINE_AA)
        cv2.putText(footer, f"{gain:+.4f}", (col_positions[3], y), FONT_SM, 0.40, color, 1, cv2.LINE_AA)
        pnr = r["psnr_gain"] or 0
        cv2.putText(footer, f"{pnr:+.2f} dB", (col_positions[4], y), FONT_SM, 0.40, color, 1, cv2.LINE_AA)
        y += 15

    # ── Assemble final grid ──────────────────────────────────
    grid = np.vstack([header] + rows + [footer])

    # ── Save ─────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"comparison_ALL6_{ts}.png")
    cv2.imwrite(out_path, grid)

    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)
    print(f"  Main grid  : {out_path}")
    print(f"  Individual : outputs/compare_{{blur|noise|jpeg|lowlight|haze|rain}}.png")
    print()

    # Print benchmark table
    print(f"  {'Degradation':<22}  {'Baseline SSIM':>13}  {'MAIR+ SSIM':>10}  {'SSIM Gain':>10}  {'PSNR Gain':>10}")
    print(f"  {'─'*78}")
    for r in benchmark_results:
        g = r['ssim_gain'] or 0
        marker = "🟢" if g > 0 else "🔴"
        print(f"  {r['name']:<22}  {r['deg_ssim']:>13.4f}  {r['res_ssim']:>10.4f}"
              f"  {g:>+10.4f}  {(r['psnr_gain'] or 0):>+8.2f} dB  {marker}")
    print(f"  {'─'*78}")

    avg_gain = sum(r['ssim_gain'] or 0 for r in benchmark_results) / len(benchmark_results)
    print(f"  {'AVERAGE':<22}  {'':>13}  {'':>10}  {avg_gain:>+10.4f}")
    print()

    return out_path, benchmark_results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _find_source_image() -> str:
    """Find a suitable clean source image."""
    candidates = [
        "outputs/test_inputs/baby.png",
        "outputs/test_inputs/bird.png",
        "datasets/benchmark/noise_test/reference/baby.png",
        "datasets/benchmark/blur_test/reference/baby.png",
        "models/SwinIR/testsets/Set5/baby.png",
        "models/SwinIR/testsets/Set5/bird.png",
        "test.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Check datasets/benchmark/*/reference/
    import glob
    refs = glob.glob("datasets/benchmark/*/reference/*.png")
    if refs:
        return refs[0]

    print("[Comparison] ERROR: No clean source image found.")
    print("  Provide one with --source path/to/clean_image.png")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ v2 — Generate 6-degradation visual comparison"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to clean reference image (auto-found if omitted)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save comparison images (default: outputs/)"
    )
    args = parser.parse_args()

    source = args.source or _find_source_image()
    if not source:
        return

    if not os.path.exists(source):
        print(f"[Comparison] ERROR: File not found: {source}")
        return

    out_path, results = generate_comparison(source, args.output_dir)
    print(f"\n  >> Open: {out_path}")


if __name__ == "__main__":
    main()
