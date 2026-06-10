"""
Expert Comparison Grid
======================
Runs ALL available experts on a single input image and assembles
a 2×3 panel grid showing each output with quality scores.

Usage:
    python experiments/compare_experts.py --input <image_path>
    python experiments/compare_experts.py --input models/SwinIr/testsets/test/test.jpg

Output:
    outputs/expert_grid_Phase2_<timestamp>.png
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experts.sr_expert       import restore_sr
from experts.denoise_expert  import restore_denoise
from experts.jpeg_expert     import restore_jpeg
from experts.deblur_expert   import restore_deblur
from experts.lowlight_expert import restore_lowlight


# ─────────────────────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────────────────────

def compute_ssim_psnr(original_path, restored_path):
    """Compute SSIM and PSNR. Handles different sizes (e.g., 4x SR)."""
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        orig = cv2.imread(original_path)
        rest = cv2.imread(restored_path)
        if orig is None or rest is None:
            return None, None

        if orig.shape != rest.shape:
            rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))

        og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rest, cv2.COLOR_BGR2GRAY)

        return (
            round(float(ssim(og, rg)), 4),
            round(float(psnr(og, rg)), 2),
        )
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────
# PANEL BUILDER
# ─────────────────────────────────────────────────────────────

PANEL_W = 400
PANEL_H = 320
BAR_H   = 56
FONT    = cv2.FONT_HERSHEY_SIMPLEX


def _build_panel(image_path, label, ssim_val=None, psnr_val=None, failed=False):
    """
    Build one grid panel: label bar + image (resized to PANEL_W×PANEL_H).
    Returns a numpy array of shape (BAR_H + PANEL_H, PANEL_W, 3).
    """
    dark    = (22,  22,  22)
    white   = (230, 230, 230)
    green   = (72,  210, 100)
    red     = (80,  80,  220)
    gray    = (130, 130, 130)

    # ── Label bar ──────────────────────────────────────────────
    bar = np.full((BAR_H, PANEL_W, 3), dark, dtype=np.uint8)
    cv2.putText(bar, label, (10, 22), FONT, 0.65, white, 2, cv2.LINE_AA)

    if failed:
        cv2.putText(bar, "FAILED / SKIPPED", (10, 44), FONT, 0.52, red, 1, cv2.LINE_AA)
    elif ssim_val is not None and psnr_val is not None:
        metric_txt = f"SSIM: {ssim_val:.4f}   PSNR: {psnr_val:.2f} dB"
        cv2.putText(bar, metric_txt, (10, 44), FONT, 0.50, green, 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, "No reference metrics", (10, 44), FONT, 0.50, gray, 1, cv2.LINE_AA)

    # ── Image panel ────────────────────────────────────────────
    if image_path and os.path.exists(image_path) and not failed:
        img = cv2.imread(image_path)
        if img is not None:
            panel = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        else:
            panel = np.full((PANEL_H, PANEL_W, 3), 40, dtype=np.uint8)
    else:
        panel = np.full((PANEL_H, PANEL_W, 3), 35, dtype=np.uint8)
        msg   = "Not available"
        tw    = cv2.getTextSize(msg, FONT, 0.65, 1)[0][0]
        cv2.putText(
            panel, msg,
            ((PANEL_W - tw) // 2, PANEL_H // 2),
            FONT, 0.65, (80, 80, 80), 1, cv2.LINE_AA
        )

    return np.vstack([bar, panel])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_comparison(input_path, output_dir="outputs"):
    print("\n" + "=" * 60)
    print("   MAIR+  Expert Comparison Grid")
    print("=" * 60)
    print(f"  Input : {input_path}\n")

    if not os.path.exists(input_path):
        print(f"ERROR: File not found — {input_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ── Define experts ────────────────────────────────────────
    experts = [
        ("Original",      None),          # no function — use input itself
        ("SR  (SwinIR x4)", restore_sr),
        ("Deblur (Restormer)", restore_deblur),
        ("Denoise (NLM)", restore_denoise),
        ("JPEG (SwinIR CAR)", restore_jpeg),
        ("Lowlight (CLAHE)", restore_lowlight),
    ]

    results = []  # list of (label, output_path, ssim, psnr, failed)

    for label, fn in experts:
        print(f"─── Running: {label}")
        if fn is None:
            # Original image panel
            results.append((label, input_path, None, None, False))
            continue

        try:
            t0      = time.time()
            out     = fn(input_path)
            elapsed = round(time.time() - t0, 1)

            if out and os.path.exists(out):
                ssim_v, psnr_v = compute_ssim_psnr(input_path, out)
                print(f"    Done in {elapsed}s  |  SSIM: {ssim_v}  PSNR: {psnr_v} dB")
                results.append((label, out, ssim_v, psnr_v, False))
            else:
                print(f"    FAILED or returned None.")
                results.append((label, None, None, None, True))

        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append((label, None, None, None, True))

    print("\n── Assembling comparison grid...")

    # ── Build panels ──────────────────────────────────────────
    panels = []
    for label, path, ssim_v, psnr_v, failed in results:
        panel = _build_panel(path, label, ssim_v, psnr_v, failed)
        panels.append(panel)

    # Ensure exactly 6 panels (pad if needed)
    while len(panels) < 6:
        panels.append(_build_panel(None, "N/A", failed=True))

    # ── Assemble 2×3 grid ─────────────────────────────────────
    gap     = np.full((PANEL_H + BAR_H, 4, 3), 50, dtype=np.uint8)
    row1    = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    row2    = np.hstack([panels[3], gap, panels[4], gap, panels[5]])
    h_gap   = np.full((6, row1.shape[1], 3), 50, dtype=np.uint8)
    grid    = np.vstack([row1, h_gap, row2])

    # ── Footer ────────────────────────────────────────────────
    footer_h = 38
    footer   = np.full((footer_h, grid.shape[1], 3), 15, dtype=np.uint8)
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        footer,
        f"MAIR+  |  Phase 2 Expert Grid  |  {ts}  |  {os.path.basename(input_path)}",
        (14, 26), FONT, 0.50, (110, 110, 110), 1, cv2.LINE_AA
    )
    grid = np.vstack([grid, footer])

    # ── Save ──────────────────────────────────────────────────
    ts_short    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path    = os.path.join(output_dir, f"expert_grid_Phase2_{ts_short}.png")
    cv2.imwrite(out_path, grid)

    print(f"\n  Grid saved → {out_path}")
    print("  Open this file to see all experts side by side.")
    print("=" * 60 + "\n")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIR+ Expert Comparison Grid")
    parser.add_argument(
        "--input",
        type=str,
        default="models/SwinIr/testsets/test/test.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the grid image",
    )
    args = parser.parse_args()
    run_comparison(args.input, args.output_dir)
