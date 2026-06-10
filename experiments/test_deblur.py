"""
Deblur Test — Real Blurry Image
=================================
Runs Restormer motion deblurring on an EXISTING blurry image
and produces a side-by-side before/after comparison.

Default test images (real camera-shake photos from Restormer demo):
    models/Restormer/demo/degraded/couple.jpg
    models/Restormer/demo/degraded/engagement.jpg
    models/Restormer/demo/degraded/portrait.jpg

Usage:
    python experiments/test_deblur.py
    python experiments/test_deblur.py --input models/Restormer/demo/degraded/portrait.jpg
    python experiments/test_deblur.py --input <any_blurry_image.jpg>
"""

import os
import sys
import argparse
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experts.deblur_expert import restore_deblur


# ─────────────────────────────────────────────────────────────
# SHARPNESS MEASUREMENT
# ─────────────────────────────────────────────────────────────

def laplacian_sharpness(image_path):
    """Return Laplacian variance — higher = sharper."""
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


# ─────────────────────────────────────────────────────────────
# PANEL BUILDER
# ─────────────────────────────────────────────────────────────

FONT    = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 500
PANEL_H = 420
BAR_H   = 72


def _panel(img_path, title, line2=None, title_color=(230, 230, 230)):
    dark = (18, 18, 18)
    bar  = np.full((BAR_H, PANEL_W, 3), dark, dtype=np.uint8)

    cv2.putText(bar, title, (12, 28), FONT, 0.75, title_color, 2, cv2.LINE_AA)
    if line2:
        cv2.putText(bar, line2, (12, 56), FONT, 0.52, (120, 210, 120), 1, cv2.LINE_AA)

    img   = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None
    panel = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA) \
            if img is not None else np.full((PANEL_H, PANEL_W, 3), 35, dtype=np.uint8)

    return np.vstack([bar, panel])


def save_comparison(blurry_path, deblurred_path, sharp_before, sharp_after, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    delta = round(sharp_after - sharp_before, 2)
    sign  = "+" if delta >= 0 else ""

    p1 = _panel(
        blurry_path,
        "BLURRY INPUT",
        f"Laplacian variance: {sharp_before}  (lower = more blur)",
        title_color=(80, 140, 220)
    )
    p2 = _panel(
        deblurred_path,
        "DEBLURRED  (Restormer)",
        f"Laplacian variance: {sharp_after}  ({sign}{delta} improvement)",
        title_color=(72, 210, 100)
    )

    gap  = np.full((PANEL_H + BAR_H, 6, 3), 55, dtype=np.uint8)
    grid = np.hstack([p1, gap, p2])

    footer = np.full((36, grid.shape[1], 3), 10, dtype=np.uint8)
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(footer,
        f"MAIR+  |  Restormer Motion Deblur  |  {ts}  |  {os.path.basename(blurry_path)}",
        (12, 24), FONT, 0.48, (100, 100, 100), 1, cv2.LINE_AA)
    grid = np.vstack([grid, footer])

    ts_s     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"deblur_result_{ts_s}.png")
    cv2.imwrite(out_path, grid)
    return out_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Restormer Deblur on Real Blurry Image")
    parser.add_argument(
        "--input", type=str,
        default="models/Restormer/demo/degraded/couple.jpg",
        help="Path to a real blurry image"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    blurry_path = args.input

    print("\n" + "=" * 60)
    print("   Restormer Deblur — Real Blurry Image")
    print("=" * 60)
    print(f"  Input : {blurry_path}\n")

    if not os.path.exists(blurry_path):
        print(f"  ERROR: File not found — {blurry_path}")
        print("\n  Available real blurry images:")
        for f in ["couple.jpg", "engagement.jpg", "portrait.jpg"]:
            p = f"models/Restormer/demo/degraded/{f}"
            print(f"    {p}")
        return

    # ── Measure sharpness before ──────────────────────────────
    sharp_before = laplacian_sharpness(blurry_path)
    print(f"  Sharpness (input)  : {sharp_before}  (Laplacian variance)")

    # ── Run Restormer ─────────────────────────────────────────
    print("\n── Running Restormer deblur expert...\n")
    deblurred_path = restore_deblur(blurry_path)

    if deblurred_path is None:
        print("  ERROR: Deblur expert returned None.")
        return

    # ── Measure sharpness after ───────────────────────────────
    sharp_after = laplacian_sharpness(deblurred_path)
    delta       = round(sharp_after - sharp_before, 2)
    sign        = "+" if delta >= 0 else ""
    print(f"  Sharpness (output) : {sharp_after}  ({sign}{delta})")

    # ── Save comparison ───────────────────────────────────────
    out_path = save_comparison(
        blurry_path, deblurred_path,
        sharp_before, sharp_after,
        args.output_dir
    )

    print(f"\n  Comparison saved → {out_path}")
    print("  Open this file to see:  Blurry  |  Deblurred")

    print("\n── Try other real blurry images:")
    for f in ["couple.jpg", "engagement.jpg", "portrait.jpg"]:
        p = f"models/Restormer/demo/degraded/{f}"
        if p != blurry_path:
            print(f"   python experiments/test_deblur.py --input {p}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
