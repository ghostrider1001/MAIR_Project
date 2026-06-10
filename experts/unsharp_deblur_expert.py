"""
Unsharp Mask Deblur Expert (C12 — Fast Fallback)
=================================================
A lightweight, CPU-only blur-reduction expert using unsharp masking.

Unsharp masking subtracts a blurred version of the image from itself,
amplifying high-frequency detail. It does NOT remove motion blur (that
requires blind deconvolution), but perceptually sharpens slightly blurry
images with no GPU and in <100ms.

Registered as a very_fast, low-quality fallback for the voting ensemble
(C12) when Restormer is unavailable or time budget is exceeded.

stage   = "imaging"
handles = ["blur"]
speed   = "very_fast"
quality = "low"
"""

import os
import time
import cv2
import numpy as np


def restore_unsharp_deblur(input_path: str) -> str | None:
    """
    Apply unsharp masking to perceptually sharpen a slightly blurry image.

    Pipeline:
      1. Gaussian blur (sigma=2.0) to extract low-frequency content
      2. Weighted addition: sharpened = 1.5 × original − 0.5 × blurred
      3. Clip to [0, 255]

    Returns:
        Path to sharpened output PNG, or None on failure.
    """
    print("\n===================================")
    print("   UNSHARP DEBLUR EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Unsharp Deblur] Input Path : {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[Unsharp Deblur] ERROR: Cannot load image.")
        return None

    # ── Unsharp mask ─────────────────────────────────────────
    # sigma=2.0 extracts edge detail without amplifying noise too much
    blurred  = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Save ─────────────────────────────────────────────────
    output_dir  = os.path.join("outputs", "deblurred")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_unsharp.png")
    cv2.imwrite(output_path, sharpened)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Unsharp Deblur] Output saved     : {output_path}")
    print(f"[Unsharp Deblur] Processing Time  : {elapsed}s")

    print("\n===================================")
    print("   UNSHARP DEBLUR EXPERT FINISHED")
    print("===================================\n")

    return output_path
