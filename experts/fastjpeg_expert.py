"""
Fast JPEG Artifact Removal Expert (C12 — Fast Fallback)
========================================================
A lightweight, CPU-only JPEG artifact reduction expert using OpenCV's
Non-Local Means denoising tuned for blocking artifacts.

JPEG blocking artifacts manifest as high-frequency noise concentrated
at 8×8 block boundaries. NLM with a small h parameter effectively
smooths these without over-blurring, at a fraction of SwinIR's compute.

Registered as a very_fast, low-quality fallback for the voting ensemble
(C12) when SwinIR-JPEG is unavailable or time budget is exceeded.

stage   = "compression"
handles = ["jpeg"]
speed   = "very_fast"
quality = "low"
"""

import os
import time
import cv2


def restore_fast_jpeg(input_path: str) -> str | None:
    """
    Remove JPEG compression artifacts using fast Non-Local Means denoising.

    NLM parameters tuned for JPEG blocking noise:
      h=10      — moderate denoising strength (JPEG noise sigma ≈ 10–15)
      templateWindowSize=7  — patch size
      searchWindowSize=21   — search window

    Returns:
        Path to artifact-reduced output PNG, or None on failure.
    """
    print("\n===================================")
    print("    FAST JPEG EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Fast JPEG] Input Path : {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[Fast JPEG] ERROR: Cannot load image.")
        return None

    # ── Fast NLM denoising (JPEG-tuned) ──────────────────────
    # h=10 is calibrated for JPEG quality 10-25 artifacts
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,                # luminance denoising strength
        hColor=10,           # color denoising strength
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # ── Save ─────────────────────────────────────────────────
    output_dir  = os.path.join("outputs", "jpeg")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_fastjpeg.png")
    cv2.imwrite(output_path, denoised)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Fast JPEG] Output saved     : {output_path}")
    print(f"[Fast JPEG] Processing Time  : {elapsed}s")

    print("\n===================================")
    print("    FAST JPEG EXPERT FINISHED")
    print("===================================\n")

    return output_path
