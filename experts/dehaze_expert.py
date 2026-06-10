"""
Dehazing Expert — DCP (C1)
===========================
MAIR+ Contribution C1.

Removes haze / atmospheric scattering from images using the
Dark Channel Prior (He et al., 2011) — a physics-based method
that requires no GPU and no model download.

Stage assignment: "scene" (Stage 3) — same as lowlight.
Handles: ["haze"]

DehazeFormer integration (optional):
  If models/DehazeFormer/pretrained/dehazeformer-b.pth exists,
  restore_dehazeformer() will be available as a higher-quality
  alternative registered in the tool registry.

Returns:
    Path to dehazed output image, or None on failure.
"""

import os
import time
import cv2
import numpy as np

from core.dark_channel_prior import dehaze_dcp


# ─────────────────────────────────────────────────────────────
# DCP DEHAZE EXPERT
# ─────────────────────────────────────────────────────────────

def restore_dcp(input_path: str) -> str | None:
    """
    Remove haze from an image using the Dark Channel Prior.

    Physics-based: no GPU, no model download required.
    Estimated runtime: 0.5–2s depending on image resolution.

    Returns:
        Path to dehazed output PNG, or None on failure.
    """
    print("\n===================================")
    print("      DEHAZE (DCP) EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Dehaze DCP] Input Path : {input_path}")

    # ── Load image ────────────────────────────────────────────
    img = cv2.imread(input_path)
    if img is None:
        print(f"[Dehaze DCP] ERROR: Cannot load image: {input_path}")
        return None

    h, w = img.shape[:2]
    print(f"[Dehaze DCP] Resolution : {w}×{h}")

    # ── Apply DCP dehazing ────────────────────────────────────
    print("[Dehaze DCP] Running Dark Channel Prior dehazing...")
    try:
        dehazed = dehaze_dcp(
            img,
            patch_size=15,
            omega=0.95,
            t_min=0.10,
            refine=True,
        )
    except Exception as e:
        print(f"[Dehaze DCP] ERROR: DCP dehazing failed: {e}")
        return None

    # ── Post-processing: contrast enhancement ─────────────────
    # DCP can produce slightly washed-out results; mild CLAHE helps
    lab      = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enh    = clahe.apply(l)
    dehazed  = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)

    # ── Save output ───────────────────────────────────────────
    output_dir  = os.path.join("outputs", "dehazed")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_dehazed.png")
    cv2.imwrite(output_path, dehazed)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Dehaze DCP] Output saved      : {output_path}")
    print(f"[Dehaze DCP] Processing Time   : {elapsed}s")

    print("\n===================================")
    print("      DEHAZE (DCP) EXPERT FINISHED")
    print("===================================\n")

    return output_path


# ─────────────────────────────────────────────────────────────
# DEHAZEFORMER EXPERT (optional — gated on weights existence)
# ─────────────────────────────────────────────────────────────

DEHAZEFORMER_WEIGHTS = "models/DehazeFormer/pretrained/dehazeformer-b.pth"


def restore_dehazeformer(input_path: str) -> str | None:
    """
    Remove haze using DehazeFormer (transformer-based, GPU-accelerated).

    Only available if pretrained weights are placed at:
        models/DehazeFormer/pretrained/dehazeformer-b.pth

    Falls back gracefully to DCP if weights are missing.

    Returns:
        Path to dehazed output PNG, or None on failure.
    """
    if not os.path.exists(DEHAZEFORMER_WEIGHTS):
        print(f"[DehazeFormer] Weights not found: {DEHAZEFORMER_WEIGHTS}")
        print("[DehazeFormer] Falling back to DCP dehazing.")
        return restore_dcp(input_path)

    print("\n===================================")
    print("    DEHAZE (DehazeFormer) ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[DehazeFormer] Input Path : {input_path}")

    try:
        import torch
        # Lazy import — only needed if weights exist
        # DehazeFormer model loading would go here
        # For now, this is a placeholder that falls through to DCP
        print("[DehazeFormer] NOTE: DehazeFormer inference not yet implemented.")
        print("[DehazeFormer] Falling back to DCP.")
        return restore_dcp(input_path)

    except ImportError:
        print("[DehazeFormer] PyTorch not installed — using DCP fallback.")
        return restore_dcp(input_path)
    except Exception as e:
        print(f"[DehazeFormer] ERROR: {e} — using DCP fallback.")
        return restore_dcp(input_path)
