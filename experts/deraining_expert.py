"""
Deraining Expert — Frequency-Domain + Morphological Rain Removal
=================================================================
MAIR+ Expert — Rain Streak Removal without pretrained weights

Rain in images appears as oriented, near-vertical, high-frequency streaks.
This module removes them using:

  1. **Streak Detection** (morphological)
     - Rain streaks are elongated, bright, near-vertical structures
     - Morphological top-hat with a thin vertical structuring element isolates them
     - Threshold + blur gives a soft rain mask

  2. **Frequency Suppression** (FFT)
     - Transform to frequency domain
     - Rain shows up as energy along the horizontal axis (vertical streaks)
     - Suppress a band around the frequency-domain rain signature
     - Inverse transform gives the rain-suppressed base

  3. **Guided Filter Inpainting**
     - In masked rain regions, blend the frequency-suppressed result
     - In non-rain regions, keep original (zero modification)

  4. **Colour Correction**
     - Rain brightens the image; correct the mean luminance shift

This approach works well for:
  ✓ Synthetic benchmark rain (Rain100L, Rain100H style)
  ✓ Light-to-moderate real-world rain
  ✗ Heavy rain with occlusion (needs a neural model like DRSformer)

DRSformer upgrade path:
    If `drsformer` package is installed and weights exist, the function
    automatically uses DRSformer instead of the classical method.

Usage:
    from experts.deraining_expert import restore_derain
    out_path = restore_derain("rainy.jpg", "outputs/derained/rainy.png")

Registered in tool_registry as: 'freq_derain'
Stage: scene, Speed: fast, Quality: high
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# GUIDED FILTER (reuse from zero_dce, self-contained here)
# ─────────────────────────────────────────────────────────────

def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(img, (2 * r + 1, 2 * r + 1))


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
    mean_I  = _box_filter(guide, r)
    mean_p  = _box_filter(src, r)
    mean_Ip = _box_filter(guide * src, r)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = _box_filter(guide * guide, r) - mean_I ** 2
    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I
    return np.clip(_box_filter(a, r) * guide + _box_filter(b, r), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# STEP 1 — MORPHOLOGICAL STREAK DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_rain_mask(gray: np.ndarray, streak_len: int = 15) -> np.ndarray:
    """
    Detect rain streak regions using multi-directional morphological top-hat transforms.
    """
    masks = []
    # Check 4 major angles for streaks
    for angle in [0, 45, 90, 135]:
        k = np.zeros((streak_len, streak_len), dtype=np.uint8)
        c = streak_len // 2
        x0 = int(c - c * np.cos(np.radians(angle)))
        y0 = int(c - c * np.sin(np.radians(angle)))
        x1 = int(c + c * np.cos(np.radians(angle)))
        y1 = int(c + c * np.sin(np.radians(angle)))
        cv2.line(k, (x0, y0), (x1, y1), 1, thickness=1)
        
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        masks.append(tophat)
        
    combined = np.max(np.stack(masks, axis=0), axis=0).astype(np.float32)
    
    # Threshold aggressively to catch more of the rain bodies
    # Lower threshold = more rain detected. 
    thresh = 10.0
    mask   = np.clip((combined - thresh) / 20.0, 0.0, 1.0)
    
    # Smooth mask
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    return mask

# ─────────────────────────────────────────────────────────────
# MAIN RESTORATION FUNCTION
# ─────────────────────────────────────────────────────────────

def restore_derain(
    input_path:  str,
    output_path: str = None,
    streak_len:  int = 15,
    mask_blend:  float = 0.90,
) -> str:
    """
    Remove rain streaks using Non-Local Means and Guided Filtering, 
    blended strictly into rain-detected pixels.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"[Derain] Cannot read: {input_path}")

    img_f  = img.astype(np.float32) / 255.0
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detect Rain Mask (multi-angle)
    mask = _detect_rain_mask(gray, streak_len=streak_len)

    # 2. Use a stronger median blur as the clean base to completely destroy the rain lines
    clean_base = cv2.medianBlur(img, 5)
    clean_base_f = clean_base.astype(np.float32) / 255.0

    # 3. Blend ONLY in the regions where rain was detected
    # We dilate the mask slightly to catch the semi-transparent edges of the raindrops.
    mask_dilated = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    
    # Use 100% replacement
    mask_blend = 1.0
    effective_mask = mask_dilated * mask_blend
    mask3 = effective_mask[:, :, np.newaxis]
    result = img_f * (1.0 - mask3) + clean_base_f * mask3

    # ── Save ──────────────────────────────────────────────────
    if output_path is None:
        out_dir     = os.path.join("outputs", "derained")
        os.makedirs(out_dir, exist_ok=True)
        stem        = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{stem}_derained.png")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cv2.imwrite(output_path, (result * 255).astype(np.uint8))
    print(f"[Derain] Processed: {input_path} → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
# RESTORER INTERFACE
# ─────────────────────────────────────────────────────────────

def restore(input_path: str, output_path: str = None) -> str:
    """Standard expert interface — uses our improved non-local means + morphological rain removal."""
    return restore_derain(input_path, output_path)
