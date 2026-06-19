"""
Zero-DCE Style Lowlight Expert
================================
MAIR+ Expert — Adaptive Curve-Based Low-Light Enhancement

Implements the core mathematical insight from Zero-DCE (2020) and Zero-DCE++ (2021):
    LE(x, α) = x + α · x · (1 − x)      [iterative self-adjusting curve]

Where α ∈ [−1, 1] controls enhancement strength per pixel.

Unlike the paper (which trains a CNN to estimate α), this module estimates α
analytically from the image's local illumination statistics — giving quality
that is substantially better than CLAHE while requiring zero pretrained weights.

Key advantages over CLAHE (current):
  • Per-channel curve (not histogram equalization) → no colour clipping
  • Spatially adaptive α map → bright regions not blown out
  • Guided filter refinement → edges preserved, halos eliminated
  • Multi-iteration curve → controllable, non-destructive enhancement

Architecture:
  1. Estimate illumination map L (max-channel, guided-filtered)
  2. Compute α map = clip(-k · (L - target), -1, 0)  [negative α = brighten]
  3. Apply iterative curve: x ← x + α · x · (1 − x)   [n_iters times]
  4. Mild colour saturation recovery (DCE curves slightly desaturate)
  5. Clip and return

Usage:
    from experts.zero_dce_expert import restore_zero_dce
    out_path = restore_zero_dce("dark.jpg", "outputs/enhanced/dark.png")

Registered in tool_registry as: 'zero_dce_lowlight'
Stage: scene, Speed: fast, Quality: high
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# GUIDED FILTER (lightweight, no ximgproc required)
# ─────────────────────────────────────────────────────────────

def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Fast box filter using cv2.blur."""
    ksize = 2 * r + 1
    return cv2.blur(img, (ksize, ksize))


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 16, eps: float = 1e-3) -> np.ndarray:
    """
    Guided image filter (He et al., 2013).
    guide: [H×W] float32 in [0,1]
    src  : [H×W] float32 in [0,1]
    Returns filtered version of src, edge-aware using guide.
    """
    mean_I  = _box_filter(guide, r)
    mean_p  = _box_filter(src,   r)
    mean_Ip = _box_filter(guide * src, r)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(guide * guide, r)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# ILLUMINATION ESTIMATION
# ─────────────────────────────────────────────────────────────

def _estimate_illumination(img_f: np.ndarray, radius: int = 16) -> np.ndarray:
    """
    Estimate per-pixel illumination map from max-channel image.
    Returns a [H×W] float32 map in [0, 1].
    """
    # Max channel gives a good luminance proxy for natural images
    illum = np.max(img_f, axis=2)                      # [H×W]
    # Guided filter with itself → smooth illumination, preserve edges
    illum_smooth = _guided_filter(illum, illum, r=radius, eps=1e-2)
    return illum_smooth


# ─────────────────────────────────────────────────────────────
# DCE CURVE APPLICATION
# ─────────────────────────────────────────────────────────────

def _apply_dce_curve(
    img_f:   np.ndarray,
    alpha:   np.ndarray,
    n_iters: int = 8,
) -> np.ndarray:
    """
    Apply iterative self-adjusting enhancement curve.
    Each iteration: x ← x + α · x · (1 − x)

    With negative α this brightens dark pixels and leaves bright ones alone.
    n_iters controls total enhancement strength.
    """
    x = img_f.copy()
    for _ in range(n_iters):
        # Broadcast alpha [H×W] over channels [H×W×3]
        x = x + alpha[:, :, np.newaxis] * x * (1.0 - x)
    return np.clip(x, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# COLOUR SATURATION RECOVERY
# ─────────────────────────────────────────────────────────────

def _recover_saturation(
    original:  np.ndarray,
    enhanced:  np.ndarray,
    strength:  float = 0.3,
) -> np.ndarray:
    """
    The DCE curve slightly desaturates (pushes toward white).
    Recover a fraction of the original image's saturation by mixing
    in a small amount of a saturation-boosted version.
    """
    hsv_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_enh  = cv2.cvtColor((enhanced  * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    # Blend saturation channels
    hsv_enh[:, :, 1] = np.clip(
        hsv_enh[:, :, 1] * (1.0 - strength) + hsv_orig[:, :, 1] * strength,
        0, 255
    )
    recovered = cv2.cvtColor(hsv_enh.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return np.clip(recovered, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# MAIN RESTORATION FUNCTION
# ─────────────────────────────────────────────────────────────

def restore_zero_dce(
    input_path:   str,
    output_path:  str | None = None,
    target_mean:  float = 0.45,   # target brightness after enhancement
    n_iters:      int   = 8,      # DCE curve iterations
    guide_radius: int   = 16,     # guided filter spatial radius
) -> str:
    """
    Enhance a low-light image using Zero-DCE-style adaptive curves.

    Args:
        input_path   : path to input (dark) image
        output_path  : where to save the result (auto-generated if None)
        target_mean  : target illumination mean (0.45 = natural daylight look)
        n_iters      : number of curve iterations (more = brighter, 8 is good)
        guide_radius : guided filter radius in pixels

    Returns:
        Path to the saved output image.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"[ZeroDCE] Cannot read: {input_path}")

    img_f = img.astype(np.float32) / 255.0           # [H×W×3] in [0,1]

    # ── 1. Estimate illumination ──────────────────────────────
    illum = _estimate_illumination(img_f, radius=guide_radius)  # [H×W]

    # ── 2. Compute per-pixel alpha map ────────────────────────
    # alpha < 0 → brightening; magnitude proportional to darkness
    # k scales the sensitivity: k=1.5 gives strong enhancement
    current_mean = float(illum.mean())
    k = (target_mean - current_mean) / max(current_mean * (1.0 - current_mean), 1e-4)
    k = np.clip(k, 0.0, 3.0)          # cap enhancement to avoid blown-out

    # Per-pixel: darker pixels get stronger alpha
    # In LE(x, a) = x + a * x * (1 - x), a > 0 brightens the pixel!
    alpha = k * (1.0 - illum)        # [H×W], values >= 0
    alpha = np.clip(alpha, 0.0, 1.0)

    # ── 3. Apply iterative DCE curve ─────────────────────────
    enhanced = _apply_dce_curve(img_f, alpha, n_iters=n_iters)

    # ── 4. Refine illumination with guided filter ─────────────
    # Edge-aware smoothing of the luminance channel
    enhanced_lum = np.max(enhanced, axis=2)
    refined_lum  = _guided_filter(enhanced_lum, illum, r=guide_radius // 2, eps=1e-3)

    # Scale each channel by the ratio of refined to raw luminance.
    # This applies the smoother illumination estimate back to the output,
    # suppressing halo over-enhancement at bright edges (previously refined_lum
    # was computed but never used — dead code).
    lum_ratio = np.clip(refined_lum / (enhanced_lum + 1e-6), 0.5, 1.5)
    enhanced_refined = np.clip(enhanced * lum_ratio[:, :, np.newaxis], 0.0, 1.0)

    # Soft blend between guided-refined and original
    blend  = 0.7
    output = enhanced_refined * blend + img_f * (1.0 - blend)
    output = np.clip(output, 0.0, 1.0)

    # ── 5. Colour saturation recovery ────────────────────────
    output = _recover_saturation(img_f, output, strength=0.25)

    # ── 6. Save ───────────────────────────────────────────────
    if output_path is None:
        out_dir     = os.path.join("outputs", "lowlight")
        os.makedirs(out_dir, exist_ok=True)
        stem        = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{stem}_zero_dce.png")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cv2.imwrite(output_path, (output * 255).astype(np.uint8))
    print(f"[ZeroDCE] Enhanced: {input_path} → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
# RESTORER INTERFACE (called by tool_registry / scheduler)
# ─────────────────────────────────────────────────────────────

def restore(input_path: str, output_path: str | None = None) -> str:
    """Standard expert interface — wraps restore_zero_dce."""
    return restore_zero_dce(input_path, output_path)
