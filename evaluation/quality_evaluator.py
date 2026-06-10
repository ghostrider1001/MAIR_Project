"""
Quality Evaluator — SSIM + PSNR + Optional LPIPS (C6)
=======================================================
MAIR+ Contribution C6: Adds perceptual LPIPS distance to the quality
scoring pipeline so the reflection engine can detect over-smoothing and
GAN hallucinations that SSIM misses.

Two modes:
  evaluate_quality(orig, rest)        → float  (backward-compatible, SSIM+PSNR)
  evaluate_quality_full(orig, rest)   → dict   (SSIM, PSNR, LPIPS, composite)

LPIPS is fully optional — if `lpips` is not installed the evaluator falls
back to SSIM-only without crashing.

Composite score formula (when LPIPS available):
  composite = 0.5 × ssim_score + 0.5 × (1 − lpips_distance)

where ssim_score = 0.7 × SSIM + 0.3 × min(PSNR/50, 1.0)
"""

import cv2
import time
import numpy as np

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


# ─────────────────────────────────────────────────────────────
# LPIPS AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────

try:
    import torch
    import lpips as _lpips_lib
    _lpips_model  = _lpips_lib.LPIPS(net="alex", verbose=False)
    _lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _lpips_model.to(_lpips_device)
    _lpips_model.eval()
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
except Exception:
    HAS_LPIPS = False


# ─────────────────────────────────────────────────────────────
# IMAGE LOADING HELPER
# ─────────────────────────────────────────────────────────────

def _load_pair(original_path: str, restored_path: str):
    """
    Load two images and resize restored to match original if sizes differ.
    Returns (original_bgr, restored_bgr) or (None, None) on failure.
    """
    original = cv2.imread(original_path)
    restored = cv2.imread(restored_path)

    if original is None or restored is None:
        return None, None

    if original.shape != restored.shape:
        restored = cv2.resize(
            restored,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return original, restored


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────

def _compute_ssim_psnr(original_bgr, restored_bgr) -> tuple[float, float]:
    """Return (ssim_score, psnr_score) on grayscale images."""
    orig_g = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    rest_g = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
    ssim_score = float(ssim_fn(orig_g, rest_g))
    psnr_score = float(psnr_fn(orig_g, rest_g))
    return ssim_score, psnr_score


def compute_lpips(original_path: str, restored_path: str) -> float | None:
    """
    Compute LPIPS perceptual distance between two images.

    Returns:
        LPIPS distance in [0, 1] (lower = more perceptually similar), or
        None if lpips is not installed or computation fails.
    """
    if not HAS_LPIPS:
        return None

    try:
        import torch

        original, restored = _load_pair(original_path, restored_path)
        if original is None:
            return None

        def _to_tensor(bgr_img):
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(rgb.astype(np.float32) / 255.0)
            t   = t.permute(2, 0, 1).unsqueeze(0)        # 1×C×H×W
            t   = t * 2.0 - 1.0                           # normalize to [-1, 1]
            return t.to(_lpips_device)

        with torch.no_grad():
            dist = _lpips_model(_to_tensor(original), _to_tensor(restored))

        return float(dist.item())

    except Exception as e:
        print(f"[QualityEval] LPIPS computation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# FULL QUALITY DICT (C6)
# ─────────────────────────────────────────────────────────────

def evaluate_quality_full(original_path: str, restored_path: str) -> dict:
    """
    Compute all available quality metrics and return a structured dict.

    Returns:
        {
            "ssim":            float,
            "psnr":            float,
            "lpips":           float | None,
            "ssim_score":      float,   # 0.7×SSIM + 0.3×min(PSNR/50,1)
            "composite_score": float,   # 0.5×ssim_score + 0.5×(1-lpips) if available
            "quality_label":   str,
            "has_lpips":       bool,
        }
    """
    original, restored = _load_pair(original_path, restored_path)

    if original is None or restored is None:
        return {
            "ssim": None, "psnr": None, "lpips": None,
            "ssim_score": None, "composite_score": None,
            "quality_label": "N/A", "has_lpips": HAS_LPIPS,
        }

    ssim_val, psnr_val = _compute_ssim_psnr(original, restored)

    # Baseline SSIM+PSNR score (backward-compatible)
    ssim_score = round(ssim_val * 0.7 + min(psnr_val / 50.0, 1.0) * 0.3, 4)

    # LPIPS
    lpips_val = compute_lpips(original_path, restored_path)

    # Composite
    if lpips_val is not None:
        lpips_component   = 1.0 - lpips_val          # higher = better
        composite_score   = round(0.5 * ssim_score + 0.5 * lpips_component, 4)
    else:
        composite_score   = ssim_score               # fall back to SSIM-only

    composite_score = max(0.0, min(1.0, composite_score))

    # Label
    if composite_score >= 0.85:   label = "Excellent"
    elif composite_score >= 0.70: label = "Good"
    elif composite_score >= 0.50: label = "Moderate"
    else:                         label = "Poor"

    return {
        "ssim":            round(ssim_val,  4),
        "psnr":            round(psnr_val,  2),
        "lpips":           round(lpips_val, 4) if lpips_val is not None else None,
        "ssim_score":      ssim_score,
        "composite_score": composite_score,
        "quality_label":   label,
        "has_lpips":       HAS_LPIPS,
    }


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE SCALAR EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_quality(original_path: str, restored_path: str) -> float:
    """
    Compute a single composite quality score (backward-compatible).

    When LPIPS is available, returns:
        0.5 × ssim_score + 0.5 × (1 − lpips)
    Otherwise:
        0.7 × SSIM + 0.3 × min(PSNR/50, 1.0)

    Returns float in [0, 1] — higher is better.
    Prints a verbose evaluation summary (kept for traceability).
    """
    print("\n===================================")
    print("      QUALITY EVALUATOR ACTIVE")
    print("===================================\n")

    start_time = time.time()
    print("[Quality Evaluator] Loading images...")

    result = evaluate_quality_full(original_path, restored_path)

    if result["ssim"] is None:
        print("[Quality Evaluator] ERROR: Unable to load images.")
        print("\n===================================")
        print("      QUALITY EVALUATION DONE")
        print("===================================\n")
        return 0.0

    print(f"[Quality Evaluator] SSIM           : {result['ssim']:.4f}")
    print(f"[Quality Evaluator] PSNR           : {result['psnr']:.2f} dB")
    if result["lpips"] is not None:
        print(f"[Quality Evaluator] LPIPS          : {result['lpips']:.4f}  (lower=better)")
    else:
        print("[Quality Evaluator] LPIPS          : not available (pip install lpips)")

    score = result["composite_score"]
    label = result["quality_label"]
    score_type = "Composite (SSIM+LPIPS)" if result["has_lpips"] and result["lpips"] is not None else "SSIM+PSNR"
    print(f"[Quality Evaluator] Final Score    : {score}  [{score_type}]")
    print(f"[Quality Evaluator] Quality Level  : {label}")
    print(f"[Quality Evaluator] Eval Time      : {round(time.time() - start_time, 2)}s")

    print("\n===================================")
    print("      QUALITY EVALUATION DONE")
    print("===================================\n")

    return score