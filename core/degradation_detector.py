import cv2
import numpy as np
import time

from core.dark_channel_prior import compute_haze_score


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL SIGNAL ESTIMATORS
# ─────────────────────────────────────────────────────────────

def _estimate_blur(gray):
    """
    Laplacian variance method.
    Sharp images have high variance. Blurry images have low variance.
    Returns score in [0, 1] where 1 = very blurry.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Natural sharp image: variance > 300
    # Severely blurry: < 20
    score = max(0.0, 1.0 - (laplacian_var / 300.0))
    return round(min(1.0, score), 3)


def _estimate_sr_need(width, height):
    """
    Resolution-based low-resolution detection.
    Returns score in [0, 1] where 1 = very low resolution.

    Note: SR score is capped at 0.75 so it cannot override strong scene
    signals (lowlight, noise) that have genuine degradation confidence.
    This prevents the resolution heuristic from masking real degradations.
    """
    pixel_count = width * height
    # Below 480p (640×480 = 307200) is clearly low-res
    # Below 720p (1280×720 = 921600) may benefit from SR
    reference = 921600.0
    score = max(0.0, 1.0 - (pixel_count / reference))
    # Cap at 0.75 so explicit degradation signals can override
    score = min(0.75, score)
    return round(score, 3)


def _estimate_lowlight(gray):
    """
    Mean brightness-based low-light detection.
    Returns score in [0, 1] where 1 = very dark.

    Calibrated so that gamma=3.5 darkened images (used in test data
    generator) score reliably above 0.5. Mean brightness of ~15-25
    should score 0.7-0.9.
    """
    brightness = float(np.mean(gray))
    # Very dark: < 30  → score ~0.8-1.0
    # Somewhat dark: 30-60 → score ~0.5-0.8
    # Acceptable: 60-100 → score ~0.2-0.5
    # Normal: > 100 → score ~0
    score = max(0.0, 1.0 - (brightness / 60.0))
    return round(min(1.0, score), 3)


def _estimate_jpeg(gray):
    """
    Detect JPEG blocking artifacts by comparing gradient magnitudes
    at 8-pixel boundaries vs the overall average gradient.
    Returns score in [0, 1] where 1 = strong JPEG artifacts.
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    # Collect column indices at 8-pixel boundaries (7, 15, 23, ...)
    col_indices = list(range(7, w - 1, 8))
    if len(col_indices) < 3:
        return 0.0

    next_cols = [c + 1 for c in col_indices]
    boundary_diffs = np.abs(
        gray_f[:, col_indices] - gray_f[:, next_cols]
    )
    avg_boundary = float(np.mean(boundary_diffs))

    # Compare against overall horizontal gradient
    all_h_diffs = np.abs(np.diff(gray_f, axis=1))
    avg_all = float(np.mean(all_h_diffs)) + 1e-6

    # If boundary gradient is ~2x the average, JPEG artifacts are present
    ratio = avg_boundary / avg_all
    score = max(0.0, min(1.0, (ratio - 1.0) / 3.0))
    return round(score, 3)


def _estimate_noise(gray):
    """
    Noise estimation via median-filter residual standard deviation.
    Returns score in [0, 1] where 1 = very noisy.
    """
    median = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    noise_std = float(np.std(residual))
    # Std of 25 maps to score 1.0 (visibly noisy)
    score = min(1.0, noise_std / 25.0)
    return round(score, 3)


def _estimate_haze_dcp(img_bgr: np.ndarray) -> float:
    """
    Haze confidence using Dark Channel Prior (C3).
    Returns score in [0, 1] where 1 = heavy haze/fog.
    DCP dark channel mean directly measures atmospheric scattering.
    """
    return compute_haze_score(img_bgr, patch_size=15)


def _estimate_rain(gray: np.ndarray) -> float:
    """
    Rain streak detection using morphological top-hat transform.

    Rain appears as near-vertical, bright, elongated streaks in images.
    A tall vertical structuring element (1px wide, 15px tall) responds
    strongly to rain and weakly to natural scene edges.

    Score calibration:
      - No rain       : top-hat mean < 3     → score ≈ 0.0
      - Light rain    : top-hat mean  3–10    → score ≈ 0.2–0.4
      - Moderate rain : top-hat mean 10–20    → score ≈ 0.4–0.7
      - Heavy rain    : top-hat mean > 20     → score ≈ 0.7–1.0

    Returns score in [0, 1] where 1 = heavy rain.
    """
    # Morphological top-hat with vertical kernel
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    mean_val = float(np.mean(tophat))
    # Normalise: 20 grey-level mean = score 1.0 (heavy rain)
    score    = min(1.0, mean_val / 20.0)
    return round(score, 3)


# ─────────────────────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────

def detect_degradation(image_path, verbose: bool = True):
    """
    Analyze image degradation and return a full confidence dict.

    Args:
        image_path : path to the image to analyze
        verbose    : if True (default), print the detection banner and score
                     breakdown. Pass False for silent re-detection calls (C2).

    Returns:
        {
            "primary":    "blur",
            "confidence": 0.83,
            "scores": {
                "blur":     0.83,
                "sr":       0.41,
                "jpeg":     0.18,
                "denoise":  0.22,
                "lowlight": 0.05
            }
        }
    """
    if verbose:
        print("\n===================================")
        print("     DEGRADATION DETECTOR ACTIVE")
        print("===================================\n")

    start_time = time.time()
    if verbose:
        print(f"[Detector] Loading image : {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        if verbose:
            print("[Detector] ERROR: Unable to load image.")
        return {
            "primary": "sr",
            "confidence": 0.5,
            "scores": {
                "blur": 0.0, "sr": 0.5, "jpeg": 0.0,
                "denoise": 0.0, "lowlight": 0.0
            }
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if verbose:
        print(f"[Detector] Resolution : {width} x {height}")

    # Compute all degradation signals
    blur_score     = _estimate_blur(gray)
    sr_score       = _estimate_sr_need(width, height)
    lowlight_score = _estimate_lowlight(gray)
    jpeg_score     = _estimate_jpeg(gray)
    denoise_score  = _estimate_noise(gray)
    haze_score     = _estimate_haze_dcp(image)    # C3: DCP physics signal
    rain_score     = _estimate_rain(gray)          # Phase 3: morphological

    scores = {
        "blur":     blur_score,
        "sr":       sr_score,
        "jpeg":     jpeg_score,
        "denoise":  denoise_score,
        "lowlight": lowlight_score,
        "haze":     haze_score,    # C3
        "rain":     rain_score,    # Phase 3
    }

    # C11: image_size for adaptive ranking
    image_size = {"width": width, "height": height, "pixels": width * height}

    # Primary = highest-scoring degradation
    primary    = max(scores, key=lambda k: scores[k])
    confidence = scores[primary]

    if verbose:
        print(f"[Detector] Blur Score     : {blur_score:.3f}")
        print(f"[Detector] SR Score       : {sr_score:.3f}")
        print(f"[Detector] JPEG Score     : {jpeg_score:.3f}")
        print(f"[Detector] Noise Score    : {denoise_score:.3f}")
        print(f"[Detector] Lowlight Score : {lowlight_score:.3f}")
        print(f"[Detector] Haze Score     : {haze_score:.3f}  [DCP]")
        print(f"[Detector] Rain Score     : {rain_score:.3f}  [morphological]")
        print(f"[Detector] ─────────────────────────────────────────")
        print(f"[Detector] Primary        : {primary}  (confidence: {confidence:.3f})")
        print(f"[Detector] Image Size     : {width}×{height}  ({width*height:,} px)")
        print(f"[Detector] Detection Time : {round(time.time() - start_time, 2)}s")

        print("\n===================================")
        print("     DEGRADATION ANALYSIS DONE")
        print("===================================\n")

    return {
        "primary":    primary,
        "confidence": round(confidence, 3),
        "scores":     scores,
        "image_size": image_size,  # C11
    }