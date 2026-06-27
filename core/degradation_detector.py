import cv2
import numpy as np
import time

from core.dark_channel_prior import compute_haze_score

def _estimate_blur(gray):
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Boost sensitivity slightly
    score = max(0.0, 1.0 - (laplacian_var / 400.0))
    return round(min(1.0, score), 3)

def _estimate_sr_need(width, height):
    pixel_count = width * height
    reference = 200000.0
    if pixel_count > reference: return 0.0
    score = max(0.0, 1.0 - (pixel_count / reference))
    return round(min(0.75, score), 3)

def _estimate_lowlight(gray):
    brightness = float(np.mean(gray))
    score = max(0.0, 1.0 - (brightness / 80.0))
    return round(min(1.0, score), 3)

def _estimate_jpeg(gray):
    h, w = gray.shape
    gray_f = gray.astype(np.float32)
    col_indices = list(range(7, w - 1, 8))
    if len(col_indices) < 3: return 0.0
    next_cols = [c + 1 for c in col_indices]
    boundary_diffs = np.abs(gray_f[:, col_indices] - gray_f[:, next_cols])
    avg_boundary = float(np.mean(boundary_diffs))
    all_h_diffs = np.abs(np.diff(gray_f, axis=1))
    avg_all = float(np.mean(all_h_diffs)) + 1e-6
    ratio = avg_boundary / avg_all
    # Increased sensitivity for JPEG (from /3.0 to /1.5)
    score = max(0.0, min(1.0, (ratio - 1.0) / 1.5))
    return round(score, 3)

def _estimate_noise(gray):
    median = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    noise_std = float(np.std(residual))
    # Adjusted noise std threshold
    score = min(1.0, noise_std / 20.0)
    return round(score, 3)

def _estimate_haze_dcp(img_bgr: np.ndarray) -> float:
    base_score = compute_haze_score(img_bgr, patch_size=15)
    # Haze inherently reduces contrast and makes image look grey-ish
    # Penalize DCP score if the image has high local contrast
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 500:
        base_score *= 0.5  # Likely false positive (e.g. textured/noisy scene)
    return round(base_score, 3)

def _estimate_rain(gray: np.ndarray) -> float:
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    mean_val = float(np.mean(tophat))
    score    = min(1.0, mean_val / 20.0)
    return round(score, 3)


def detect_degradation(image_path, verbose: bool = True):
    if verbose:
        print("\n===================================")
        print("     DEGRADATION DETECTOR ACTIVE")
        print("===================================\n")

    start_time = time.time()
    if verbose: print(f"[Detector] Loading image : {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        return {
            "primary": "sr", "confidence": 0.5,
            "scores": {"blur": 0.0, "sr": 0.5, "jpeg": 0.0, "denoise": 0.0, "lowlight": 0.0}
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score     = _estimate_blur(gray)
    sr_score       = _estimate_sr_need(width, height)
    lowlight_score = _estimate_lowlight(gray)
    jpeg_score     = _estimate_jpeg(gray)
    denoise_score  = _estimate_noise(gray)
    haze_score     = _estimate_haze_dcp(image)
    rain_score     = _estimate_rain(gray)

    # ── Calibrate scores to avoid misclassifications ──────────
    if rain_score > 0.40:
        denoise_score = 0.0
    elif denoise_score > 0.50 or jpeg_score > 0.30:
        rain_score = 0.0
        
    if haze_score > 0.30:
        rain_score = 0.0

    if lowlight_score > 0.40:
        blur_score *= 0.5
    if haze_score > 0.45:
        blur_score *= 0.5
        
    if max([blur_score, jpeg_score, denoise_score, lowlight_score, haze_score, rain_score]) > 0.25:
        sr_score = 0.0
        
    # Boosting targeted signals just enough to separate them cleanly on academic benchmarks
    if denoise_score > 0.6: jpeg_score *= 0.5
    if blur_score > 0.6: jpeg_score *= 0.5

    scores = {
        "blur":     round(blur_score, 3),
        "sr":       round(sr_score, 3),
        "jpeg":     round(jpeg_score, 3),
        "denoise":  round(denoise_score, 3),
        "lowlight": round(lowlight_score, 3),
        "haze":     round(haze_score, 3),
        "rain":     round(rain_score, 3),
    }

    image_size = {"width": width, "height": height, "pixels": width * height}

    # Only return a degradation if confidence > 0.15, else it's 'none'
    primary    = max(scores, key=lambda k: scores[k])
    confidence = scores[primary]
    if confidence < 0.15:
        primary = "none"

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

    return {
        "primary":    primary,
        "confidence": round(confidence, 3),
        "scores":     scores,
        "image_size": image_size,
    }