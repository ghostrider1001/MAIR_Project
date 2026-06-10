"""
Dark Channel Prior (DCP) — Physics-Grounded Haze Module (C3)
=============================================================
MAIR+ Contribution C3.

Implements He et al. (2011) "Single Image Haze Removal Using Dark Channel Prior."
Used in two places:
  1. core/degradation_detector.py  — `compute_haze_score()` as a haze confidence signal
  2. experts/dehaze_expert.py      — full DCP restoration pipeline

Algorithm summary:
  1. Dark channel: J_dark(x) = min_{y∈Ω(x)} min_c J^c(y)
     In hazy images, the dark channel has high values (fog/haze lifts dark pixels).
  2. Atmospheric light A: estimated as the intensity at the top 0.1% of dark channel pixels.
  3. Transmission map t(x) = 1 − ω × dark_channel(I/A)
     where ω=0.95 preserves a small amount of haze for depth realism.
  4. Guided filter refinement (or bilateral fallback) to reduce halo artifacts.
  5. Scene radiance: J = (I − A) / max(t, t_min) + A

Haze score:
  The mean of the dark channel normalized to [0,1].
  Clear images: dark channel mean ≈ 0.0–0.05
  Hazy images:  dark channel mean ≈ 0.3–0.8

References:
  He, K., Sun, J., & Tang, X. (2011). Single image haze removal using dark channel prior.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(12), 2341-2353.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# GUIDED FILTER AVAILABILITY
# ─────────────────────────────────────────────────────────────

try:
    import cv2.ximgproc
    _HAS_GUIDED_FILTER = True
except ImportError:
    _HAS_GUIDED_FILTER = False


# ─────────────────────────────────────────────────────────────
# CORE DCP OPERATIONS
# ─────────────────────────────────────────────────────────────

def compute_dark_channel(img_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.

    For each pixel, the dark channel value is the minimum pixel
    value in a local patch across all color channels.

    Args:
        img_bgr    : uint8 BGR image [H×W×3]
        patch_size : size of the minimum filter window (default 15)

    Returns:
        dark_channel: float32 array [H×W] in [0, 1]
    """
    img_f     = img_bgr.astype(np.float32) / 255.0
    min_chan  = np.min(img_f, axis=2)           # channel-wise minimum → [H×W]
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dark_chan = cv2.erode(min_chan, kernel)      # spatial minimum (erosion)
    return dark_chan


def estimate_atmospheric_light(
    img_bgr:      np.ndarray,
    dark_channel: np.ndarray,
    top_fraction: float = 0.001,
) -> np.ndarray:
    """
    Estimate atmospheric light A from the brightest pixels in the dark channel.

    Args:
        img_bgr      : uint8 BGR image [H×W×3]
        dark_channel : float32 dark channel [H×W]
        top_fraction : fraction of pixels to consider (default 0.1%)

    Returns:
        atmospheric_light: float32 array [3] (one value per channel)
    """
    h, w       = dark_channel.shape
    num_pixels = h * w
    n_top      = max(1, int(num_pixels * top_fraction))

    # Flatten and find top-n indices by dark channel intensity
    flat_dark  = dark_channel.flatten()
    top_idx    = np.argsort(flat_dark)[-n_top:]   # highest dark channel values

    # Extract corresponding pixels from the original image
    img_f      = img_bgr.astype(np.float32) / 255.0
    flat_img   = img_f.reshape(-1, 3)
    top_pixels = flat_img[top_idx, :]             # [n_top × 3]

    # Atmospheric light = mean of the brightest pixels
    atm_light  = np.mean(top_pixels, axis=0)
    return atm_light.astype(np.float32)


def estimate_transmission(
    img_bgr:    np.ndarray,
    atm_light:  np.ndarray,
    patch_size: int   = 15,
    omega:      float = 0.95,
) -> np.ndarray:
    """
    Estimate the transmission map t(x).

    t(x) = 1 − ω × dark_channel(I^c(y) / A^c)

    Args:
        img_bgr    : uint8 BGR image [H×W×3]
        atm_light  : float32 atmospheric light [3]
        patch_size : dark channel patch size (default 15)
        omega      : haze preservation factor (default 0.95 = almost fully dehaze)

    Returns:
        transmission: float32 array [H×W] in [0, 1]
    """
    img_f      = img_bgr.astype(np.float32) / 255.0
    # Normalize by atmospheric light
    norm_img   = img_f / (atm_light + 1e-6)       # prevent division by zero
    # Dark channel of normalized image
    norm_dark  = compute_dark_channel(
        (np.clip(norm_img, 0, 1) * 255).astype(np.uint8),
        patch_size=patch_size,
    )
    transmission = 1.0 - omega * norm_dark
    return transmission.astype(np.float32)


def refine_transmission(
    img_bgr:      np.ndarray,
    transmission: np.ndarray,
    radius:       int   = 60,
    eps:          float = 1e-3,
) -> np.ndarray:
    """
    Refine the transmission map using guided filter or bilateral fallback.

    Guided filter removes block artifacts while preserving edges (no halos).
    Falls back to bilateral filter if cv2.ximgproc is unavailable.

    Args:
        img_bgr      : uint8 BGR image [H×W×3] — used as guide
        transmission : float32 transmission map [H×W]
        radius       : filter radius (guided) or diameter (bilateral)
        eps          : regularization term for guided filter

    Returns:
        Refined float32 transmission map [H×W]
    """
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if _HAS_GUIDED_FILTER:
        # Guided filter — best quality, no halos
        t_u8      = (transmission * 255).astype(np.uint8)
        refined   = cv2.ximgproc.guidedFilter(
            guide=guide, src=t_u8, radius=radius, eps=eps * 255 * 255
        )
        refined   = refined.astype(np.float32) / 255.0
    else:
        # Bilateral filter fallback — adequate quality
        t_u8      = (transmission * 255).astype(np.uint8)
        refined   = cv2.bilateralFilter(
            t_u8, d=radius // 3, sigmaColor=60, sigmaSpace=60
        )
        refined   = refined.astype(np.float32) / 255.0

    return np.clip(refined, 0.0, 1.0)


def recover_scene_radiance(
    img_bgr:      np.ndarray,
    atm_light:    np.ndarray,
    transmission: np.ndarray,
    t_min:        float = 0.10,
) -> np.ndarray:
    """
    Recover the haze-free scene radiance.

    J(x) = (I(x) − A) / max(t(x), t_min) + A

    Args:
        img_bgr      : uint8 BGR image [H×W×3]
        atm_light    : float32 atmospheric light [3]
        transmission : float32 transmission map [H×W]
        t_min        : minimum transmission to prevent division by zero

    Returns:
        Dehazed uint8 BGR image [H×W×3]
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    t     = np.maximum(transmission, t_min)[:, :, np.newaxis]  # [H×W×1]
    J     = (img_f - atm_light) / t + atm_light
    J     = np.clip(J, 0.0, 1.0)
    return (J * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# HAZE SCORE (used by degradation detector — C3)
# ─────────────────────────────────────────────────────────────

def compute_haze_score(img_bgr: np.ndarray, patch_size: int = 15) -> float:
    """
    Compute a haze confidence score for an image using the dark channel prior.

    Score interpretation:
        0.0 – 0.05 : clear image (very low haze)
        0.05 – 0.20: mild atmospheric scattering
        0.20 – 0.50: moderate haze — DCP restoration will help
        0.50 – 1.00: heavy haze / fog

    Returns:
        float in [0, 1] where 1 = maximally hazy
    """
    dark_chan = compute_dark_channel(img_bgr, patch_size=patch_size)
    score     = float(np.mean(dark_chan))   # mean of dark channel as haze proxy
    return round(min(1.0, score), 3)


# ─────────────────────────────────────────────────────────────
# FULL DCP PIPELINE (used by dehaze expert — C1)
# ─────────────────────────────────────────────────────────────

def dehaze_dcp(
    img_bgr:    np.ndarray,
    patch_size: int   = 15,
    omega:      float = 0.95,
    t_min:      float = 0.10,
    refine:     bool  = True,
) -> np.ndarray:
    """
    Full single-image dehazing pipeline using Dark Channel Prior.

    Steps:
      1. Compute dark channel
      2. Estimate atmospheric light
      3. Estimate transmission
      4. Refine transmission (guided / bilateral filter)
      5. Recover scene radiance

    Args:
        img_bgr    : uint8 BGR input image
        patch_size : dark channel patch size
        omega      : haze preservation factor (0.95 = almost complete removal)
        t_min      : minimum allowed transmission
        refine     : whether to apply guided filter refinement (recommended)

    Returns:
        Dehazed uint8 BGR image
    """
    dark_chan = compute_dark_channel(img_bgr, patch_size=patch_size)
    atm_light = estimate_atmospheric_light(img_bgr, dark_chan)
    trans_map = estimate_transmission(img_bgr, atm_light, patch_size=patch_size, omega=omega)

    if refine:
        trans_map = refine_transmission(img_bgr, trans_map)

    dehazed   = recover_scene_radiance(img_bgr, atm_light, trans_map, t_min=t_min)
    return dehazed
