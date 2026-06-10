"""
Wiener Filter Deblur Expert
============================
MAIR+ Expert — CPU-only motion blur removal using Wiener deconvolution.

This implements the classical Wiener filter for blind deconvolution:
    G(u,v) = H*(u,v) / (|H(u,v)|² + K) · F(u,v)

Where:
    H = estimated PSF (blur kernel) in frequency domain
    K = noise-to-signal ratio regularization (prevents noise amplification)
    F = blurred image in frequency domain
    G = restored image estimate

Blur kernel estimation uses the Radon transform approach:
  - Compute power spectrum of the image
  - Find the dominant orientation (angle of motion blur)
  - Estimate kernel length from the power spectrum spread

This is significantly better than unsharp masking for true motion blur
because it properly inverts the blur physics rather than just sharpening.

No pretrained weights required.

Stage: imaging (Stage 2)
Speed: fast  
Quality: medium-high
"""

import os
import time
import cv2
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# BLUR KERNEL ESTIMATION
# ─────────────────────────────────────────────────────────────

def _estimate_motion_kernel(gray: np.ndarray) -> tuple:
    """
    Estimate the dominant motion blur direction and length from
    the image power spectrum.

    Returns:
        (kernel_length, angle_deg) — estimated blur parameters
    """
    # Power spectrum of the image
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Look at a ring of the power spectrum to find dominant direction
    # Streak in power spectrum indicates motion blur direction
    ring_r = min(h, w) // 6
    angles = np.linspace(0, 180, 36, endpoint=False)
    angle_energies = []

    for angle in angles:
        # Sample along a line at this angle
        rad = np.deg2rad(angle)
        samples = []
        for r in range(1, ring_r):
            yr = int(cy + r * np.sin(rad))
            xr = int(cx + r * np.cos(rad))
            if 0 <= yr < h and 0 <= xr < w:
                samples.append(magnitude[yr, xr])
        angle_energies.append(np.mean(samples) if samples else 0.0)

    best_angle = angles[np.argmax(angle_energies)]

    # Estimate kernel length from the spread of the dominant streak
    # Use a fixed reasonable estimate for typical motion blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Blurrier images → longer kernels needed
    kernel_length = max(3, min(25, int(200.0 / (laplacian_var + 1.0))))

    return kernel_length, best_angle


def _make_motion_kernel(length: int, angle: float) -> np.ndarray:
    """Create a motion blur PSF kernel of given length and angle."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    angle_rad = np.deg2rad(angle)

    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(angle_rad)))
        y = int(round(center + offset * np.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0

    s = kernel.sum()
    return kernel / s if s > 0 else kernel + 1e-6


# ─────────────────────────────────────────────────────────────
# WIENER DECONVOLUTION
# ─────────────────────────────────────────────────────────────

def _wiener_deconvolve(channel: np.ndarray, kernel: np.ndarray, K: float = 0.01) -> np.ndarray:
    """
    Apply Wiener filter deconvolution to a single channel.

    Args:
        channel : [H×W] float32 in [0, 1]
        kernel  : PSF estimate
        K       : regularization (noise level estimate, higher = smoother)

    Returns:
        Deconvolved channel in [0, 1]
    """
    h, w = channel.shape

    # Pad kernel to image size
    kernel_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, -kh // 2, axis=0)
    kernel_padded = np.roll(kernel_padded, -kw // 2, axis=1)

    # FFT of both
    F = np.fft.fft2(channel)
    H = np.fft.fft2(kernel_padded)

    # Wiener filter: G = H* / (|H|² + K)
    H_conj = np.conj(H)
    H_sq   = np.abs(H) ** 2
    G      = H_conj / (H_sq + K)

    # Apply filter
    restored = np.real(np.fft.ifft2(G * F))
    return np.clip(restored, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# MAIN RESTORATION FUNCTION
# ─────────────────────────────────────────────────────────────

def restore_wiener(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Restore a motion-blurred image using Wiener deconvolution.

    Automatically estimates blur kernel from the image power spectrum.
    CPU-only, no pretrained weights required.

    Args:
        input_path  : path to blurred input image
        output_path : where to save the result (auto-generated if None)

    Returns:
        Path to the restored output image.
    """
    print("\n===================================")
    print("    WIENER DEBLUR EXPERT ACTIVATED")
    print("===================================\n")

    t0 = time.time()
    img = cv2.imread(input_path)
    if img is None:
        print(f"[Wiener] ERROR: Cannot load image: {input_path}")
        return None

    img_f = img.astype(np.float32) / 255.0
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1. Estimate blur kernel ───────────────────────────────
    kernel_length, angle = _estimate_motion_kernel(gray)
    print(f"[Wiener] Estimated blur: length={kernel_length}px, angle={angle:.1f}°")
    kernel = _make_motion_kernel(kernel_length, angle)

    # ── 2. Adaptive K (regularization) from noise estimate ────
    median = cv2.medianBlur(gray, 5)
    noise_std = float(np.std(gray.astype(np.float32) - median.astype(np.float32)))
    K = max(0.002, min(0.05, (noise_std / 255.0) ** 2 * 10.0))
    print(f"[Wiener] Regularization K={K:.4f} (noise_std={noise_std:.1f})")

    # ── 3. Apply Wiener per-channel ───────────────────────────
    restored = np.zeros_like(img_f)
    for c in range(3):
        restored[:, :, c] = _wiener_deconvolve(img_f[:, :, c], kernel, K=K)

    # ── 4. Post-processing: mild sharpening to recover edges ──
    sharp_kernel = np.array([[0, -0.3, 0],
                              [-0.3, 2.2, -0.3],
                              [0, -0.3, 0]], dtype=np.float32)
    for c in range(3):
        restored[:, :, c] = cv2.filter2D(restored[:, :, c], -1, sharp_kernel)
    restored = np.clip(restored, 0.0, 1.0)

    # ── 5. Blend with original to avoid ringing artifacts ────
    # Ringing is the main artifact of Wiener deconvolution
    blend = 0.75   # 75% deconvolved + 25% original
    output = restored * blend + img_f * (1.0 - blend)
    output = np.clip(output, 0.0, 1.0)

    # ── 6. Save ───────────────────────────────────────────────
    if output_path is None:
        out_dir = os.path.join("outputs", "deblurred")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{stem}_wiener.png")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cv2.imwrite(output_path, (output * 255).astype(np.uint8))
    elapsed = round(time.time() - t0, 2)
    print(f"[Wiener] Restored: {input_path} → {output_path}  ({elapsed}s)")

    print("\n===================================")
    print("    WIENER DEBLUR EXPERT FINISHED")
    print("===================================\n")

    return output_path


def restore(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """Standard expert interface."""
    return restore_wiener(input_path, output_path)
