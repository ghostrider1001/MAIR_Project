import cv2
import numpy as np
import os
from datetime import datetime


def save_comparison(
    original_path,
    restored_path,
    quality_scores=None,
    output_dir="outputs",
    label="result",
):
    """
    Save a side-by-side before/after comparison image.

    The comparison shows:
      - Left panel:  Original image with label
      - Right panel: Restored image with quality scores

    Args:
        original_path  : path to the original (degraded) image
        restored_path  : path to the restored image
        quality_scores : dict with 'ssim' and 'psnr' keys (optional)
        output_dir     : directory to save comparison PNG
        label          : short label for the filename (e.g. "Phase1")

    Returns:
        Absolute path to the saved comparison PNG, or None on error.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load images ──────────────────────────────────────────
    original = cv2.imread(original_path)
    restored = cv2.imread(restored_path)

    if original is None:
        print(f"[Visualizer] Cannot load original: {original_path}")
        return None
    if restored is None:
        print(f"[Visualizer] Cannot load restored: {restored_path}")
        return None

    orig_h, orig_w = original.shape[:2]

    # ── Resize restored for side-by-side display ─────────────
    # SR output is 4× larger — downscale for display only
    if restored.shape[:2] != (orig_h, orig_w):
        display_restored = cv2.resize(
            restored, (orig_w, orig_h), interpolation=cv2.INTER_AREA
        )
    else:
        display_restored = restored.copy()

    # ── Build label bars ─────────────────────────────────────
    bar_h  = 64
    font   = cv2.FONT_HERSHEY_SIMPLEX
    dark   = (28,  28,  28)
    white  = (230, 230, 230)
    green  = (72,  210, 100)
    yellow = (60,  200, 220)

    orig_bar = np.full((bar_h, orig_w, 3), dark, dtype=np.uint8)
    rest_bar = np.full((bar_h, orig_w, 3), dark, dtype=np.uint8)

    # Original label
    cv2.putText(
        orig_bar, "ORIGINAL (Input)",
        (14, 42), font, 0.75, white, 2, cv2.LINE_AA
    )

    # Restored label with scores
    if quality_scores:
        ssim_v = quality_scores.get("ssim", 0.0)
        psnr_v = quality_scores.get("psnr", 0.0)
        rest_label = f"RESTORED  |  SSIM: {ssim_v:.4f}  |  PSNR: {psnr_v:.2f} dB"
    else:
        rest_label = "RESTORED (Output)"

    cv2.putText(
        rest_bar, rest_label,
        (14, 42), font, 0.68, green, 2, cv2.LINE_AA
    )

    # ── Assemble columns ─────────────────────────────────────
    orig_col = np.vstack([orig_bar, original])
    rest_col = np.vstack([rest_bar, display_restored])

    # Thin vertical divider
    divider = np.full((orig_h + bar_h, 5, 3), 90, dtype=np.uint8)

    comparison = np.hstack([orig_col, divider, rest_col])

    # ── Footer bar with metadata ─────────────────────────────
    footer_h   = 36
    footer     = np.full((footer_h, comparison.shape[1], 3), 18, dtype=np.uint8)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_txt = f"MAIR+  |  {label}  |  {timestamp}  |  Input: {os.path.basename(original_path)}"
    cv2.putText(
        footer, footer_txt,
        (14, 24), font, 0.50, (130, 130, 130), 1, cv2.LINE_AA
    )
    comparison = np.vstack([comparison, footer])

    # ── Save ─────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename    = f"comparison_{label}_{ts}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, comparison)

    print(f"[Visualizer] Comparison saved → {output_path}")
    return output_path
