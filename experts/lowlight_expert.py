import cv2
import numpy as np
import os
import time


def restore_lowlight(input_path):
    """
    Enhance a low-light image using CLAHE + adaptive gamma correction.

    Pipeline:
        1. Convert BGR → LAB color space
        2. Apply CLAHE to L (luminance) channel
        3. Apply adaptive gamma correction based on mean brightness
        4. Convert LAB → BGR and save

    No model download required.

    Returns:
        Path to enhanced output image, or None on failure.
    """
    print("\n===================================")
    print("     LOWLIGHT EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Lowlight Expert] Input Path : {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[Lowlight Expert] ERROR: Cannot load image: {input_path}")
        return None

    # ── Measure input brightness ──────────────────────────────
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_bright = float(np.mean(gray))
    print(f"[Lowlight Expert] Mean brightness (input) : {mean_bright:.1f}")

    # ── Step 1: Convert to LAB ────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # ── Step 2: CLAHE on L channel ────────────────────────────
    # Adaptive clip limit: darker images get stronger CLAHE
    clip_limit   = max(1.5, min(4.0, 200.0 / (mean_bright + 1.0)))
    clahe        = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced   = clahe.apply(l_channel)
    print(f"[Lowlight Expert] CLAHE clip limit         : {clip_limit:.2f}")

    # ── Step 3: Adaptive gamma correction ────────────────────
    # Gamma < 1 brightens the image; darker images get stronger gamma
    gamma        = max(0.4, min(0.85, mean_bright / 128.0))
    inv_gamma    = 1.0 / gamma
    table        = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    l_gamma      = cv2.LUT(l_enhanced, table)
    print(f"[Lowlight Expert] Gamma value               : {gamma:.3f}  (inv: {inv_gamma:.3f})")

    # ── Step 4: Merge and convert back to BGR ─────────────────
    lab_enhanced = cv2.merge([l_gamma, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── Measure output brightness ─────────────────────────────
    gray_out     = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    mean_out     = float(np.mean(gray_out))
    print(f"[Lowlight Expert] Mean brightness (output) : {mean_out:.1f}  (+{mean_out - mean_bright:.1f})")

    # ── Save output ───────────────────────────────────────────
    output_dir  = os.path.join("outputs", "lowlight")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_lowlight.png")
    cv2.imwrite(output_path, enhanced_bgr)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Lowlight Expert] Output saved      : {output_path}")
    print(f"[Lowlight Expert] Processing Time   : {elapsed}s")

    print("\n===================================")
    print("     LOWLIGHT EXPERT FINISHED")
    print("===================================\n")

    return output_path
