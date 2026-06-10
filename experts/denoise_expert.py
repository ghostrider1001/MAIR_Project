import cv2
import numpy as np
import os
import time


def restore_denoise(input_path):
    """
    Denoise an image using OpenCV Non-Local Means (NLM) denoising.
    Denoising strength is estimated adaptively from the image noise level.

    Returns:
        Path to denoised output image, or None on failure.
    """
    print("\n===================================")
    print("      DENOISE EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Denoise Expert] Input Path : {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print("[Denoise Expert] ERROR: Cannot load image.")
        return None

    # ── Estimate noise level from median-filter residual ──
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(residual))

    # Map noise level to NLM h parameter [3, 20]
    h_param = int(max(3, min(noise_std * 0.8, 20)))

    print(f"[Denoise Expert] Estimated noise std : {noise_std:.2f}")
    print(f"[Denoise Expert] NLM strength (h)    : {h_param}")
    print("[Denoise Expert] Applying Non-Local Means denoising...")

    # ── Apply NLM denoising ──
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h_param,
        hColor=h_param,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # ── Save output ──
    output_dir = os.path.join("outputs", "denoised")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_denoised.png")
    cv2.imwrite(output_path, denoised)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Denoise Expert] Output saved       : {output_path}")
    print(f"[Denoise Expert] Processing Time    : {elapsed}s")

    print("\n===================================")
    print("      DENOISE EXPERT FINISHED")
    print("===================================\n")

    return output_path