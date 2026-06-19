import cv2
import os
import numpy as np
from datasets.generate_benchmark import apply_haze
from experts.dehaze_expert import restore_dcp
from evaluation.quality_evaluator import _compute_ssim_psnr

clear_path = r'datasets\synthetic_smoke\clear\TLH_2_001_0019.png'
img_clear = cv2.imread(clear_path)

if img_clear is None:
    print("Could not load image")
    exit(1)

# Resize to max 512 for speed
h, w = img_clear.shape[:2]
if max(h, w) > 512:
    scale = 512 / max(h, w)
    img_clear = cv2.resize(img_clear, (int(w * scale), int(h * scale)))

# Generate extreme smoke/haze so the restoration is obvious
img_hazy = apply_haze(img_clear, beta=3.0, atm_light=0.95)

# Save temporary hazy image
os.makedirs('outputs', exist_ok=True)
tmp_hazy_path = r'outputs\tmp_hazy.png'
cv2.imwrite(tmp_hazy_path, img_hazy)

out_path = restore_dcp(tmp_hazy_path)
img_restored = cv2.imread(out_path)

ssim_deg, psnr_deg = _compute_ssim_psnr(img_clear, img_hazy)
ssim_res, psnr_res = _compute_ssim_psnr(img_clear, img_restored)

print(f"Degraded: PSNR={psnr_deg:.2f}, SSIM={ssim_deg:.4f}")
print(f"Restored: PSNR={psnr_res:.2f}, SSIM={ssim_res:.4f}")

grid_path = r'outputs\comparison_grids\medical_synthetic_smoke_comparison.png'
os.makedirs(os.path.dirname(grid_path), exist_ok=True)

# Add headers
font = cv2.FONT_HERSHEY_SIMPLEX
bar_h = 40
bar1 = np.full((bar_h, img_hazy.shape[1], 3), (0,0,0), dtype=np.uint8)
cv2.putText(bar1, "1. Degraded", (10, 30), font, 1.0, (0,0,255), 2)
col1 = np.vstack([bar1, img_hazy])

bar2 = np.full((bar_h, img_restored.shape[1], 3), (0,0,0), dtype=np.uint8)
cv2.putText(bar2, "2. MAIR+ Restored", (10, 30), font, 1.0, (255,255,0), 2)
col2 = np.vstack([bar2, img_restored])

comparison = np.hstack([col1, col2])
cv2.imwrite(grid_path, comparison)

print(f"Saved better grid to {grid_path}")
