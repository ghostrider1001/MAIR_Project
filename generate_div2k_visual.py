import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from scheduler.scheduler import run_scheduler

def add_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def main():
    DIV2K_DIR = "datasets/DIV2K/ground_truth"
    TMP_DEG = "tmp_div2k_visual.png"
    
    images = glob.glob(os.path.join(DIV2K_DIR, "*.*"))
    if not images: return
    
    # Process just the first image
    path = images[0]
    gt_img = cv2.imread(path)
    
    # Crop to a 1024x1024 patch so the visual details are visible
    h, w = gt_img.shape[:2]
    ch, cw = h // 2, w // 2
    gt_patch = gt_img[ch-512:ch+512, cw-512:cw+512]
    
    # Degrade
    deg_patch = add_noise(gt_patch)
    cv2.imwrite(TMP_DEG, deg_patch)
    
    # Restore
    print("Generating visual... Routing through AI...")
    rest_path = run_scheduler(TMP_DEG)
    rest_patch = cv2.imread(rest_path)
    
    # Convert BGR to RGB for matplotlib
    gt_rgb = cv2.cvtColor(gt_patch, cv2.COLOR_BGR2RGB)
    deg_rgb = cv2.cvtColor(deg_patch, cv2.COLOR_BGR2RGB)
    rest_rgb = cv2.cvtColor(rest_patch, cv2.COLOR_BGR2RGB)
    
    # Plot side by side
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gt_rgb)
    plt.title("Ground Truth (2K Patch)", fontsize=16)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(deg_rgb)
    plt.title("Degraded (Noise/Rain Detection)", fontsize=16)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(rest_rgb)
    plt.title("Restored (MAIR+)", fontsize=16)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("div2k_comparison.png", dpi=150)
    print("✅ Saved comparison visual to div2k_comparison.png")

if __name__ == "__main__":
    main()
