import os
import glob
import cv2
import numpy as np

SUBSETS = {
    "BSD68_subset": "noise",      # Denoising
    "Set14_subset": "noise",      # Denoising
    "Kodak_subset": "noise",      # Denoising
    "LIVE1_subset": "noise"       # Denoising
}
BASE_DIR = "datasets/academic_subsets"

def add_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def main():
    print("="*60)
    print("  APPLYING STANDARD ACADEMIC DEGRADATIONS (OPTIMIZED)")
    print("="*60)
    
    for subset, deg_type in SUBSETS.items():
        subset_dir = os.path.join(BASE_DIR, subset)
        gt_dir = os.path.join(subset_dir, "ground_truth")
        deg_dir = os.path.join(subset_dir, "degraded")
        
        os.makedirs(deg_dir, exist_ok=True)
        
        if not os.path.exists(gt_dir):
            continue
            
        images = glob.glob(os.path.join(gt_dir, "*.*"))
        if not images: continue
        
        for gt_path in images:
            img = cv2.imread(gt_path)
            if img is None: continue
            
            deg = add_noise(img)
            fname = os.path.basename(gt_path)
            cv2.imwrite(os.path.join(deg_dir, os.path.splitext(fname)[0] + ".png"), deg)
            
    print("\nDone! Mathematically optimized degradations applied.")

if __name__ == "__main__":
    main()
