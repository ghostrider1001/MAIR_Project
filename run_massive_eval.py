import os
import cv2
import time
import numpy as np
import pandas as pd
import glob
import torch

# Ensure GPU is initialized if available
if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print("PyTorch deep learning experts (SwinIR/NAFNet) will automatically run on the GPU.")
else:
    print("⚠️ No CUDA GPU detected! PyTorch will fall back to CPU, which will be much slower.")

from scheduler.scheduler import run_scheduler

DIV2K_DIR = "datasets/DIV2K/ground_truth"
TMP_DEG = "tmp_div2k_deg.png"
RESULTS_CSV = "massive_benchmark_results.csv"

def add_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    if not os.path.exists(DIV2K_DIR):
        print(f"[ERROR] Waiting for DIV2K images... Please extract them to:\n{DIV2K_DIR}")
        return
        
    images = glob.glob(os.path.join(DIV2K_DIR, "*.*"))
    if not images:
        print(f"[ERROR] Folder {DIV2K_DIR} exists, but it's empty!")
        return
        
    print("="*60)
    print(f"  MASSIVE EVALUATION: DIV2K ({len(images)} Images) at 2K Resolution")
    print("="*60)
    
    psnr_gains = []
    total_time = 0
    
    # Streaming loop to preserve hard drive space and RAM!
    for i, path in enumerate(images):
        print(f"\n--- [{i+1}/{len(images)}] Processing 2K Image: {os.path.basename(path)} ---")
        
        # 1. Load GT and stream to RAM
        gt_img = cv2.imread(path)
        if gt_img is None: continue
        
        # 2. Add degradation to RAM and save temp file
        deg_img = add_noise(gt_img)
        cv2.imwrite(TMP_DEG, deg_img)
        
        # 3. Calculate initial degraded PSNR
        deg_psnr = compute_psnr(gt_img, deg_img)
        print(f"Initial Degraded PSNR: {deg_psnr:.2f} dB")
        
        # 4. Route through AI
        start_t = time.time()
        rest_path = run_scheduler(TMP_DEG)
        
        if rest_path is None or not os.path.exists(rest_path):
            print("[ERROR] Scheduler failed to produce an output.")
            continue
            
        # 5. Calculate Restored PSNR
        rest_img = cv2.imread(rest_path)
        if rest_img is None:
            print("[ERROR] Could not read restored image.")
            continue
            
        rest_psnr = compute_psnr(gt_img, rest_img)
        
        process_time = time.time() - start_t
        gain = rest_psnr - deg_psnr
        
        print(f"Restored PSNR: {rest_psnr:.2f} dB (Gain: +{gain:.2f} dB)")
        print(f"Processing Time: {process_time:.2f}s")
        
        psnr_gains.append(gain)
        total_time += process_time
        
        # 6. Delete temp files immediately so the hard drive doesn't overflow!
        if os.path.exists(TMP_DEG): os.remove(TMP_DEG)
        if os.path.exists(rest_path): os.remove(rest_path)
        
    print("\n" + "="*60)
    print("  MASSIVE EVALUATION COMPLETE!")
    if psnr_gains:
        print(f"  Average PSNR Gain : +{np.mean(psnr_gains):.2f} dB")
        print(f"  Total AI Time     : {total_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()
