import os
import cv2
import time
import numpy as np
import pandas as pd
import glob
import torch

from scheduler.scheduler import run_scheduler

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    RESIDE_BASE = "datasets/RESIDE"
    TMP_IMG = "tmp_reside.png"
    
    subsets = ["indoor", "outdoor"]
    results = []
    
    for subset in subsets:
        print("="*60)
        print(f"  EVALUATING RESIDE SOTS: {subset.upper()}")
        print("="*60)
        
        hazy_dir = os.path.join(RESIDE_BASE, subset, "hazy")
        gt_dir = os.path.join(RESIDE_BASE, subset, "ground_truth")
        
        hazy_images = sorted(glob.glob(os.path.join(hazy_dir, "*.*")))
        if not hazy_images:
            print(f"[SKIP] No images found in {hazy_dir}")
            continue
            
        psnr_gains = []
        total_time = 0
        
        for i, hazy_path in enumerate(hazy_images):
            filename = os.path.basename(hazy_path)
            # Find the corresponding GT image. SOTS hazy files are often named '1400_1.png' while GT is '1400.png'
            base_id = filename.split('_')[0].split('.')[0]
            gt_path = os.path.join(gt_dir, f"{base_id}.png")
            
            if not os.path.exists(gt_path):
                # Fallback to direct name matching
                gt_path = os.path.join(gt_dir, filename)
                if not os.path.exists(gt_path):
                    continue
            
            print(f"\n--- [{i+1}/{len(hazy_images)}] Dehazing: {filename} ---")
            
            hazy_img = cv2.imread(hazy_path)
            gt_img = cv2.imread(gt_path)
            if hazy_img is None or gt_img is None: continue
            
            # Ensure they are the same size for PSNR
            if hazy_img.shape != gt_img.shape:
                hazy_img = cv2.resize(hazy_img, (gt_img.shape[1], gt_img.shape[0]))
                cv2.imwrite(hazy_path, hazy_img)
            
            deg_psnr = compute_psnr(gt_img, hazy_img)
            print(f"Initial Hazy PSNR: {deg_psnr:.2f} dB")
            
            # Route through AI
            start_t = time.time()
            rest_path = run_scheduler(hazy_path)
            
            if rest_path is None or not os.path.exists(rest_path):
                print("[ERROR] Scheduler failed to produce an output.")
                continue
                
            rest_img = cv2.imread(rest_path)
            rest_psnr = compute_psnr(gt_img, rest_img)
            
            process_time = time.time() - start_t
            gain = rest_psnr - deg_psnr
            
            print(f"Restored PSNR: {rest_psnr:.2f} dB (Gain: +{gain:.2f} dB)")
            print(f"Processing Time: {process_time:.2f}s")
            
            psnr_gains.append(gain)
            total_time += process_time
            
            # Delete tmp restored image to save space
            if os.path.exists(rest_path): os.remove(rest_path)
            
        if psnr_gains:
            avg_gain = np.mean(psnr_gains)
            print("\n" + "="*60)
            print(f"  {subset.upper()} EVALUATION COMPLETE!")
            print(f"  Average PSNR Gain : +{avg_gain:.2f} dB")
            print(f"  Total AI Time     : {total_time:.2f} seconds")
            print("="*60)
            results.append({"Subset": f"RESIDE_{subset.capitalize()}", "Images": len(hazy_images), "Task": "Dehazing", "PSNR_Gain": avg_gain, "Avg_Time": total_time/len(hazy_images)})

    if results:
        df = pd.DataFrame(results)
        df.to_csv("reside_benchmark_results.csv", index=False)
        print("✅ Saved results to reside_benchmark_results.csv")

if __name__ == "__main__":
    main()
