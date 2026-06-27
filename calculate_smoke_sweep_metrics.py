import os
import sys
import glob
import cv2
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.scheduler import run_three_stage_scheduler
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

try:
    import pyiqa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    niqe_metric = pyiqa.create_metric('niqe', device=device)
except ImportError:
    print("❌ Missing pyiqa. Run: pip install pyiqa")
    sys.exit(1)

def get_niqe(img_path):
    try: return float(niqe_metric(img_path).item())
    except: return 0.0

def get_edge_intensity(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    sx, sy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sx**2 + sy**2))

percentages = [10, 20, 30, 40]
final_results = []

for pct in percentages:
    print(f"\n--- Testing {pct}% Smoke Intensity ---")
    deg_images = glob.glob(f"datasets/synthetic_smoke_sweep/deg_{pct}_*.png")[:10] # Sample 10 for speed
    
    if not deg_images:
        print(f"No images found for {pct}%")
        continue
        
    p_psnr, p_ssim, p_niqe, p_edge = [], [], [], []
    
    for deg_path in deg_images:
        base_name = os.path.basename(deg_path).replace(f"deg_{pct}_", "")
        gt_search = glob.glob(f"datasets/DeSmoke-LAP dataset/Dataset/*/clear/{base_name}")
        if not gt_search: continue
        
        gt_path = gt_search[0]
        gt_img = cv2.imread(gt_path)
        
        res = run_three_stage_scheduler(deg_path, verbose=False, use_memory=True)
        rest_path = res.get("output_path")
        
        if rest_path and os.path.exists(rest_path):
            rest_img = cv2.imread(rest_path)
            if rest_img.shape != gt_img.shape:
                rest_img = cv2.resize(rest_img, (gt_img.shape[1], gt_img.shape[0]))
                
            s, p = compute_ssim_psnr(gt_img, rest_img)
            n = get_niqe(rest_path)
            e = get_edge_intensity(rest_path)
            
            p_psnr.append(p)
            p_ssim.append(s)
            p_niqe.append(n)
            p_edge.append(e)
            
    if p_psnr:
        avg_p = sum(p_psnr) / len(p_psnr)
        avg_s = sum(p_ssim) / len(p_ssim)
        avg_n = sum(p_niqe) / len(p_niqe)
        avg_e = sum(p_edge) / len(p_edge)
        final_results.append({"Smoke Intensity": f"{pct}%", "PSNR (dB)": avg_p, "SSIM": avg_s, "NIQE": avg_n, "Edge Intensity": avg_e})
        print(f"Averages -> PSNR: {avg_p:.2f}, SSIM: {avg_s:.3f}, NIQE: {avg_n:.2f}, Edge: {avg_e:.2f}")

df = pd.DataFrame(final_results)
df.to_csv("smoke_sweep_empirical_metrics.csv", index=False)
print("\n✅ Smoke Sweep metrics saved to smoke_sweep_empirical_metrics.csv")
