import os
import glob
import cv2
import numpy as np
import pandas as pd
from datasets.generate_synthetic_smoke import add_synthetic_smoke
from scheduler.scheduler import run_scheduler
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

# Try to load pyiqa for NIQE
try:
    import pyiqa
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    HAS_NIQE = True
except ImportError:
    HAS_NIQE = False
    print("Warning: pyiqa not installed. NIQE will be skipped (reported as 0.0).")
    print("Run: pip install pyiqa torch torchvision")

def compute_edge_intensity(img):
    """Compute Edge Intensity (Sharpness) using Sobel operators."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(magnitude)

def compute_niqe(img_path):
    if not HAS_NIQE: return 0.0
    try:
        return float(niqe_metric(img_path).item())
    except Exception:
        return 0.0

def main():
    base_dir = "datasets/DeSmoke-LAP dataset/Dataset"
    if not os.path.exists(base_dir):
        print(f"Error: Could not find DeSmoke-LAP at {base_dir}")
        return
        
    clear_images = glob.glob(os.path.join(base_dir, "*", "clear", "*.png"))
    if not clear_images:
        print("No images found.")
        return
        
    print(f"Found {len(clear_images)} clear images in DeSmoke-LAP dataset.")
    
    intensities = [0.1, 0.2, 0.3, 0.4] # 10%, 20%, 30%, 40%
    results = []
    
    out_dir = "datasets/synthetic_smoke_sweep"
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*60)
    print(" STARTING SMOKE INTENSITY SWEEP (10%, 20%, 30%, 40%)")
    print(" Metrics: PSNR, SSIM, NIQE, Edge Intensity")
    print("="*60)
    
    for intensity in intensities:
        pct = int(intensity * 100)
        print(f"\n--- Generating and Evaluating Smoke Level: {pct}% ---")
        psnr_list, ssim_list, niqe_list, edge_list = [], [], [], []
        
        for img_path in clear_images:
            basename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Generate degraded image for this specific intensity
            hazy_img = add_synthetic_smoke(img, intensity=intensity)
            deg_path = os.path.join(out_dir, f"deg_{pct}_{basename}")
            cv2.imwrite(deg_path, hazy_img)
            
            # Run MAIR+ routing & restoration
            rest_path = run_scheduler(deg_path, verbose=False, three_stage=True)
            if rest_path and os.path.exists(rest_path):
                rest_img = cv2.imread(rest_path)
                if rest_img is not None:
                    if rest_img.shape != img.shape:
                        rest_img = cv2.resize(rest_img, (img.shape[1], img.shape[0]))
                    ssim, psnr = compute_ssim_psnr(img, rest_img)
                    niqe = compute_niqe(rest_path)
                    edge = compute_edge_intensity(rest_img)
                    
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    niqe_list.append(niqe)
                    edge_list.append(edge)
                    
        avg_psnr = np.mean(psnr_list) if psnr_list else 0
        avg_ssim = np.mean(ssim_list) if ssim_list else 0
        avg_niqe = np.mean(niqe_list) if niqe_list else 0
        avg_edge = np.mean(edge_list) if edge_list else 0
        
        print(f"Result for {pct}% -> PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | NIQE: {avg_niqe:.2f} | Edge: {avg_edge:.2f}")
        
        results.append({
            "Smoke Intensity": f"{pct}%",
            "PSNR (dB)": round(avg_psnr, 2),
            "SSIM": round(avg_ssim, 4),
            "NIQE": round(avg_niqe, 2),
            "Edge Intensity": round(avg_edge, 2)
        })
        
    df = pd.DataFrame(results)
    df.to_csv("smoke_sweep_results.csv", index=False)
    
    print("\n" + "="*60)
    print(" FINAL SWEEP RESULTS ")
    print("="*60)
    print(df.to_string(index=False))
    print("\nResults saved to smoke_sweep_results.csv")

if __name__ == "__main__":
    np.random.seed(42) # For deterministic scattering
    main()
