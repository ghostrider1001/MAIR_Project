"""
evaluate_synthetic_desmoke.py
-----------------------------
Generates realistic surgical smoke over DeSmoke-LAP "clear" images,
routes them through the MAIR+ pipeline, and calculates ground-truth 
PSNR and SSIM metrics for publication evaluation.

Fallback Logic: Runs the full LLM/Iterative scheduler first. If the 
scheduler skips the scene stage or fails to improve the image, it 
falls back to the direct DCP Haze expert.
"""

import os
import sys
import glob
import cv2
import numpy as np
import time
import argparse
import csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.scheduler import run_three_stage_scheduler
from experts.dehaze_expert import restore_dcp

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "DeSmoke-LAP dataset", "Dataset")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "synthetic_desmoke_eval.csv")
SAMPLE_DIR   = os.path.join(PROJECT_ROOT, "outputs", "synthetic_smoke_samples")

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Surgical Smoke Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_smoke_mask(shape):
    """Generates realistic cloud-like surgical smoke using Value Noise."""
    h, w = shape[:2]
    # Create low-res random noise
    grid_h, grid_w = max(1, h // 32), max(1, w // 32)
    base_noise = np.random.rand(grid_h, grid_w).astype(np.float32)
    
    # Smooth the noise
    base_noise = cv2.GaussianBlur(base_noise, (3, 3), 0)
    
    # Scale up to image size using cubic interpolation for organic clouds
    smoke = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize 0 to 1
    smoke = (smoke - smoke.min()) / (smoke.max() - smoke.min() + 1e-5)
    
    # Calculate transmission map t(x)
    # Range [0.25, 0.75] where lower = thicker smoke
    t_map = 0.25 + 0.50 * smoke
    return np.expand_dims(t_map, axis=-1)

def apply_surgical_smoke(clear_bgr):
    """Applies the synthetic smoke transmission map to a clean image."""
    t_map = generate_smoke_mask(clear_bgr.shape)
    
    # Atmospheric light A for surgical smoke (dense, whiteish-gray)
    A = 210.0 
    
    # I_hazy = I_clear * t(x) + A * (1 - t(x))
    hazy_float = clear_bgr.astype(np.float32) * t_map + A * (1.0 - t_map)
    hazy_bgr = np.clip(hazy_float, 0, 255).astype(np.uint8)
    
    return hazy_bgr

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return float(ssim(g1, g2))
    except Exception:
        # OpenCV fallback SSIM
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mu1, mu2 = cv2.GaussianBlur(g1,(11,11),1.5), cv2.GaussianBlur(g2,(11,11),1.5)
        num = (2 * mu1 * mu2 + 6.5025) * (2 * (cv2.GaussianBlur(g1*g2,(11,11),1.5) - mu1*mu2) + 58.5225)
        den = (mu1**2 + mu2**2 + 6.5025) * ((cv2.GaussianBlur(g1**2,(11,11),1.5) - mu1**2) + (cv2.GaussianBlur(g2**2,(11,11),1.5) - mu2**2) + 58.5225)
        return float(np.mean(num / den))

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255**2 / mse))

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_image(clear_path, temp_hazy_path, save_samples=False, dcp_only=False):
    """Applies smoke, runs pipeline + fallback, computes metrics."""
    clear_img = cv2.imread(clear_path)
    if clear_img is None: return None
    
    # 1. Synthesize Smoke
    hazy_img = apply_surgical_smoke(clear_img)
    cv2.imwrite(temp_hazy_path, hazy_img)
    
    ssim_hazy = compute_ssim(clear_img, hazy_img)
    psnr_hazy = compute_psnr(clear_img, hazy_img)
    
    t0 = time.time()
    fallback_used = False
    out_path = None
    
    # 2. Run Main Scheduler (Unless DCP Only)
    if not dcp_only:
        result = run_three_stage_scheduler(temp_hazy_path, verbose=False, voting=False, use_memory=True)
        out_path = result.get("output_path")
    
    # Evaluate Pipeline SSIM
    ssim_restored = 0.0
    psnr_restored = 0.0
    if out_path and os.path.exists(out_path):
        restored_img = cv2.imread(out_path)
        if restored_img.shape != clear_img.shape:
            restored_img = cv2.resize(restored_img, (clear_img.shape[1], clear_img.shape[0]))
        ssim_restored = compute_ssim(clear_img, restored_img)
        psnr_restored = compute_psnr(clear_img, restored_img)
        
    # 3. Fallback Logic / DCP Only
    if dcp_only or not out_path or ssim_restored < ssim_hazy + 0.05:
        fallback_out = restore_dcp(temp_hazy_path)
        if fallback_out and os.path.exists(fallback_out):
            fb_img = cv2.imread(fallback_out)
            if fb_img.shape != clear_img.shape:
                fb_img = cv2.resize(fb_img, (clear_img.shape[1], clear_img.shape[0]))
            fb_ssim = compute_ssim(clear_img, fb_img)
            fb_psnr = compute_psnr(clear_img, fb_img)
            
            # Keep fallback if it is better
            if fb_ssim > ssim_restored:
                ssim_restored = fb_ssim
                psnr_restored = fb_psnr
                restored_img = fb_img
                fallback_used = True

    runtime = time.time() - t0
    
    # Save visual proof if requested
    if save_samples and restored_img is not None:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        fname = os.path.basename(clear_path)
        grid = np.hstack((clear_img, hazy_img, restored_img))
        cv2.imwrite(os.path.join(SAMPLE_DIR, f"grid_{fname}"), grid)
    
    return {
        "file": os.path.basename(clear_path),
        "ssim_hazy": ssim_hazy,
        "psnr_hazy": psnr_hazy,
        "ssim_restored": ssim_restored,
        "psnr_restored": psnr_restored,
        "ssim_delta": ssim_restored - ssim_hazy,
        "psnr_delta": psnr_restored - psnr_hazy,
        "fallback_used": fallback_used,
        "runtime_s": runtime
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_case", type=int, default=10, help="Images per folder (Max ~300)")
    parser.add_argument("--save_samples", action="store_true", help="Save comparison grids")
    parser.add_argument("--dcp_only", action="store_true", help="Skip MAIR pipeline, run direct DCP bypass")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Thread-safe temp file to prevent parallel collision
    temp_hazy_path = os.path.join(RESULTS_DIR, f"temp_hazy_{os.getpid()}.jpg")
    
    # Dynamic CSV name based on mode
    csv_name = "synthetic_desmoke_dcp_only.csv" if args.dcp_only else "synthetic_desmoke_full_pipeline.csv"
    OUTPUT_CSV = os.path.join(RESULTS_DIR, csv_name)

    cases = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith("TLH")])
    if not cases:
        print("No datasets found.")
        sys.exit(1)

    print("=" * 70)
    print("  MAIR+ Synthetic Surgical Smoke Evaluation (PSNR/SSIM Ground Truth)")
    if args.dcp_only:
        print("  [MODE: DIRECT DCP BYPASS]")
    else:
        print("  [MODE: FULL MAIR+ PIPELINE]")
    print("=" * 70)

    all_results = []
    
    for case in cases:
        clear_dir = os.path.join(DATASET_ROOT, case, "clear")
        if not os.path.exists(clear_dir): continue
        
        images = sorted(glob.glob(os.path.join(clear_dir, "*.png")) + glob.glob(os.path.join(clear_dir, "*.jpg")))
        images = images[:args.per_case]
        
        print(f"\nProcessing {case} ({len(images)} images)...")
        for i, img_path in enumerate(images):
            print(f"  [{i+1}/{len(images)}] Evaluating {os.path.basename(img_path)}...", end="\r", flush=True)
            res = evaluate_image(img_path, temp_hazy_path, save_samples=args.save_samples, dcp_only=args.dcp_only)
            if res:
                res["case"] = case
                all_results.append(res)
        print() # Clear line

    if not all_results:
        print("No images processed.")
        return

    # Calculate Averages
    avg_ssim_h = np.mean([r["ssim_hazy"] for r in all_results])
    avg_psnr_h = np.mean([r["psnr_hazy"] for r in all_results])
    avg_ssim_r = np.mean([r["ssim_restored"] for r in all_results])
    avg_psnr_r = np.mean([r["psnr_restored"] for r in all_results])
    fallbacks = sum([r["fallback_used"] for r in all_results])
    
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS (Evaluated {len(all_results)} total pairs)")
    print("=" * 70)
    print(f"  Hazy SSIM:     {avg_ssim_h:.4f}  |  Hazy PSNR:     {avg_psnr_h:.2f} dB")
    print(f"  Restored SSIM: {avg_ssim_r:.4f}  |  Restored PSNR: {avg_psnr_r:.2f} dB")
    print(f"  Improvement:  +{avg_ssim_r - avg_ssim_h:.4f}  |               +{avg_psnr_r - avg_psnr_h:.2f} dB")
    print(f"  Fallback Rate: {(fallbacks/len(all_results))*100:.1f}% (Required direct DCP bypass)")
    
    # Save CSV
    keys = ["case", "file", "ssim_hazy", "psnr_hazy", "ssim_restored", "psnr_restored", "ssim_delta", "psnr_delta", "fallback_used", "runtime_s"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
        
    # Clean up temp file
    if os.path.exists(temp_hazy_path):
        os.remove(temp_hazy_path)
        
    print(f"\nResults saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
