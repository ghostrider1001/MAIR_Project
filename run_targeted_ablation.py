import os
import glob
import cv2
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler.scheduler import run_three_stage_scheduler
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr
import core.degradation_detector as detector_module

def eval_image(deg_path, gt_path, force_degradation, **kwargs):
    # Setup mock detector for this specific target
    old_detect = detector_module.detect_degradation
    def mock_detect(path, **k):
        # We also need to mock stage so it triggers. 
        # Haze is scene, blur is imaging, noise is imaging.
        return {
            "primary": force_degradation, 
            "confidence": 1.0, 
            "scores": {force_degradation: 1.0}, 
            "image_size": {"width": 256, "height": 256, "pixels": 65536}
        }
    
    import scheduler.scheduler as sched_mod
    sched_mod.detect_degradation = mock_detect
    
    gt_img = cv2.imread(gt_path)
    deg_img = cv2.imread(deg_path)
    if gt_img is None or deg_img is None: return 0, 0
    base_ssim, base_psnr = compute_ssim_psnr(gt_img, deg_img)
    
    # Run pipeline (with forced detector)
    result = run_three_stage_scheduler(deg_path, verbose=False, **kwargs)
    rest_path = result.get("output_path")
    
    # Teardown mock
    sched_mod.detect_degradation = old_detect
    
    if rest_path and os.path.exists(rest_path):
        rest_img = cv2.imread(rest_path)
        if rest_img is not None:
            if rest_img.shape != gt_img.shape:
                rest_img = cv2.resize(rest_img, (gt_img.shape[1], gt_img.shape[0]))
            rest_ssim, rest_psnr = compute_ssim_psnr(gt_img, rest_img)
        else:
            rest_ssim, rest_psnr = base_ssim, base_psnr
    else:
        rest_ssim, rest_psnr = base_ssim, base_psnr
        
    return rest_psnr - base_psnr, rest_ssim - base_ssim

def run_targeted(subset, force_degradation):
    print(f"\n--- Targeted Activation: Forcing '{force_degradation}' on {subset} ---")
    deg_dir = os.path.join("datasets", "academic_subsets", subset, "degraded")
    gt_dir = os.path.join("datasets", "academic_subsets", subset, "ground_truth")
    
    images = []
    found = glob.glob(os.path.join(deg_dir, "*.png"))[:10]
    for f in found:
        images.append((f, os.path.join(gt_dir, os.path.basename(f))))
        
    psnr_gains, ssim_gains = [], []
    for idx, (deg_path, gt_path) in enumerate(images):
        print(f"  [{idx+1}/{len(images)}] Processing {os.path.basename(deg_path)}...", end="\r")
        p_g, s_g = eval_image(deg_path, gt_path, force_degradation, use_memory=False)
        psnr_gains.append(p_g)
        ssim_gains.append(s_g)
        
    avg_psnr = sum(psnr_gains) / len(psnr_gains) if psnr_gains else 0
    avg_ssim = sum(ssim_gains) / len(ssim_gains) if ssim_gains else 0
    print(f"\nResult: PSNR Gain = +{avg_psnr:.2f} dB, SSIM Gain = +{avg_ssim:.3f}")
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    print("=== TARGETED ABLATION STUDY ===")
    print("By forcefully bypassing the detector bias, we can prove the individual mathematical efficacy of the experts.\n")
    
    results = {}
    
    # Force DCP Dehaze on RESIDE
    results["DCP Expert (Forced on Haze)"] = run_targeted("RESIDE_subset", "haze")
    
    # Force NAFNet Denoiser on BSD68
    results["NAFNet Expert (Forced on Noise)"] = run_targeted("BSD68_subset", "denoise")
    
    # Force Wiener Deblur on BSD68 (as a proxy for blurring, or we just see how it handles noise)
    results["Wiener Expert (Forced on Blur)"] = run_targeted("BSD68_subset", "blur")
    
    print("\n=== FINAL TARGETED ABLATION RESULTS ===")
    for k, v in results.items():
        print(f"{k}: +{v[0]:.2f} dB / +{v[1]:.3f} SSIM")
