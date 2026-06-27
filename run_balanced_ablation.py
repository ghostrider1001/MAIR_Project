import os
import glob
import time
import pandas as pd
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.scheduler import run_three_stage_scheduler
from core.tool_registry import REGISTRY
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

import scheduler.scheduler as sched_module
import core.degradation_detector as detector_module

# 1. Define Balanced Dataset
SUBSETS = ["BSD68_subset", "RESIDE_subset", "LIVE1_subset", "LOL_subset"]
images = []

for subset in SUBSETS:
    deg_dir = os.path.join("datasets", "academic_subsets", subset, "degraded")
    gt_dir = os.path.join("datasets", "academic_subsets", subset, "ground_truth")
    
    # Grab first 10 images from each subset to keep runtime manageable while ensuring diversity
    found = glob.glob(os.path.join(deg_dir, "*.png"))[:10]
    for f in found:
        images.append((f, os.path.join(gt_dir, os.path.basename(f))))

print(f"Loaded {len(images)} images for Balanced Ablation Study.")

def eval_image(deg_path, gt_path, **scheduler_kwargs):
    gt_img = cv2.imread(gt_path)
    deg_img = cv2.imread(deg_path)
    if gt_img is None or deg_img is None: return 0, 0
    base_ssim, base_psnr = compute_ssim_psnr(gt_img, deg_img)
    
    result = run_three_stage_scheduler(deg_path, verbose=False, **scheduler_kwargs)
    rest_path = result.get("output_path")
    
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

def run_ablation(name, setup_fn, teardown_fn, **kwargs):
    print(f"\n--- Running: {name} ---")
    setup_fn()
    psnr_gains, ssim_gains = [], []
    
    for idx, (deg_path, gt_path) in enumerate(images):
        print(f"  [{idx+1}/{len(images)}] {os.path.basename(deg_path)}", end="\r")
        p_g, s_g = eval_image(deg_path, gt_path, **kwargs)
        psnr_gains.append(p_g)
        ssim_gains.append(s_g)
        
    teardown_fn()
    avg_psnr = sum(psnr_gains) / len(psnr_gains) if psnr_gains else 0
    avg_ssim = sum(ssim_gains) / len(ssim_gains) if ssim_gains else 0
    print(f"\nResult: PSNR Gain = +{avg_psnr:.2f} dB, SSIM Gain = +{avg_ssim:.3f}")
    return avg_psnr, avg_ssim

results = {}

# MOCKS
old_qg = sched_module.QUALITY_GATE_MIN
def setup_no_qg(): sched_module.QUALITY_GATE_MIN = -100.0
def teardown_no_qg(): sched_module.QUALITY_GATE_MIN = old_qg

old_detect = detector_module.detect_degradation
def mock_detect(*args, **kwargs):
    if not kwargs.get('verbose', True):
        return {"primary": "none", "confidence": 0.0, "scores": {}, "image_size": None}
    return old_detect(*args, **kwargs)
def setup_no_redetect(): detector_module.detect_degradation = mock_detect
def teardown_no_redetect(): detector_module.detect_degradation = old_detect

def setup_no_dcp():
    if 'dark_channel_dehaze' in REGISTRY: REGISTRY['dark_channel_dehaze']['handles'] = []
def teardown_no_dcp():
    if 'dark_channel_dehaze' in REGISTRY: REGISTRY['dark_channel_dehaze']['handles'] = ['haze']

def setup_no_wiener():
    if 'wiener_deconv' in REGISTRY: REGISTRY['wiener_deconv']['handles'] = []
def teardown_no_wiener():
    if 'wiener_deconv' in REGISTRY: REGISTRY['wiener_deconv']['handles'] = ['blur']

# RUNS
results["Full MAIR+ v2"] = run_ablation("Full MAIR+ v2", lambda: None, lambda: None, use_memory=True)
results["- Quality Gate"] = run_ablation("- Quality Gate", setup_no_qg, teardown_no_qg, use_memory=True)
results["- Re-detection"] = run_ablation("- Re-detection", setup_no_redetect, teardown_no_redetect, use_memory=True)
results["- CaseStore Memory"] = run_ablation("- CaseStore Memory", lambda: None, lambda: None, use_memory=False)
results["- DCP Expert"] = run_ablation("- DCP Expert", setup_no_dcp, teardown_no_dcp, use_memory=True)
results["- Wiener Expert"] = run_ablation("- Wiener Expert", setup_no_wiener, teardown_no_wiener, use_memory=True)
results["Detector Only (no routing)"] = (0.00, 0.000)

df = pd.DataFrame([{"Configuration": k, "PSNR_Gain": v[0], "SSIM_Gain": v[1]} for k, v in results.items()])
df.to_csv("balanced_ablation_results.csv", index=False)
print("\n✅ Ablation data saved to balanced_ablation_results.csv")
