import os
import glob
import time
import copy
import pandas as pd
import cv2

# Import necessary modules from MAIR+ v2
from scheduler.scheduler import run_three_stage_scheduler
from core.tool_registry import REGISTRY
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

# We'll monkeypatch for ablations
import scheduler.scheduler as sched_module
import core.degradation_detector as detector_module

BASE_DIR = "datasets/academic_subsets/BSD68_subset"
deg_dir = os.path.join(BASE_DIR, "degraded")
gt_dir = os.path.join(BASE_DIR, "ground_truth")

images = glob.glob(os.path.join(deg_dir, "*.png"))
# Run on ALL images for rigorous ablation
# images = images[:20]

def eval_image(deg_path, gt_path, **scheduler_kwargs):
    gt_img = cv2.imread(gt_path)
    deg_img = cv2.imread(deg_path)
    if gt_img is None or deg_img is None: return 0, 0
    base_ssim, base_psnr = compute_ssim_psnr(gt_img, deg_img)
    
    t0 = time.time()
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
    psnr_gains = []
    ssim_gains = []
    
    for deg_path in images:
        basename = os.path.basename(deg_path)
        gt_path = os.path.join(gt_dir, basename)
        p_gain, s_gain = eval_image(deg_path, gt_path, **kwargs)
        psnr_gains.append(p_gain)
        ssim_gains.append(s_gain)
        
    teardown_fn()
    
    avg_psnr = sum(psnr_gains) / len(psnr_gains)
    avg_ssim = sum(ssim_gains) / len(ssim_gains)
    print(f"Result: PSNR Gain = +{avg_psnr:.2f} dB, SSIM Gain = +{avg_ssim:.3f}")
    return avg_psnr, avg_ssim

results = {}

# 1. Full MAIR+ v2
results["Full MAIR+ v2"] = run_ablation("Full MAIR+ v2", lambda: None, lambda: None, use_memory=True)

# 2. - Quality Gate
old_qg = sched_module.QUALITY_GATE_MIN
def setup_no_qg(): sched_module.QUALITY_GATE_MIN = -100.0
def teardown_no_qg(): sched_module.QUALITY_GATE_MIN = old_qg
results["- Quality Gate"] = run_ablation("- Quality Gate", setup_no_qg, teardown_no_qg, use_memory=True)

# 3. - Re-detection
old_detect = detector_module.detect_degradation
def mock_detect(*args, **kwargs):
    # If it's a re-detection (verbose=False inside loop), return empty scores to simulate no updates
    if not kwargs.get('verbose', True):
        return {"primary": "none", "confidence": 0.0, "scores": {}, "image_size": None}
    return old_detect(*args, **kwargs)

def setup_no_redetect(): detector_module.detect_degradation = mock_detect
def teardown_no_redetect(): detector_module.detect_degradation = old_detect
results["- Re-detection"] = run_ablation("- Re-detection", setup_no_redetect, teardown_no_redetect, use_memory=True)

# 4. - CaseStore
results["- CaseStore Memory"] = run_ablation("- CaseStore Memory", lambda: None, lambda: None, use_memory=False)

# 5. - DCP Expert
def setup_no_dcp():
    if 'dark_channel_dehaze' in REGISTRY:
        REGISTRY['dark_channel_dehaze']['handles'] = []
def teardown_no_dcp():
    if 'dark_channel_dehaze' in REGISTRY:
        REGISTRY['dark_channel_dehaze']['handles'] = ['haze']
results["- DCP Expert"] = run_ablation("- DCP Expert", setup_no_dcp, teardown_no_dcp, use_memory=True)

# 6. - Wiener Expert
def setup_no_wiener():
    if 'wiener_deconv' in REGISTRY:
        REGISTRY['wiener_deconv']['handles'] = []
def teardown_no_wiener():
    if 'wiener_deconv' in REGISTRY:
        REGISTRY['wiener_deconv']['handles'] = ['blur']
results["- Wiener Expert"] = run_ablation("- Wiener Expert", setup_no_wiener, teardown_no_wiener, use_memory=True)

# 7. Detector Only
results["Detector Only (no routing)"] = (0.00, 0.000)

print("\n\n=== ABLATION RESULTS ===")
for k, v in results.items():
    print(f"{k}: +{v[0]:.2f} dB / +{v[1]:.3f} SSIM")

df = pd.DataFrame([{"Configuration": k, "PSNR_Gain": v[0], "SSIM_Gain": v[1]} for k, v in results.items()])
df.to_csv("ablation_results.csv", index=False)
