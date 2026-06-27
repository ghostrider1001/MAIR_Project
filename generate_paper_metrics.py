import os
import glob
import cv2
import pandas as pd
import numpy as np
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.scheduler import run_three_stage_scheduler
import scheduler.scheduler as sched_module
import core.degradation_detector as detector_module
from core.degradation_detector import detect_degradation
from core.tool_registry import REGISTRY
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

DATASETS = {
    "GoPro_subset": "blur",
    "BSD68_subset": "noise",
    "LIVE1_subset": "jpeg",
    "LOL_subset": "lowlight",
    "RESIDE_subset": "haze"
}

def load_dataset(n_per_class=20):
    images = []
    for subset, true_label in DATASETS.items():
        deg_dir = os.path.join("datasets", "academic_subsets", subset, "degraded")
        gt_dir = os.path.join("datasets", "academic_subsets", subset, "ground_truth")
        found = glob.glob(os.path.join(deg_dir, "*.png"))[:n_per_class]
        for f in found:
            gt_path = os.path.join(gt_dir, os.path.basename(f))
            if os.path.exists(gt_path):
                images.append({"deg_path": f, "gt_path": gt_path, "true_label": true_label})
    return images

def run_detector_eval():
    print("--- Evaluating Detector Accuracy ---")
    images = load_dataset(100)
    classes = ["blur", "noise", "jpeg", "lowlight", "haze"]
    conf_matrix = {t: {p: 0 for p in classes + ["none"]} for t in classes}
    
    for img_info in images:
        deg_path = img_info["deg_path"]
        true_label = img_info["true_label"]
        res = detect_degradation(deg_path, verbose=False)
        pred_label = res["primary"]
        
        # FIX: map "denoise" to "noise"
        if pred_label in ["denoise", "sensor_noise", "poisson_noise"]: pred_label = "noise"
        if pred_label == "motion_blur": pred_label = "blur"
        if pred_label == "jpeg_artifacts": pred_label = "jpeg"
        if pred_label not in classes: pred_label = "none"
        
        conf_matrix[true_label][pred_label] += 1
        
    df = pd.DataFrame(conf_matrix).T
    df.fillna(0, inplace=True)
    df_pct = df.div(df.sum(axis=1).replace(0, 1), axis=0) * 100
    df_pct.to_csv("detector_confusion_matrix_actual.csv")
    print("\nDetector Confusion Matrix (%)")
    print(df_pct)
    return df_pct

def eval_image(deg_path, gt_path, **scheduler_kwargs):
    gt_img = cv2.imread(gt_path)
    deg_img = cv2.imread(deg_path)
    if gt_img is None or deg_img is None: return 0, 0
    base_ssim, base_psnr = compute_ssim_psnr(gt_img, deg_img)
    try:
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
    except Exception as e:
        print(f"Error evaluating {deg_path}: {e}")
        rest_ssim, rest_psnr = base_ssim, base_psnr
    return rest_psnr - base_psnr, rest_ssim - base_ssim

def run_ablation_eval():
    print("\n--- Evaluating Ablation Study ---")
    images = load_dataset(2) # VERY FAST for ablation: 2 per class = 10 images total
    print(f"Loaded {len(images)} images for Mixed Degradation Ablation.")
    
    old_qg = getattr(sched_module, 'QUALITY_GATE_MIN', 0.5)
    def setup_no_qg(): sched_module.QUALITY_GATE_MIN = -100.0
    def teardown_no_qg(): sched_module.QUALITY_GATE_MIN = old_qg
    
    old_detect = detector_module.detect_degradation
    def mock_detect(*args, **kwargs):
        return {"primary": "none", "confidence": 0.0, "scores": {}, "image_size": None}
    def setup_no_redetect(): detector_module.detect_degradation = mock_detect
    def teardown_no_redetect(): detector_module.detect_degradation = old_detect
    
    # CORRECT KEYS FOR ABLATION
    def setup_no_dcp():
        if 'dcp_dehaze' in REGISTRY: REGISTRY['dcp_dehaze']['handles'] = []
    def teardown_no_dcp():
        if 'dcp_dehaze' in REGISTRY: REGISTRY['dcp_dehaze']['handles'] = ['haze']
        
    def setup_no_wiener():
        if 'wiener_deblur' in REGISTRY: REGISTRY['wiener_deblur']['handles'] = []
    def teardown_no_wiener():
        if 'wiener_deblur' in REGISTRY: REGISTRY['wiener_deblur']['handles'] = ['blur']

    def setup_no_nafnet():
        if 'nafnet_lite_denoise' in REGISTRY: REGISTRY['nafnet_lite_denoise']['handles'] = []
    def teardown_no_nafnet():
        if 'nafnet_lite_denoise' in REGISTRY: REGISTRY['nafnet_lite_denoise']['handles'] = ['denoise', 'blur']

    def setup_no_swinir():
        if 'swinir_jpeg' in REGISTRY: REGISTRY['swinir_jpeg']['handles'] = []
    def teardown_no_swinir():
        if 'swinir_jpeg' in REGISTRY: REGISTRY['swinir_jpeg']['handles'] = ['jpeg']

    configs = [
        ("Full MAIR++", lambda: None, lambda: None, True),
        ("- Quality Gate", setup_no_qg, teardown_no_qg, True),
        ("- Re-detection", setup_no_redetect, teardown_no_redetect, True),
        ("- CaseStore Memory", lambda: None, lambda: None, False),
        ("- FastJPEG Expert", setup_no_swinir, teardown_no_swinir, True),
        ("- NAFNet Expert", setup_no_nafnet, teardown_no_nafnet, True),
        ("- DCP Expert", setup_no_dcp, teardown_no_dcp, True),
        ("- Wiener Expert", setup_no_wiener, teardown_no_wiener, True)
    ]
    
    results = {}
    for name, setup_fn, teardown_fn, use_mem in configs:
        setup_fn()
        psnr_gains, ssim_gains = [], []
        t0 = time.time()
        for idx, img_info in enumerate(images):
            p_g, s_g = eval_image(img_info["deg_path"], img_info["gt_path"], use_memory=use_mem)
            psnr_gains.append(p_g)
            ssim_gains.append(s_g)
        teardown_fn()
        avg_p = np.mean(psnr_gains) if psnr_gains else 0
        avg_s = np.mean(ssim_gains) if ssim_gains else 0
        results[name] = {"Overall PSNR Gain": avg_p, "Overall SSIM Gain": avg_s, "Time (s)": time.time()-t0}
        
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv("mixed_ablation_results_actual.csv")
    print("\n" + str(df))
    
if __name__ == "__main__":
    run_detector_eval()
    run_ablation_eval()
