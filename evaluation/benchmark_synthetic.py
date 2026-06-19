import os
import glob
import json
import time
import sys
from datetime import datetime
from tqdm import tqdm

# Add the project root to the START of sys.path to override the pip-installed 'evaluation' module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.quality_evaluator import evaluate_quality_full
from scheduler.scheduler import run_three_stage_scheduler

def run_synthetic_benchmark():
    print("==================================================")
    print("   MAIR+ v2 — SYNTHETIC DUAL-EVAL BENCHMARK")
    print("==================================================")

    hazy_dir = "datasets/synthetic_smoke/hazy"
    clear_dir = "datasets/synthetic_smoke/clear"
    output_dir = "results/synthetic_benchmark"
    
    os.makedirs(output_dir, exist_ok=True)

    hazy_images = sorted(glob.glob(os.path.join(hazy_dir, "*.png")))
    
    if not hazy_images:
        print(f"No hazy images found in {hazy_dir}. Did you run generate_synthetic_smoke.py?")
        return

    print(f"Found {len(hazy_images)} synthetic images. Starting benchmark...")
    
    results = []
    
    base_ssims = []
    base_psnrs = []
    
    rest_ssims = []
    rest_psnrs = []
    
    start_time = time.time()
    
    for hazy_path in tqdm(hazy_images):
        filename = os.path.basename(hazy_path)
        clear_path = os.path.join(clear_dir, filename)
        
        if not os.path.exists(clear_path):
            print(f"Warning: Ground truth missing for {filename}. Skipping.")
            continue
            
        # 1. Evaluate baseline (Hazy vs Clear)
        base_metrics = evaluate_quality_full(clear_path, hazy_path)
        base_ssim = base_metrics.get("ssim", 0.0)
        base_psnr = base_metrics.get("psnr", 0.0)
        
        base_ssims.append(base_ssim)
        base_psnrs.append(base_psnr)
        
        # 2. Run MAIR+ Pipeline
        # We run it normally (clinical_eval=False) because we actually WANT to use
        # the standard SSIM/LPIPS quality gate, since this dataset HAS ground truth.
        # But wait, run_three_stage_scheduler takes image path directly and saves 
        # to a folder, but it also overwrites in place if we aren't careful.
        # It's better to just invoke the scheduler.
        
        try:
            # We don't want the pipeline to print everything, but the scheduler does.
            result_dict = run_three_stage_scheduler(hazy_path, clinical_eval=False)
            restored_path = result_dict.get("output_path", hazy_path)
            
            if restored_path is None:
                restored_path = hazy_path
            
            # 3. Evaluate restored (Restored vs Clear)
            rest_metrics = evaluate_quality_full(clear_path, restored_path)
            rest_ssim = rest_metrics.get("ssim", 0.0)
            rest_psnr = rest_metrics.get("psnr", 0.0)
            
            rest_ssims.append(rest_ssim)
            rest_psnrs.append(rest_psnr)
            
            results.append({
                "image": filename,
                "baseline_ssim": base_ssim,
                "baseline_psnr": base_psnr,
                "restored_ssim": rest_ssim,
                "restored_psnr": rest_psnr,
                "ssim_gain": round(rest_ssim - base_ssim, 4),
                "psnr_gain": round(rest_psnr - base_psnr, 2)
            })
            
        except Exception as e:
            print(f"\nFailed to process {filename}: {e}")
            continue

    total_time = time.time() - start_time
    
    # Calculate Averages
    avg_base_ssim = sum(base_ssims) / len(base_ssims) if base_ssims else 0
    avg_base_psnr = sum(base_psnrs) / len(base_psnrs) if base_psnrs else 0
    
    avg_rest_ssim = sum(rest_ssims) / len(rest_ssims) if rest_ssims else 0
    avg_rest_psnr = sum(rest_psnrs) / len(rest_psnrs) if rest_psnrs else 0
    
    avg_ssim_gain = avg_rest_ssim - avg_base_ssim
    avg_psnr_gain = avg_rest_psnr - avg_base_psnr
    
    print("\n==================================================")
    print("   SYNTHETIC BENCHMARK RESULTS")
    print("==================================================")
    print(f"Images Evaluated : {len(results)}")
    print(f"Total Time       : {total_time:.2f}s")
    print("--------------------------------------------------")
    print("Baseline (Hazy vs Clear Ground Truth):")
    print(f"  Avg SSIM : {avg_base_ssim:.4f}")
    print(f"  Avg PSNR : {avg_base_psnr:.2f} dB")
    print("\nRestored (MAIR+ Output vs Clear Ground Truth):")
    print(f"  Avg SSIM : {avg_rest_ssim:.4f}  (Gain: +{avg_ssim_gain:.4f})")
    print(f"  Avg PSNR : {avg_rest_psnr:.2f} dB (Gain: +{avg_psnr_gain:.2f} dB)")
    print("==================================================")

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"synthetic_benchmark_{timestamp}.json")
    
    final_report = {
        "metadata": {
            "timestamp": timestamp,
            "images_processed": len(results),
            "total_time_seconds": round(total_time, 2)
        },
        "averages": {
            "baseline_ssim": round(avg_base_ssim, 4),
            "baseline_psnr": round(avg_base_psnr, 2),
            "restored_ssim": round(avg_rest_ssim, 4),
            "restored_psnr": round(avg_rest_psnr, 2),
            "ssim_gain": round(avg_ssim_gain, 4),
            "psnr_gain": round(avg_psnr_gain, 2)
        },
        "details": results
    }
    
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Saved detailed JSON report to {report_file}")

if __name__ == "__main__":
    run_synthetic_benchmark()
