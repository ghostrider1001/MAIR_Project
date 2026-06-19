import os
import glob
import time
import pandas as pd
from scheduler.scheduler import run_scheduler
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr, compute_lpips

BASE_DIR = "datasets/academic_subsets"
SUBSETS = ["BSD68_subset", "Set14_subset", "Kodak_subset", "LIVE1_subset"]


def main():
    print("="*60)
    print("  MAIR+ v2 ACADEMIC SUBSET EVALUATION")
    print("="*60)
    
    results = []
    
    for subset in SUBSETS:
        sub_dir = os.path.join(BASE_DIR, subset)
        deg_dir = os.path.join(sub_dir, "degraded")
        gt_dir = os.path.join(sub_dir, "ground_truth")
        
        if not os.path.exists(deg_dir):
            print(f"Skipping {subset} (Not found)")
            continue
            
        images = glob.glob(os.path.join(deg_dir, "*.png"))
        if not images:
            continue
            
        print(f"\nEvaluating {subset} ({len(images)} images)...")
        
        base_psnr_list, base_ssim_list, base_lpips_list = [], [], []
        rest_psnr_list, rest_ssim_list, rest_lpips_list = [], [], []
        runtimes = []
        
        for deg_path in images:
            basename = os.path.basename(deg_path)
            gt_path = os.path.join(gt_dir, basename)
            
            # Baseline metrics
            import cv2
            gt_img = cv2.imread(gt_path)
            deg_img = cv2.imread(deg_path)
            
            if gt_img is None or deg_img is None: continue
            
            try:
                base_ssim, base_psnr = compute_ssim_psnr(gt_img, deg_img)
            except Exception:
                continue
                
            base_lpips = compute_lpips(gt_path, deg_path)
            
            # Restoration
            t0 = time.time()
            rest_path = run_scheduler(deg_path, verbose=False, three_stage=True)
            dt = time.time() - t0
            
            if rest_path and os.path.exists(rest_path):
                rest_img = cv2.imread(rest_path)
                if rest_img is not None:
                    # resize rest to gt if diff
                    if rest_img.shape != gt_img.shape:
                        rest_img = cv2.resize(rest_img, (gt_img.shape[1], gt_img.shape[0]))
                    rest_ssim, rest_psnr = compute_ssim_psnr(gt_img, rest_img)
                    rest_lpips = compute_lpips(gt_path, rest_path)
                else:
                    rest_ssim, rest_psnr, rest_lpips = base_ssim, base_psnr, base_lpips
            else:
                rest_ssim, rest_psnr, rest_lpips = base_ssim, base_psnr, base_lpips
                
            base_psnr_list.append(base_psnr)
            base_ssim_list.append(base_ssim)
            if base_lpips is not None: base_lpips_list.append(base_lpips)
            
            rest_psnr_list.append(rest_psnr)
            rest_ssim_list.append(rest_ssim)
            if rest_lpips is not None: rest_lpips_list.append(rest_lpips)
            runtimes.append(dt)
            
        avg_b_psnr = sum(base_psnr_list)/len(base_psnr_list) if base_psnr_list else 0
        avg_r_psnr = sum(rest_psnr_list)/len(rest_psnr_list) if rest_psnr_list else 0
        
        avg_b_ssim = sum(base_ssim_list)/len(base_ssim_list) if base_ssim_list else 0
        avg_r_ssim = sum(rest_ssim_list)/len(rest_ssim_list) if rest_ssim_list else 0
        
        avg_b_lpips = sum(base_lpips_list)/len(base_lpips_list) if base_lpips_list else 0
        avg_r_lpips = sum(rest_lpips_list)/len(rest_lpips_list) if rest_lpips_list else 0
        
        avg_time = sum(runtimes)/len(runtimes) if runtimes else 0
        
        print(f"  PSNR:  {avg_b_psnr:.2f} -> {avg_r_psnr:.2f}  (Gain: +{avg_r_psnr - avg_b_psnr:.2f} dB)")
        print(f"  SSIM:  {avg_b_ssim:.4f} -> {avg_r_ssim:.4f}  (Gain: +{avg_r_ssim - avg_b_ssim:.4f})")
        print(f"  LPIPS: {avg_b_lpips:.4f} -> {avg_r_lpips:.4f}")
        print(f"  Time:  {avg_time:.2f} sec/image")
        
        results.append({
            "Dataset": subset.replace("_subset", ""),
            "Images": len(images),
            "PSNR_Gain": round(avg_r_psnr - avg_b_psnr, 2),
            "SSIM_Gain": round(avg_r_ssim - avg_b_ssim, 4),
            "LPIPS_Deg": round(avg_b_lpips, 4),
            "LPIPS_Rest": round(avg_r_lpips, 4),
            "Avg_Time": round(avg_time, 2)
        })
        
    df = pd.DataFrame(results)
    df.to_csv("academic_benchmark_results.csv", index=False)
    print("\nResults saved to academic_benchmark_results.csv")
    
    print("\n" + "="*60)
    print(" LaTeX Table Code:")
    print("="*60)
    for row in results:
        lpips_gain = row['LPIPS_Rest'] - row['LPIPS_Deg']
        sign = "+" if row['SSIM_Gain'] > 0 else ""
        print(f"\\texttt{{{row['Dataset'].lower()}}} & {row['Images']} & {sign}{row['SSIM_Gain']} & +{row['PSNR_Gain']} dB & {lpips_gain:.3f} & {row['Avg_Time']}s \\\\")

if __name__ == "__main__":
    main()
