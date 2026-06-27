import os
import glob
import cv2
import json
import sys

def compute_summary():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr
    except ImportError:
        print("Failed to import compute_ssim_psnr. Are you running this from the project root?")
        sys.exit(1)

    candidates = glob.glob(os.path.join("datasets", "synthetic_smoke_sweep", "deg_*_*.png"))
    
    # We will accumulate psnr_gain for each folder and each percentile
    # Data structure: data[folder][percentile] = list of (deg_psnr, rest_psnr)
    data = {}
    
    for deg_path in candidates:
        base = os.path.basename(deg_path)
        parts = base.split('_')
        if len(parts) >= 4:
            pct = parts[1]
            folder = f"{parts[2]}_{parts[3]}" # e.g. TLH_10
            
            base_name = base.replace(f"deg_{pct}_", "")
            orig_clear_search = glob.glob(os.path.join("datasets", "DeSmoke-LAP dataset", "Dataset", "*", "clear", base_name))
            if not orig_clear_search: continue
            orig_clear_img = cv2.imread(orig_clear_search[0])
            
            rest_search = glob.glob(os.path.join("outputs", "*", f"deg_{pct}_{base_name.replace('.png', '')}*.png"))
            if not rest_search: continue
            rest_path = rest_search[0]
            
            deg_img = cv2.imread(deg_path)
            rest_img = cv2.imread(rest_path)
            
            if rest_img is None or deg_img is None or orig_clear_img is None: continue
            
            if rest_img.shape != orig_clear_img.shape:
                rest_img = cv2.resize(rest_img, (orig_clear_img.shape[1], orig_clear_img.shape[0]))
            if deg_img.shape != orig_clear_img.shape:
                deg_img = cv2.resize(deg_img, (orig_clear_img.shape[1], orig_clear_img.shape[0]))
                
            _, deg_psnr = compute_ssim_psnr(orig_clear_img, deg_img)
            _, rest_psnr = compute_ssim_psnr(orig_clear_img, rest_img)
            
            if folder not in data:
                data[folder] = {}
            if pct not in data[folder]:
                data[folder][pct] = []
            
            data[folder][pct].append((deg_psnr, rest_psnr))

    summary = {}
    for folder, pcts in data.items():
        summary[folder] = {}
        for pct, scores in pcts.items():
            avg_deg = sum(x[0] for x in scores) / len(scores)
            avg_rest = sum(x[1] for x in scores) / len(scores)
            avg_gain = avg_rest - avg_deg
            summary[folder][pct] = {
                'avg_deg_psnr': avg_deg,
                'avg_rest_psnr': avg_rest,
                'avg_gain_psnr': avg_gain,
                'count': len(scores)
            }
            
    with open("synthetic_smoke_sweep_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("Summary calculation complete! Results saved to synthetic_smoke_sweep_summary.json")
    
if __name__ == "__main__":
    compute_summary()
