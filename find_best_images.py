import os
import sys
import glob
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation.quality_evaluator import _compute_ssim_psnr as compute_ssim_psnr

def get_best_image_for_pct(pct):
    candidates = glob.glob(f"datasets/synthetic_smoke_sweep/deg_{pct}_*.png")
    best_candidate = None
    best_psnr_gain = -999.0
    
    for deg_path in candidates:
        base_name = os.path.basename(deg_path).replace(f"deg_{pct}_", "")
        
        orig_clear_search = glob.glob(f"datasets/DeSmoke-LAP dataset/Dataset/*/clear/{base_name}")
        if not orig_clear_search:
            continue
            
        orig_clear_img = cv2.imread(orig_clear_search[0])
        
        rest_search = glob.glob(f"outputs/*/deg_{pct}_{base_name.replace('.png', '')}*.png")
        if not rest_search:
            continue
            
        rest_path = rest_search[0]
        
        deg_img = cv2.imread(deg_path)
        rest_img = cv2.imread(rest_path)
        
        if rest_img is None or deg_img is None or orig_clear_img is None:
            continue
            
        if rest_img.shape != orig_clear_img.shape:
            rest_img = cv2.resize(rest_img, (orig_clear_img.shape[1], orig_clear_img.shape[0]))
        if deg_img.shape != orig_clear_img.shape:
            deg_img = cv2.resize(deg_img, (orig_clear_img.shape[1], orig_clear_img.shape[0]))
            
        _, deg_psnr = compute_ssim_psnr(orig_clear_img, deg_img)
        _, rest_psnr = compute_ssim_psnr(orig_clear_img, rest_img)
        
        gain = rest_psnr - deg_psnr
        if gain > best_psnr_gain:
            best_psnr_gain = gain
            
            best_candidate = {
                'pct': pct,
                'base_name': base_name,
                'deg_path': deg_path,
                'rest_path': rest_path,
            }
            
    return best_candidate

def main():
    print("Finding best images for each intensity...")
    for pct in [10, 20, 30, 40]:
        data = get_best_image_for_pct(pct)
        if data:
            print(f"[{pct}% Smoke] Best Image: {data['base_name']}")
            print(f"   Degraded Path: {data['deg_path']}")
            print(f"   Restored Path: {data['rest_path']}\n")
        else:
            print(f"[{pct}% Smoke] No valid images found.\n")

if __name__ == "__main__":
    main()
