import os
import cv2
import json
import numpy as np
import argparse
from glob import glob

def get_latest_json(test_name):
    """Find the most recent evaluation JSON for a given test."""
    files = glob(f"results/benchmark_{test_name}_*.json")
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def build_comparison_grid(image_name):
    print(f"============================================================")
    print(f" Generating 1x3 Visual Proof Grid for: {image_name}")
    print(f"============================================================")
    
    out_dir = os.path.join("outputs", "comparison_grids")
    os.makedirs(out_dir, exist_ok=True)
    
    tests = ["blur", "noise", "haze", "lowlight", "jpeg", "mixed", "rain"]
    
    for t in tests:
        clean_path = os.path.join("datasets", "benchmark", f"{t}_test", "reference", image_name)
        deg_path = os.path.join("datasets", "benchmark", f"{t}_test", "degraded", image_name)
        
        if not os.path.exists(clean_path) or not os.path.exists(deg_path):
            print(f"[{t.upper()}] Skipping: Image not found in dataset.")
            continue
            
        latest_json = get_latest_json(f"{t}_test")
        if not latest_json:
            print(f"[{t.upper()}] Skipping: No evaluation results found.")
            continue
            
        restored_path = None
        metrics = {}
        with open(latest_json, "r") as f:
            data = json.load(f)
            for item in data.get("per_image", []):
                if item["file"] == image_name:
                    restored_path = item.get("restored_path")
                    metrics = item
                    break
                    
        if not restored_path or not os.path.exists(restored_path):
            print(f"[{t.upper()}] Skipping: Restored image not found.")
            continue
            
        # Load images
        img_clean = cv2.imread(clean_path)
        img_deg = cv2.imread(deg_path)
        img_rest = cv2.imread(restored_path)
        
        if img_clean is None or img_deg is None or img_rest is None:
            continue
            
        # Ensure all are same height for horizontal concatenation
        h = max(img_clean.shape[0], img_deg.shape[0], img_rest.shape[0])
        w = img_clean.shape[1]
        
        def resize_to_h(img, target_h):
            if img.shape[0] != target_h:
                aspect = img.shape[1] / img.shape[0]
                new_w = int(target_h * aspect)
                return cv2.resize(img, (new_w, target_h))
            return img
            
        img_clean = resize_to_h(img_clean, h)
        img_deg = resize_to_h(img_deg, h)
        img_rest = resize_to_h(img_rest, h)
        
        # Add labels directly onto the images
        def add_label(img, text, color=(0, 255, 0)):
            # Dark background box for text readability
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0,0,0), -1)
            cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img
            
        img_clean = add_label(img_clean.copy(), "1. Original Clean")
        
        deg_label = f"1. Degraded ({t.upper()})"
        img_deg = add_label(img_deg.copy(), deg_label, color=(0, 0, 255))
        
        rest_label = f"2. MAIR+ Restored"
        img_rest = add_label(img_rest.copy(), rest_label, color=(255, 255, 0))
        
        # Concatenate horizontally (1x2 Grid)
        grid = np.hstack([img_deg, img_rest])
        
        # Save Grid
        save_path = os.path.join(out_dir, f"{os.path.splitext(image_name)[0]}_{t}_comparison.png")
        cv2.imwrite(save_path, grid)
        
        print(f"\n[{t.upper()} TEST]")
        print(f"  Saved Grid To : {save_path}")
        print(f"  Expert Used   : {metrics.get('primary', 'unknown')}")
        print(f"  Baseline PSNR : {metrics.get('baseline_psnr', 0):.2f} dB")
        print(f"  Restored PSNR : {metrics.get('restored_psnr', 0):.2f} dB  (Gain: +{metrics.get('psnr_gain', 0):.2f} dB)")
        print(f"  Baseline SSIM : {metrics.get('baseline_ssim', 0):.4f}")
        print(f"  Restored SSIM : {metrics.get('restored_ssim', 0):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Side-by-Side Visual Proof Grid")
    parser.add_argument("--image", type=str, default="bird.png", help="The name of the image to test (e.g. bird.png, baby.png)")
    args = parser.parse_args()
    
    build_comparison_grid(args.image)
