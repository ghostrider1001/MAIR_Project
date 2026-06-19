import os
import urllib.request
import cv2
import sys
import numpy as np
from scheduler.scheduler import run_scheduler

def _create_real_comparison_grid(degraded_path, restored_path, out_path, title):
    img_deg = cv2.imread(degraded_path)
    img_rest = cv2.imread(restored_path)
    if img_deg is None or img_rest is None: return
    
    h = max(img_deg.shape[0], img_rest.shape[0])
    w = img_deg.shape[1]
    
    def resize_to_h(img, target_h):
        if img.shape[0] != target_h:
            aspect = img.shape[1] / img.shape[0]
            return cv2.resize(img, (int(target_h * aspect), target_h))
        return img
        
    img_deg = resize_to_h(img_deg, h)
    img_rest = resize_to_h(img_rest, h)
    
    def add_label(img, text, color=(0, 255, 0)):
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0,0,0), -1)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img
        
    img_deg = add_label(img_deg.copy(), f"1. Real {title} (Academic Dataset)", color=(0, 0, 255))
    img_rest = add_label(img_rest.copy(), "2. MAIR+ Restored", color=(255, 255, 0))
    
    grid = np.hstack([img_deg, img_rest])
    cv2.imwrite(out_path, grid)

# Using raw GitHub URLs from famous computer vision repositories! 
# GitHub raw URLs NEVER rate-limit like Wikipedia does.
ACADEMIC_IMAGES = {
    "blur": "https://raw.githubusercontent.com/Megvii-CSG/NAFNet/main/demo/blur/GOPR0854_11_00-000090.png",
    "noise": "https://raw.githubusercontent.com/Megvii-CSG/NAFNet/main/demo/noisy/noisy.png",
    "haze": "https://raw.githubusercontent.com/Li-Chongyi/Dehamer/master/classic_test_image/canyon.png",
}

OUT_DIR = "datasets/real_world_tests/degraded"
GRID_DIR = "outputs/comparison_grids"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

print("=" * 60)
print("  MAIR+ ACADEMIC DATASET REAL-WORLD TEST")
print("=" * 60)

for deg_type, url in ACADEMIC_IMAGES.items():
    print(f"\n[1] Downloading Academic '{deg_type.upper()}' image...")
    img_path = os.path.join(OUT_DIR, f"academic_{deg_type}.png")
    try:
        # Standard urllib request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"Failed to download {deg_type} image: {e}")
        continue
    
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        # Resize if huge to save CPU computation time
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imwrite(img_path, img)

    print(f"[2] Running MAIR+ Multi-Agent Scheduler on: {deg_type.upper()}")
    restored_path = run_scheduler(img_path, verbose=True, three_stage=True)
    
    if restored_path and os.path.exists(restored_path):
        print(f"[3] Generating Visual Comparison Grid...")
        grid_path = os.path.join(GRID_DIR, f"academic_{deg_type}_comparison.png")
        _create_real_comparison_grid(
            degraded_path=img_path,
            restored_path=restored_path,
            out_path=grid_path,
            title=deg_type.title()
        )
        print(f"    -> Saved Grid to: {grid_path}")
    else:
        print(f"    -> ERROR: Restoration rolled back or failed!")

print("\n" + "=" * 60)
print("  All real-world tests completed! Check outputs/comparison_grids/")
print("=" * 60)
