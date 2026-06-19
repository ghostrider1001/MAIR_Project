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
    
    # Ensure same height
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
        
    img_deg = add_label(img_deg.copy(), f"1. Real Degraded ({title})", color=(0, 0, 255))
    img_rest = add_label(img_rest.copy(), "2. MAIR+ Restored", color=(255, 255, 0))
    
    grid = np.hstack([img_deg, img_rest])
    cv2.imwrite(out_path, grid)

REAL_IMAGES = {
    "haze": "https://loremflickr.com/640/480/smog,fog",
    "lowlight": "https://loremflickr.com/640/480/night,dark",
    "blur": "https://loremflickr.com/640/480/motion,blur",
    "noise": "https://loremflickr.com/640/480/grain,high-iso",
    "rain": "https://loremflickr.com/640/480/rain,street",
}

OUT_DIR = "datasets/real_world_tests/degraded"
GRID_DIR = "outputs/comparison_grids"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

print("=" * 60)
print("  MAIR+ REAL-WORLD DEGRADATION TEST")
print("=" * 60)

for deg_type, url in REAL_IMAGES.items():
    print(f"\n[1] Downloading Real '{deg_type.upper()}' image...")
    img_path = os.path.join(OUT_DIR, f"real_{deg_type}.jpg")
    try:
        # User-agent header to avoid 403/400 Forbidden
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"Failed to download {deg_type} image: {e}")
        continue
    
    # Optional: Resize if it's too large to save computation time
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imwrite(img_path, img)

    print(f"[2] Running MAIR+ Multi-Agent Scheduler on: {deg_type.upper()}")
    # We pass three_stage=True so it uses the full dynamic pipeline
    restored_path = run_scheduler(img_path, verbose=True, three_stage=True)
    
    if restored_path and os.path.exists(restored_path):
        print(f"[3] Generating Visual Comparison Grid...")
        grid_path = os.path.join(GRID_DIR, f"real_{deg_type}_comparison.png")
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
