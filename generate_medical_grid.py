import os
import cv2
import sys
import numpy as np
from scheduler.scheduler import run_three_stage_scheduler

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_medical_grid.py <input_path> <output_name>")
        return
        
    deg_path = sys.argv[1]
    out_name = sys.argv[2]
    
    print(f"Generating Medical Grid for {deg_path} -> {out_name}...")
    
    out_dir = os.path.join("outputs", "comparison_grids")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(deg_path):
        print(f"Error: Could not find {deg_path}")
        return
        
    print(f"Running pipeline on {deg_path}")
    result = run_three_stage_scheduler(deg_path, verbose=False, use_memory=False, clinical_eval=True)
    restored_path = result.get("output_path")
    
    img_deg = cv2.imread(deg_path)
    img_rest = cv2.imread(restored_path) if restored_path else img_deg.copy()
    
    def add_label(img, text, color=(0, 255, 0)):
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0,0,0), -1)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img
        
    img_deg = add_label(img_deg.copy(), "1. Degraded", color=(0, 0, 255))
    img_rest = add_label(img_rest.copy(), "2. MAIR+ Restored", color=(255, 255, 0))
    
    grid = np.hstack([img_deg, img_rest])
    
    save_path = os.path.join(out_dir, out_name)
    cv2.imwrite(save_path, grid)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
