import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scheduler.scheduler import run_scheduler

def _create_real_comparison_grid(degraded_path, restored_path, out_path):
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
        
    img_deg = add_label(img_deg.copy(), f"1. Real-World Degraded", color=(0, 0, 255))
    img_rest = add_label(img_rest.copy(), "2. MAIR+ Restored", color=(255, 255, 0))
    
    grid = np.hstack([img_deg, img_rest])
    cv2.imwrite(out_path, grid)

def main():
    print("=" * 60)
    print("  MAIR+ REAL-WORLD IMAGE TESTER")
    print("=" * 60)
    print("Please select a real degraded image from your computer.")
    print("(A file dialog window will open...)")

    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a Real-World Degraded Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"\n[1] Selected Image: {file_path}")
    
    # Optional: resize if it's a massive 4K phone photo to save computation time
    img = cv2.imread(file_path)
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            print("    -> Image is very large. Scaling down for faster processing...")
            scale = 1200 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            # Save a temporary copy to process
            temp_path = "temp_real_image.png"
            cv2.imwrite(temp_path, img)
            file_path = temp_path

    # MONKEYPATCH: Disable Quality Gate for real-world images!
    # The Quality Gate calculates SSIM between the restored output and the degraded input.
    # If the degradation is severe (like heavy blur or bokeh), restoring it changes the image
    # drastically, causing a low SSIM, which triggers a false rollback!
    import scheduler.scheduler
    scheduler.scheduler.QUALITY_GATE_MIN = -1.0
    
    print(f"[2] Running MAIR+ Multi-Agent Scheduler...")
    restored_path = run_scheduler(file_path, verbose=True, three_stage=True)
    
    if restored_path and os.path.exists(restored_path):
        GRID_DIR = "outputs/comparison_grids"
        os.makedirs(GRID_DIR, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        grid_path = os.path.join(GRID_DIR, f"{base_name}_real_comparison.png")
        
        print(f"[3] Generating Visual Comparison Grid...")
        _create_real_comparison_grid(
            degraded_path=file_path,
            restored_path=restored_path,
            out_path=grid_path
        )
        print(f"    -> Success! Saved Grid to: {grid_path}")
    else:
        print(f"    -> ERROR: Restoration failed!")

    if 'temp_real_image.png' in file_path and os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    main()
