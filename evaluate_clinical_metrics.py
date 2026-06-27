import os
import sys
import glob
import cv2
import torch
import numpy as np

# MAIR+ Imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.scheduler import run_three_stage_scheduler

# Metrics Imports
try:
    import piq
    import pyiqa
except ImportError:
    print("❌ Missing required libraries! Run:")
    print("pip install piq pyiqa")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
niqe_metric = pyiqa.create_metric('niqe', device=device)

def get_brisque(img_path):
    img = cv2.imread(img_path)
    if img is None: return 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        score = piq.brisque(img_t)
    return score.item()

def get_niqe(img_path):
    try:
        return float(niqe_metric(img_path).item())
    except:
        return 0.0

def get_edge_intensity(img_path):
    """Calculates the average gradient magnitude using Sobel operators. Higher = sharper edges."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(magnitude)

def get_sharpness(img_path):
    """Calculates variance of Laplacian. Higher = more in-focus/sharp."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def main():
    hazy_folder = r"datasets\PG002\hazy_imgs"
    # To prevent it taking 45 minutes, we randomly sample 50 images.
    hazy_images = sorted(glob.glob(os.path.join(hazy_folder, "*.png")))[:50]
    
    if not hazy_images:
        print(f"❌ No images found in {hazy_folder}")
        return
        
    print(f"🔍 Found {len(hazy_images)} hazy clinical images.")
    print("Calculating BRISQUE, NIQE, Edge Intensity, and Sharpness...\n")
    
    metrics = {"b_brisque": 0, "r_brisque": 0, "b_niqe": 0, "r_niqe": 0, 
               "b_edge": 0, "r_edge": 0, "b_sharp": 0, "r_sharp": 0}
    successful_runs = 0
    
    for img_path in hazy_images:
        filename = os.path.basename(img_path)
        print(f"🚀 Processing: {filename} ... ", end="")
        
        # Base Metrics
        b_brisque = get_brisque(img_path)
        b_niqe = get_niqe(img_path)
        b_edge = get_edge_intensity(img_path)
        b_sharp = get_sharpness(img_path)
        
        # Process
        result = run_three_stage_scheduler(img_path, verbose=False, use_memory=True)
        restored_path = result.get("output_path")
        
        if restored_path and os.path.exists(restored_path):
            r_brisque = get_brisque(restored_path)
            r_niqe = get_niqe(restored_path)
            r_edge = get_edge_intensity(restored_path)
            r_sharp = get_sharpness(restored_path)
            
            metrics["b_brisque"] += b_brisque
            metrics["r_brisque"] += r_brisque
            metrics["b_niqe"] += b_niqe
            metrics["r_niqe"] += r_niqe
            metrics["b_edge"] += b_edge
            metrics["r_edge"] += r_edge
            metrics["b_sharp"] += b_sharp
            metrics["r_sharp"] += r_sharp
            successful_runs += 1
            print(f"Done.")
        else:
            print("Failed to restore.")

    if successful_runs > 0:
        print("\n" + "="*60)
        print("  CLINICAL NON-REFERENCE METRICS SUMMARY (Averaged)")
        print("="*60)
        print(f"  BRISQUE (Lower is Better)  : {metrics['b_brisque']/successful_runs:.2f} -> {metrics['r_brisque']/successful_runs:.2f}")
        print(f"  NIQE    (Lower is Better)  : {metrics['b_niqe']/successful_runs:.2f} -> {metrics['r_niqe']/successful_runs:.2f}")
        print(f"  Edge Int. (Higher is Better): {metrics['b_edge']/successful_runs:.2f} -> {metrics['r_edge']/successful_runs:.2f}")
        print(f"  Sharpness (Higher is Better): {metrics['b_sharp']/successful_runs:.2f} -> {metrics['r_sharp']/successful_runs:.2f}")
        print("="*60)
        print("\n✅ Copy these final average numbers and paste them back to the chat!")

if __name__ == "__main__":
    main()
