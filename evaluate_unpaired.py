import os
import sys
import glob
import cv2
import torch
import numpy as np
import shutil
import csv

# Make sure we can import from MAIR+
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.scheduler import run_three_stage_scheduler

try:
    import piq
except ImportError:
    print("❌ Missing required library for non-reference metrics!")
    print("Please run this exact command in your terminal first:")
    print("pip install piq")
    sys.exit(1)

def get_brisque(img_path):
    """Calculate BRISQUE. LOWER is better."""
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        score = piq.brisque(img_t)
    return score.item()

def main():
    hazy_folder = r"datasets\PG002\hazy_imgs"
    output_folder = r"outputs\desmoke_results\PG002"
    os.makedirs(output_folder, exist_ok=True)
    
    hazy_images = sorted(glob.glob(os.path.join(hazy_folder, "*.png")))
    
    if not hazy_images:
        print(f"❌ No images found in {hazy_folder}")
        return
        
    print(f"🔍 Found {len(hazy_images)} hazy images. Processing the first 10 with BRISQUE (Non-Reference Metric).")
    print(f"NOTE: For BRISQUE, a LOWER score means HIGHER quality / less distortion.\n")
    
    results = []
    
    for img_path in hazy_images[:10]:
        filename = os.path.basename(img_path)
        print(f"==========================================")
        print(f"🚀 Processing: {filename}")
        
        # Calculate Baseline Metrics (Before)
        baseline_brisque = get_brisque(img_path)
        print(f"📊 Baseline Quality -> BRISQUE: {baseline_brisque:.2f}")
        
        result = run_three_stage_scheduler(
            input_path=img_path,
            verbose=False,
            voting=True,
            use_memory=True
        )
        
        restored_path = result.get("output_path")
        
        if restored_path and os.path.exists(restored_path):
            new_path = os.path.join(output_folder, f"restored_{filename}")
            shutil.copy(restored_path, new_path)
            
            # Calculate Restored Metrics (After)
            restored_brisque = get_brisque(new_path)
            brisque_gain = baseline_brisque - restored_brisque
            
            print(f"📊 Restored Quality -> BRISQUE: {restored_brisque:.2f}")
            print(f"📈 Improvement      -> BRISQUE: {brisque_gain:+.2f}")
            
            results.append({
                "filename": filename,
                "baseline_brisque": baseline_brisque,
                "restored_brisque": restored_brisque,
                "brisque_gain": brisque_gain
            })
        else:
            print(f"❌ Failed to restore {filename}")

    # Print final summary table
    if results:
        print("\n" + "="*50)
        print("  NON-REFERENCE EVALUATION SUMMARY (PG002)")
        print("  * Note: Lower BRISQUE means better perceptual quality")
        print("="*50)
        print(f"  {'FILE':<20} | {'BRISQUE (B -> R)':<20}")
        print("-" * 50)
        for r in results:
            brisq_str = f"{r['baseline_brisque']:.1f} -> {r['restored_brisque']:.1f}"
            print(f"  {r['filename']:<20} | {brisq_str:<20}")
        
        avg_brisque_gain = sum(r['brisque_gain'] for r in results) / len(results)
        
        print("-" * 50)
        print(f"  AVERAGE IMPROVEMENT: BRISQUE improved by {avg_brisque_gain:.2f}")
        print("="*50)

if __name__ == "__main__":
    main()
