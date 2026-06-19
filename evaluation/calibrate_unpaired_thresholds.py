import os
import sys
import json
import argparse
import itertools
import time
from datetime import datetime
import glob
import cv2
import torch
import shutil

# Make sure we can import from MAIR+
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import piq
except ImportError:
    print("❌ Missing required library for non-reference metrics!")
    print("Please run this exact command in your terminal first:")
    print("pip install piq")
    sys.exit(1)

from scheduler.scheduler import run_three_stage_scheduler

CONFIG_PATH = os.path.join("config", "thresholds.json")

# Default threshold search grid for surgical smoke (focused on lower thresholds)
DEFAULT_GRID = [0.05, 0.10, 0.15, 0.20]

def get_brisque(img_path):
    """Calculate BRISQUE. LOWER is better."""
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        score = piq.brisque(img_t)
    return score.item()

def _patch_thresholds(thresholds: dict) -> None:
    """Write a temporary thresholds.json so the scheduler picks them up."""
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    data = {
        **thresholds,
        "calibrated":  False,   # still False during search
        "calibrated_on": None,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _save_calibrated(thresholds: dict, best_gain: float, grid: list) -> None:
    """Write the final calibrated thresholds."""
    data = {
        "_comment":           "MAIR+ calibrated thresholds for unpaired medical datasets (BRISQUE)",
        "compression":        thresholds["compression"],
        "imaging":            thresholds["imaging"],
        "scene":              thresholds["scene"],
        "calibrated":         True,
        "calibrated_on":      datetime.now().isoformat(),
        "calibration_metric": "avg_brisque_gain",
        "best_avg_brisque_gain": best_gain,
        "search_range":       grid,
    }
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Calibrate Unpaired] Saved calibrated thresholds → {CONFIG_PATH}")
    print(f"  compression : {thresholds['compression']}")
    print(f"  imaging     : {thresholds['imaging']}")
    print(f"  scene       : {thresholds['scene']}")
    print(f"  best BRISQUE Δ : +{best_gain:.4f} (Higher is better improvement)")

def _eval_with_thresholds(images: list, thresholds: dict) -> float:
    """
    Run the pipeline on all images with the given stage thresholds.
    Returns average BRISQUE gain (Baseline - Restored). Higher gain is better.
    """
    _patch_thresholds(thresholds)

    gains = []
    try:
        for img_path in images:
            try:
                base_brisque = get_brisque(img_path)
                if base_brisque is None:
                    continue
                    
                result = run_three_stage_scheduler(
                    input_path=img_path, 
                    verbose=False, 
                    use_memory=False
                )
                restored = result["output_path"]
                if restored and os.path.exists(restored):
                    rest_brisque = get_brisque(restored)
                    if rest_brisque is not None:
                        # BRISQUE goes down = quality goes up.
                        # Gain = Baseline - Restored (positive is good)
                        gains.append(base_brisque - rest_brisque)
            except Exception as e:
                print(f"    [Calibrate Unpaired] Error on {os.path.basename(img_path)}: {e}")
    finally:
        pass

    return round(sum(gains) / len(gains), 4) if gains else -999.0

def calibrate(dataset_folder: str, grid: list = DEFAULT_GRID, max_images: int = 5):
    print("\n" + "=" * 60)
    print("  MAIR+ Unpaired Threshold Calibration (BRISQUE)")
    print("=" * 60)

    if not os.path.exists(dataset_folder):
        print(f"❌ Cannot find dataset folder: {dataset_folder}")
        return
        
    all_images = sorted(glob.glob(os.path.join(dataset_folder, "*.png")))
    if not all_images:
        print(f"❌ No PNG images found in {dataset_folder}")
        return
        
    if max_images > 0 and max_images < len(all_images):
        all_images = all_images[:max_images]

    combinations = list(itertools.product(grid, repeat=3))
    total = len(combinations)
    
    print(f"\n  Grid     : {grid}")
    print(f"  Combos   : {total}  (compression × imaging × scene)")
    print(f"  Images   : {len(all_images)}")
    
    best_gain   = -999.0
    best_thresh = {"compression": 0.20, "imaging": 0.20, "scene": 0.20}

    t_cal_start = time.time()
    for idx, (c_t, i_t, s_t) in enumerate(combinations, 1):
        thresholds = {"compression": c_t, "imaging": i_t, "scene": s_t}
        print(f"\n  [{idx:3d}/{total}] comp={c_t}  img={i_t}  scene={s_t}", end="  ", flush=True)

        gain = _eval_with_thresholds(all_images, thresholds)
        print(f"→ avg BRISQUE gain = {gain:+.4f}", end="")

        if gain > best_gain:
            best_gain   = gain
            best_thresh = thresholds
            print("  ★ NEW BEST", end="")
        print()

    elapsed = round(time.time() - t_cal_start, 1)
    print(f"\n  Calibration complete in {elapsed}s.")

    _save_calibrated(best_thresh, best_gain, grid)
    return best_thresh

def main():
    parser = argparse.ArgumentParser(description="MAIR+ Unpaired Threshold Calibration using BRISQUE")
    parser.add_argument("--dataset", type=str, required=True, help="Folder containing unpaired hazy images")
    parser.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID, help=f"Threshold values to search (default: {DEFAULT_GRID})")
    parser.add_argument("--max_images", type=int, default=5, help="Maximum images to evaluate per combination (default: 5)")
    args = parser.parse_args()

    calibrate(dataset_folder=args.dataset, grid=args.grid, max_images=args.max_images)

if __name__ == "__main__":
    main()
