"""
Threshold Calibration Script (C8)
===================================
MAIR+ Contribution C8: Calibrated Stage Thresholds.

Sweeps a range of stage activation thresholds on a small held-out
calibration set and writes the best-performing thresholds to
config/thresholds.json.

Algorithm:
  For each combination of (compression_t, imaging_t, scene_t) from
  a predefined grid, run the pipeline on each calibration image and
  compute average SSIM gain. The combination with the highest average
  gain is saved as the calibrated thresholds.

The calibration set should be small (≤10 images per degradation type)
and separate from the benchmark test set. Place it under:
    datasets/calibration/<set_name>/degraded/
    datasets/calibration/<set_name>/reference/

Usage:
    python evaluation/calibrate_thresholds.py
    python evaluation/calibrate_thresholds.py --max_images 5 --fast_only
    python evaluation/calibrate_thresholds.py --grid 0.15 0.20 0.25 0.30

Output:
    config/thresholds.json  (updated in place)

WARNING: Full grid search is expensive. Use --fast_only or --max_images
to limit runtime. Typically 15–30 minutes on CPU with default settings.
"""

import os
import sys
import json
import argparse
import itertools
import time
from datetime import datetime

# make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import list_image_pairs, compute_metrics, baseline_metrics
from core.degradation_detector import detect_degradation


CALIBRATION_ROOT = os.path.join("datasets", "calibration")
CONFIG_PATH      = os.path.join("config", "thresholds.json")

# Default threshold search grid
DEFAULT_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

# Stage names
STAGES = ["compression", "imaging", "scene"]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def find_calibration_sets(root: str = CALIBRATION_ROOT) -> list:
    """Return paths to valid calibration sets (have degraded/ and reference/)."""
    sets = []
    if not os.path.isdir(root):
        return sets
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            if (os.path.isdir(os.path.join(path, "degraded")) and
                    os.path.isdir(os.path.join(path, "reference"))):
                sets.append(path)
    return sets


def _eval_with_thresholds(
    pairs:      list,
    thresholds: dict,
    fast_only:  bool = False,
    max_images: int  = 0,
) -> float:
    """
    Run the pipeline on all pairs with the given stage thresholds.
    Returns average SSIM gain (higher is better).
    """
    import core.tool_registry as _tr
    from scheduler.scheduler import run_three_stage_scheduler

    FAST_EXPERTS = {"opencv_denoise", "clahe_lowlight", "opencv_fast_jpeg",
                    "opencv_unsharp_deblur"}

    if max_images and max_images < len(pairs):
        pairs = pairs[:max_images]

    # Temporarily patch thresholds JSON on disk
    _patch_thresholds(thresholds)

    orig_registry = None
    if fast_only:
        orig_registry = dict(_tr.REGISTRY)
        _tr.REGISTRY  = {k: v for k, v in orig_registry.items() if k in FAST_EXPERTS}

    gains = []
    try:
        for deg_path, ref_path in pairs:
            try:
                base    = baseline_metrics(deg_path, ref_path)
                result  = run_three_stage_scheduler(deg_path, verbose=False, use_memory=False)
                restored = result["output_path"]
                if restored and os.path.exists(restored):
                    metrics  = compute_metrics(ref_path, restored)
                    if base["ssim"] is not None and metrics["ssim"] is not None:
                        gains.append(metrics["ssim"] - base["ssim"])
            except Exception as e:
                print(f"    [Calibrate] Error on {os.path.basename(deg_path)}: {e}")
    finally:
        if orig_registry is not None:
            _tr.REGISTRY = orig_registry

    return round(sum(gains) / len(gains), 4) if gains else -999.0


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
        "_comment":           "MAIR+ calibrated thresholds (C8)",
        "compression":        thresholds["compression"],
        "imaging":            thresholds["imaging"],
        "scene":              thresholds["scene"],
        "calibrated":         True,
        "calibrated_on":      datetime.now().isoformat(),
        "calibration_metric": "avg_ssim_gain",
        "best_avg_ssim_gain": best_gain,
        "search_range":       grid,
    }
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Calibrate] Saved calibrated thresholds → {CONFIG_PATH}")
    print(f"  compression : {thresholds['compression']}")
    print(f"  imaging     : {thresholds['imaging']}")
    print(f"  scene       : {thresholds['scene']}")
    print(f"  best SSIM Δ : +{best_gain:.4f}")


# ─────────────────────────────────────────────────────────────
# MAIN CALIBRATION LOOP
# ─────────────────────────────────────────────────────────────

def calibrate(
    grid:       list = DEFAULT_GRID,
    fast_only:  bool = False,
    max_images: int  = 0,
    dry_run:    bool = False,
) -> dict:
    """
    Grid-search optimal stage thresholds on the calibration set.

    Args:
        grid        : threshold values to search over (per stage)
        fast_only   : use only CPU-fast experts during calibration
        max_images  : limit images per calibration set
        dry_run     : print what would be searched without running

    Returns:
        Best threshold dict.
    """
    print("\n" + "=" * 60)
    print("  MAIR+ Threshold Calibration (C8)")
    print("=" * 60)

    # Collect all calibration pairs
    cal_sets = find_calibration_sets()
    if not cal_sets:
        print(f"\n[Calibrate] No calibration sets found under '{CALIBRATION_ROOT}'.")
        print(f"  Create: {CALIBRATION_ROOT}/<set_name>/degraded/ and reference/")
        print(f"  Using default thresholds (all stages = 0.20).")
        _save_calibrated(
            {"compression": 0.20, "imaging": 0.20, "scene": 0.20},
            best_gain=0.0,
            grid=grid,
        )
        return {"compression": 0.20, "imaging": 0.20, "scene": 0.20}

    all_pairs = []
    for s in cal_sets:
        pairs = list_image_pairs(s)
        print(f"  Found calibration set: {os.path.basename(s)} ({len(pairs)} pairs)")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("[Calibrate] No valid image pairs found.")
        return {}

    # Build grid combinations
    combinations = list(itertools.product(grid, repeat=3))
    total = len(combinations)
    print(f"\n  Grid     : {grid}")
    print(f"  Combos   : {total}  (compression × imaging × scene)")
    print(f"  Images   : {len(all_pairs)}" + (f" (limited to {max_images})" if max_images else ""))
    print(f"  FastOnly : {fast_only}")

    if dry_run:
        print(f"\n[Calibrate] DRY RUN — would search {total} combinations.")
        return {}

    best_gain   = -999.0
    best_thresh = {"compression": 0.20, "imaging": 0.20, "scene": 0.20}

    t_cal_start = time.time()
    for idx, (c_t, i_t, s_t) in enumerate(combinations, 1):
        thresholds = {"compression": c_t, "imaging": i_t, "scene": s_t}
        print(f"\n  [{idx:3d}/{total}] comp={c_t}  img={i_t}  scene={s_t}", end="  ", flush=True)

        gain = _eval_with_thresholds(all_pairs, thresholds, fast_only, max_images)
        print(f"→ avg SSIM gain = {gain:+.4f}", end="")

        if gain > best_gain:
            best_gain   = gain
            best_thresh = thresholds
            print("  ★ NEW BEST", end="")
        print()

    elapsed = round(time.time() - t_cal_start, 1)
    print(f"\n  Calibration complete in {elapsed}s.")

    _save_calibrated(best_thresh, best_gain, grid)
    return best_thresh


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ Threshold Calibration (C8) — grid-search optimal stage thresholds"
    )
    parser.add_argument(
        "--grid", type=float, nargs="+",
        default=DEFAULT_GRID,
        metavar="T",
        help=f"Threshold values to search (default: {DEFAULT_GRID})",
    )
    parser.add_argument(
        "--fast_only", action="store_true",
        help="Use only CPU-fast experts during calibration (much faster)",
    )
    parser.add_argument(
        "--max_images", type=int, default=3,
        help="Maximum images per calibration set (default: 3, use 0 for all)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be searched without running anything",
    )
    args = parser.parse_args()

    calibrate(
        grid=args.grid,
        fast_only=args.fast_only,
        max_images=args.max_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
