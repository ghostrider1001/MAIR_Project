"""
MAIR+ v2 — Ablation Study Runner
===================================
Systematic comparison of pipeline configurations to validate
each contribution's individual impact.

Ablation experiments:
    A1: TSF vs Legacy (no three-stage ordering)
    A2: Memory ON vs OFF (C9 CaseStore)
    A3: Voting ON vs OFF (C12 ensemble)
    A4: With Quality Gate vs Without (C4 rollback)
    A5: Full v2 vs Baseline (all contributions combined)

Usage:
    python experiments/run_ablation.py --experiment A1
    python experiments/run_ablation.py --all
    python experiments/run_ablation.py --all --max_images 3 --fast_only
    python experiments/run_ablation.py --experiment A5 --dataset blur_test

Results saved to:
    results/ablation_<experiment>_<timestamp>.json
    results/ablation_<experiment>_<timestamp>.csv
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import (
    find_benchmark_sets, list_image_pairs,
    compute_metrics, baseline_metrics,
)
from core.degradation_detector import detect_degradation
from scheduler.scheduler import run_three_stage_scheduler, run_scheduler


RESULTS_DIR = "results"


# ─────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A1": {
        "name":        "Three-Stage Framework vs Legacy",
        "description": "Compares C2/C3-stage ordering against flat confidence-based selection",
        "configs": [
            {"label": "TSF (Three-Stage)",   "three_stage": True,  "voting": False, "use_memory": False},
            {"label": "Legacy (no TSF)",      "three_stage": False, "voting": False, "use_memory": False},
        ],
    },
    "A2": {
        "name":        "Memory-Augmented Planning (C9)",
        "description": "Compares CaseStore memory bias vs no memory",
        "configs": [
            {"label": "With Memory (C9)",    "three_stage": True, "voting": False, "use_memory": True},
            {"label": "No Memory",           "three_stage": True, "voting": False, "use_memory": False},
        ],
    },
    "A3": {
        "name":        "Expert Voting Ensemble (C12)",
        "description": "Compares top-2 voting vs single best expert per stage",
        "configs": [
            {"label": "Voting ON (C12)",     "three_stage": True, "voting": True,  "use_memory": True},
            {"label": "Standard (no vote)",  "three_stage": True, "voting": False, "use_memory": True},
        ],
    },
    "A4": {
        "name":        "Quality Gate Rollback (C4)",
        "description": "Compares full pipeline with C4 quality gate vs disabled (by patching threshold to 0.0)",
        "configs": [
            {"label": "Gate ON (C4)",        "three_stage": True, "voting": False, "use_memory": False, "gate_min": 0.50},
            {"label": "Gate OFF",            "three_stage": True, "voting": False, "use_memory": False, "gate_min": 0.00},
        ],
    },
    "A5": {
        "name":        "Full MAIR+ v2 vs v1 Baseline",
        "description": "All 12 contributions combined vs original TSF-only baseline",
        "configs": [
            {"label": "MAIR+ v2 (all C1-C12)", "three_stage": True,  "voting": True,  "use_memory": True},
            {"label": "MAIR+ v1 (baseline)",   "three_stage": True,  "voting": False, "use_memory": False},
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# SINGLE CONFIG EVALUATION
# ─────────────────────────────────────────────────────────────

def _run_config(pairs: list, config: dict, fast_only: bool = False, max_images: int = 0) -> dict:
    """Run one ablation configuration on all image pairs. Returns per-image results."""
    import core.tool_registry as _tr

    FAST_EXPERTS = {"opencv_denoise", "clahe_lowlight", "opencv_fast_jpeg", "opencv_unsharp_deblur"}

    if max_images and max_images < len(pairs):
        pairs = pairs[:max_images]

    orig_registry = None
    if fast_only:
        orig_registry = dict(_tr.REGISTRY)
        _tr.REGISTRY  = {k: v for k, v in orig_registry.items() if k in FAST_EXPERTS}

    # Patch quality gate if specified
    gate_min = config.get("gate_min", None)
    orig_gate = None
    if gate_min is not None:
        import scheduler.scheduler as _sched
        orig_gate = _sched.QUALITY_GATE_MIN
        _sched.QUALITY_GATE_MIN = gate_min

    per_image = []
    try:
        for deg_path, ref_path in pairs:
            filename = os.path.basename(deg_path)
            base     = baseline_metrics(deg_path, ref_path)
            t0       = time.time()
            try:
                if config["three_stage"]:
                    result   = run_three_stage_scheduler(
                        deg_path,
                        verbose=False,
                        voting=config.get("voting", False),
                        use_memory=config.get("use_memory", True),
                    )
                    restored = result["output_path"]
                    n_calls  = result["invocation_count"]
                    n_rollbacks = sum(
                        1 for s in result.get("stage_results", {}).values()
                        if s.get("rolled_back")
                    )
                else:
                    restored = run_scheduler(deg_path, three_stage=False, verbose=False)
                    n_calls  = None
                    n_rollbacks = 0
            except Exception as e:
                print(f"    [Ablation] Error on {filename}: {e}")
                restored    = None
                n_calls     = None
                n_rollbacks = 0

            elapsed = round(time.time() - t0, 2)

            if restored and os.path.exists(restored):
                metrics   = compute_metrics(ref_path, restored)
                ssim_gain = round(metrics["ssim"] - base["ssim"], 4) if base["ssim"] and metrics["ssim"] else None
                psnr_gain = round(metrics["psnr"] - base["psnr"], 2) if base["psnr"] and metrics["psnr"] else None
            else:
                metrics   = {"ssim": None, "psnr": None, "lpips": None}
                ssim_gain = None
                psnr_gain = None

            per_image.append({
                "file":          filename,
                "ssim_gain":     ssim_gain,
                "psnr_gain":     psnr_gain,
                "time_s":        elapsed,
                "invocations":   n_calls,
                "rollbacks":     n_rollbacks,
            })
    finally:
        if orig_registry is not None:
            _tr.REGISTRY = orig_registry
        if orig_gate is not None:
            _sched.QUALITY_GATE_MIN = orig_gate

    return per_image


# ─────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────

def run_experiment(
    exp_key:    str,
    dataset_filter: str | None = None,
    fast_only:  bool = False,
    max_images: int  = 3,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """Run a single ablation experiment across all benchmark sets."""
    exp = EXPERIMENTS[exp_key]
    print(f"\n{'='*68}")
    print(f"  Ablation {exp_key}: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"{'='*68}")

    # Collect datasets
    all_sets = find_benchmark_sets()
    if not all_sets:
        print("[Ablation] No benchmark sets found. Run: python datasets/generate_benchmark.py")
        return {}

    if dataset_filter:
        all_sets = [s for s in all_sets if dataset_filter in os.path.basename(s)]
        if not all_sets:
            print(f"[Ablation] No sets matching '{dataset_filter}'")
            return {}

    all_results = {}

    for dataset_dir in all_sets:
        set_name = os.path.basename(dataset_dir)
        pairs    = list_image_pairs(dataset_dir)
        if not pairs:
            continue

        print(f"\n  Dataset: {set_name} ({len(pairs)} images)")
        config_results = {}

        for config in exp["configs"]:
            label = config["label"]
            print(f"\n    ── {label}")
            per_img = _run_config(pairs, config, fast_only, max_images)

            # Aggregate
            gains = [r["ssim_gain"] for r in per_img if r["ssim_gain"] is not None]
            avg   = round(sum(gains) / len(gains), 4) if gains else None
            print(f"       avg SSIM gain = {avg:+.4f}" if avg is not None else "       avg SSIM gain = N/A")

            config_results[label] = {
                "avg_ssim_gain": avg,
                "per_image":     per_img,
            }

        # Print comparison
        print(f"\n  Comparison — {set_name}:")
        for label, res in config_results.items():
            g = res["avg_ssim_gain"]
            bar = "█" * int(max(0, (g or 0) * 40)) if g else ""
            print(f"    {label:<35} {('+' if g and g >= 0 else '')}{g or 'N/A':.4f}  {bar}")

        all_results[set_name] = config_results

    # ── Save results ──────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"ablation_{exp_key}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "experiment":  exp_key,
            "name":        exp["name"],
            "description": exp["description"],
            "timestamp":   datetime.now().isoformat(),
            "results":     all_results,
        }, f, indent=2)
    print(f"\n  Saved → {json_path}")

    return all_results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ Ablation Study Runner — compare pipeline configurations"
    )
    parser.add_argument(
        "--experiment", type=str, choices=list(EXPERIMENTS.keys()),
        help="Ablation experiment to run (A1–A5)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all ablation experiments"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Filter to a specific benchmark set name (e.g. blur_test)"
    )
    parser.add_argument(
        "--max_images", type=int, default=3,
        help="Max images per dataset (default: 3, use 0 for all)"
    )
    parser.add_argument(
        "--fast_only", action="store_true",
        help="Use only CPU-fast experts (much faster, lower quality)"
    )
    parser.add_argument(
        "--results_dir", type=str, default=RESULTS_DIR,
        help=f"Output directory (default: {RESULTS_DIR})"
    )
    args = parser.parse_args()

    if args.all:
        for key in EXPERIMENTS:
            run_experiment(key, args.dataset, args.fast_only, args.max_images, args.results_dir)
    elif args.experiment:
        run_experiment(args.experiment, args.dataset, args.fast_only, args.max_images, args.results_dir)
    else:
        print("\nAvailable ablation experiments:")
        for key, exp in EXPERIMENTS.items():
            print(f"  {key}: {exp['name']}")
            print(f"       {exp['description']}")
        print("\nUsage:")
        print("  python experiments/run_ablation.py --experiment A1")
        print("  python experiments/run_ablation.py --all --fast_only --max_images 3")
        parser.print_help()


if __name__ == "__main__":
    main()
