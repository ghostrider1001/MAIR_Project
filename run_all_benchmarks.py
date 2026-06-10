"""
MAIR+ v2 — Master Setup and Benchmark Runner
=============================================
Run this script to:
  1. Generate missing haze + rain datasets
  2. Fix benchmark datasets (top 5 per type)
  3. Run v2 full benchmark (all degradation types, with voting)
  4. Run ablation studies A1-A5
  5. Generate the 6-degradation visual comparison

Run from project root:
    python run_all_benchmarks.py

Estimated time: 5-15 minutes (CPU-only mode)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_benchmark import generate_all, DEGRADATIONS
from evaluation.benchmark import run_benchmark, find_benchmark_sets
from experiments.run_ablation import run_experiment, EXPERIMENTS


def step(n, total, label):
    print(f"\n{'='*70}")
    print(f"  STEP {n}/{total}: {label}")
    print(f"{'='*70}")


def main():
    t_total = time.time()
    print("\n" + "="*70)
    print("  MAIR+ v2 — FULL BENCHMARK SUITE")
    print("="*70)

    # ── STEP 1: Generate missing datasets ───────────────────────────────
    step(1, 5, "Generating missing haze + rain datasets")
    
    # Source: use existing reference images
    source_dir = "datasets/benchmark/noise_test/reference"
    if not os.path.isdir(source_dir):
        source_dir = ""  # fallback to auto-discovery
    
    missing = []
    for dtype in ["haze", "rain"]:
        set_dir = os.path.join("datasets/benchmark", f"{dtype}_test")
        if not os.path.isdir(os.path.join(set_dir, "degraded")):
            missing.append(dtype)
    
    if missing:
        print(f"  Generating: {missing}")
        generate_all(missing, source_dir, n=5)
    else:
        print("  ✓ haze_test and rain_test already exist")
    
    # List all available datasets
    sets = find_benchmark_sets()
    print(f"\n  Available benchmark sets ({len(sets)}):")
    for s in sets:
        print(f"    {os.path.basename(s)}")

    # ── STEP 2: Run v2 full benchmark ────────────────────────────────────
    step(2, 5, "Running v2 full benchmark (all types, fast mode)")
    
    benchmark_results = {}
    for ds in sets:
        set_name = os.path.basename(ds)
        print(f"\n  ── {set_name}")
        result = run_benchmark(
            ds,
            save_outputs=False,
            results_dir="results",
            max_images=3,       # 3 images per set = fast preview
            fast_only=True,     # CPU-only: skip Restormer/SwinIR
            three_stage=True,
            voting=False,       # standard mode first
            use_memory=True,    # C9 memory on
        )
        benchmark_results[set_name] = result
    
    # Save combined results
    combined_path = "results/benchmark_v2_combined.json"
    with open(combined_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n  Saved combined results: {combined_path}")

    # ── STEP 3: Run voting mode benchmark ────────────────────────────────
    step(3, 5, "Running v2 voting benchmark (C12 ensemble mode)")
    
    voting_results = {}
    for ds in sets:
        set_name = os.path.basename(ds)
        result = run_benchmark(
            ds,
            save_outputs=False,
            results_dir="results",
            max_images=3,
            fast_only=True,
            three_stage=True,
            voting=True,        # C12 voting ON
            use_memory=True,
        )
        voting_results[set_name] = result
    
    voting_path = "results/benchmark_v2_voting.json"
    with open(voting_path, "w") as f:
        json.dump(voting_results, f, indent=2)
    print(f"\n  Saved voting results: {voting_path}")

    # ── STEP 4: Run ablation studies ─────────────────────────────────────
    step(4, 5, "Running ablation studies A1-A5")
    
    for exp_key in ["A1", "A2", "A3", "A5"]:
        print(f"\n  ── Ablation {exp_key}: {EXPERIMENTS[exp_key]['name']}")
        try:
            run_experiment(
                exp_key,
                fast_only=True,
                max_images=2,   # quick run
                results_dir="results",
            )
        except Exception as e:
            print(f"  [WARNING] {exp_key} failed: {e}")

    # ── STEP 5: Generate visual comparison ───────────────────────────────
    step(5, 5, "Generating 6-degradation visual comparison")
    
    # Find a clean source image
    source_candidates = [
        "datasets/benchmark/noise_test/reference/baby.png",
        "datasets/benchmark/noise_test/reference/bird.png",
        "datasets/benchmark/blur_test/reference/baby.png",
    ]
    source = next((p for p in source_candidates if os.path.exists(p)), None)
    
    if source:
        from generate_visual_comparison import generate_comparison
        out_path, results = generate_comparison(source, "outputs")
        print(f"\n  ✓ Visual comparison saved: {out_path}")
    else:
        print("  [SKIP] No clean source image found for comparison")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    total_time = round(time.time() - t_total, 1)
    print(f"\n{'='*70}")
    print(f"  MAIR+ v2 BENCHMARK SUITE COMPLETE  ({total_time}s)")
    print(f"{'='*70}")
    print()
    print(f"  Results saved to:  results/")
    print(f"  Visuals saved to:  outputs/")
    print()
    
    # Print summary table
    print(f"  {'Dataset':<22}  {'Baseline SSIM':>13}  {'MAIR+ SSIM':>10}  {'SSIM Gain':>10}")
    print(f"  {'─'*65}")
    for name, res in benchmark_results.items():
        if res and res.get("avg_ssim_gain") is not None:
            g = res["avg_ssim_gain"]
            marker = "🟢" if g > 0 else "🔴"
            print(f"  {name:<22}  {res.get('avg_baseline_ssim', 0):>13.4f}"
                  f"  {res.get('avg_restored_ssim', 0):>10.4f}"
                  f"  {g:>+10.4f}  {marker}")
    print()
    print("  Next steps:")
    print("    1. Open outputs/comparison_ALL6_*.png to see visual results")
    print("    2. Run:  python MAIR_Paper_Presentation.py  to open the HTML presentation")


if __name__ == "__main__":
    main()
