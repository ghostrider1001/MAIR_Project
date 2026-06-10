"""
MAIR+ Evaluation Benchmark
===========================
Runs the MAIR+ pipeline on a test set and computes quality metrics
against ground-truth reference images (true PSNR/SSIM for the paper).

Directory layout expected:
    datasets/benchmark/<set_name>/
        degraded/   ← degraded input images
        reference/  ← clean ground-truth images (same filenames)

Usage:
    python evaluation/benchmark.py --list_sets
    python evaluation/benchmark.py --all
    python evaluation/benchmark.py --all --max_images 3
    python evaluation/benchmark.py --all --fast_only
    python evaluation/benchmark.py --dataset datasets/benchmark/blur_test

Tips:
    --max_images 3    Limit to 3 images per set (fast preview, ~5 min)
    --fast_only       Skip Restormer/SwinIR; use only CPU-fast experts
    --no_tsf          Disable Three-Stage Framework (ablation mode)
"""

import os
import sys
import csv
import json
import argparse
import time
from datetime import datetime

import cv2
import numpy as np

# ── make project root importable ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.scheduler import run_scheduler, run_three_stage_scheduler
from core.degradation_detector import detect_degradation

# Fast experts that run in seconds (no deep model loading)
FAST_EXPERTS = {"opencv_denoise", "clahe_lowlight"}


# ─────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────

def compute_metrics(ref_path: str, restored_path: str) -> dict:
    """
    Compute SSIM, PSNR, and optionally LPIPS (C6) between a restored image
    and clean reference. Handles size mismatch (e.g. SR 4× output).

    Returns:
        dict with keys: ssim, psnr, lpips  (None values on failure)
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn

        ref  = cv2.imread(ref_path)
        rest = cv2.imread(restored_path)

        if ref is None or rest is None:
            return {"ssim": None, "psnr": None, "lpips": None}

        # Resize restored to match reference if needed
        if ref.shape != rest.shape:
            rest = cv2.resize(rest, (ref.shape[1], ref.shape[0]),
                              interpolation=cv2.INTER_AREA)

        ref_g  = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
        rest_g = cv2.cvtColor(rest, cv2.COLOR_BGR2GRAY)

        ssim_val = round(float(ssim_fn(ref_g, rest_g)), 4)
        psnr_val = round(float(psnr_fn(ref_g, rest_g)), 2)

        # C6: LPIPS (optional — requires `pip install lpips`)
        lpips_val = None
        try:
            from evaluation.quality_evaluator import compute_lpips
            lpips_val = compute_lpips(ref_path, restored_path)
        except Exception:
            pass

        return {"ssim": ssim_val, "psnr": psnr_val, "lpips": lpips_val}
    except Exception as e:
        print(f"  [Benchmark] Metric error: {e}")
        return {"ssim": None, "psnr": None, "lpips": None}


def baseline_metrics(degraded_path: str, ref_path: str) -> dict:
    """Compute SSIM/PSNR/LPIPS for degraded input vs reference (no restoration)."""
    return compute_metrics(ref_path, degraded_path)


# ─────────────────────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_benchmark_sets(root: str = "datasets/benchmark") -> list:
    """
    Find all benchmark sets under root.
    A valid set has both degraded/ and reference/ subdirectories.

    Returns list of absolute paths to valid set directories.
    """
    sets = []
    if not os.path.isdir(root):
        return sets
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            has_deg = os.path.isdir(os.path.join(path, "degraded"))
            has_ref = os.path.isdir(os.path.join(path, "reference"))
            if has_deg and has_ref:
                sets.append(path)
    return sets


def list_image_pairs(dataset_dir: str) -> list:
    """
    Return list of (degraded_path, reference_path) tuples for all
    matching image filenames in the dataset's degraded/ and reference/ dirs.
    """
    deg_dir = os.path.join(dataset_dir, "degraded")
    ref_dir = os.path.join(dataset_dir, "reference")

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    deg_files = {
        f for f in os.listdir(deg_dir)
        if os.path.splitext(f)[1].lower() in EXTS
    }
    ref_files = {
        f for f in os.listdir(ref_dir)
        if os.path.splitext(f)[1].lower() in EXTS
    }

    # Match by stem (filename without extension)
    matched = []
    for df in sorted(deg_files):
        stem = os.path.splitext(df)[0]
        # Find matching reference (any extension)
        rf_match = next(
            (rf for rf in ref_files if os.path.splitext(rf)[0] == stem),
            None
        )
        if rf_match:
            matched.append((
                os.path.join(deg_dir, df),
                os.path.join(ref_dir, rf_match),
            ))
        else:
            print(f"  [Benchmark] WARNING: No reference found for '{df}' — skipping.")

    return matched


# ─────────────────────────────────────────────────────────────
# SINGLE-IMAGE EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_image(
    degraded_path: str,
    ref_path:      str,
    save_outputs:  bool = False,
    output_dir:    str  = "outputs/benchmark",
    fast_only:     bool = False,
    three_stage:   bool = True,
    voting:        bool = False,   # C12
    use_memory:    bool = True,    # C9
) -> dict:
    """
    Run the MAIR+ pipeline on one image and evaluate against reference.

    Args:
        fast_only   : if True, skip deep models (Restormer/SwinIR)
        three_stage : if False, use legacy single-expert mode (ablation)
        voting      : C12 — run top-2 experts, keep best
        use_memory  : C9  — enable CaseStore memory planning

    Returns a result dict with all metrics, timing, and expert info.
    """
    filename = os.path.basename(degraded_path)
    print(f"\n  ── {filename}")

    # ── Baseline: degraded vs reference ──────────────────────
    baseline = baseline_metrics(degraded_path, ref_path)
    lpips_str = f"  LPIPS={baseline['lpips']:.4f}" if baseline.get("lpips") is not None else ""
    print(f"     Baseline  SSIM={baseline['ssim']}  PSNR={baseline['psnr']} dB{lpips_str}")

    # ── Detect degradation type ───────────────────────────────
    deg_result = detect_degradation(degraded_path)
    primary    = deg_result["primary"]
    confidence = deg_result["confidence"]

    # ── Fast-only: patch registry to only allow fast experts ──
    if fast_only:
        from core import tool_registry as _tr
        _orig_registry = dict(_tr.REGISTRY)
        _tr.REGISTRY = {
            k: v for k, v in _orig_registry.items()
            if k in FAST_EXPERTS
        }

    # ── Run pipeline ─────────────────────────────────────────
    t0 = time.time()
    n_rollbacks = 0
    try:
        if three_stage:
            result      = run_three_stage_scheduler(
                degraded_path,
                verbose=False,
                voting=voting,
                use_memory=use_memory,
            )
            restored    = result["output_path"]
            n_calls     = result["invocation_count"]
            # Count rollback events across stages (C4)
            for sr in result.get("stage_results", {}).values():
                if sr.get("rolled_back"):
                    n_rollbacks += 1
        else:
            restored = run_scheduler(degraded_path, three_stage=False, verbose=False)
            n_calls  = None
    finally:
        if fast_only:
            _tr.REGISTRY = _orig_registry
    elapsed = round(time.time() - t0, 2)

    # ── Evaluate restored vs reference ────────────────────────
    if restored and os.path.exists(restored):
        restored_metrics = compute_metrics(ref_path, restored)
        mode_tag = "fast" if fast_only else ("TSF" if three_stage else "legacy")
        lpips_r  = restored_metrics.get("lpips")
        lpips_str = f"  LPIPS={lpips_r:.4f}" if lpips_r is not None else ""
        print(f"     Restored  SSIM={restored_metrics['ssim']}  PSNR={restored_metrics['psnr']} dB"
              f"{lpips_str}  ({elapsed}s, {mode_tag}, calls={n_calls})")

        ssim_gain  = None
        psnr_gain  = None
        lpips_gain = None
        if baseline["ssim"] is not None and restored_metrics["ssim"] is not None:
            ssim_gain = round(restored_metrics["ssim"] - baseline["ssim"], 4)
            psnr_gain = round(restored_metrics["psnr"] - baseline["psnr"], 2)
        if baseline.get("lpips") is not None and lpips_r is not None:
            # LPIPS: lower is better, so gain = baseline_lpips - restored_lpips
            lpips_gain = round(baseline["lpips"] - lpips_r, 4)

        if ssim_gain is not None:
            lpips_g_str = f"  Δ LPIPS={lpips_gain:+.4f}" if lpips_gain is not None else ""
            print(f"     Δ SSIM={ssim_gain:+.4f}  Δ PSNR={psnr_gain:+.2f} dB{lpips_g_str}")
        if n_rollbacks:
            print(f"     Rollbacks: {n_rollbacks} stage(s) rolled back (C4 quality gate)")
    else:
        restored_metrics = {"ssim": None, "psnr": None, "lpips": None}
        ssim_gain        = None
        psnr_gain        = None
        lpips_gain       = None
        n_calls          = None
        print("     Restored  FAILED — no output produced")

    return {
        "file":             filename,
        "primary":          primary,
        "confidence":       confidence,
        "baseline_ssim":    baseline["ssim"],
        "baseline_psnr":    baseline["psnr"],
        "baseline_lpips":   baseline.get("lpips"),
        "restored_ssim":    restored_metrics["ssim"],
        "restored_psnr":    restored_metrics["psnr"],
        "restored_lpips":   restored_metrics.get("lpips"),
        "ssim_gain":        ssim_gain,
        "psnr_gain":        psnr_gain,
        "lpips_gain":       lpips_gain,
        "rollbacks":        n_rollbacks,
        "time_s":           elapsed,
        "invocations":      n_calls,
        "restored_path":    restored or "",
    }


# ─────────────────────────────────────────────────────────────
# DATASET-LEVEL EVALUATION
# ─────────────────────────────────────────────────────────────

def run_benchmark(
    dataset_dir:  str,
    save_outputs: bool = False,
    results_dir:  str  = "results",
    max_images:   int  = 0,
    fast_only:    bool = False,
    three_stage:  bool = True,
    voting:       bool = False,   # C12
    use_memory:   bool = True,    # C9
) -> dict:
    """
    Args:
        max_images  : limit images per set (0 = no limit, use ≤3 for quick tests)
        fast_only   : skip deep models, use only CPU-fast experts
        three_stage : False = legacy single-expert mode (ablation)
    """
    """
    Evaluate MAIR+ on an entire benchmark dataset.

    Returns a summary dict with per-image results and dataset averages.
    """
    set_name = os.path.basename(dataset_dir)
    pairs    = list_image_pairs(dataset_dir)

    if not pairs:
        print(f"[Benchmark] No valid image pairs found in '{dataset_dir}'.")
        return {}

    # Apply image limit
    if max_images and max_images < len(pairs):
        print(f"[Benchmark] Limiting to {max_images} images (use --max_images 0 for all)")
        pairs = pairs[:max_images]

    mode_desc = "fast-only" if fast_only else ("TSF" if three_stage else "legacy (no TSF)")
    print(f"\n{'='*60}")
    print(f"  MAIR+ Benchmark  —  {set_name}  ({len(pairs)} images, {mode_desc})")
    print(f"{'='*60}")

    os.makedirs(results_dir, exist_ok=True)
    all_results = []

    for deg_path, ref_path in pairs:
        result = evaluate_image(
            degraded_path=deg_path,
            ref_path=ref_path,
            save_outputs=save_outputs,
            output_dir=os.path.join(results_dir, "benchmark_outputs", set_name),
            fast_only=fast_only,
            three_stage=three_stage,
            voting=voting,
            use_memory=use_memory,
        )
        all_results.append(result)

    # ── Aggregate statistics ──────────────────────────────────
    def _avg(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "dataset":            set_name,
        "n_images":           len(pairs),
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "avg_baseline_ssim":  _avg("baseline_ssim"),
        "avg_baseline_psnr":  _avg("baseline_psnr"),
        "avg_baseline_lpips": _avg("baseline_lpips"),
        "avg_restored_ssim":  _avg("restored_ssim"),
        "avg_restored_psnr":  _avg("restored_psnr"),
        "avg_restored_lpips": _avg("restored_lpips"),
        "avg_ssim_gain":      _avg("ssim_gain"),
        "avg_psnr_gain":      _avg("psnr_gain"),
        "avg_lpips_gain":     _avg("lpips_gain"),
        "total_rollbacks":    sum(r.get("rollbacks", 0) for r in all_results),
        "avg_time_s":         _avg("time_s"),
        "per_image":          all_results,
    }

    # ── Print summary table ───────────────────────────────────
    _print_summary(summary)

    # ── Save CSV ──────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"benchmark_{set_name}_{ts}.csv")
    _save_csv(all_results, csv_path)

    # ── Save JSON ─────────────────────────────────────────────
    json_path = os.path.join(results_dir, f"benchmark_{set_name}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved:")
    print(f"    CSV  → {csv_path}")
    print(f"    JSON → {json_path}")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────
# REPORT PRINTING
# ─────────────────────────────────────────────────────────────

def _print_summary(summary: dict) -> None:
    """Print a formatted benchmark results table."""
    W = 74
    has_lpips = any(
        r.get("restored_lpips") is not None for r in summary["per_image"]
    )
    print(f"\n{'─'*W}")
    print(f"  BENCHMARK RESULTS — {summary['dataset']}")
    print(f"{'─'*W}")
    hdr = f"  {'FILE':<28}  {'B-SSIM':>7}  {'R-SSIM':>7}  {'Δ SSIM':>8}  {'Δ PSNR':>8}"
    if has_lpips:
        hdr += f"  {'Δ LPIPS':>8}"
    print(hdr)
    print(f"  {'─'*(W-2)}")

    for r in summary["per_image"]:
        b_ssim = f"{r['baseline_ssim']:.4f}" if r["baseline_ssim"] is not None else "  N/A "
        r_ssim = f"{r['restored_ssim']:.4f}" if r["restored_ssim"]  is not None else "  N/A "
        d_ssim = f"{r['ssim_gain']:+.4f}"   if r["ssim_gain"] is not None else "   N/A  "
        d_psnr = f"{r['psnr_gain']:+.2f}"   if r["psnr_gain"] is not None else "   N/A  "
        rb     = " ⚠" if r.get("rollbacks") else ""
        fname  = r["file"][:28]
        row    = f"  {fname:<28}  {b_ssim:>7}  {r_ssim:>7}  {d_ssim:>8}  {d_psnr:>8}{rb}"
        if has_lpips:
            d_lpips = f"{r['lpips_gain']:+.4f}" if r.get("lpips_gain") is not None else "   N/A  "
            row += f"  {d_lpips:>8}"
        print(row)

    print(f"  {'─'*(W-2)}")
    avg_row = (f"  {'AVERAGE':<28}  "
               f"{summary['avg_baseline_ssim'] or 'N/A':>7}  "
               f"{summary['avg_restored_ssim'] or 'N/A':>7}  "
               f"{('+' if (summary['avg_ssim_gain'] or 0) >= 0 else '') + str(summary['avg_ssim_gain'] or 'N/A'):>8}  "
               f"{('+' if (summary['avg_psnr_gain'] or 0) >= 0 else '') + str(summary['avg_psnr_gain'] or 'N/A'):>8}")
    if has_lpips:
        avg_lp = summary.get("avg_lpips_gain")
        avg_row += f"  {('+' if (avg_lp or 0) >= 0 else '') + str(avg_lp or 'N/A'):>8}"
    print(avg_row)
    rollbacks = summary.get("total_rollbacks", 0)
    if rollbacks:
        print(f"  ⚠  {rollbacks} stage rollback(s) by quality gate (C4)")
    print(f"{'─'*W}")


def _save_csv(results: list, csv_path: str) -> None:
    """Save per-image results as CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    fieldnames = [k for k in fieldnames if k != "restored_path"]  # omit long paths
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ Evaluation Benchmark — measure pipeline quality vs. ground truth"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to benchmark dataset dir (must contain degraded/ and reference/)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all benchmark sets found under datasets/benchmark/"
    )
    parser.add_argument(
        "--list_sets", action="store_true",
        help="List all available benchmark sets and exit"
    )
    parser.add_argument(
        "--save_outputs", action="store_true",
        help="Save restored images to results/benchmark_outputs/"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to write CSV and JSON results (default: results/)"
    )
    parser.add_argument(
        "--max_images", type=int, default=3,
        help="Max images per benchmark set (default: 3, use 0 for all)"
    )
    parser.add_argument(
        "--fast_only", action="store_true",
        help="Use only fast CPU experts (NLM, CLAHE) — skips Restormer/SwinIR"
    )
    parser.add_argument(
        "--no_tsf", action="store_true",
        help="Disable Three-Stage Framework (ablation comparison mode)"
    )
    parser.add_argument(
        "--voting", action="store_true",
        help="Enable voting ensemble per stage (C12) — run top-2, keep best"
    )
    parser.add_argument(
        "--no_memory", action="store_true",
        help="Disable CaseStore memory-augmented planning (C9)"
    )
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────
    if args.list_sets:
        sets = find_benchmark_sets()
        if sets:
            print("\n  Available benchmark sets:")
            for s in sets:
                pairs = list_image_pairs(s)
                print(f"    {os.path.basename(s):30}  ({len(pairs)} image pairs)")
        else:
            print("\n  No benchmark sets found under datasets/benchmark/")
            print("  Create: datasets/benchmark/<set_name>/degraded/ and reference/")
        return

    use_tsf    = not args.no_tsf
    use_memory = not args.no_memory

    # ── Run specific dataset ──────────────────────────────────
    if args.dataset:
        run_benchmark(
            args.dataset, args.save_outputs, args.results_dir,
            max_images=args.max_images,
            fast_only=args.fast_only,
            three_stage=use_tsf,
            voting=args.voting,
            use_memory=use_memory,
        )
        return

    # ── Run all datasets ──────────────────────────────────────
    if args.all:
        sets = find_benchmark_sets()
        if not sets:
            print("[Benchmark] No datasets found. Run: python datasets/generate_benchmark.py")
            return
        for ds in sets:
            run_benchmark(
                ds, args.save_outputs, args.results_dir,
                max_images=args.max_images,
                fast_only=args.fast_only,
                three_stage=use_tsf,
                voting=args.voting,
                use_memory=use_memory,
            )
        return

    # ── Default: show help ────────────────────────────────────
    print("\n  MAIR+ Benchmark — Usage:")
    print("    python evaluation/benchmark.py --list_sets")
    print("    python evaluation/benchmark.py --dataset datasets/benchmark/<set_name>")
    print("    python evaluation/benchmark.py --all --save_outputs")
    print()
    parser.print_help()


if __name__ == "__main__":
    main()
