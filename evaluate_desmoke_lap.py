"""
evaluate_desmoke_lap.py
-----------------------
Evaluate MAIR+ on the DeSmoke-LAP dataset using:
  - BRISQUE  (no-reference, lower = better)  — always available
  - SSIM     (paired, higher = better)        — available (clear/ folders exist)
  - PSNR dB  (paired, higher = better)        — available (clear/ folders exist)

Usage:
    # Quick smoke-test (2 images per case, ~2 min)
    python evaluate_desmoke_lap.py --per_case 2

    # Standard evaluation (10 images per case, ~15 min on CPU)
    python evaluate_desmoke_lap.py

    # Specific TLH cases only
    python evaluate_desmoke_lap.py --cases TLH_2 TLH_6 TLH_7

    # Full run (30 images per case)
    python evaluate_desmoke_lap.py --per_case 30
"""

import os
import sys
import glob
import json
import csv
import time
import argparse

import cv2
import numpy as np

# ── Ensure project root on path ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Check piq (BRISQUE) ───────────────────────────────────────────────────────
try:
    import torch
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("[WARN] piq not found. BRISQUE will be skipped.")
    print("       Install with: .\\venv\\Scripts\\pip install piq")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "DeSmoke-LAP dataset", "Dataset")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUTPUT_JSON  = os.path.join(RESULTS_DIR, "desmoke_lap_eval.json")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "desmoke_lap_eval.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "desmoke_lap_eval")

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_brisque(img_bgr):
    """BRISQUE score for a BGR numpy image. Lower = better quality."""
    if not HAS_PIQ:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t   = torch.tensor(rgb, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
        with torch.no_grad():
            score = piq.brisque(t, data_range=1.0)
        return float(score)
    except Exception as e:
        return None


def compute_ssim(img1_bgr, img2_bgr):
    """SSIM between two BGR images (same size). Higher = better."""
    try:
        from skimage.metrics import structural_similarity as ssim
        g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
        return float(ssim(g1, g2))
    except ImportError:
        # Fallback: OpenCV-based approximate SSIM
        g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        C1, C2 = 6.5025, 58.5225
        mu1, mu2 = cv2.GaussianBlur(g1,(11,11),1.5), cv2.GaussianBlur(g2,(11,11),1.5)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        s1  = cv2.GaussianBlur(g1**2,(11,11),1.5) - mu1_sq
        s2  = cv2.GaussianBlur(g2**2,(11,11),1.5) - mu2_sq
        s12 = cv2.GaussianBlur(g1*g2,(11,11),1.5) - mu1_mu2
        num = (2*mu1_mu2+C1)*(2*s12+C2)
        den = (mu1_sq+mu2_sq+C1)*(s1+s2+C2)
        return float(np.mean(num / (den + 1e-10)))
    except Exception:
        return None


def compute_psnr(img1_bgr, img2_bgr):
    """PSNR (dB) between two BGR images. Higher = better."""
    try:
        mse = np.mean((img1_bgr.astype(np.float64) - img2_bgr.astype(np.float64))**2)
        if mse < 1e-10:
            return 100.0
        return float(10 * np.log10(255**2 / mse))
    except Exception:
        return None


def align_images(ref, tgt):
    """Resize tgt to match ref dimensions if they differ."""
    if ref.shape != tgt.shape:
        tgt = cv2.resize(tgt, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    return tgt


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def run_dcp_direct(img_path):
    """
    Bypass the scheduler and run DCP dehaze expert directly.
    Used for surgical smoke where the re-detection after denoising
    causes the scene stage to be skipped (haze score drops post-NLM).
    """
    try:
        from experts.dehaze_expert import restore_dcp
        out = restore_dcp(img_path)
        return out if out and os.path.exists(out) else None
    except Exception as e:
        print(f"    [DCP ERROR] {type(e).__name__}: {e}")
        return None

def run_mair_pipeline(img_path, verbose=False, diag=False):
    """Run MAIR+ on a single image. Returns output path or None."""
    try:
        from scheduler.scheduler import run_three_stage_scheduler
        result = run_three_stage_scheduler(
            input_path=img_path,   # correct parameter name
            verbose=verbose,
            voting=False,          # faster for bulk eval
            use_memory=False       # skip memory for speed in bulk eval
        )
        out = result.get("output_path")
        if diag:
            print(f"    [Diag] pipeline result keys: {list(result.keys())}")
            print(f"    [Diag] output_path: {out}")
            for stage, sr in result.get("stage_results", {}).items():
                skipped = sr.get("skipped", False)
                best    = sr.get("best_expert", "—")
                score   = sr.get("best_score")
                rb      = sr.get("rolled_back", False)
                print(f"    [Diag]   stage={stage:12}  skipped={skipped}  expert={best}  score={score}  rollback={rb}")
        return out if out and os.path.exists(out) else None
    except Exception as e:
        print(f"    [Pipeline ERROR] {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_image(hazy_path, clear_path=None, save_dir=None,
                   verbose=False, diag=False, bypass_scheduler=False):
    """
    Evaluate one image pair through MAIR+.
    Returns a dict with all metrics.
    """
    row = {
        "file":             os.path.basename(hazy_path),
        "brisque_hazy":     None,
        "brisque_restored": None,
        "brisque_delta":    None,
        "ssim_hazy":        None,
        "ssim_restored":    None,
        "psnr_hazy":        None,
        "psnr_restored":    None,
        "runtime_s":        None,
        "pipeline_ok":      False,
    }

    hazy_img = cv2.imread(hazy_path)
    if hazy_img is None:
        return row

    # ── Baseline metrics on hazy frame ───────────────────────────────────────
    row["brisque_hazy"] = compute_brisque(hazy_img)

    # ── Paired SSIM/PSNR — only if clear frame with same filename exists ─────
    # NOTE: DeSmoke-LAP hazy/clear folders are NOT frame-aligned;
    #       clear frames are different timestamps. SSIM is only computed
    #       when the exact filename exists in clear/ (some overlap exists).
    clear_img = None
    if clear_path and os.path.exists(clear_path):
        clear_img = cv2.imread(clear_path)
        if clear_img is not None:
            hazy_aligned = align_images(clear_img, hazy_img)
            row["ssim_hazy"] = compute_ssim(clear_img, hazy_aligned)
            row["psnr_hazy"] = compute_psnr(clear_img, hazy_aligned)

    # ── Run pipeline or direct expert ────────────────────────────────────────
    t0 = time.time()
    if bypass_scheduler:
        restored_path = run_dcp_direct(hazy_path)
    else:
        restored_path = run_mair_pipeline(hazy_path, verbose=verbose, diag=diag)
    row["runtime_s"] = round(time.time() - t0, 2)

    if restored_path:
        row["pipeline_ok"] = True
        rest_img = cv2.imread(restored_path)

        if rest_img is not None:
            row["brisque_restored"] = compute_brisque(rest_img)

            if row["brisque_hazy"] is not None and row["brisque_restored"] is not None:
                row["brisque_delta"] = round(row["brisque_hazy"] - row["brisque_restored"], 4)

            if clear_img is not None:
                rest_aligned = align_images(clear_img, rest_img)
                row["ssim_restored"] = compute_ssim(clear_img, rest_aligned)
                row["psnr_restored"] = compute_psnr(clear_img, rest_aligned)

        # Copy restored image to dedicated output folder
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            import shutil
            out_name = "restored_" + os.path.basename(hazy_path)
            shutil.copy(restored_path, os.path.join(save_dir, out_name))

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Per-case evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(case_dir, per_case=10, verbose=False, diag_first=False,
                  bypass_scheduler=False):
    """Evaluate one TLH case. Returns list of per-image result dicts."""
    case_name = os.path.basename(case_dir)
    hazy_dir  = os.path.join(case_dir, "hazy")
    clear_dir = os.path.join(case_dir, "clear")

    if not os.path.exists(hazy_dir):
        print(f"  [SKIP] {case_name}: no hazy/ folder found")
        return []

    all_hazy = sorted(glob.glob(os.path.join(hazy_dir, "*.png")) +
                      glob.glob(os.path.join(hazy_dir, "*.jpg")))
    if not all_hazy:
        print(f"  [SKIP] {case_name}: no images in hazy/")
        return []

    # Build set of available clear filenames for fast lookup
    clear_fnames = set()
    if os.path.exists(clear_dir):
        clear_fnames = {f for f in os.listdir(clear_dir)
                        if f.lower().endswith(('.png', '.jpg'))}

    # Sample evenly across the video frames for representative coverage
    if per_case < len(all_hazy):
        indices  = [int(i * len(all_hazy) / per_case) for i in range(per_case)]
        selected = [all_hazy[i] for i in indices]
    else:
        selected = all_hazy

    paired_count = sum(1 for p in selected if os.path.basename(p) in clear_fnames)
    print(f"  [{case_name}] {len(selected)} hazy frames selected, {paired_count} have matching clear frame")

    save_dir = os.path.join(OUTPUT_DIR, case_name)
    rows     = []

    for i, hazy_path in enumerate(selected):
        fname      = os.path.basename(hazy_path)
        # Only use clear_path if filename actually exists in clear/
        clear_path = os.path.join(clear_dir, fname) if fname in clear_fnames else None

        is_diag = diag_first and i == 0   # verbose first image only for diagnosis
        if is_diag:
            print(f"    [DIAG] Running first image with verbose output...")

        print(f"    [{i+1}/{len(selected)}] {fname}", end="\r", flush=True)

        row = evaluate_image(hazy_path, clear_path, save_dir=save_dir,
                             verbose=is_diag, diag=is_diag,
                             bypass_scheduler=bypass_scheduler)
        row["case"] = case_name
        rows.append(row)

    print()  # newline after \r progress
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(rows):
    return {
        "n":                len(rows),
        "pipeline_ok":      sum(1 for r in rows if r["pipeline_ok"]),
        "brisque_hazy":     _avg([r["brisque_hazy"]     for r in rows]),
        "brisque_restored": _avg([r["brisque_restored"] for r in rows]),
        "brisque_delta":    _avg([r["brisque_delta"]    for r in rows]),
        "ssim_hazy":        _avg([r["ssim_hazy"]        for r in rows]),
        "ssim_restored":    _avg([r["ssim_restored"]    for r in rows]),
        "psnr_hazy":        _avg([r["psnr_hazy"]        for r in rows]),
        "psnr_restored":    _avg([r["psnr_restored"]    for r in rows]),
        "avg_runtime_s":    _avg([r["runtime_s"]        for r in rows]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAIR+ DeSmoke-LAP BRISQUE/SSIM/PSNR Evaluation")
    parser.add_argument("--per_case",  type=int,   default=10,
                        help="Images to sample per TLH case (default: 10)")
    parser.add_argument("--cases",    nargs="*",   default=None,
                        help="Specific cases e.g. TLH_2 TLH_6 (default: all)")
    parser.add_argument("--bypass",    action="store_true",
                        help="Bypass scheduler: run DCP dehaze expert directly on each image (recommended for smoke)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Full verbose pipeline output for every image")
    parser.add_argument("--diag",      action="store_true",
                        help="Print pipeline diagnostic for first image of first case only (fast debug)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,  exist_ok=True)

    print("=" * 62)
    print("  MAIR+ DeSmoke-LAP Evaluation  (BRISQUE + SSIM + PSNR)")
    print("=" * 62)

    if not os.path.exists(DATASET_ROOT):
        print(f"[ERROR] Dataset not found: {DATASET_ROOT}")
        sys.exit(1)

    # Discover TLH cases
    all_cases = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("TLH")
    ])
    if args.cases:
        all_cases = [c for c in all_cases if c in args.cases]

    if not all_cases:
        print("[ERROR] No TLH cases found.")
        sys.exit(1)

    print(f"  Cases     : {', '.join(all_cases)}")
    print(f"  Per case  : {args.per_case} images")
    if args.bypass:
        print("  Mode      : BYPASS (DCP dehaze direct — no scheduler routing)")
    print(f"  Total est : ~{len(all_cases) * args.per_case} images")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Restored  : {OUTPUT_DIR}")
    print()

    all_rows     = []
    case_results = {}
    t_total      = time.time()

    first_case = True
    for case_name in all_cases:
        case_dir = os.path.join(DATASET_ROOT, case_name)
        print(f"── {case_name} " + "─" * (50 - len(case_name)))

        # --diag: run verbose diagnostics on first image of first case only
        diag_this = args.diag and first_case
        rows = evaluate_case(case_dir, per_case=args.per_case,
                             verbose=args.verbose, diag_first=diag_this,
                             bypass_scheduler=args.bypass)
        first_case = False
        if not rows:
            continue

        agg = aggregate(rows)
        case_results[case_name] = agg
        all_rows.extend(rows)

        # Per-case summary line
        bq_h = f"{agg['brisque_hazy']:.1f}"   if agg['brisque_hazy']     is not None else "N/A"
        bq_r = f"{agg['brisque_restored']:.1f}"if agg['brisque_restored'] is not None else "N/A"
        bq_d = f"+{agg['brisque_delta']:.2f}"  if agg['brisque_delta']    is not None else "N/A"
        ss_r = f"{agg['ssim_restored']:.4f}"   if agg['ssim_restored']    is not None else "N/A"
        pn_r = f"{agg['psnr_restored']:.2f}dB" if agg['psnr_restored']    is not None else "N/A"
        rt   = f"{agg['avg_runtime_s']:.1f}s"  if agg['avg_runtime_s']   is not None else "N/A"

    print(f"  BRISQUE: {bq_h} → {bq_r}  (Δ {bq_d})   SSIM: {ss_r}   PSNR: {pn_r}   {rt}/img")

    # ── Overall aggregate ─────────────────────────────────────────────────────
    elapsed = round(time.time() - t_total, 1)
    overall = aggregate(all_rows) if all_rows else {}

    print()
    print("=" * 62)
    print("  OVERALL RESULTS")
    print("=" * 62)

    header  = f"{'Case':<12} {'N':>4}  {'BRISQUE↓(hazy)':>14}  {'BRISQUE↓(rest)':>14}  {'ΔBRISQUE':>9}  {'SSIM↑':>7}  {'PSNR↑':>8}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    for case_name, agg in case_results.items():
        bq_h = f"{agg['brisque_hazy']:.1f}"    if agg['brisque_hazy']     is not None else " N/A"
        bq_r = f"{agg['brisque_restored']:.1f}" if agg['brisque_restored'] is not None else " N/A"
        bq_d = f"+{agg['brisque_delta']:.2f}"   if agg['brisque_delta']    is not None else " N/A"
        ss_r = f"{agg['ssim_restored']:.4f}"    if agg['ssim_restored']    is not None else "  N/A"
        pn_r = f"{agg['psnr_restored']:.2f}"    if agg['psnr_restored']    is not None else "  N/A"
        print(f"{case_name:<12} {agg['n']:>4}  {bq_h:>14}  {bq_r:>14}  {bq_d:>9}  {ss_r:>7}  {pn_r:>8}")

    print(divider)
    if overall:
        bq_h = f"{overall['brisque_hazy']:.1f}"    if overall['brisque_hazy']     is not None else " N/A"
        bq_r = f"{overall['brisque_restored']:.1f}" if overall['brisque_restored'] is not None else " N/A"
        bq_d = f"+{overall['brisque_delta']:.2f}"   if overall['brisque_delta']    is not None else " N/A"
        ss_r = f"{overall['ssim_restored']:.4f}"    if overall['ssim_restored']    is not None else "  N/A"
        pn_r = f"{overall['psnr_restored']:.2f}"    if overall['psnr_restored']    is not None else "  N/A"
        print(f"{'OVERALL':<12} {overall['n']:>4}  {bq_h:>14}  {bq_r:>14}  {bq_d:>9}  {ss_r:>7}  {pn_r:>8}")

    print(f"\n  Total time: {elapsed}s")

    # ── Save results ──────────────────────────────────────────────────────────
    full_results = {
        "overall":      overall,
        "per_case":     case_results,
        "per_image":    all_rows,
        "elapsed_s":    elapsed,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_results, f, indent=2)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n  Saved JSON -> {OUTPUT_JSON}")
    print(f"  Saved CSV  -> {OUTPUT_CSV}")
    print("=" * 62)


if __name__ == "__main__":
    main()
