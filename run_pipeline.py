import os
import argparse
import time
import cv2

from scheduler.scheduler import run_scheduler, run_three_stage_scheduler
from utils.visualizer import save_comparison
from utils.report_generator import generate_report  # C7
from evaluation.clinical_evaluator import print_clinical_report  # Clinical Evaluation Expert


def _compute_scores(original_path, restored_path):
    """
    Compute SSIM and PSNR between original and restored images.
    Resizes restored to original dimensions if they differ (e.g. 4x SR output).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        orig = cv2.imread(original_path)
        rest = cv2.imread(restored_path)

        if orig is None or rest is None:
            return {}

        if orig.shape != rest.shape:
            rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))

        orig_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        rest_g = cv2.cvtColor(rest, cv2.COLOR_BGR2GRAY)

        return {
            "ssim": round(float(ssim(orig_g, rest_g)), 4),
            "psnr": round(float(psnr(orig_g, rest_g)), 2),
        }

    except Exception as e:
        print(f"[Pipeline] Score computation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="MAIR+: Multi-Agent Intelligent Image Restoration"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="models/SwinIr/testsets/test/test.jpg",
        help="Path to the input image (default: SwinIR test image)",
    )
    parser.add_argument(
        "--no_tsf",
        action="store_true",
        help="Disable Three-Stage Framework — use legacy single-expert mode (for ablation)",
    )
    # ── v2 flags ────────────────────────────────────────────
    parser.add_argument(
        "--report",
        choices=["console", "html", "both"],
        default="console",
        help="Report format: console (default), html, or both (C7)",
    )
    parser.add_argument(
        "--voting",
        action="store_true",
        help="Enable voting ensemble: run top-2 experts per stage, keep best (C12)",
    )
    parser.add_argument(
        "--no_memory",
        action="store_true",
        help="Disable memory-augmented planning (CaseStore) (C9)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Time budget in seconds — skip slow stages near expiry (C11)",
    )
    parser.add_argument(
        "--clinical_eval",
        action="store_true",
        help="Enable the Clinical Evaluation Expert (NR-IQA metrics) instead of SSIM/PSNR for unpaired medical data.",
    )
    args = parser.parse_args()
    input_image = args.input
    use_tsf     = not args.no_tsf
    use_memory  = not args.no_memory

    # ── Header ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   MAIR+  Multi-Agent Image Restoration")
    print("=" * 50)
    print(f"  Input : {input_image}\n")

    if not os.path.exists(input_image):
        print(f"[Pipeline] ERROR: File not found — {input_image}")
        return

    # ── Run pipeline ─────────────────────────────────────────
    mode_label = "Three-Stage" if use_tsf else "Legacy (no TSF)"
    print(f"  Mode  : {mode_label}\n")

    t0 = time.time()
    if use_tsf:
        result          = run_three_stage_scheduler(
            input_image,
            voting=args.voting,
            use_memory=use_memory,
            budget_seconds=args.budget,
            clinical_eval=args.clinical_eval
        )
        restored_output = result["output_path"]
        total_time      = result["total_time_s"]
        n_invocations   = result["invocation_count"]
        stage_results   = result.get("stage_results", {})
        mem_bias        = result.get("memory_bias_applied", False)
    else:
        restored_output = run_scheduler(input_image, three_stage=False)
        total_time      = round(time.time() - t0, 2)
        n_invocations   = None
        stage_results   = {}
        mem_bias        = False

    # ── Generate visual comparison ────────────────────────────
    if restored_output and os.path.exists(restored_output):
        scores = _compute_scores(input_image, restored_output)
        comparison_path = save_comparison(
            original_path=input_image,
            restored_path=restored_output,
            quality_scores=scores,
            output_dir="outputs",
            label="Phase1",
        )

        # ── C7: Generate per-stage report ────────────────────────
        if stage_results:   # only in three-stage mode
            report_str = generate_report(
                stage_results=stage_results,
                input_path=input_image,
                final_output=restored_output,
                total_time_s=total_time,
                invocation_count=n_invocations,
                memory_bias_applied=mem_bias,
                format=args.report,
            )
            print(report_str)

        # ── Results summary ─────────────────────────────────
        print("\n" + "=" * 50)
        print("  RESULTS")
        print("=" * 50)
        print(f"  Mode     : {mode_label}")
        print(f"  Input    : {input_image}")
        print(f"  Restored : {restored_output}")
        if args.clinical_eval:
            print(f"  Eval Mode: Clinical (NR-IQA)")
            # Run and print the clinical evaluation expert report
            print_clinical_report(input_image, restored_output)
        else:
            if scores:
                print(f"  SSIM     : {scores.get('ssim', 'N/A')}")
                print(f"  PSNR     : {scores.get('psnr', 'N/A')} dB")
        
        print(f"  Time     : {total_time}s")
        if n_invocations is not None:
            print(f"  Calls    : {n_invocations} expert invocation(s)")
        if mem_bias:
            print(f"  Memory   : ✓ Memory bias applied (CaseStore)")
        if comparison_path:
            print(f"  Visual   : {comparison_path}")
            print(f"\n  >> Open the comparison image to see the result!")
        print("=" * 50 + "\n")

    else:
        print("\n[Pipeline] No output was produced.")
        print("[Pipeline] Check that 'python install_phase1_deps.py' has been run.")


if __name__ == "__main__":
    main()