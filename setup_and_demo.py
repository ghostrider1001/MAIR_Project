"""
MAIR+ One-Shot Setup & Demo
============================
Run this once to:
  1. Install all required packages (cv2, scikit-image, einops)
  2. Generate a degraded demo image
  3. Run the full MAIR+ pipeline on it
  4. Generate all benchmark datasets (blur, jpeg, noise, lowlight, haze, mixed)
  5. Run benchmarks on all sets (fast mode, 3 images each)

Usage:
    python setup_and_demo.py
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def pip_install(*packages):
    print(f"\n[Setup] Installing: {' '.join(packages)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    )

# ── STEP 1: Install dependencies ────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Installing dependencies")
print("="*60)

pip_install("opencv-python", "scikit-image", "numpy", "einops", "Pillow")

print("[Setup] Core packages installed ✓")
print("[Setup] Optional: torch/lpips (for LPIPS metric) — install separately if needed")

# ── STEP 2: Generate a demo test image ─────────────────────────
print("\n" + "="*60)
print("  STEP 2: Creating demo test image")
print("="*60)

import cv2
import numpy as np

demo_dir = os.path.join(ROOT, "demo_inputs")
os.makedirs(demo_dir, exist_ok=True)

# Create a clean synthetic test image (gradient + shapes)
H, W = 256, 256
clean = np.zeros((H, W, 3), dtype=np.uint8)

# Background gradient
for i in range(H):
    for j in range(W):
        clean[i, j] = [int(200 * j / W), int(150 * i / H), int(100 + 50 * (i + j) / (H + W))]

# Add some structure (circles and lines)
cv2.circle(clean, (80, 80), 40, (220, 180, 100), -1)
cv2.circle(clean, (180, 160), 55, (100, 200, 180), -1)
cv2.rectangle(clean, (30, 170), (120, 230), (180, 100, 200), -1)
cv2.line(clean, (0, 128), (255, 128), (255, 255, 200), 2)
cv2.putText(clean, "MAIR+", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

clean_path = os.path.join(demo_dir, "clean_reference.png")
cv2.imwrite(clean_path, clean)

# Apply noise degradation for demo
noise = np.random.normal(0, 35, clean.shape).astype(np.float32)
noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
noisy_path = os.path.join(demo_dir, "noisy_input.png")
cv2.imwrite(noisy_path, noisy)

# Also apply lowlight for second demo
lut = np.array([((i / 255.0) ** 3.5) * 255 for i in range(256)], dtype=np.uint8)
lowlight = cv2.LUT(clean, lut)
lowlight_path = os.path.join(demo_dir, "lowlight_input.png")
cv2.imwrite(lowlight_path, lowlight)

# Apply haze
img_f = clean.astype(np.float32) / 255.0
gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
depth = 1.0 - gray
depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=W // 8)
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
t = np.exp(-1.5 * depth).astype(np.float32)
t = np.clip(t, 0.1, 1.0)[:, :, np.newaxis]
hazy = img_f * t + 0.85 * (1.0 - t)
hazy = np.clip(hazy, 0.0, 1.0)
hazy_path = os.path.join(demo_dir, "hazy_input.png")
cv2.imwrite(hazy_path, (hazy * 255).astype(np.uint8))

print(f"[Setup] Demo images saved:")
print(f"  Clean     : {clean_path}")
print(f"  Noisy     : {noisy_path}")
print(f"  Lowlight  : {lowlight_path}")
print(f"  Hazy      : {hazy_path}")

# ── STEP 3: Run MAIR+ pipeline on demo images ───────────────────
print("\n" + "="*60)
print("  STEP 3: Running MAIR+ pipeline (demo)")
print("="*60)

sys.path.insert(0, ROOT)

from scheduler.scheduler import run_three_stage_scheduler
from utils.visualizer import save_comparison
from utils.report_generator import generate_report

demo_results = {}
for label, path in [("noisy", noisy_path), ("lowlight", lowlight_path), ("hazy", hazy_path)]:
    print(f"\n[Demo] Running on: {label} image...")
    try:
        result = run_three_stage_scheduler(path, verbose=True, use_memory=True)
        output = result["output_path"]
        if output and os.path.exists(output):
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            ref   = cv2.cvtColor(cv2.imread(clean_path), cv2.COLOR_BGR2GRAY)
            rest  = cv2.cvtColor(cv2.imread(output), cv2.COLOR_BGR2GRAY)
            if ref.shape != rest.shape:
                rest = cv2.resize(rest, (ref.shape[1], ref.shape[0]))
            ssim_v = round(float(ssim_fn(ref, rest)), 4)
            psnr_v = round(float(psnr_fn(ref, rest)), 2)

            comp = save_comparison(
                original_path=path,
                restored_path=output,
                quality_scores={"ssim": ssim_v, "psnr": psnr_v},
                output_dir="outputs",
                label=f"demo_{label}",
            )
            demo_results[label] = {
                "output": output,
                "ssim": ssim_v,
                "psnr": psnr_v,
                "comparison": comp,
                "time": result["total_time_s"],
                "calls": result["invocation_count"],
                "stage_results": result.get("stage_results", {}),
            }
            print(f"[Demo] ✓ {label}: SSIM={ssim_v}, PSNR={psnr_v} dB ({result['total_time_s']}s)")

            # Generate HTML report
            report_str = generate_report(
                stage_results=result.get("stage_results", {}),
                input_path=path,
                final_output=output,
                total_time_s=result["total_time_s"],
                invocation_count=result["invocation_count"],
                memory_bias_applied=result.get("memory_bias_applied", False),
                format="html",
            )
        else:
            print(f"[Demo] ✗ {label}: Pipeline produced no output")
            demo_results[label] = {"output": None}
    except Exception as e:
        print(f"[Demo] ✗ {label}: Error — {e}")
        demo_results[label] = {"output": None, "error": str(e)}

# ── STEP 4: Generate all benchmark datasets ─────────────────────
print("\n" + "="*60)
print("  STEP 4: Generating benchmark datasets")
print("="*60)

from datasets.generate_benchmark import generate_all, DEGRADATIONS

# Use our generated clean image as source
source_dir = demo_dir
generate_all(
    types=["blur", "jpeg", "noise", "lowlight", "haze", "mixed"],
    source_dir=source_dir,
    n=5,
)

# ── STEP 5: Run benchmarks ──────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5: Running benchmarks (fast mode, all sets)")
print("="*60)

from evaluation.benchmark import run_benchmark, find_benchmark_sets

sets = find_benchmark_sets()
print(f"[Benchmark] Found {len(sets)} benchmark sets: {[os.path.basename(s) for s in sets]}")

all_summaries = {}
for ds in sets:
    name = os.path.basename(ds)
    print(f"\n[Benchmark] Running: {name}")
    try:
        summary = run_benchmark(
            ds,
            max_images=3,
            fast_only=True,   # fast for demo; remove for full quality
            three_stage=True,
            use_memory=True,
        )
        all_summaries[name] = summary
    except Exception as e:
        print(f"[Benchmark] ERROR on {name}: {e}")

# ── FINAL SUMMARY ──────────────────────────────────────────────
print("\n" + "="*60)
print("  MAIR+ DEMO COMPLETE — SUMMARY")
print("="*60)

print("\n  Demo Pipeline Results:")
print(f"  {'Input':<15} {'SSIM':>8} {'PSNR':>10} {'Time':>8} {'Calls':>6}")
print(f"  {'-'*50}")
for label, r in demo_results.items():
    if r.get("ssim"):
        print(f"  {label:<15} {r['ssim']:>8.4f} {r['psnr']:>9.2f}dB {r['time']:>7.2f}s {r.get('calls',0):>6}")

print("\n  Benchmark Averages (v2, fast mode):")
print(f"  {'Dataset':<20} {'Avg SSIM Gain':>14} {'Avg PSNR Gain':>14}")
print(f"  {'-'*50}")
for name, s in all_summaries.items():
    if s and s.get("avg_ssim_gain") is not None:
        sg = s['avg_ssim_gain']
        pg = s['avg_psnr_gain']
        icon = "🟢" if sg > 0.01 else ("🔴" if sg < -0.01 else "🟡")
        print(f"  {name:<20} {sg:>+14.4f} {pg:>+13.2f} dB  {icon}")

print(f"\n  Comparison images: outputs/ directory")
print(f"  Benchmark results: results/ directory (CSV + JSON)")
print(f"  HTML reports:      outputs/reports/")
print("="*60)
print("\n  ✅ Ready for professor demo! Open the comparison PNGs to see results.\n")
