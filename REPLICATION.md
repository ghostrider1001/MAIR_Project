# 🔁 MAIR+ v2 — Full Replication Guide

> **Paper:** Jiang et al., *"Multi-Agent Image Restoration"*, IJCV 2026  
> **This guide** walks you through completely reproducing all benchmark results, ablation studies, and evaluation outputs from scratch.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Verify Installation](#3-verify-installation)
4. [Generate Benchmark Datasets](#4-generate-benchmark-datasets)
5. [Run the Pipeline (Sanity Check)](#5-run-the-pipeline-sanity-check)
6. [Run Full Benchmark Evaluation](#6-run-full-benchmark-evaluation)
7. [Run Ablation Studies](#7-run-ablation-studies)
8. [Run Smoke / Desmoke Evaluation](#8-run-smoke--desmoke-evaluation)
9. [Generate Reports](#9-generate-reports)
10. [Expected Outputs](#10-expected-outputs)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | 3.10 or 3.11 recommended |
| pip | latest | `python -m pip install --upgrade pip` |
| OpenCV | 4.x | installed via pip |
| RAM | 4 GB+ | 8 GB recommended |
| GPU | Optional | CUDA GPU speeds up deep experts; all CPU experts work without one |
| Disk Space | ~2 GB | for datasets, results, and output images |

---

## 2. Clone & Install

### Step 2.1 — Clone the repository

```bash
git clone https://github.com/ghostrider1001/MAIR_Project.git
cd MAIR_Project
```

### Step 2.2 — Install Phase 1 (core dependencies — required)

```bash
python install_phase1_deps.py
```

This installs: `opencv-python`, `scikit-image`, `numpy`, `einops`, `scipy`, `matplotlib`, `tqdm`

**Or manually:**
```bash
pip install opencv-python scikit-image numpy einops scipy matplotlib tqdm
```

### Step 2.3 — Install Phase 2 (optional — deep learning models)

```bash
python install_phase2_deps.py
```

This installs: `torch`, `torchvision`, `lpips`, and clones SwinIR + Restormer model repos.

> ✅ Skip Phase 2 if you don't have a GPU or don't need deep-model experts. The entire pipeline runs on CPU-only experts (NAFNet-Lite, Wiener deblur, Zero-DCE, DCP, etc.).

### Windows users

```powershell
.\install_packages.ps1
```

---

## 3. Verify Installation

Run this quick check to confirm everything is wired up correctly:

```bash
python test_thresholds.py
```

Expected output:
```
✓ Config loaded: config/thresholds.json
✓ Detector imports OK
✓ Scheduler imports OK
✓ All experts importable
```

If you installed Phase 2, also test:
```bash
python test_restormer.py
```

---

## 4. Generate Benchmark Datasets

This creates synthetic degraded images from clean source images for all degradation types.

```bash
python datasets/generate_benchmark.py
```

This generates 10 images per degradation type across 7 categories:

| Category | What it creates |
|----------|----------------|
| `blur_test` | Gaussian + motion-blurred images |
| `noise_test` | Gaussian noise σ=30 images |
| `jpeg_test` | JPEG-compressed quality=10 images |
| `lowlight_test` | Gamma-darkened (γ=3.5) images |
| `haze_test` | Atmospheric haze β=1.5 images |
| `rain_test` | Synthetic rain streak images |
| `smoke_test` | Synthetic smoke overlay images |
| `mixed_test` | JPEG + Noise combined images |

> **Source images:** The generator uses images from `datasets/Set14/`, `datasets/BSDS100/`, or any images you place in `datasets/natural_clean/`. These datasets are not included in the repo due to size — download them separately or let the generator use any images you have.

To generate specific types only:
```bash
python datasets/generate_benchmark.py --types haze blur noise
```

To generate with more images per set:
```bash
python datasets/generate_benchmark.py --n 20
```

---

## 5. Run the Pipeline (Sanity Check)

Before running full benchmarks, verify the pipeline works end-to-end on a single image.

```bash
# Run on any test image with full console report
python run_pipeline.py --input datasets/Set14/lenna.png --report console
```

Expected console output:
```
[MAIR+] Detecting degradations...
  blur:     0.12   noise:    0.41   jpeg:     0.08
  lowlight: 0.15   haze:     0.04   sr:       0.00   rain: 0.02

[MAIR+] Stage 1 — COMPRESSION
  Skipping (no JPEG artifact detected above threshold)

[MAIR+] Stage 2 — IMAGING
  Selected expert: nafnet_lite_denoise  (confidence: HIGH)
  SSIM: 0.812 → 0.934  |  PSNR: 26.4 → 28.9 dB  ✓ ACCEPTED

[MAIR+] Stage 3 — SCENE
  Skipping (no scene degradation detected)

[MAIR+] Done. Final SSIM: 0.934  PSNR: 28.9 dB
```

Try with HTML report:
```bash
python run_pipeline.py --input datasets/Set14/lenna.png --report html
```
Report saved to `outputs/reports/`.

---

## 6. Run Full Benchmark Evaluation

This runs the MAIR+ pipeline against all generated benchmark sets and saves results to `results/`.

### Step 6.1 — Quick preview (3 images per set)

```bash
python evaluation/benchmark.py --all --max_images 3
```

### Step 6.2 — Full evaluation

```bash
python evaluation/benchmark.py --all
```

### Step 6.3 — With voting ensemble (C12)

```bash
python evaluation/benchmark.py --all --voting
```

Results are saved as `results/benchmark_<type>_<timestamp>.csv` and `.json`.

### Step 6.4 — Academic benchmark suite

```bash
python run_academic_eval.py
```

### Step 6.5 — Massive benchmark (all datasets, all configs)

```bash
python run_all_benchmarks.py
```

### Expected benchmark results

| Dataset | Avg SSIM Gain | Avg PSNR Gain |
|---------|:---:|:---:|
| `noise_test` | **+0.29** | **+3.7 dB** |
| `mixed_test` | **+0.24** | **+2.9 dB** |
| `jpeg_test` | **+0.03** | **+0.5 dB** |
| `haze_test` | **+0.07** | **+1.2 dB** |
| `blur_test` | ~0.00 | ~0.00 |
| `lowlight_test` | ~0.00 | ~0.00 |

> Blur and low-light showing ~0 gain is **expected** — the C4 Quality Gate detects these experts are not helping and rolls back.

---

## 7. Run Ablation Studies

These experiments isolate the contribution of each v2 component.

### A1 — Three-Stage Framework vs Legacy

Tests whether the reverse-causal stage ordering improves over flat expert selection.

```bash
python experiments/run_ablation.py --experiment A1
```

### A2 — Memory-Augmented Planning (C9)

Tests whether the CaseStore memory improves expert selection over time.

```bash
python experiments/run_ablation.py --experiment A2
```

### A3 — Expert Voting Ensemble (C12)

Tests whether running top-2 experts and keeping the better result helps.

```bash
python experiments/run_ablation.py --experiment A3
```

### A4 — Quality Gate Rollback (C4)

Tests the impact of the SSIM-based rollback (gate at 0.50 vs disabled).

```bash
python experiments/run_ablation.py --experiment A4
```

### A5 — Full v2 vs v1 Baseline

Compares all 13 contributions active vs TSF-only baseline.

```bash
python experiments/run_ablation.py --experiment A5
```

### Run all ablations at once

```bash
python experiments/run_ablation.py --all --fast_only
```

### Balanced ablation (equal images per degradation type)

```bash
python run_balanced_ablation.py
```

### Targeted ablation (per-degradation breakdown)

```bash
python run_targeted_ablation.py
```

### Quality Gate statistics

```bash
python run_quality_gate_stats.py
```

---

## 8. Run Smoke / Desmoke Evaluation

### Step 8.1 — Generate synthetic smoke test images

```bash
python make_better_smoke.py
```

### Step 8.2 — Run the desmoke pipeline

```bash
python run_desmoke.py --input datasets/synthetic_smoke/
```

### Step 8.3 — Sweep across smoke density levels

```bash
python run_smoke_sweep.py
```

This tests smoke densities at 10%, 20%, 30%, 40% and saves outputs to `outputs/`.

### Step 8.4 — Compute sweep metrics

```bash
python calculate_smoke_sweep_metrics.py
python summarize_smoke.py
```

### Step 8.5 — DeSmoke-LAP dataset evaluation

```bash
python evaluate_desmoke_lap.py
```

### Step 8.6 — Synthetic desmoke evaluation

```bash
python evaluate_synthetic_desmoke.py
```

### Step 8.7 — Clinical quality metrics

```bash
python evaluate_clinical_metrics.py
```

---

## 9. Generate Reports

### Per-image HTML report card (C7)

```bash
python run_pipeline.py --input image.jpg --report html
# → saved to outputs/reports/
```

### Paper metrics table

```bash
python generate_paper_metrics.py
```

### Academic benchmarks report

```bash
python generate_academic_benchmarks.py
```

### Full automated demo (everything at once)

```bash
python setup_and_demo.py
```

This single script:
1. Generates benchmark datasets
2. Runs all benchmark evaluations
3. Runs ablation studies
4. Produces HTML reports
5. Prints a summary table

---

## 10. Expected Outputs

After running the full replication, you should find:

| Location | Contents |
|----------|---------|
| `results/benchmark_*.csv` | Per-image SSIM/PSNR metrics for each benchmark run |
| `results/benchmark_*.json` | Detailed JSON results with per-stage breakdown |
| `results/desmoke_lap_eval.*` | DeSmoke-LAP evaluation results |
| `results/synthetic_desmoke_eval.csv` | Synthetic smoke experiment results |
| `outputs/comparison_*.png` | Before/after visual comparisons |
| `outputs/deblurred/` | Deblur expert visual outputs |
| `outputs/dehazed/` | Dehazing expert visual outputs |
| `outputs/desmoke_results/` | Desmoke pipeline outputs |
| `outputs/reports/` | HTML per-stage report cards |

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'einops'`
```bash
pip install einops
```

### Pipeline runs but SSIM doesn't improve on blur/low-light
This is **expected** — C4 (Quality Gate) is working correctly. The pipeline detects the expert correction is not helpful and rolls back. Check `results/` for `c4_rollback: true` entries in the JSON output.

### Deep experts (Restormer, SwinIR) not loading
These require Phase 2 installation and model weights. Run:
```bash
python install_phase2_deps.py
```
Then download pretrained weights into `models/Restormer/` and `models/SwinIR/` as described in README.md.

### Out of memory on large images
Use the `--budget` flag to limit time (which limits which heavy models are selected):
```bash
python run_pipeline.py --input large_image.jpg --budget 15
```

### Results vary slightly between runs
This is normal if memory (C9) is enabled — the CaseStore biases expert selection based on past successes. To get fully deterministic results, disable memory:
```bash
python run_pipeline.py --input image.jpg --no_memory
```

---

<div align="center">

*MAIR+ v2 — Full Replication Guide · June 2026*

</div>
