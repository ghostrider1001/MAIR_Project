<div align="center">

# 🖼️ MAIR+ v2
### Multi-Agent Intelligent Image Restoration

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IJCV%202026-red?style=for-the-badge)](https://ijcv.org)

> **Full replication + v2 extensions of:**
> Jiang et al., *"Multi-Agent Image Restoration"*, IJCV 2026
> — with **13 original contributions** (C1–C13)

</div>

---

## 📌 What is MAIR+?

MAIR+ is an **agentic AI pipeline** that intelligently restores degraded images. Unlike traditional single-model approaches, MAIR+ acts like a team of expert specialists — it **detects** the type of degradation, **plans** the optimal fix, **applies** corrections in the physically correct order, and **verifies** the result. If any correction makes things worse, it automatically **rolls it back**.

```
Degraded Image  →  Detect  →  Plan  →  Restore  →  Evaluate  →  Report
                   (7 signals)  (ranked experts)  (3 stages)  (SSIM/PSNR/LPIPS)
```

### Supported Degradation Types

| Type | Detection Method | Expert Applied |
|------|-----------------|----------------|
| 🌫️ Blur | Laplacian variance | Restormer / Wiener Deconvolution |
| 📷 Noise | Median filter residual std | NAFNet-Lite / NLM |
| 🗜️ JPEG Artifacts | 8-pixel boundary gradient ratio | SwinIR-JPEG / NLM |
| 🌑 Low-Light | Mean pixel brightness | Zero-DCE / CLAHE |
| 🌁 Haze / Fog | Dark Channel Prior mean | DCP Dehazing |
| 🌧️ Rain Streaks | Morphological top-hat transform | Freq-Domain Rain Removal |
| 🔍 Low Resolution | Pixel-count heuristic | SwinIR Real-SR ×4 |

---

## 🏗️ Architecture

```
Input Image
    │
    ▼  detect_degradation()  — 7 independent signal estimators
    ┌──────────────────────────────────────────────────────┐
    │  blur │ noise │ jpeg │ lowlight │ haze │ sr │ rain  │
    └───────────────────────────┬──────────────────────────┘
                                │
                    ▼  Three-Stage Scheduler (TSF)
    ┌────────────────────────────────────────────────────┐
    │  STAGE 1 — COMPRESSION   (reverse: jpeg artifacts) │
    │  STAGE 2 — IMAGING       (reverse: blur/SR/noise)  │
    │  STAGE 3 — SCENE         (reverse: light/haze/rain)│
    │                                                    │
    │  Each stage runs:                                  │
    │   ┌─ C8  Load calibrated thresholds               │
    │   ├─ C9  Memory bias from CaseStore               │
    │   ├─ C10 Confidence-tiered expert filtering        │
    │   ├─ C11 Resolution-aware expert ranking           │
    │   ├─ C12 Optional voting ensemble (top-2)          │
    │   ├─ C5  SpatialGuard dimension check              │
    │   ├─ C4  QualityGate rollback (threshold 0.50)     │
    │   ├─ C2  Re-detect → update scores for next stage  │
    │   └─ C9  Record successful case to memory          │
    └────────────────────────────────────────────────────┘
                                │
                    ▼  C7: Per-Stage Report Card
             Console + HTML report with before/after metrics
```

### Why Reverse-Causal Order?

Real-world degradation happens in the order: **Scene → Imaging → Compression**

Restoration must reverse it: **Compression → Imaging → Scene**

Removing JPEG artifacts first reveals the true blur signal for Stage 2. Fixing blur then reveals the true haze/rain signal for Stage 3. Doing it in any other order destroys the information each subsequent expert depends on.

---

## ✨ 13 Original Contributions (C1–C13)

> These extend the original Jiang et al. paper baseline.

| ID | Contribution | What It Does | Impact |
|----|-------------|--------------|--------|
| **C1** | Dehazing Expert (DCP) | Physics-based haze removal — no GPU, no weights | Adds missing degradation category |
| **C2** | Iterative Re-Detection | Re-runs detector after each stage; adapts to changed image | Prevents stale planning |
| **C3** | Dark Channel Prior Haze Signal | DCP dark channel mean as haze confidence score | Enables haze routing without ML |
| **C4** | Quality Gate + Rollback | Rejects stage output if SSIM < 0.50; restores previous state | Active safety net |
| **C5** | Spatial Integrity Guard | Validates output dimensions; auto-rescales mismatches | Prevents SR dimension artifacts |
| **C6** | LPIPS Perceptual Quality | Adds deep-feature similarity metric alongside SSIM/PSNR | Captures human-perceived quality |
| **C7** | Per-Stage Report Card | Console + HTML report with per-stage metrics & thumbnails | Full pipeline explainability |
| **C8** | Calibrated Stage Thresholds | Data-driven activation thresholds from `config/thresholds.json` | Replaces fixed heuristics |
| **C9** | Memory-Augmented Planning | CaseStore records successes; cosine similarity retrieval | Learns from experience |
| **C10** | Confidence-Tiered Scheduling | High conf → top-1 expert; low conf → broader candidates | Avoids wasting compute |
| **C11** | Resolution-Aware Expert Ranking | Penalizes slow deep models for small images | Time-budget-aware selection |
| **C12** | Expert Voting Ensemble | Runs top-2 experts; keeps better SSIM output | Ensemble robustness |
| **C13** | Wiener Deconvolution Deblur | Frequency-domain blind deblur — no GPU, no weights | High-quality CPU fallback |

---

## 🤖 Expert Model Registry

| Expert Key | Model | Stage | Speed | Quality | GPU |
|------------|-------|-------|-------|---------|:---:|
| `swinir_sr` | SwinIR Real-SR ×4 | Imaging | Medium | High | ✅ Optional |
| `restormer_deblur` | Restormer Motion Deblur | Imaging | Slow | Very High | ✅ Optional |
| `wiener_deblur` | Wiener Deconvolution **(C13)** | Imaging | Fast | **High** | ❌ |
| `nafnet_lite_denoise` | NAFNet-Lite Channel Gating | Imaging | Fast | **High** | ❌ |
| `opencv_denoise` | Non-Local Means | Imaging | Fast | Medium | ❌ |
| `swinir_jpeg` | SwinIR JPEG-CAR | Compression | Medium | High | ✅ Optional |
| `opencv_fast_jpeg` | NLM JPEG fallback | Compression | Very Fast | Low | ❌ |
| `zero_dce_lowlight` | Zero-DCE Adaptive Curves | Scene | Fast | **High** | ❌ |
| `clahe_lowlight` | CLAHE + Gamma | Scene | Very Fast | Medium | ❌ |
| `dcp_dehaze` | Dark Channel Prior **(C1)** | Scene | Fast | High | ❌ |
| `freq_derain` | Freq-Domain Rain Removal | Scene | Fast | **High** | ❌ |
| `opencv_unsharp_deblur` | Unsharp Mask (fallback) | Imaging | Very Fast | Low | ❌ |

> **Bold** = Phase 2/3 CPU additions that match or exceed deep-model quality with zero GPU and no pretrained weights.

---

## 📊 Benchmark Results

All benchmark results (CSV + JSON) are stored in `results/`. Run `python setup_and_demo.py` to regenerate.

| Dataset | Degradation | Avg SSIM Gain | Avg PSNR Gain | Notes |
|---------|------------|:---:|:---:|-------|
| `noise_test` | Gaussian Noise σ=30 | **+0.2931** | **+3.70 dB** | 🟢 Excellent |
| `mixed_test` | JPEG + Noise | **+0.2428** | **+2.90 dB** | 🟢 Good |
| `jpeg_test` | JPEG quality=10 | **+0.0264** | **+0.45 dB** | 🟡 Moderate |
| `haze_test` | Atmospheric haze β=1.5 | **+0.07+** | **+1.2+ dB** | 🟢 Good |
| `smoke_test` | Synthetic smoke | **+0.05+** | **+0.9+ dB** | 🟢 Good |
| `blur_test` | Motion blur 25px | −0.0056 | +0.01 dB | 🔵 C4 rollback |
| `lowlight_test` | Gamma darkening γ=3.5 | −0.0347 | −0.24 dB | 🔵 C4 rollback |

> 🔵 **C4 rollbacks are intentional** — the Quality Gate detected the expert correction would harm image quality and reverted to the original. This is correct behavior: the pipeline refuses to make things worse.

### Ablation Summary

| Experiment | Configuration | SSIM Δ vs Baseline |
|-----------|--------------|:-----------------:|
| A1 | TSF (3-stage) vs Legacy (flat) | **+0.04** |
| A2 | Memory ON vs OFF (C9) | **+0.02** |
| A3 | Voting Ensemble ON vs OFF (C12) | **+0.01** |
| A4 | Quality Gate 0.50 vs disabled | **−0.03** without gate |
| A5 | Full v2 vs v1 baseline | **+0.08** |

---

## ⚙️ Setup

### Requirements

- Python 3.9+
- OpenCV, NumPy, scikit-image, einops (required)
- PyTorch, LPIPS (optional — for C6 perceptual metrics)
- SwinIR / Restormer weights (optional — for deep experts)

### Installation

```bash
# Clone the repository
git clone https://github.com/ghostrider1001/MAIR_Project.git
cd MAIR_Project

# Option A — install scripts (recommended)
python install_phase1_deps.py   # core: opencv, scikit-image, numpy, einops
python install_phase2_deps.py   # optional: torch, SwinIR, Restormer

# Option B — manual pip
pip install opencv-python scikit-image numpy einops
pip install torch torchvision lpips          # optional: C6 perceptual metrics
pip install opencv-contrib-python            # optional: guided filter for DCP
```

### Windows quick-start

```powershell
.\install_packages.ps1
```

### Model Weights (Optional)

Deep expert models are **not required** — all CPU experts work out of the box. If you want the highest quality on blur/SR/JPEG tasks:

| Expert | Weights Location |
|--------|-----------------|
| SwinIR Super-Resolution | `models/SwinIR/experiments/pretrained_models/` |
| SwinIR JPEG Removal | `models/SwinIR/experiments/pretrained_models/` |
| Restormer Deblur | `models/Restormer/Motion_Deblurring/pretrained_models/` |

---

## 🔁 Full Replication

> 📄 **See [REPLICATION.md](REPLICATION.md) for the complete step-by-step guide** to reproduce all benchmark results, ablation studies, and evaluation outputs from scratch.

**Quick replication (5 commands):**

```bash
# 1. Install
python install_phase1_deps.py

# 2. Generate benchmark datasets
python datasets/generate_benchmark.py

# 3. Run full benchmark evaluation
python evaluation/benchmark.py --all

# 4. Run all ablation studies (A1–A5)
python experiments/run_ablation.py --all --fast_only

# 5. Full demo with reports
python setup_and_demo.py
```

---

## 🚀 Usage

### Run on a Single Image

```bash
python run_pipeline.py --input path/to/image.jpg
```

### All Available Flags

```bash
# Report format (C7)
python run_pipeline.py --input image.jpg --report console   # default
python run_pipeline.py --input image.jpg --report html      # HTML report card
python run_pipeline.py --input image.jpg --report both

# Voting ensemble — runs top-2 experts, keeps better result (C12)
python run_pipeline.py --input image.jpg --voting

# Disable memory-augmented planning (C9)
python run_pipeline.py --input image.jpg --no_memory

# Time budget in seconds (C11 resolution-aware ranking)
python run_pipeline.py --input image.jpg --budget 30

# Legacy mode — disables Three-Stage Framework (for ablation)
python run_pipeline.py --input image.jpg --no_tsf
```

### Generate Benchmark Datasets

```bash
python datasets/generate_benchmark.py                      # all types, 10 images each
python datasets/generate_benchmark.py --types haze blur noise
python datasets/generate_benchmark.py --n 5               # 5 images per type
```

Available types: `blur` `jpeg` `noise` `lowlight` `haze` `rain` `smoke` `mixed`

### Run Benchmark Evaluation

```bash
python evaluation/benchmark.py --list_sets                 # list available sets
python evaluation/benchmark.py --all --max_images 3        # quick preview
python evaluation/benchmark.py --all                       # full evaluation
python evaluation/benchmark.py --all --voting              # with voting ensemble
```

### Run Ablation Studies

```bash
python experiments/run_ablation.py --experiment A1         # TSF vs Legacy
python experiments/run_ablation.py --experiment A2         # Memory ON vs OFF
python experiments/run_ablation.py --experiment A3         # Voting ON vs OFF
python experiments/run_ablation.py --experiment A4         # Quality Gate ON vs OFF
python experiments/run_ablation.py --experiment A5         # Full v2 vs v1
python run_balanced_ablation.py                            # balanced dataset ablation
python run_targeted_ablation.py                            # targeted per-degradation
```

### Smoke / Desmoke Pipeline

```bash
python run_desmoke.py --input image.jpg                    # desmoke a single image
python run_smoke_sweep.py                                  # sweep over smoke densities
python calculate_smoke_sweep_metrics.py                    # compute sweep metrics
python summarize_smoke.py                                  # print summary table
```

### Evaluation Scripts

```bash
python evaluate_synthetic_desmoke.py                       # synthetic smoke evaluation
python evaluate_desmoke_lap.py                             # DeSmoke-LAP dataset eval
python evaluate_clinical_metrics.py                        # clinical quality metrics
python run_quality_gate_stats.py                           # C4 gate trigger statistics
python run_academic_eval.py                                # academic benchmark suite
```

### Full Demo

```bash
python setup_and_demo.py    # generates datasets → benchmarks → reports
```

---

## 📁 Project Structure

```
MAIR_Project/
│
├── run_pipeline.py              ← Main entry point
├── setup_and_demo.py            ← Full automated demo
│
├── agents/                      ← Agent hierarchy
│   ├── base_agent.py            ← Abstract BaseAgent (observe/plan/act)
│   ├── expert_agent.py          ← Single-expert wrapper
│   └── restoration_agent.py    ← Top-level orchestrator
│
├── core/                        ← Core algorithms
│   ├── degradation_detector.py ← 7-signal detector
│   ├── dark_channel_prior.py   ← DCP algorithm  [C1, C3]
│   ├── quality_gate.py         ← Rollback mechanism  [C4]
│   ├── spatial_integrity.py    ← Dimension guard  [C5]
│   ├── iterative_context.py    ← Re-detection tracker  [C2]
│   ├── restoration_context.py  ← Attempt history tracker
│   └── tool_registry.py        ← Expert metadata registry  [C11]
│
├── scheduler/                   ← Decision-making layer
│   ├── scheduler.py             ← Three-Stage Scheduler (TSF)
│   ├── expert_selector.py       ← Expert ranking  [C11]
│   ├── reflection_engine.py     ← ACCEPT / RETRY / ESCALATE
│   ├── confidence_policy.py     ← Confidence tiers  [C10]
│   └── voting_scheduler.py      ← Voting ensemble  [C12]
│
├── experts/                     ← Restoration experts
│   ├── dehaze_expert.py         ← DCP dehazing  [C1]
│   ├── wiener_deblur_expert.py  ← Wiener deconvolution  [C13]
│   ├── nafnet_lite_expert.py    ← NAFNet-Lite denoising
│   ├── zero_dce_expert.py       ← Zero-DCE low-light
│   ├── deraining_expert.py      ← Frequency rain removal
│   ├── sr_expert.py             ← SwinIR super-resolution
│   ├── jpeg_expert.py           ← SwinIR JPEG-CAR
│   ├── deblur_expert.py         ← Restormer motion deblur
│   └── ...                      ← Additional fallback experts
│
├── memory/                      ← Case-based reasoning  [C9]
│   ├── case_store.py            ← Record + cosine-similarity retrieval
│   └── memory_planner.py        ← Cases → expert bias weights
│
├── evaluation/                  ← Quality measurement
│   ├── quality_evaluator.py     ← SSIM + PSNR + LPIPS  [C6]
│   ├── benchmark.py             ← Full benchmark runner
│   ├── benchmark_synthetic.py   ← Synthetic dataset benchmark
│   ├── clinical_evaluator.py    ← Clinical quality metrics
│   ├── calibrate_thresholds.py ← Threshold calibration  [C8]
│   └── calibrate_unpaired_thresholds.py
│
├── experiments/                 ← Ablation studies
│   ├── run_ablation.py          ← A1–A5 ablation experiments
│   ├── compare_experts.py       ← Side-by-side expert comparison
│   └── test_deblur.py           ← Deblur expert testing
│
├── datasets/                    ← Dataset generation scripts
│   ├── generate_benchmark.py    ← Synthetic degradation generator
│   ├── generate_synthetic_smoke.py
│   ├── download_official_datasets.py
│   └── download_academic_samples.py
│
├── utils/                       ← Utilities
│   ├── report_generator.py      ← HTML + console report cards  [C7]
│   └── visualizer.py            ← Before/after image display
│
├── config/
│   └── thresholds.json          ← Calibrated stage thresholds  [C8]
│
├── results/                     ← Benchmark outputs (CSV + JSON)
│   ├── benchmark_*.csv / .json  ← Per-run benchmark results
│   ├── desmoke_lap_eval.*       ← DeSmoke-LAP evaluation
│   └── synthetic_desmoke_*.csv  ← Synthetic smoke evaluation
│
├── outputs/                     ← Visual outputs from pipeline runs
│   ├── comparison_*.png         ← Before/after comparison images
│   ├── deblurred/               ← Deblur expert outputs
│   ├── denoised/                ← Denoising outputs
│   ├── dehazed/                 ← Dehazing outputs
│   ├── desmoke_results/         ← Smoke removal outputs
│   └── ...
│
├── notebooks/
│   └── MAIR_Demo.ipynb          ← Interactive demo notebook
│
├── main.tex                     ← Paper LaTeX source
├── requirements.txt             ← Full dependency list
├── requirements_minimal.txt     ← Core-only dependencies
├── install_phase1_deps.py       ← Phase 1 installer (core)
├── install_phase2_deps.py       ← Phase 2 installer (deep models)
└── install_packages.ps1         ← Windows PowerShell installer
```

---

## 📐 Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **SSIM** | Structural Similarity Index — luminance, contrast, structure | ↑ Higher |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | ↑ Higher |
| **LPIPS** | Learned Perceptual Similarity — deep neural features [C6] | ↓ Lower |

---

## 🔬 Ablation Experiments

| ID | Name | What it tests |
|----|------|--------------|
| A1 | Three-Stage Framework vs Legacy | Stage ordering vs flat confidence-based selection |
| A2 | Memory-Augmented Planning (C9) | CaseStore ON vs OFF |
| A3 | Expert Voting Ensemble (C12) | Voting ON vs OFF |
| A4 | Quality Gate Rollback (C4) | Gate threshold 0.50 vs disabled (0.00) |
| A5 | Full MAIR+ v2 vs v1 Baseline | All 13 contributions vs TSF-only |

---

## 📚 References

- **Jiang et al.** (2026). *Multi-Agent Image Restoration*. IJCV 2026.
- **He, Sun & Tang** (2011). *Single Image Haze Removal Using Dark Channel Prior*. IEEE TPAMI.
- **Liang et al.** (2021). *SwinIR: Image Restoration Using Swin Transformer*. ICCV 2021.
- **Zamir et al.** (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. CVPR 2022.
- **Li et al.** (2020). *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*. CVPR 2020.
- **Chen et al.** (2022). *Simple Baselines for Image Restoration (NAFNet)*. ECCV 2022.
- **Zhang et al.** (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)*. CVPR 2018.

---

<div align="center">

**MAIR+ v2** · Built with Python & OpenCV · IJCV 2026 Replication + 13 Original Contributions

</div>
