# MAIR+ v2 — Multi-Agent Intelligent Image Restoration

> **Jiang et al., "Multi-Agent Image Restoration," IJCV 2026**
> Full v2 implementation with 12 contributions (C1–C12)

---

## What is MAIR+?

MAIR+ is an **agentic image restoration pipeline** that:

1. **Detects** what is wrong with an image (blur, noise, JPEG artifacts, low-light, haze, low-resolution)
2. **Plans** which expert model to apply, in what order, at what confidence
3. **Applies** restoration in reverse-causal order (Compression → Imaging → Scene)
4. **Evaluates** quality after each stage and rolls back if the expert made things worse
5. **Learns** from past successes using case-based memory (CaseStore)
6. **Reports** a per-stage breakdown with SSIM, PSNR, and optional LPIPS metrics

---

## Architecture

```
Input Image
    │
    ▼ detect_degradation()
    ├── blur score       (Laplacian variance)
    ├── sr score         (resolution heuristic, capped at 0.75)
    ├── jpeg score       (8-pixel boundary gradient ratio)
    ├── noise score      (median residual std)
    ├── lowlight score   (mean brightness)
    └── haze score       (Dark Channel Prior mean) ← C3
         │
         ▼ Three-Stage Scheduler
    ┌────────────────────────────────────────────┐
    │  Stage 1 — COMPRESSION  (jpeg artifacts)  │
    │  Stage 2 — IMAGING      (blur/SR/noise)   │
    │  Stage 3 — SCENE        (light/haze)      │
    │                                            │
    │  Each stage:                               │
    │    • Memory bias from CaseStore (C9)       │
    │    • Resolution penalty for slow experts   │
    │    • Confidence tiering (C10)              │
    │    • Optional voting ensemble (C12)        │
    │    • SpatialGuard size check (C5)          │
    │    • LPIPS-aware quality eval (C6)         │
    │    • QualityGate rollback (C4)             │
    │    • Re-detect → update scores (C2)        │
    │    • Record case to memory (C9)            │
    └────────────────────────────────────────────┘
         │
         ▼ Report Card (C7)
    Console / HTML per-stage report
```

---

## 12 Contributions

| ID | Name | Key File(s) |
|----|------|------------|
| C1 | Dehazing Expert (DCP) | `experts/dehaze_expert.py`, `core/dark_channel_prior.py` |
| C2 | Iterative Re-Detection | `core/iterative_context.py`, `scheduler/scheduler.py` |
| C3 | Dark Channel Prior Haze Signal | `core/dark_channel_prior.py`, `core/degradation_detector.py` |
| C4 | Quality Gate with Rollback | `core/quality_gate.py`, `scheduler/scheduler.py` |
| C5 | Spatial Integrity Guard | `core/spatial_integrity.py` |
| C6 | LPIPS Perceptual Quality | `evaluation/quality_evaluator.py` |
| C7 | Per-Stage Report Card | `utils/report_generator.py`, `run_pipeline.py` |
| C8 | Calibrated Stage Thresholds | `evaluation/calibrate_thresholds.py`, `config/thresholds.json` |
| C9 | Memory-Augmented Planning | `memory/case_store.py`, `memory/memory_planner.py` |
| C10 | Confidence-Tiered Scheduling | `scheduler/confidence_policy.py` |
| C11 | Resolution-Aware Ranking | `core/tool_registry.py`, `scheduler/expert_selector.py` |
| C12 | Expert Voting Ensemble | `scheduler/voting_scheduler.py`, `experts/unsharp_deblur_expert.py` |


---

## Setup

### 1. Install dependencies

```bash
# Core (required)
pip install opencv-python scikit-image numpy

# Optional: LPIPS perceptual metrics (C6)
pip install torch torchvision lpips

# Optional: guided filter for DCP (C3)
pip install opencv-contrib-python
```

Or use the provided install scripts:
```bash
python install_phase1_deps.py   # core dependencies
python install_phase2_deps.py   # deep learning models (SwinIR, Restormer)
```

### 2. Download model weights (for deep experts)

| Expert | Weights Path |
|--------|-------------|
| SwinIR SR | `models/SwinIR/experiments/pretrained_models/` |
| SwinIR JPEG | `models/SwinIR/experiments/pretrained_models/` |
| Restormer Deblur | `models/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth` |

DCP dehazing (C1) and all fast experts require **no model weights** — pure OpenCV.

---

## Usage

### Run on a single image

```bash
python run_pipeline.py --input path/to/image.jpg
```

### All v2 flags

```bash
# Report options (C7)
python run_pipeline.py --input image.jpg --report console   # default
python run_pipeline.py --input image.jpg --report html      # saves HTML report
python run_pipeline.py --input image.jpg --report both

# Voting ensemble per stage (C12)
python run_pipeline.py --input image.jpg --voting

# Disable memory-augmented planning (C9)
python run_pipeline.py --input image.jpg --no_memory

# Time budget in seconds (C11)
python run_pipeline.py --input image.jpg --budget 30

# Legacy mode — disable Three-Stage Framework (ablation)
python run_pipeline.py --input image.jpg --no_tsf
```

### Generate benchmark datasets

```bash
python datasets/generate_benchmark.py                   # all types, 10 images each
python datasets/generate_benchmark.py --types haze blur # specific types
python datasets/generate_benchmark.py --n 5             # 5 images per type
```

Degradation types: `blur`, `jpeg`, `noise`, `lowlight`, `haze`, `rain`, `mixed`

### Run benchmark evaluation

```bash
python evaluation/benchmark.py --list_sets              # list available sets
python evaluation/benchmark.py --all --max_images 3     # quick preview (3 imgs)
python evaluation/benchmark.py --all                    # full evaluation
python evaluation/benchmark.py --all --voting           # with voting ensemble
python evaluation/benchmark.py --dataset datasets/benchmark/blur_test
```

### Calibrate thresholds (C8)

```bash
# Create calibration sets first (same layout as benchmark sets)
mkdir -p datasets/calibration/my_set/degraded datasets/calibration/my_set/reference

# Run calibration
python evaluation/calibrate_thresholds.py --fast_only --max_images 5
```

### Run ablation study

```bash
python experiments/run_ablation.py --experiment A1      # TSF vs Legacy
python experiments/run_ablation.py --experiment A5      # Full v2 vs baseline
python experiments/run_ablation.py --all --fast_only    # all experiments, fast
```

---

## Expert Registry

| Key | Model | Stage | Speed | Quality | GPU |
|-----|-------|-------|-------|---------|-----|
| `swinir_sr` | SwinIR Real-SR ×4 | Imaging | Medium | High | ✓ |
| `restormer_deblur` | Restormer Motion Deblur | Imaging | Slow | Very High | ✓ |
| `opencv_denoise` | NLM Denoising | Imaging | Fast | Medium | ✗ |
| `nafnet_lite_denoise` | NAFNet-Lite Channel Gating | Imaging | Fast | **High** | ✗ |
| `swinir_jpeg` | SwinIR JPEG-CAR | Compression | Medium | High | ✓ |
| `clahe_lowlight` | CLAHE + Gamma | Scene | Very Fast | Medium | ✗ |
| `zero_dce_lowlight` | Zero-DCE Adaptive Curves | Scene | Fast | **High** | ✗ |
| `dcp_dehaze` | Dark Channel Prior | Scene | Fast | High | ✗ |
| `freq_derain` | Freq-Domain Rain Removal | Scene | Fast | **High** | ✗ |
| `opencv_unsharp_deblur` | Unsharp Mask (fallback) | Imaging | Very Fast | Low | ✗ |
| `wiener_deblur` | Wiener Deconvolution **(C13)** | Imaging | Fast | **High** | ✗ |
| `opencv_fast_jpeg` | NLM JPEG (fallback) | Compression | Very Fast | Low | ✗ |

**Bold** = Phase 3 additions that outperform the original baselines without any pretrained weights.

---

## Output Structure

```
outputs/
    pipeline_tmp/    ← inter-stage handoff images
    deblurred/       ← deblur expert outputs
    jpeg/            ← JPEG expert outputs
    dehazed/         ← dehazing expert outputs
    sr/              ← super-resolution outputs
    memory/
        case_memory.json   ← CaseStore memory (C9)
    reports/
        report_<name>_<ts>.html   ← HTML report cards (C7)

results/
    benchmark_<set>_<ts>.csv     ← benchmark metrics
    benchmark_<set>_<ts>.json
    ablation_<A1>_<ts>.json      ← ablation results
```

---

## Benchmark Results (v1 baseline, 2026-06-03)

| Dataset | Avg SSIM Gain | Avg PSNR Gain | Status |
|---------|---------------|---------------|--------|
| noise_test | **+0.2931** | **+3.70 dB** | 🟢 Excellent |
| mixed_test | **+0.2428** | **+2.90 dB** | 🟢 Good |
| jpeg_test | **+0.0264** | **+0.45 dB** | 🟡 Moderate |
| blur_test | **−0.0056** | **+0.01 dB** | 🔴 Regression (C4+C12 fix) |
| lowlight_test | **−0.0347** | **−0.24 dB** | 🔴 Regression (C4 fix) |

*v2 results with all 12 contributions to follow after benchmark run.*

---

## Project Structure

```
MAIR_Project/
├── agents/                 # BaseAgent, ExpertAgent, RestorationAgent
├── config/                 # thresholds.json (C8)
├── core/                   # Detector, QualityGate, SpatialGuard, DCP, IterativeCtx
├── datasets/               # generate_benchmark.py (6 degradation types)
├── evaluation/             # benchmark.py, quality_evaluator.py, calibrate_thresholds.py
├── experiments/            # compare_experts.py, run_ablation.py
├── experts/                # 8 expert modules
├── memory/                 # CaseStore + MemoryPlanner (C9)
├── scheduler/              # scheduler.py, expert_selector.py, confidence_policy.py, voting_scheduler.py
├── utils/                  # visualizer.py, report_generator.py (C7)
└── run_pipeline.py         # main entry point
```
