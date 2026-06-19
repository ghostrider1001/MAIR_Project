"""
Tool Registry
=============
Central catalog of all available restoration models.

Every expert is registered here with:
  - name        : human-readable label
  - fn          : the callable that performs restoration
  - task        : primary degradation it targets
  - handles     : list of degradation types it can improve
  - stage       : restoration stage per the three-stage framework (paper §3.2)
                  "compression" | "imaging" | "scene"
  - speed       : "very_fast" | "fast" | "medium" | "slow"
  - quality     : "low" | "medium" | "high" | "very_high"
  - gpu_boost   : True if GPU gives significant speedup
  - description : one-line summary

Three-Stage Framework (Jiang et al., IJCV 2026):
  Real-world degradations occur in order:
      Scene → Imaging → Compression
  Restoration reverses them in opposite order:
      Stage 1 (compression): JPEG artifact removal  ← always first
      Stage 2 (imaging)    : Deblur, Denoise, SR    ← always second
      Stage 3 (scene)      : Low-light, Rain, Haze  ← always third

Adding a new model never requires changes to the scheduler.
Just add an entry here with the correct stage.
"""

from experts.sr_expert              import restore_sr
from experts.deblur_expert          import restore_deblur
from experts.denoise_expert         import restore_denoise
from experts.jpeg_expert            import restore_jpeg
from experts.lowlight_expert        import restore_lowlight
from experts.dehaze_expert          import restore_dcp              # C1
from experts.unsharp_deblur_expert  import restore_unsharp_deblur   # C12 fallback
from experts.wiener_deblur_expert   import restore as restore_wiener # C13: CPU Wiener deblur
from experts.fastjpeg_expert        import restore_fast_jpeg         # C12
from experts.zero_dce_expert        import restore as restore_zero_dce     # Phase 3
from experts.deraining_expert       import restore as restore_derain        # Phase 3
from experts.nafnet_lite_expert     import restore as restore_nafnet_lite   # Phase 3


# ─────────────────────────────────────────────────────────────
# Three-Stage Framework — ordered stage constants
# ─────────────────────────────────────────────────────────────
STAGE_ORDER = ["compression", "imaging", "scene"]

# Degradation type → stage mapping
DEGRADATION_STAGE = {
    "jpeg":     "compression",   # Stage 1
    "blur":     "imaging",       # Stage 2
    "denoise":  "imaging",       # Stage 2
    "sr":       "imaging",       # Stage 2
    "lowlight": "scene",         # Stage 3
    "haze":     "scene",         # Stage 3  ← C3
    "rain":     "scene",         # Stage 3  ← Phase 3
}

# ─────────────────────────────────────────────────────────────
# Speed → numeric weight for ranking (higher = faster)
# ─────────────────────────────────────────────────────────────
SPEED_WEIGHT = {
    "very_fast": 1.0,
    "fast":      0.8,
    "medium":    0.6,
    "slow":      0.3,
}

# Quality → numeric weight for ranking (higher = better)
QUALITY_WEIGHT = {
    "low":       0.4,
    "medium":    0.6,
    "high":      0.85,
    "very_high": 1.0,
}


# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────
REGISTRY = {
    "swinir_sr": {
        "name":        "SwinIR Super Resolution x4",
        "fn":          restore_sr,
        "task":        "sr",
        "handles":     ["sr"],
        "stage":       "imaging",        # Stage 2
        "speed":       "medium",
        "quality":     "high",
        "gpu_boost":   True,
        "preserves_size": False,          # 4x upscale changes dimensions
        "description": "SwinIR Real-SR x4 (GAN) — upscales low-resolution images 4×",
    },
    "nafnet_lite_denoise": {
        "name":        "NAFNet-Lite Channel Gating",
        "fn":          restore_nafnet_lite,
        "task":        "denoise",
        "handles":     ["denoise", "blur"],  # Use NAFNet for blur too to avoid Restormer white blob
        "stage":       "imaging",        # Stage 2
        "speed":       "fast",
        "quality":     "high",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "NAFNet-inspired variance gating — handles noise and blur safely",
    },
    "opencv_denoise": {
        "name":        "OpenCV NLM Denoising",
        "fn":          restore_denoise,
        "task":        "denoise",
        "handles":     ["denoise"],       # FIX: removed 'blur' — prevents denoiser routing on blur images
        "stage":       "imaging",        # Stage 2
        "speed":       "fast",
        "quality":     "medium",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Non-Local Means denoising — fast CPU-based Gaussian noise removal",
    },
    "swinir_jpeg": {
        "name":        "SwinIR JPEG Artifact Removal",
        "fn":          restore_jpeg,
        "task":        "jpeg",
        "handles":     ["jpeg"],
        "stage":       "compression",    # Stage 1
        "speed":       "medium",
        "quality":     "high",
        "gpu_boost":   True,
        "preserves_size": True,
        "description": "SwinIR CAR — removes JPEG compression blocking artifacts (q=10)",
    },
    "clahe_lowlight": {
        "name":        "CLAHE Lowlight Enhancement",
        "fn":          restore_lowlight,
        "task":        "lowlight",
        "handles":     ["lowlight"],
        "stage":       "scene",          # Stage 3
        "speed":       "very_fast",
        "quality":     "medium",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "CLAHE + adaptive gamma — enhances dark/underexposed images",
    },
    # ── C1: Dehazing Expert (DCP) ────────────────────────────────────
    "dcp_dehaze": {
        "name":        "DCP Dehazing (Dark Channel Prior)",
        "fn":          restore_dcp,
        "task":        "haze",
        "handles":     ["haze"],
        "stage":       "scene",          # Stage 3
        "speed":       "fast",
        "quality":     "high",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Physics-based haze removal via Dark Channel Prior (He et al. 2011)",
    },
    # ── C12: Fast Fallback Experts ─────────────────────────────────
    "opencv_unsharp_deblur": {
        "name":        "Unsharp Mask Deblur (fast fallback)",
        "fn":          restore_unsharp_deblur,
        "task":        "blur",
        "handles":     ["blur"],
        "stage":       "imaging",
        "speed":       "very_fast",
        "quality":     "low",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Unsharp masking — very fast CPU blur reduction (last-resort fallback)",
    },
    # Wiener Deconvolution — CPU-based physics-correct deblur
    "wiener_deblur": {
        "name":        "Wiener Deconvolution Deblur",
        "fn":          restore_wiener,
        "task":        "blur",
        "handles":     ["blur"],
        "stage":       "imaging",
        "speed":       "fast",
        "quality":     "high",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Wiener filter deconvolution — physics-correct motion deblur, no weights needed",
    },
    "opencv_fast_jpeg": {
        "name":        "Fast JPEG NLM (fast fallback)",
        "fn":          restore_fast_jpeg,
        "task":        "jpeg",
        "handles":     ["jpeg"],
        "stage":       "compression",
        "speed":       "very_fast",
        "quality":     "low",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "NLM denoising tuned for JPEG blocks — C12 voting fallback",
    },
    # ── Phase 3: SOTA-class experts (no weights needed) ─────────────
    "zero_dce_lowlight": {
        "name":        "Zero-DCE Adaptive Curve Enhancement",
        "fn":          restore_zero_dce,
        "task":        "lowlight",
        "handles":     ["lowlight"],
        "stage":       "scene",          # Stage 3
        "speed":       "fast",
        "quality":     "very_high",      # FIX: boosted so it outranks CLAHE in all confidence tiers
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Zero-DCE-style adaptive pixel curves — far better than CLAHE, no weights",
    },
    "freq_derain": {
        "name":        "Frequency-Domain Deraining",
        "fn":          restore_derain,
        "task":        "rain",
        "handles":     ["rain"],
        "stage":       "scene",          # Stage 3
        "speed":       "fast",
        "quality":     "high",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "Morphological + FFT rain streak removal — DRSformer upgrade path ready",
    },
    "nafnet_lite_denoise": {
        "name":        "NAFNet-Lite Channel Gating Denoiser",
        "fn":          restore_nafnet_lite,
        "task":        "denoise",
        "handles":     ["denoise"],       # FIX: removed 'blur' — prevents denoiser routing on blur images
        "stage":       "imaging",        # Stage 2
        "speed":       "fast",
        "quality":     "high",
        "gpu_boost":   False,
        "preserves_size": True,
        "description": "NAFNet-inspired variance gating — better than NLM on textures, no weights",
    },
}


# ─────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_expert(name: str) -> dict:
    """Return registry entry by key. Raises KeyError if not found."""
    if name not in REGISTRY:
        raise KeyError(f"Expert '{name}' not in registry. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


def get_experts_for_task(task: str) -> list:
    """Return all registry entries that handle a given degradation type."""
    return [
        (key, entry)
        for key, entry in REGISTRY.items()
        if task in entry["handles"]
    ]


def list_all() -> None:
    """Print a formatted summary of all registered experts."""
    print("\n" + "─" * 68)
    print(f"  {'KEY':<22}  {'STAGE':<12}  {'TASK':<10}  {'SPEED':<12}  {'QUALITY'}")
    print("─" * 68)
    for stage in STAGE_ORDER:
        for key, entry in REGISTRY.items():
            if entry.get("stage") == stage:
                print(
                    f"  {key:<22}  {stage:<12}  {entry['task']:<10}  "
                    f"{entry['speed']:<12}  {entry['quality']}"
                )
                print(f"    {entry['description']}")
    print("─" * 68 + "\n")


def get_experts_for_stage(stage: str) -> list:
    """
    Return all registry entries belonging to a given restoration stage.

    Args:
        stage: "compression" | "imaging" | "scene"

    Returns:
        List of (expert_key, expert_entry) tuples.
    """
    return [
        (key, entry)
        for key, entry in REGISTRY.items()
        if entry.get("stage") == stage
    ]


def expert_score(
    entry:       dict,
    confidence:  float = 1.0,
    pixel_count: int | None = None,
) -> float:
    """
    Compute a single ranking score for an expert given degradation confidence.

    Score = quality_weight × confidence + speed_weight × (1 - confidence)

    C11 resolution penalty: slow experts on large images (>2MP) receive
    a penalty proportional to image size, preventing Restormer from being
    chosen over faster alternatives on 4K inputs.

    At high confidence: prioritize quality.
    At low confidence: prioritize speed (don't waste time on heavy models).
    """
    q = QUALITY_WEIGHT.get(entry["quality"], 0.5)
    s = SPEED_WEIGHT.get(entry["speed"],    0.5)
    base_score = q * confidence + s * (1.0 - confidence)

    # C11: resolution-aware penalty for slow models
    if pixel_count is not None and entry.get("speed") == "slow":
        # Penalize slow experts linearly for images above 2MP (1920×1080 ≈ 2M px)
        penalty = max(0.0, min(0.20, (pixel_count - 2_000_000) / 10_000_000))
        base_score = max(0.0, base_score - penalty)

    return round(base_score, 4)
