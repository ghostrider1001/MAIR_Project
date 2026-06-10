"""
MAIR+ Benchmark Dataset Generator
===================================
Synthesizes degraded/reference image pairs for evaluating the pipeline.

Takes clean reference images and applies controlled degradations to create
matched LQ (degraded) / HQ (reference) pairs — the standard IR evaluation setup.

Output layout:
    datasets/benchmark/
        blur_test/
            degraded/   ← blurred images
            reference/  ← original clean images
        jpeg_test/
            degraded/   ← JPEG-compressed images
            reference/
        noise_test/
            degraded/   ← noisy images
            reference/
        lowlight_test/
            degraded/   ← darkened images
            reference/
        mixed_test/
            degraded/   ← multi-degradation images
            reference/

Usage:
    python datasets/generate_benchmark.py
    python datasets/generate_benchmark.py --source_dir my_clean_images/
    python datasets/generate_benchmark.py --types blur jpeg --n 5
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path

# ── make project root importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARK_ROOT = "datasets/benchmark"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ─────────────────────────────────────────────────────────────
# DEGRADATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def apply_motion_blur(img: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Apply horizontal motion blur to an image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_blur(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    k = max(3, int(6 * sigma) | 1)  # kernel size odd, at least 3
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_jpeg_compression(img: np.ndarray, quality: int = 10) -> np.ndarray:
    """Apply JPEG compression artifacts by encode/decode cycle."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_gaussian_noise(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_lowlight(img: np.ndarray, gamma: float = 3.5) -> np.ndarray:
    """Darken an image with gamma correction (simulates low-light capture)."""
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, table)


def apply_mixed(img: np.ndarray) -> np.ndarray:
    """Apply a combination: JPEG + noise (simulates real-world mixed degradations)."""
    img = apply_jpeg_compression(img, quality=15)
    img = apply_gaussian_noise(img, sigma=20.0)
    return img


def apply_rain(
    img:         np.ndarray,
    n_streaks:   int   = 800,
    length_range: tuple = (15, 60),
    thickness:   int   = 1,
    alpha:       float = 0.55,
    angle_deg:   float = -15.0,   # slight diagonal (realistic rain angle)
) -> np.ndarray:
    """
    Synthesize rain streaks using additive line rendering.

    Each streak is a short, bright line segment at a near-vertical angle,
    matching Rain100L/Rain100H benchmark characteristics.

    Args:
        n_streaks    : number of rain streaks to render
        length_range : (min, max) streak length in pixels
        thickness    : streak width in pixels (1 = realistic)
        alpha        : opacity of rain layer (0.4–0.6 is natural)
        angle_deg    : mean angle from vertical (negative = leans right)

    Returns:
        uint8 BGR image with synthetic rain overlaid.
    """
    rain_layer = np.zeros_like(img, dtype=np.float32)
    h, w       = img.shape[:2]
    rng        = np.random.default_rng(42)    # deterministic for reproducibility

    angle_rad   = np.deg2rad(angle_deg)
    dx_per_unit = np.sin(angle_rad)           # x displacement per unit length
    dy_per_unit = np.cos(angle_rad)           # y displacement per unit length

    for _ in range(n_streaks):
        length    = rng.integers(*length_range)
        brightness = rng.uniform(200, 255)     # near-white streaks

        # Random start point anywhere in the image (allow some off-screen)
        x0 = rng.integers(-20, w + 20)
        y0 = rng.integers(-20, h + 20)
        x1 = int(x0 + dx_per_unit * length)
        y1 = int(y0 + dy_per_unit * length)

        color = (int(brightness),) * 3         # white streak
        cv2.line(rain_layer, (x0, y0), (x1, y1), color, thickness=thickness,
                 lineType=cv2.LINE_AA)

    # Blend: original + alpha * rain_layer
    img_f  = img.astype(np.float32)
    result = img_f + alpha * rain_layer
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_haze(
    img:      np.ndarray,
    beta:     float = 1.5,
    atm_light: float = 0.85,
) -> np.ndarray:
    """
    Simulate atmospheric haze using the physics-based scattering equation:
        I(x) = J(x) · t(x) + A · (1 − t(x))
    where:
        J(x)  = clean scene radiance (the original image)
        t(x)  = transmission map (depth-dependent, estimated from distance)
        A     = atmospheric light (here: constant bright sky value)
        beta  = scattering coefficient (higher = denser haze)

    Transmission is derived as t = exp(−beta × depth), where depth is
    estimated from the luminance (dark = far away) for a plausible spatial
    haze distribution without needing an actual depth map.

    Args:
        img        : uint8 BGR clean image
        beta       : scattering strength (1.5 = moderate haze)
        atm_light  : atmospheric light intensity in [0, 1] (0.85 = bright sky)
    """
    img_f     = img.astype(np.float32) / 255.0
    h, w      = img_f.shape[:2]

    # Estimate depth from luminance (bright = near, dark = far)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Invert: dark regions treated as distant (more haze)
    depth     = 1.0 - gray
    # Blur depth for spatial coherence
    depth     = cv2.GaussianBlur(depth, (0, 0), sigmaX=w // 8)
    depth     = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # Transmission: exp(−beta × depth), deeper = less transmission
    t         = np.exp(-beta * depth).astype(np.float32)  # [H×W]
    t         = np.clip(t, 0.1, 1.0)[:, :, np.newaxis]    # [H×W×1]

    # Atmospheric scattering
    A         = atm_light
    hazy      = img_f * t + A * (1.0 - t)
    hazy      = np.clip(hazy, 0.0, 1.0)
    return (hazy * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# DEGRADATION CONFIG
# ─────────────────────────────────────────────────────────────

DEGRADATIONS = {
    "blur": {
        "fn":          apply_motion_blur,
        "description": "Motion blur (kernel=25px)",
        "stage":       "imaging",
    },
    "jpeg": {
        "fn":          apply_jpeg_compression,
        "description": "JPEG compression (quality=10)",
        "stage":       "compression",
    },
    "noise": {
        "fn":          apply_gaussian_noise,
        "description": "Gaussian noise (σ=30)",
        "stage":       "imaging",
    },
    "lowlight": {
        "fn":          apply_lowlight,
        "description": "Low-light / underexposure (γ=3.5)",
        "stage":       "scene",
    },
    "haze": {
        "fn":          apply_haze,
        "description": "Atmospheric haze (β=1.5, physics-based scattering) [C1/C3]",
        "stage":       "scene",
    },
    "rain": {
        "fn":          apply_rain,
        "description": "Synthetic rain streaks (800 streaks, −15° angle, additive) [Phase 3]",
        "stage":       "scene",
    },
    "mixed": {
        "fn":          apply_mixed,
        "description": "Mixed: JPEG (q=15) + Gaussian noise (σ=20)",
        "stage":       "compression+imaging",
    },
}


# ─────────────────────────────────────────────────────────────
# SOURCE IMAGE COLLECTION
# ─────────────────────────────────────────────────────────────

def _find_source_images(source_dir: str, n: int) -> list:
    """
    Find up to n clean source images from source_dir.
    Falls back to images already in the project's test sets if dir is empty.
    """
    images = []

    if os.path.isdir(source_dir):
        for f in sorted(Path(source_dir).rglob("*")):
            if f.suffix.lower() in EXTS:
                images.append(str(f))
            if len(images) >= n:
                break

    # Fallback: use SwinIR testset images
    if not images:
        fallback_dirs = [
            "models/SwinIR/testsets/Set5",
            "models/SwinIR/testsets/Set14",
            "models/SwinIR/testsets/test",
            "models/SwinIR/testsets/RealSRSet+5images",
        ]
        for d in fallback_dirs:
            if os.path.isdir(d):
                for f in sorted(Path(d).rglob("*")):
                    if f.suffix.lower() in EXTS:
                        images.append(str(f))
                    if len(images) >= n:
                        break
            if len(images) >= n:
                break

    if not images:
        print("[Generator] ERROR: No source images found.")
        print("  Place clean PNG/JPG images in a directory and use --source_dir")

    return images[:n]


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def generate_set(
    degradation_name: str,
    source_images:    list,
    output_root:      str = BENCHMARK_ROOT,
) -> int:
    """
    Generate one benchmark set (degraded + reference pairs).

    Returns number of pairs successfully written.
    """
    if degradation_name not in DEGRADATIONS:
        print(f"[Generator] Unknown degradation: {degradation_name}")
        return 0

    cfg         = DEGRADATIONS[degradation_name]
    degrade_fn  = cfg["fn"]
    set_name    = f"{degradation_name}_test"

    deg_dir = os.path.join(output_root, set_name, "degraded")
    ref_dir = os.path.join(output_root, set_name, "reference")
    os.makedirs(deg_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    count = 0
    for src_path in source_images:
        img = cv2.imread(src_path)
        if img is None:
            print(f"  [Generator] Cannot load {src_path} — skipping")
            continue

        # Resize to a consistent size for fair comparison (512×512 max)
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        stem    = Path(src_path).stem
        out_ext = ".png"  # always save as PNG for lossless reference

        # Save clean reference
        ref_out = os.path.join(ref_dir, stem + out_ext)
        cv2.imwrite(ref_out, img)

        # Apply degradation and save
        try:
            degraded = degrade_fn(img)
        except Exception as e:
            print(f"  [Generator] Degradation failed for {stem}: {e}")
            continue

        deg_out = os.path.join(deg_dir, stem + out_ext)
        cv2.imwrite(deg_out, degraded)

        count += 1
        print(f"  [{degradation_name:10}] {stem:<30}  ✓")

    return count


def generate_all(
    types:      list,
    source_dir: str,
    n:          int,
    output_root: str = BENCHMARK_ROOT,
) -> None:
    """Generate all requested degradation sets."""
    print("\n" + "=" * 60)
    print("  MAIR+ Benchmark Dataset Generator")
    print("=" * 60)

    source_images = _find_source_images(source_dir, n)
    if not source_images:
        return

    print(f"  Source images : {len(source_images)}")
    print(f"  Types         : {types}")
    print(f"  Output root   : {output_root}\n")

    total = 0
    for dtype in types:
        print(f"\n─── Generating: {dtype}_test")
        count = generate_set(dtype, source_images, output_root)
        print(f"    → {count} pairs written")
        total += count

    print(f"\n{'=' * 60}")
    print(f"  Done. {total} total image pairs generated.")
    print(f"  Run benchmark:")
    print(f"    python evaluation/benchmark.py --all")
    print(f"    python evaluation/benchmark.py --list_sets")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ — Generate benchmark datasets with synthetic degradations"
    )
    parser.add_argument(
        "--source_dir", type=str, default="",
        help="Directory of clean source images (auto-discovers if omitted)"
    )
    parser.add_argument(
        "--types", nargs="+",
        default=list(DEGRADATIONS.keys()),
        choices=list(DEGRADATIONS.keys()),
        help="Degradation types to generate (default: all)"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Max number of images per set (default: 10)"
    )
    parser.add_argument(
        "--output_root", type=str, default=BENCHMARK_ROOT,
        help=f"Root output directory (default: {BENCHMARK_ROOT})"
    )
    args = parser.parse_args()

    generate_all(
        types=args.types,
        source_dir=args.source_dir,
        n=args.n,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
