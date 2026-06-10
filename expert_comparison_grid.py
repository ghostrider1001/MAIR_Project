"""
MAIR+ v2 — Expert-by-Expert Comparison Grid
=============================================
Takes ONE input image and passes it through ALL available expert models
individually, showing the result of each expert on the same input.

This makes it easy to see exactly what each expert contributes and compare
them side-by-side in a single frame.

Output:  outputs/expert_comparison/expert_comparison_grid.png

Usage:
    python expert_comparison_grid.py
    python expert_comparison_grid.py --input path/to/your/image.jpg
    python expert_comparison_grid.py --input baby.png --degradation noise

Degradation options:
    none       -- use the image as-is (clean)
    noise      -- add Gaussian noise σ=25 (good for showing denoising)
    blur       -- motion blur kernel=21px
    jpeg       -- JPEG quality=15 artifacts
    lowlight   -- γ=3.5 dark image
    haze       -- atmospheric haze
    rain       -- 600 rain streaks
    mixed      -- JPEG + noise
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── make project root importable ──────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ─── METRICS ───────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def compute_metrics(ref, test):
    if not HAS_SKIMAGE or ref is None:
        return None, None
    try:
        if ref.shape != test.shape:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
        rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        tg = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        s = round(float(ssim_fn(rg, tg)), 4)
        p = round(float(psnr_fn(rg, tg)), 2)
        return s, p
    except Exception:
        return None, None


# ─── DEGRADATION FUNCTIONS ─────────────────────────────────────
def degrade_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def degrade_blur(img):
    k = 21
    kernel = np.zeros((k, k)); kernel[k//2, :] = 1.0/k
    return cv2.filter2D(img, -1, kernel)

def degrade_jpeg(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def degrade_lowlight(img):
    table = np.array([((i/255.0)**3.5)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

def degrade_haze(img):
    img_f = img.astype(np.float32)/255.0
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    depth = cv2.GaussianBlur(1.0-gray, (0,0), sigmaX=img.shape[1]//8)
    depth = (depth-depth.min())/(depth.max()-depth.min()+1e-6)
    t     = np.clip(np.exp(-1.5*depth)[:,:,None], 0.1, 1.0)
    hazy  = np.clip(img_f*t + 0.85*(1.0-t), 0, 1)
    return (hazy*255).astype(np.uint8)

def degrade_rain(img):
    layer = np.zeros_like(img, dtype=np.float32)
    h, w  = img.shape[:2]; rng = np.random.default_rng(42)
    for _ in range(600):
        length = rng.integers(15, 55); br = rng.uniform(200, 255)
        x0, y0 = rng.integers(-10, w+10), rng.integers(-10, h+10)
        ang = np.deg2rad(-15)
        x1  = int(x0+np.sin(ang)*length); y1 = int(y0+np.cos(ang)*length)
        cv2.line(layer, (x0,y0), (x1,y1), (int(br),)*3, 1, cv2.LINE_AA)
    return np.clip(img.astype(np.float32)+0.5*layer, 0, 255).astype(np.uint8)

def degrade_mixed(img: np.ndarray) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        out = img.copy()
    noise = np.random.normal(0, 18, out.shape).astype(np.float32)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

DEGRADERS = {
    "none":      lambda img: img.copy(),
    "noise":     degrade_noise,
    "blur":      degrade_blur,
    "jpeg":      degrade_jpeg,
    "lowlight":  degrade_lowlight,
    "haze":      degrade_haze,
    "rain":      degrade_rain,
    "mixed":     degrade_mixed,
}


# ─── EXPERT DEFINITIONS ────────────────────────────────────────
def _load_experts():
    """Load all expert functions, gracefully skip those with missing deps."""
    from experts.denoise_expert      import restore_denoise
    from experts.nafnet_lite_expert  import restore_nafnet
    from experts.lowlight_expert     import restore_lowlight
    from experts.zero_dce_expert     import restore_zero_dce
    from experts.fastjpeg_expert     import restore_fast_jpeg
    from experts.unsharp_deblur_expert import restore_unsharp_deblur
    from experts.wiener_deblur_expert  import restore as restore_wiener
    from experts.dehaze_expert         import restore_dcp
    from experts.deraining_expert      import restore as restore_derain
    from experts.zero_dce_expert       import restore_zero_dce

    experts = [
        # ── DENOISING ──────────────────────────────────────
        {
            "name":     "NLM Denoiser",
            "key":      "nlm_denoise",
            "stage":    "Imaging",
            "color":    (79, 142, 255),    # blue
            "tag":      "C12",
            "fn":       restore_denoise,
        },
        {
            "name":     "NAFNet-Lite",
            "key":      "nafnet_denoise",
            "stage":    "Imaging",
            "color":    (100, 180, 255),
            "tag":      "New",
            "fn":       restore_nafnet,
        },
        # ── DEBLURRING ─────────────────────────────────────
        {
            "name":     "Wiener Deblur",
            "key":      "wiener_deblur",
            "stage":    "Imaging",
            "color":    (150, 100, 255),
            "tag":      "C13",
            "fn":       restore_wiener,
        },
        {
            "name":     "Unsharp Deblur",
            "key":      "unsharp_deblur",
            "stage":    "Imaging",
            "color":    (180, 130, 255),
            "tag":      "C12",
            "fn":       restore_unsharp_deblur,
        },
        # ── LOW-LIGHT ──────────────────────────────────────
        {
            "name":     "Zero-DCE\nLow-Light",
            "key":      "zero_dce",
            "stage":    "Scene",
            "color":    (50, 220, 160),    # teal-green
            "tag":      "New",
            "fn":       restore_zero_dce,
        },
        {
            "name":     "CLAHE\nLow-Light",
            "key":      "clahe_lowlight",
            "stage":    "Scene",
            "color":    (80, 200, 130),
            "tag":      "Baseline",
            "fn":       restore_lowlight,
        },
        # ── JPEG ───────────────────────────────────────────
        {
            "name":     "NLM JPEG\nFallback",
            "key":      "nlm_jpeg",
            "stage":    "Compression",
            "color":    (255, 180, 50),    # orange
            "tag":      "C12",
            "fn":       restore_fast_jpeg,
        },
        # ── HAZE ───────────────────────────────────────────
        {
            "name":     "DCP Dehaze",
            "key":      "dcp_dehaze",
            "stage":    "Scene",
            "color":    (80, 210, 200),    # cyan
            "tag":      "C1",
            "fn":       restore_dcp,
        },
        # ── RAIN ───────────────────────────────────────────
        {
            "name":     "Freq-Domain\nDerain",
            "key":      "freq_derain",
            "stage":    "Scene",
            "color":    (120, 200, 255),   # sky blue
            "tag":      "New",
            "fn":       restore_derain,
        },
    ]

    # Try to also add Restormer deblur if available
    try:
        from experts.deblur_expert import restore_deblur
        experts.append({
            "name":     "Restormer\nDeblur",
            "key":      "restormer_deblur",
            "stage":    "Imaging",
            "color":    (220, 80, 120),
            "tag":      "GPU",
            "fn":       restore_deblur,
        })
    except Exception:
        pass

    return experts


# ─── PANEL DRAWING ─────────────────────────────────────────────

PANEL_W   = 340
PANEL_H   = 280
LABEL_H   = 85
BORDER    = 3
FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_SM   = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

STAGE_COLORS = {
    "Imaging":     (79,  142, 255),
    "Scene":       (50,  220, 160),
    "Compression": (255, 180, 50),
}


def bgr(rgb_tuple):
    """Convert (R,G,B) → (B,G,R) for OpenCV."""
    r, g, b = rgb_tuple
    return (b, g, r)


def draw_panel(img, name, stage, tag, ssim, psnr, elapsed,
               border_color=(60, 80, 100), is_original=False):
    """Draw a single labelled image panel."""
    total_h = PANEL_H + LABEL_H
    panel   = np.zeros((total_h, PANEL_W, 3), dtype=np.uint8)

    # Background gradient
    for row in range(total_h):
        t = row / total_h
        r = int(18 + t*8); g = int(22 + t*8); b = int(36 + t*8)
        panel[row, :] = (b, g, r)

    # Image (resize + paste)
    img_area_w = PANEL_W - 2*BORDER
    img_area_h = PANEL_H - 2*BORDER
    img_r = cv2.resize(img, (img_area_w, img_area_h), interpolation=cv2.INTER_AREA)
    panel[BORDER:PANEL_H-BORDER, BORDER:PANEL_W-BORDER] = img_r

    # Coloured border
    bc = bgr(border_color)
    cv2.rectangle(panel, (0,0), (PANEL_W-1, PANEL_H-1), bc, BORDER)

    # ── Label area ────────────────────────────────────────
    label_bg_start = PANEL_H
    # Stage stripe (top of label)
    stage_c = bgr(STAGE_COLORS.get(stage, (80,80,80)))
    cv2.rectangle(panel, (0, label_bg_start), (PANEL_W, label_bg_start+3), stage_c, -1)

    # Name (can be multiline with \n)
    lines = name.split("\n")
    y = label_bg_start + 22
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, FONT, 0.60, 1)
        tx = max(8, (PANEL_W - tw)//2)
        cv2.putText(panel, line, (tx, y), FONT, 0.60, (235,235,245), 1, cv2.LINE_AA)
        y += 20

    # Stage + tag pill
    stage_str = stage
    (sw, sh), _ = cv2.getTextSize(stage_str, FONT_SM, 0.38, 1)
    sx = 10
    sy = label_bg_start + 56
    # Stage chip
    cv2.rectangle(panel, (sx-2, sy-10), (sx+sw+6, sy+3), stage_c, -1)
    cv2.putText(panel, stage_str, (sx, sy), FONT_SM, 0.38, (10,10,10), 1, cv2.LINE_AA)
    # Tag chip
    if tag:
        tx2 = sx + sw + 14
        tc  = bgr(border_color)
        (tw2, _), _ = cv2.getTextSize(tag, FONT_SM, 0.38, 1)
        cv2.rectangle(panel, (tx2-2, sy-10), (tx2+tw2+6, sy+3), tc, -1)
        cv2.putText(panel, tag, (tx2, sy), FONT_SM, 0.38, (240,240,240), 1, cv2.LINE_AA)

    # Metrics row
    if ssim is not None and psnr is not None:
        ssim_str = f"SSIM {ssim:.4f}"
        psnr_str = f"PSNR {psnr:.1f}dB"

        # Colour by quality
        if ssim >= 0.80:      mc = (60, 220, 80)    # green
        elif ssim >= 0.65:    mc = (60, 210, 220)   # cyan
        elif ssim >= 0.50:    mc = (80, 180, 255)   # blue
        else:                 mc = (80, 80, 220)    # purple

        my = label_bg_start + 73
        cv2.putText(panel, ssim_str, (10, my), FONT_SM, 0.40, mc, 1, cv2.LINE_AA)
        cv2.putText(panel, psnr_str, (PANEL_W//2+5, my), FONT_SM, 0.40, mc, 1, cv2.LINE_AA)

    if elapsed is not None:
        ts = f"{elapsed:.2f}s"
        (tw, _), _ = cv2.getTextSize(ts, FONT_SM, 0.36, 1)
        cv2.putText(panel, ts, (PANEL_W-tw-8, label_bg_start+57),
                    FONT_SM, 0.36, (100,110,140), 1, cv2.LINE_AA)

    if is_original:
        cv2.putText(panel, "REFERENCE", (8, label_bg_start+75),
                    FONT_SM, 0.38, (255,220,80), 1, cv2.LINE_AA)

    return panel


def draw_header(width, degradation_name, source_name, n_experts):
    """Top title banner."""
    hh     = 100
    header = np.zeros((hh, width, 3), dtype=np.uint8)

    # Gradient bg
    for row in range(hh):
        t = row/hh
        b = int(12+t*5); g = int(18+t*5); r = int(30+t*5)
        header[row, :] = (b, g, r)

    # Bottom glow line
    header[hh-2:hh, :] = (80, 120, 200)

    title = "MAIR+ v2  |  Expert-by-Expert Comparison"
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.90, 2)
    cv2.putText(header, title, ((width-tw)//2, 38),
                FONT, 0.90, (230, 210, 255), 2, cv2.LINE_AA)

    sub = f"Source: {source_name}   |   Degradation: {degradation_name.upper()}   |   {n_experts} Experts"
    (sw, _), _ = cv2.getTextSize(sub, FONT_SM, 0.50, 1)
    cv2.putText(header, sub, ((width-sw)//2, 62),
                FONT_SM, 0.50, (140, 150, 180), 1, cv2.LINE_AA)

    ts = datetime.now().strftime("Generated: %Y-%m-%d  %H:%M")
    cv2.putText(header, ts, (16, 88), FONT_SM, 0.40, (80, 90, 120), 1, cv2.LINE_AA)

    return header


def draw_footer(width, results):
    """Bottom summary bar with all metrics."""
    fh     = 130
    footer = np.zeros((fh, width, 3), dtype=np.uint8)
    footer[:] = (14, 20, 32)
    footer[0:2, :] = (80, 120, 200)

    # Header row
    cv2.putText(footer, "BENCHMARK SUMMARY", (20, 22),
                FONT, 0.55, (200, 180, 255), 1, cv2.LINE_AA)

    # Column headers
    cols = [20, 200, 340, 460, 570, 680]
    hdrs = ["Expert", "Stage", "SSIM", "PSNR (dB)", "Time (s)", "SSIM vs Input"]
    for c, h in zip(cols, hdrs):
        cv2.putText(footer, h, (c, 42), FONT_SM, 0.38, (100, 130, 180), 1, cv2.LINE_AA)
    cv2.line(footer, (20, 46), (width-20, 46), (40, 50, 80), 1)

    y = 62
    # Sort by SSIM descending
    sorted_r = sorted(results, key=lambda r: r.get("ssim") or 0, reverse=True)
    for r in sorted_r[:8]:   # max 8 rows in footer
        name   = r["name"].replace("\n"," ")
        stage  = r["stage"]
        ssim   = r.get("ssim")
        psnr   = r.get("psnr")
        t_s    = r.get("elapsed")
        gain   = r.get("ssim_gain")
        stage_c = bgr(STAGE_COLORS.get(stage, (100,100,100)))

        cv2.putText(footer, name[:18],  (cols[0], y), FONT_SM, 0.40, (200,200,220), 1, cv2.LINE_AA)
        cv2.putText(footer, stage[:10], (cols[1], y), FONT_SM, 0.40, stage_c,      1, cv2.LINE_AA)
        if ssim is not None:
            mc = (60,220,80) if ssim>=0.75 else (80,200,255) if ssim>=0.55 else (80,80,220)
            cv2.putText(footer, f"{ssim:.4f}", (cols[2], y), FONT_SM, 0.40, mc, 1, cv2.LINE_AA)
        if psnr is not None:
            cv2.putText(footer, f"{psnr:.1f}", (cols[3], y), FONT_SM, 0.40, (180,180,220), 1, cv2.LINE_AA)
        if t_s is not None:
            cv2.putText(footer, f"{t_s:.2f}", (cols[4], y), FONT_SM, 0.40, (140,140,180), 1, cv2.LINE_AA)
        if gain is not None:
            gc = (60,220,80) if gain>0 else (80,80,220)
            cv2.putText(footer, f"{gain:+.4f}", (cols[5], y), FONT_SM, 0.40, gc, 1, cv2.LINE_AA)
        y += 14

    # Best expert callout
    if sorted_r and sorted_r[0].get("ssim"):
        best = sorted_r[0]
        txt  = f"Best Expert: {best['name'].replace(chr(10),' ')}  SSIM={best['ssim']:.4f}"
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.48, 1)
        cv2.putText(footer, txt, (width-tw-20, fh-10),
                    FONT, 0.48, (60,220,160), 1, cv2.LINE_AA)

    return footer


# ─── MAIN ──────────────────────────────────────────────────────

def run_comparison(source_path, degradation="noise", output_dir="outputs/expert_comparison"):
    """Main function: degrade input → run all experts → build grid."""

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print("\n" + "="*68)
    print("  MAIR+ v2 — Expert-by-Expert Comparison Generator")
    print("="*68)

    # ── Load + resize source ───────────────────────────────
    ref_img = cv2.imread(source_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot load: {source_path}")
    h, w = ref_img.shape[:2]
    if max(h, w) > 480:
        scale = 480 / max(h, w)
        ref_img = cv2.resize(ref_img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    print(f"  Source: {Path(source_path).name}  {ref_img.shape[1]}x{ref_img.shape[0]}")

    # ── Apply degradation ──────────────────────────────────
    if degradation not in DEGRADERS:
        print(f"  Unknown degradation '{degradation}', using 'noise'")
        degradation = "noise"
    _degraded = DEGRADERS[degradation](ref_img.copy())
    # Ensure deg_img is always a valid ndarray (never None)
    deg_img: np.ndarray = _degraded if _degraded is not None else ref_img.copy()

    # Save degraded to temp file (experts need a file path)
    deg_path = os.path.join(tmp_dir, "degraded_input.png")
    cv2.imwrite(deg_path, deg_img)
    print(f"  Degradation: {degradation}")

    # Compute degraded metrics (vs clean ref)
    deg_ssim, deg_psnr = compute_metrics(ref_img, deg_img)
    print(f"  Degraded baseline: SSIM={deg_ssim}  PSNR={deg_psnr}")

    # ── Load experts ───────────────────────────────────────
    experts = _load_experts()
    print(f"  Loaded {len(experts)} experts\n")

    # ── Run each expert ────────────────────────────────────
    results = []
    output_images = []   # (img, name, stage, tag, ssim, psnr, elapsed)

    for exp in experts:
        name  = exp["name"]
        fn    = exp["fn"]
        stage = exp["stage"]
        tag   = exp["tag"]
        color = exp["color"]

        flat_name = name.replace("\n","_")
        print(f"  [{flat_name:25}] running ...", end="", flush=True)
        t0 = time.time()

        try:
            out_path = fn(deg_path)
            elapsed  = round(time.time()-t0, 2)

            if out_path and os.path.exists(out_path):
                out_img = cv2.imread(out_path)
                if out_img is not None:
                    s, p = compute_metrics(ref_img, out_img)
                    sg   = round(s - deg_ssim, 4) if s and deg_ssim else None
                    print(f" SSIM={s:.4f}  PSNR={p:.1f}dB  ({elapsed}s)")
                    output_images.append((out_img, name, stage, tag, s, p, elapsed, color))
                    results.append({
                        "name": name, "stage": stage, "tag": tag,
                        "ssim": s, "psnr": p, "elapsed": elapsed,
                        "ssim_gain": sg,
                    })
                else:
                    print(" [output unreadable]")
                    output_images.append((deg_img, name, stage, tag, None, None, elapsed, color))
            else:
                elapsed = round(time.time() - t0, 2)
                print(f" [no output produced] ({elapsed}s)")
                output_images.append((deg_img, name, stage, tag, None, None, elapsed, color))

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f" ERROR: {e}")
            output_images.append((deg_img, name, stage, tag, None, None, elapsed, color))
            results.append({
                "name": name, "stage": stage, "tag": tag,
                "ssim": None, "psnr": None, "elapsed": elapsed, "ssim_gain": None,
            })

    # ── Build panel grid ───────────────────────────────────
    # First panel: original clean reference
    ref_panel = draw_panel(ref_img, "Original\n(Clean)", "—", "REF",
                            None, None, None,
                            border_color=(80, 200, 80), is_original=True)
    # Second panel: degraded input
    deg_panel = draw_panel(deg_img, f"Degraded\n({degradation})", "—", "INPUT",
                            deg_ssim, deg_psnr, None,
                            border_color=(220, 80, 80))

    panels = [ref_panel, deg_panel]
    for (out_img, name, stage, tag, s, p, elapsed, color) in output_images:
        panel = draw_panel(out_img, name, stage, tag, s, p, elapsed,
                            border_color=color)
        panels.append(panel)

    # Arrange in rows
    COLS     = 5
    n_panels = len(panels)
    n_rows   = (n_panels + COLS - 1) // COLS

    # Pad to full grid
    empty = np.zeros((PANEL_H+LABEL_H, PANEL_W, 3), dtype=np.uint8)
    while len(panels) % COLS != 0:
        panels.append(empty)

    rows = []
    for r in range(n_rows):
        row_panels = panels[r*COLS:(r+1)*COLS]
        row_img    = np.hstack(row_panels)
        rows.append(row_img)

    grid_body = np.vstack(rows)
    grid_w    = grid_body.shape[1]

    header = draw_header(grid_w, degradation, Path(source_path).name, len(experts))
    footer = draw_footer(grid_w, results)

    final = np.vstack([header, grid_body, footer])

    # ── Save ───────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"expert_comparison_{degradation}_{ts}.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, final)

    # Also save a "latest" copy for easy access
    latest_path = os.path.join(output_dir, "expert_comparison_LATEST.png")
    cv2.imwrite(latest_path, final)

    print("\n" + "="*68)
    print(f"  COMPARISON GRID SAVED")
    print(f"  File : {out_path}")
    print(f"  Also : {latest_path}")
    print(f"  Size : {final.shape[1]}×{final.shape[0]} px")
    print("="*68)

    # Print rankings
    if results:
        print("\n  EXPERT RANKING BY SSIM:")
        sorted_r = sorted(results, key=lambda r: r.get("ssim") or 0, reverse=True)
        for i, r in enumerate(sorted_r, 1):
            s = r.get("ssim"); p = r.get("psnr"); g = r.get("ssim_gain")
            marker = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
            name_f = r["name"].replace("\n"," ")
            print(f"  {marker} #{i:2}  {name_f:<25}  SSIM={s or 'N/A'}  Gain={g or 'N/A'}")

    return out_path


# ─── CLI ───────────────────────────────────────────────────────

def find_source():
    """Auto-find a clean source image."""
    candidates = [
        "datasets/benchmark/noise_test/reference/butterfly.png",
        "datasets/benchmark/noise_test/reference/baby.png",
        "datasets/benchmark/noise_test/reference/bird.png",
        "datasets/benchmark/blur_test/reference/baby.png",
    ]
    for p in candidates:
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            return full

    import glob
    refs = glob.glob(os.path.join(ROOT, "datasets/benchmark/*/reference/*.png"))
    if refs:
        return refs[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="MAIR+ v2 — Generate expert-by-expert comparison grid"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to clean input image (auto-found if omitted)")
    parser.add_argument("--degradation", type=str, default="noise",
                        choices=list(DEGRADERS.keys()),
                        help="Degradation to apply (default: noise)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/expert_comparison",
                        help="Output directory")
    args = parser.parse_args()

    src = args.input or find_source()
    if not src:
        print("[ERROR] No source image found. Use --input path/to/image.jpg")
        return
    if not os.path.exists(src):
        # Try relative to project root
        src2 = os.path.join(ROOT, src)
        if os.path.exists(src2):
            src = src2
        else:
            print(f"[ERROR] File not found: {src}")
            return

    run_comparison(src, args.degradation, args.output_dir)


if __name__ == "__main__":
    main()
