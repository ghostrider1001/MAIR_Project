"""
generate_professor_report_assets.py
------------------------------------
Generates:
  1. Dark-themed PSNR bar chart (matching MAIR_DeSmoke_LAP_Evaluation style)
  2. 10 before/after comparison images with PSNR + SSIM scores burned in
"""

import os, sys, random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate_synthetic_desmoke import apply_surgical_smoke
from experts.dehaze_expert import restore_dcp

CSV_PATH     = os.path.join(PROJECT_ROOT, "results", "synthetic_desmoke_dcp_only.csv")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "DeSmoke-LAP dataset", "Dataset")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
ASSETS_DIR   = os.path.join(RESULTS_DIR,  "dashboard_assets")

# ── Dark palette matching MAIR_DeSmoke_LAP_Evaluation ────────────────────────
BG_COLOR    = "#0d1117"
GRID_COLOR  = "#1f2937"
HAZY_COLOR  = "#ef4444"   # red-pink  (hazy bars)
REST_COLOR  = "#10b981"   # neon-green (restored bars)
TEXT_COLOR  = "#e2e8f0"
LABEL_COLOR = "#6ee7b7"
TICK_COLOR  = "#9ca3af"
ACCENT_BLUE = "#3b82f6"

# ─────────────────────────────────────────────────────────────────────────────
def generate_graphs():
    print("Generating statistical graphs (dark theme)...")
    df = pd.read_csv(CSV_PATH)

    case_stats = df.groupby('case')[['psnr_hazy', 'psnr_restored']].mean().reset_index()
    case_stats['delta'] = case_stats['psnr_restored'] - case_stats['psnr_hazy']
    case_stats = case_stats.sort_values('delta', ascending=True)

    # ── Chart 1: Horizontal ranking chart (like MAIR_DeSmoke_LAP slide 4) ────
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars = ax.barh(case_stats['case'], case_stats['delta'],
                   color=[HAZY_COLOR if d < 0 else REST_COLOR for d in case_stats['delta']],
                   edgecolor='none', height=0.55)

    for bar, delta in zip(bars, case_stats['delta']):
        xpos = bar.get_width() + 0.1 if delta >= 0 else bar.get_width() - 0.1
        ha   = 'left'                if delta >= 0 else 'right'
        sign = '+' if delta >= 0 else ''
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.2f} dB", va='center', ha=ha,
                color=TEXT_COLOR, fontsize=9, fontweight='bold')

    ax.axvline(0, color=TICK_COLOR, linewidth=0.8, linestyle='--')
    ax.set_xlabel("PSNR Improvement  (positive = better)", color=TICK_COLOR, fontsize=11)
    ax.set_title("Improvement Ranking by Case",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "chart_ranking.png"), dpi=200, facecolor=BG_COLOR)
    plt.close(fig)
    print("  Saved: chart_ranking.png")

    # ── Chart 2: Per-case grouped bar (like MAIR_DeSmoke_LAP slide 5) ─────────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    cases  = case_stats['case'].values
    x      = np.arange(len(cases))
    w      = 0.38

    b1 = ax.bar(x - w/2, case_stats['psnr_hazy'],     w, color=HAZY_COLOR, label="Hazy (before)", edgecolor='none')
    b2 = ax.bar(x + w/2, case_stats['psnr_restored'], w, color=REST_COLOR,  label="Restored (after)", edgecolor='none')

    for bar, hazy, rest in zip(b2, case_stats['psnr_hazy'], case_stats['psnr_restored']):
        delta = rest - hazy
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"Δ+{delta:.1f}", ha='center', va='bottom',
                color=LABEL_COLOR, fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha='right', color=TICK_COLOR, fontsize=9)
    ax.tick_params(axis='y', colors=TICK_COLOR)
    ax.set_ylabel("PSNR (dB)  —  higher is better", color=TICK_COLOR, fontsize=11)
    ax.set_title("PSNR: Hazy vs. Restored — Per TLH Case",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.legend(facecolor=GRID_COLOR, labelcolor=TEXT_COLOR, edgecolor=ACCENT_BLUE, fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "chart_per_case.png"), dpi=200, facecolor=BG_COLOR)
    plt.close(fig)
    print("  Saved: chart_per_case.png")


# ─────────────────────────────────────────────────────────────────────────────
def _burn_label(img, line1, line2, position="top"):
    """
    Burn two lines of text onto `img` (OpenCV BGR numpy array).
    position: 'top' -> top-left,  'bottom' -> bottom-left
    """
    h, w = img.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = max(0.40, w / 900)
    thickness = 1
    pad       = 8

    (_, lh1), bl1 = cv2.getTextSize(line1, font, scale, thickness)
    (_, lh2), _   = cv2.getTextSize(line2, font, scale, thickness)
    box_h  = lh1 + lh2 + pad * 3
    box_w  = w

    if position == "top":
        y0 = 0
    else:
        y0 = h - box_h

    # semi-transparent dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (box_w, y0 + box_h), (13, 17, 23), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # line 1 (white) + line 2 (neon green)
    y1 = y0 + pad + lh1
    y2 = y1 + lh2 + pad
    cv2.putText(img, line1, (pad, y1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, line2, (pad, y2), font, scale, (16, 185, 129),  thickness, cv2.LINE_AA)


def generate_comparison_images():
    print("Generating 10 comparison images with PSNR/SSIM scores...")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    df   = pd.read_csv(CSV_PATH)
    temp = os.path.join(RESULTS_DIR, "report_temp_hazy.jpg")

    # Pick the single best-PSNR-delta image from each case for maximum visual impact
    best_rows = (df.groupby('case')
                   .apply(lambda g: g.loc[g['psnr_delta'].idxmax()])
                   .reset_index(drop=True))

    if len(best_rows) > 10:
        best_rows = best_rows.head(10)

    TARGET_W, TARGET_H = 360, 270   # per sub-image

    for idx, row in best_rows.iterrows():
        case  = row['case']
        fname = row['file']
        psnr_h  = float(row['psnr_hazy'])
        psnr_r  = float(row['psnr_restored'])
        ssim_h  = float(row['ssim_hazy'])
        ssim_r  = float(row['ssim_restored'])
        psnr_d  = psnr_r - psnr_h
        pair_no = best_rows.index.get_loc(idx) + 1
        print(f"  [{pair_no}/10] {case} / {fname}  PSNR Δ={psnr_d:+.2f} dB")

        clear_path = os.path.join(DATASET_ROOT, case, "clear", fname)
        clear_img  = cv2.imread(clear_path)
        if clear_img is None:
            print(f"    WARNING: cannot read {clear_path}, skipping")
            continue

        clear_img = cv2.resize(clear_img, (TARGET_W, TARGET_H))
        hazy_img  = apply_surgical_smoke(clear_img)
        cv2.imwrite(temp, hazy_img)

        restored_path = restore_dcp(temp)
        restored_img  = cv2.imread(restored_path)
        restored_img  = cv2.resize(restored_img, (TARGET_W, TARGET_H))

        # ── Burn scores onto images ──────────────────────────────────────────
        _burn_label(hazy_img,
                    f"HAZY INPUT  —  {case}",
                    f"PSNR: {psnr_h:.2f} dB   SSIM: {ssim_h:.3f}",
                    position="bottom")

        _burn_label(restored_img,
                    f"RESTORED (MAIR+ DCP)",
                    f"PSNR: {psnr_r:.2f} dB   SSIM: {ssim_r:.3f}   Δ PSNR: {psnr_d:+.2f} dB",
                    position="bottom")

        # Neon-coloured border: red for hazy, green for restored
        cv2.rectangle(hazy_img,     (0, 0), (TARGET_W-1, TARGET_H-1), (40, 40, 200), 3)
        cv2.rectangle(restored_img, (0, 0), (TARGET_W-1, TARGET_H-1), (16, 185, 129), 3)

        # Stitch side by side with a thin separator
        sep    = np.zeros((TARGET_H, 4, 3), dtype=np.uint8) + 13
        pair   = np.hstack((hazy_img, sep, restored_img))

        out = os.path.join(ASSETS_DIR, f"pair_{pair_no}.png")
        cv2.imwrite(out, pair)

    if os.path.exists(temp):
        os.remove(temp)
    print(f"  All comparison images saved to: {ASSETS_DIR}")


if __name__ == "__main__":
    generate_graphs()
    generate_comparison_images()
    print("\nAll report assets generated successfully!")
