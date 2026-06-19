r"""
download_official_datasets.py
------------------------------
Downloads REAL academic benchmark datasets to D:\NIt\MAIR_Project\datasets\academic_subsets\

Sources (all verified working as of 2026):
  BSD68   -> GitHub: clausmichele/CBSD68-dataset  (official PNG mirror, 68 images)
  Set14   -> GitHub: jbhuang0604/SelfExSR         (original Set14 PNGs)
  GoPro   -> GitHub release (Seungjun Nah's official)
  LOL     -> Baiduyun / Google Drive (manual fallback)

Usage:
    .\venv\Scripts\python download_official_datasets.py
    .\venv\Scripts\python download_official_datasets.py --subset BSD68 Set14
    .\venv\Scripts\python download_official_datasets.py --skip-existing
"""

import os
import sys
import glob
import shutil
import zipfile
import tarfile
import argparse
import urllib.request
import subprocess
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_BASE = r"D:\NIt\MAIR_Project\datasets\academic_subsets"
MAX_IMAGES  = 68   # cap per dataset to keep disk usage small

# ── Download sources ──────────────────────────────────────────────────────────
SOURCES = {
    "BSD68": {
        "method": "git_sparse",
        "repo":   "https://github.com/cszn/KAIR.git",
        "subdir": "testsets/CBSD68",
        "desc":   "CBSD68 — 68 color images (official BSD denoising benchmark)",
    },
    "Set14": {
        "method": "git_sparse",
        "repo":   "https://github.com/cszn/KAIR.git",
        "subdir": "testsets/Set14",
        "desc":   "Set14 — 14 classic SR/restoration benchmark images",
    },
    "LIVE1": {
        "method": "git_sparse",
        "repo":   "https://github.com/cszn/KAIR.git",
        "subdir": "testsets/LIVE1",
        "desc":   "LIVE1 — 29 images for JPEG/compression quality research",
    },
    "Kodak": {
        "method": "git_sparse",
        "repo":   "https://github.com/cszn/KAIR.git",
        "subdir": "testsets/kodak24",
        "desc":   "Kodak Lossless True Color — 24 PNG images",
    },
}

# ─────────────────────────────────────────────────────────────────────────────

def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def ok(msg):  print(f"  [OK] {msg}")
def err(msg): print(f"  [FAIL] {msg}")
def info(msg):print(f"  {msg}")


def git_available():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# ── Method: git sparse-checkout (no full clone needed) ───────────────────────
def download_git_sparse(name, cfg, dest_dir, skip_existing):
    """Clone only the target subdirectory using sparse checkout."""
    if not git_available():
        err("git not found on PATH. Install Git from https://git-scm.com/")
        return 0

    clone_dir = os.path.join(dest_dir, f"_clone_{name}")
    img_dir   = os.path.join(dest_dir, "degraded")
    gt_dir    = os.path.join(dest_dir, "ground_truth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    existing = glob.glob(os.path.join(img_dir, "*.png")) + \
               glob.glob(os.path.join(img_dir, "*.jpg"))
    if skip_existing and len(existing) >= 5:
        ok(f"{name}: {len(existing)} images already present (--skip-existing)")
        return len(existing)

    info(f"Sparse-cloning {cfg['repo']} ...")
    info(f"  Subfolder: {cfg['subdir']}")

    try:
        # Init empty repo
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        os.makedirs(clone_dir)

        cmds = [
            ["git", "init"],
            ["git", "remote", "add", "origin", cfg["repo"]],
            ["git", "sparse-checkout", "init", "--cone"],
            ["git", "sparse-checkout", "set", cfg["subdir"]],
            ["git", "pull", "origin", "main", "--depth=1"],
        ]
        # Some repos use 'master' branch
        for cmd in cmds[:-1]:
            subprocess.run(cmd, cwd=clone_dir, check=True,
                           capture_output=True)

        # Try main then master
        for branch in ["main", "master"]:
            r = subprocess.run(
                ["git", "pull", "origin", branch, "--depth=1"],
                cwd=clone_dir, capture_output=True
            )
            if r.returncode == 0:
                info(f"  Branch: {branch}")
                break
        else:
            err(f"Could not pull from {cfg['repo']}")
            return 0

        # Collect images from the sparse subdir
        src_dir = os.path.join(clone_dir, cfg["subdir"])
        if not os.path.exists(src_dir):
            err(f"Subdir not found after clone: {src_dir}")
            return 0

        images = (glob.glob(os.path.join(src_dir, "*.png")) +
                  glob.glob(os.path.join(src_dir, "*.jpg")) +
                  glob.glob(os.path.join(src_dir, "*.bmp")))

        if not images:
            err(f"No images found in {src_dir}")
            return 0

        images = images[:MAX_IMAGES]
        for src in images:
            fname = os.path.basename(src)
            shutil.copy2(src, os.path.join(img_dir, fname))
            shutil.copy2(src, os.path.join(gt_dir,  fname))

        ok(f"{name}: copied {len(images)} images")
        return len(images)

    except Exception as e:
        err(f"{name} git sparse failed: {e}")
        return 0
    finally:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)


# ── Method: HTTP direct download (individual files) ───────────────────────────
def download_http_direct(name, cfg, dest_dir, skip_existing):
    """Download individual image files by URL."""
    img_dir = os.path.join(dest_dir, "degraded")
    gt_dir  = os.path.join(dest_dir, "ground_truth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    existing = glob.glob(os.path.join(img_dir, "*.png")) + \
               glob.glob(os.path.join(img_dir, "*.jpg"))
    if skip_existing and len(existing) >= 5:
        ok(f"{name}: {len(existing)} images already present (--skip-existing)")
        return len(existing)

    base = cfg["base_url"].rstrip("/")
    fnames = cfg["filenames"][:MAX_IMAGES]
    count = 0

    for fname in fnames:
        url  = f"{base}/{fname}"
        dest = os.path.join(img_dir, fname)
        gt   = os.path.join(gt_dir,  fname)

        try:
            print(f"  Downloading {fname} ...", end="\r")
            urllib.request.urlretrieve(url, dest)
            shutil.copy2(dest, gt)
            count += 1
        except Exception as e:
            print()
            err(f"  {fname}: {e}")

    print()
    ok(f"{name}: downloaded {count}/{len(fnames)} images")
    return count


# ── Method: HTTP ZIP download ─────────────────────────────────────────────────
def download_http_zip(name, cfg, dest_dir, skip_existing):
    """Download a ZIP archive and extract image files."""
    img_dir = os.path.join(dest_dir, "degraded")
    gt_dir  = os.path.join(dest_dir, "ground_truth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    existing = glob.glob(os.path.join(img_dir, "*.png")) + \
               glob.glob(os.path.join(img_dir, "*.jpg")) + \
               glob.glob(os.path.join(img_dir, "*.bmp"))
    if skip_existing and len(existing) >= 5:
        ok(f"{name}: {len(existing)} images already present (--skip-existing)")
        return len(existing)

    zip_path = os.path.join(dest_dir, f"_{name}.zip")
    glob_pat = cfg.get("image_glob", "*.png")

    for url in cfg["urls"]:
        try:
            info(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            err(f"  URL failed: {e}")
    else:
        return 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = [m for m in z.namelist()
                       if any(m.lower().endswith(ext)
                              for ext in [".png", ".jpg", ".bmp"])]
            members = members[:MAX_IMAGES]
            for m in members:
                data = z.read(m)
                fname = os.path.basename(m)
                for d in [img_dir, gt_dir]:
                    with open(os.path.join(d, fname), "wb") as f:
                        f.write(data)
        ok(f"{name}: extracted {len(members)} images")
        return len(members)
    except Exception as e:
        err(f"Zip extraction failed: {e}")
        return 0
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


# ── Manual instructions for datasets that need authentication ─────────────────
MANUAL_INSTRUCTIONS = {
    "GoPro": """
  GoPro (GOPRO_Large) — motion deblurring benchmark
  ─────────────────────────────────────────────────
  Official source: https://seungjunnah.github.io/Datasets/gopro
  1. Download GOPRO_Large.zip (~6.7 GB) from the Google Drive link on the page
  2. Extract and copy any 30 blur/*.png images to:
       D:\\NIt\\MAIR_Project\\datasets\\academic_subsets\\GoPro_subset\\degraded\\
  3. Copy the matching sharp/*.png to:
       D:\\NIt\\MAIR_Project\\datasets\\academic_subsets\\GoPro_subset\\ground_truth\\
""",
    "LOL": """
  LOL (Low-Light) — low-light image enhancement benchmark
  ────────────────────────────────────────────────────────
  Official source: https://daooshee.github.io/BMVC2018website/
  1. Download eval15.zip from the page
  2. Extract low/ images to:
       D:\\NIt\\MAIR_Project\\datasets\\academic_subsets\\LOL_subset\\degraded\\
  3. Extract high/ images to:
       D:\\NIt\\MAIR_Project\\datasets\\academic_subsets\\LOL_subset\\ground_truth\\
""",
    "RESIDE": """
  RESIDE-Standard (SOTS outdoor test set) — dehazing benchmark
  ─────────────────────────────────────────────────────────────
  Official source: https://sites.google.com/view/reside-dehaze-datasets
  1. Download RESIDE-Standard.zip from Dropbox/Baidu link
  2. Extract SOTS/outdoor/hazy/ → degraded/
  3. Extract SOTS/outdoor/gt/   → ground_truth/
""",
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download official academic benchmark datasets to D:\\NIt\\MAIR_Project\\datasets\\")
    parser.add_argument("--subset",       nargs="*", default=None,
                        help="Specific datasets to download (default: all auto-downloadable)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip datasets that already have images")
    args = parser.parse_args()

    banner("MAIR+ OFFICIAL DATASET DOWNLOADER")
    print(f"  Output: {OUTPUT_BASE}")
    print(f"  Max images per dataset: {MAX_IMAGES}")

    auto_sources = {k: v for k, v in SOURCES.items()
                    if args.subset is None or k in args.subset}
    manual       = {k: v for k, v in MANUAL_INSTRUCTIONS.items()
                    if args.subset is None or k in args.subset}

    results = {}

    for name, cfg in auto_sources.items():
        print(f"\n{'─'*60}")
        print(f"  Downloading {name} — {cfg['desc']}")
        print(f"{'─'*60}")

        dest = os.path.join(OUTPUT_BASE, f"{name}_subset")
        os.makedirs(dest, exist_ok=True)

        method = cfg["method"]
        if method == "git_sparse":
            n = download_git_sparse(name, cfg, dest, args.skip_existing)
        elif method == "http_direct":
            n = download_http_direct(name, cfg, dest, args.skip_existing)
        elif method == "http_zip":
            n = download_http_zip(name, cfg, dest, args.skip_existing)
        else:
            err(f"Unknown method: {method}")
            n = 0

        results[name] = n

    # Manual download instructions
    if manual:
        print(f"\n{'─'*60}")
        print("  MANUAL DOWNLOAD REQUIRED (authentication/large files):")
        print(f"{'─'*60}")
        for name, instructions in manual.items():
            print(f"\n  [{name}]")
            print(instructions)

    # Summary
    banner("DOWNLOAD SUMMARY")
    total = 0
    for name, n in results.items():
        status = f"{n} images" if n > 0 else "FAILED"
        print(f"  {name:<15} {status}")
        total += n

    print(f"\n  Total images downloaded: {total}")
    print(f"  Location: {OUTPUT_BASE}")

    if total > 0:
        print(f"\n  Run academic evaluation:")
        print(f"    .\\venv\\Scripts\\python run_academic_eval.py")


if __name__ == "__main__":
    main()
