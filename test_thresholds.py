import os
import glob
import cv2
import pandas as pd
import numpy as np

DATASETS = {
    "GoPro_subset": "blur",
    "BSD68_subset": "noise",
    "LIVE1_subset": "jpeg",
    "LOL_subset": "lowlight",
    "RESIDE_subset": "haze"
}

def load_dataset(n_per_class=100):
    images = []
    for subset, true_label in DATASETS.items():
        deg_dir = os.path.join("datasets", "academic_subsets", subset, "degraded")
        found = glob.glob(os.path.join(deg_dir, "*.png"))[:n_per_class]
        for f in found:
            images.append({"deg_path": f, "true_label": true_label})
    return images

def test_detector():
    images = load_dataset()
    for img_info in images:
        deg_path = img_info["deg_path"]
        true_label = img_info["true_label"]
        if true_label not in ["noise", "jpeg"]: continue
        
        image = cv2.imread(deg_path)
        if image is None: continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # test noise
        if true_label == "noise":
            median = cv2.medianBlur(gray, 5)
            residual = gray.astype(np.float32) - median.astype(np.float32)
            noise_std = float(np.std(residual))
            print(f"Noise img std: {noise_std:.2f}")
            break

    for img_info in images:
        deg_path = img_info["deg_path"]
        true_label = img_info["true_label"]
        if true_label == "jpeg":
            image = cv2.imread(deg_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            gray_f = gray.astype(np.float32)
            col_indices = list(range(7, w - 1, 8))
            next_cols = [c + 1 for c in col_indices]
            boundary_diffs = np.abs(gray_f[:, col_indices] - gray_f[:, next_cols])
            avg_boundary = float(np.mean(boundary_diffs))
            all_h_diffs = np.abs(np.diff(gray_f, axis=1))
            avg_all = float(np.mean(all_h_diffs)) + 1e-6
            ratio = avg_boundary / avg_all
            print(f"JPEG img ratio: {ratio:.3f}")
            break

if __name__ == "__main__":
    test_detector()
