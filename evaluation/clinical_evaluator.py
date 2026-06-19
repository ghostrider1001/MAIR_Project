import cv2
import numpy as np
import time

try:
    from skimage.measure import shannon_entropy
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import torch
    import torchvision.transforms.functional as TF
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False

def compute_brisque(image_path: str) -> float:
    """
    Compute the BRISQUE score using the `piq` library.
    BRISQUE measures the spatial naturalness of the image.
    LOWER score means HIGHER quality (less distortion/haze/noise).
    """
    if not HAS_PIQ:
        return -1.0

    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0)  # [1, C, H, W] in [0, 1]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_t = img_t.to(device)
        
        # BRISQUE calculation
        score = piq.brisque(img_t)
        return float(score.item())
    except Exception as e:
        print(f"[ClinicalEval] Failed to compute BRISQUE: {e}")
        return -1.0

def compute_niqe(image_path: str) -> float:
    """
    Compute the NIQE score using the `piq` library.
    LOWER score means HIGHER quality (less distortion/haze/noise).
    """
    if not HAS_PIQ:
        return -1.0
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_t = img_t.to(device)
        score = piq.niqe(img_t)
        return float(score.item())
    except Exception as e:
        print(f"[ClinicalEval] Failed to compute NIQE: {e}")
        return -1.0

def compute_piqe(image_path: str) -> float:
    """
    Compute the PIQE score using the `piq` library.
    LOWER score means HIGHER quality.
    """
    if not HAS_PIQ:
        return -1.0
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_t = img_t.to(device)
        score = piq.piqe(img_t)
        return float(score.item())
    except Exception as e:
        # piqe might not be in older piq versions
        return -1.0

def compute_edge_sharpness(image_path: str) -> float:
    """
    Compute edge sharpness using the Variance of Laplacian method.
    In laparoscopy, haze/smoke blurs edges. Removing it should increase sharpness.
    HIGHER score means SHARPER edges (more detail recovered).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
        
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)

def compute_information_entropy(image_path: str) -> float:
    """
    Compute Shannon Entropy of the image.
    Entropy represents the amount of information/detail contained in the image.
    Smoke flattens the image (low entropy). Clearing it reveals textures (high entropy).
    HIGHER score means MORE information/detail.
    """
    if not HAS_SKIMAGE:
        return -1.0
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
        
    entropy = shannon_entropy(img)
    return float(entropy)

def evaluate_clinical_quality(image_path: str) -> dict:
    """
    Run the full suite of No-Reference Image Quality Assessment (NR-IQA)
    metrics designed for clinical and unpaired evaluation.
    """
    brisque_score = compute_brisque(image_path)
    niqe_score = compute_niqe(image_path)
    piqe_score = compute_piqe(image_path)
    sharpness_score = compute_edge_sharpness(image_path)
    entropy_score = compute_information_entropy(image_path)
    
    return {
        "brisque": round(brisque_score, 2) if brisque_score != -1.0 else -1.0,
        "niqe": round(niqe_score, 2) if niqe_score != -1.0 else -1.0,
        "piqe": round(piqe_score, 2) if piqe_score != -1.0 else -1.0,
        "sharpness": round(sharpness_score, 2) if sharpness_score != -1.0 else -1.0,
        "entropy": round(entropy_score, 2) if entropy_score != -1.0 else -1.0,
    }

def evaluate_clinical_composite(original_path: str, restored_path: str) -> float:
    """
    Compute a single QualityGate score (0.0 to 1.0) using clinical metrics.
    Replaces SSIM in the scheduler.
    
    Logic:
    - Base score is derived from BRISQUE (normalized). 
    - If sharpness improves over the original, apply a multiplier bonus.
    - If sharpness degrades significantly, apply a penalty.
    """
    orig = evaluate_clinical_quality(original_path)
    rest = evaluate_clinical_quality(restored_path)
    
    # 1. Base Score from BRISQUE.
    # BRISQUE typically ranges from 0 (perfect) to 100 (awful).
    rb = rest.get('brisque', -1.0)
    if rb == -1.0:
        return 0.5  # Neutral fallback if piq fails
        
    # Invert BRISQUE so higher is better, normalize to roughly [0, 1]
    # E.g., BRISQUE 40 -> 0.60, BRISQUE 80 -> 0.20
    base_score = max(0.0, 1.0 - (rb / 100.0))
    
    # 2. Sharpness Modifier
    os_val = orig.get('sharpness', -1.0)
    rs_val = rest.get('sharpness', -1.0)
    
    modifier = 1.0
    if os_val > 0 and rs_val > 0:
        ratio = rs_val / os_val
        if ratio > 1.2:
            modifier = 1.15  # 15% bonus for significant sharpening
        elif ratio > 1.0:
            modifier = 1.05  # 5% bonus for mild sharpening
        elif ratio < 0.5:
            modifier = 0.85  # 15% penalty for severe blurring
        elif ratio < 0.8:
            modifier = 0.95  # 5% penalty for mild blurring
            
    final_score = base_score * modifier
    return min(1.0, max(0.0, final_score))

def print_clinical_report(original_path: str, restored_path: str):
    """
    Generate and print a comparison report between the original hazy image
    and the restored AI output.
    """
    print("\n==============================================")
    print("      CLINICAL EVALUATION EXPERT (NR-IQA)")
    print("==============================================")
    
    start_time = time.time()
    
    print("[Clinical Eval] Analyzing original image...")
    orig_metrics = evaluate_clinical_quality(original_path)
    
    print("[Clinical Eval] Analyzing restored image...")
    rest_metrics = evaluate_clinical_quality(restored_path)
    
    print("\n  Metric                  | Original | Restored | Trend / Meaning")
    print("  -------------------------------------------------------------------------")
    
    # BRISQUE
    ob = orig_metrics['brisque']
    rb = rest_metrics['brisque']
    if ob != -1 and rb != -1:
        diff = ob - rb
        trend = "✅ IMPROVED" if diff > 0 else "❌ REGRESSED"
        print(f"  BRISQUE (Distortion)    | {ob:8.2f} | {rb:8.2f} | {trend} (Lower is better)")
    else:
        print("  BRISQUE (Distortion)    |   N/A    |   N/A    | (Requires 'piq')")

    # NIQE
    on = orig_metrics['niqe']
    rn = rest_metrics['niqe']
    if on != -1 and rn != -1:
        diff = on - rn
        trend = "✅ IMPROVED" if diff > 0 else "❌ REGRESSED"
        print(f"  NIQE (Naturalness)      | {on:8.2f} | {rn:8.2f} | {trend} (Lower is better)")

    # PIQE
    op = orig_metrics['piqe']
    rp = rest_metrics['piqe']
    if op != -1 and rp != -1:
        diff = op - rp
        trend = "✅ IMPROVED" if diff > 0 else "❌ REGRESSED"
        print(f"  PIQE (Perception)       | {op:8.2f} | {rp:8.2f} | {trend} (Lower is better)")
        
    # Edge Sharpness
    os = orig_metrics['sharpness']
    rs = rest_metrics['sharpness']
    if os != -1 and rs != -1:
        diff = rs - os
        trend = "✅ SHARPER " if diff > 0 else "❌ BLURRIER"
        print(f"  Sharpness (Var of Lap)| {os:8.2f} | {rs:8.2f} | {trend} (Higher is better)")
        
    # Information Entropy
    oe = orig_metrics['entropy']
    re = rest_metrics['entropy']
    if oe != -1 and re != -1:
        diff = re - oe
        trend = "✅ MORE DET." if diff > 0 else "❌ FLATTER "
        print(f"  Entropy (Information)   | {oe:8.2f} | {re:8.2f} | {trend} (Higher is better)")
    else:
        print("  Entropy (Information)   |   N/A    |   N/A    | (Requires 'skimage')")
        
    print(f"\n  [Time Taken: {round(time.time() - start_time, 2)}s]")
    print("==============================================\n")
