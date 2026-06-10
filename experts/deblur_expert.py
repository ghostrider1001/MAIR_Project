import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.restormer_arch import Restormer


# ─────────────────────────────────────────────────────────────
# MODEL CONFIG  (matches Motion_Deblurring/Options/Deblurring_Restormer.yml)
# ─────────────────────────────────────────────────────────────
RESTORMER_CONFIG = {
    "inp_channels":         3,
    "out_channels":         3,
    "dim":                  48,
    "num_blocks":           [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads":                [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias":                 False,
    "LayerNorm_type":       "WithBias",
    "dual_pixel_task":      False,
}

WEIGHTS_PATH = (
    "models/Restormer/Motion_Deblurring/"
    "pretrained_models/motion_deblurring.pth"
)

# Restormer requires input dimensions to be multiples of 8
PAD_FACTOR = 8

# Module-level model cache (loaded once, reused on subsequent calls)
_model_cache = None
_device_cache = None


def _load_model():
    """Load Restormer model from weights. Cached after first call."""
    global _model_cache, _device_cache

    if _model_cache is not None:
        return _model_cache, _device_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Deblur Expert] Device : {device}")

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Restormer weights not found: {WEIGHTS_PATH}\n"
            "Expected: models/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth"
        )

    print("[Deblur Expert] Loading Restormer architecture...")
    model = Restormer(**RESTORMER_CONFIG)

    print("[Deblur Expert] Loading pretrained weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["params"])
    model.to(device)
    model.eval()

    _model_cache = model
    _device_cache = device
    print("[Deblur Expert] Model loaded successfully.")
    return model, device


def restore_deblur(input_path):
    """
    Restore a blurry image using Restormer (motion deblurring).
    Works on both CPU and GPU — auto-detects available hardware.

    Returns:
        Path to deblurred output image, or None on failure.
    """
    print("\n===================================")
    print("      DEBLUR EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[Deblur Expert] Input Path : {input_path}")

    # ── Check einops ──────────────────────────────────────────
    try:
        import einops  # noqa: F401
    except ImportError:
        print("[Deblur Expert] ERROR: 'einops' not installed.")
        print("[Deblur Expert] Run: python install_phase2_deps.py")
        return None

    # ── Load image ────────────────────────────────────────────
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"[Deblur Expert] ERROR: Cannot load image: {input_path}")
        return None

    # Convert BGR → RGB, normalize to [0, 1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f   = np.float32(img_rgb) / 255.0           # H×W×C in [0,1]

    # ── Load model ────────────────────────────────────────────
    try:
        model, device = _load_model()
    except FileNotFoundError as e:
        print(f"[Deblur Expert] {e}")
        return None
    except Exception as e:
        print(f"[Deblur Expert] Model loading failed: {e}")
        return None

    # ── Prepare tensor ────────────────────────────────────────
    # H×W×C → C×H×W → 1×C×H×W
    inp = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = inp.shape[2], inp.shape[3]
    H = ((h + PAD_FACTOR) // PAD_FACTOR) * PAD_FACTOR
    W = ((w + PAD_FACTOR) // PAD_FACTOR) * PAD_FACTOR
    pad_h = H - h if h % PAD_FACTOR != 0 else 0
    pad_w = W - w if w % PAD_FACTOR != 0 else 0
    inp   = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")

    print(f"[Deblur Expert] Input shape  : {h}×{w}")
    if pad_h or pad_w:
        print(f"[Deblur Expert] Padded to    : {H}×{W}  (multiple of {PAD_FACTOR})")
    print("[Deblur Expert] Running Restormer inference...")

    # ── Inference ─────────────────────────────────────────────
    with torch.no_grad():
        restored = model(inp)

    # Unpad, clamp, convert back to numpy
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1)
    restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    # Convert to uint8 BGR for cv2 saving
    restored_uint8 = (restored * 255.0).round().astype(np.uint8)
    restored_bgr   = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)

    # ── Save output ───────────────────────────────────────────
    output_dir  = os.path.join("outputs", "deblurred")
    os.makedirs(output_dir, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_deblurred.png")
    cv2.imwrite(output_path, restored_bgr)

    elapsed = round(time.time() - start_time, 2)
    print(f"[Deblur Expert] Output saved     : {output_path}")
    print(f"[Deblur Expert] Processing Time  : {elapsed}s")

    print("\n===================================")
    print("      DEBLUR EXPERT FINISHED")
    print("===================================\n")

    return output_path
