import os
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm

def generate_perlin_noise_2d(shape, res):
    """Generate 2D Perlin noise for realistic smoke texture."""
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def add_synthetic_smoke(img, intensity=0.6):
    """
    Apply an Atmospheric Scattering Model with Perlin noise transmission map.
    I(x) = J(x)t(x) + A(1 - t(x))
    """
    h, w = img.shape[:2]
    # Ensure divisible by 4 for Perlin resolution
    ph, pw = (h // 4) * 4, (w // 4) * 4
    img_resized = cv2.resize(img, (pw, ph))
    
    # Generate Perlin noise base (values roughly -1 to 1)
    noise = generate_perlin_noise_2d((ph, pw), (4, 4))
    
    # Normalize noise to 0..1 to act as a transmission map
    noise_norm = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    
    # Randomize scattering parameters
    A = np.random.uniform(200, 255) # Atmospheric light (white/gray smoke)
    # Transmission map t(x): 1 = clear, 0 = opaque smoke. 
    # Use exponential to make it look like depth.
    # intensity controls how thick the smoke is.
    beta = np.random.uniform(1.5, 3.0) * intensity
    t = np.exp(-beta * noise_norm)
    t = np.clip(t, 0.1, 1.0)
    
    # Expand t to 3 channels
    t_3d = np.dstack([t, t, t])
    
    # Apply scattering model
    hazy = img_resized.astype(np.float32) * t_3d + A * (1 - t_3d)
    hazy = np.clip(hazy, 0, 255).astype(np.uint8)
    
    return cv2.resize(hazy, (w, h))

def generate_dataset():
    # Source clear images
    clear_src_dir = "datasets/PG002/clear_imgs"
    if not os.path.exists(clear_src_dir):
        print(f"Error: Could not find clear image source at {clear_src_dir}")
        return

    out_clear_dir = "datasets/synthetic_smoke/clear"
    out_hazy_dir = "datasets/synthetic_smoke/hazy"
    
    os.makedirs(out_clear_dir, exist_ok=True)
    os.makedirs(out_hazy_dir, exist_ok=True)

    clear_images = glob.glob(os.path.join(clear_src_dir, "*.png"))
    
    # Limit to 50 images for a fast benchmark
    clear_images = clear_images[:50]
    
    print(f"Generating synthetic smoke for {len(clear_images)} images...")
    
    for img_path in tqdm(clear_images):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Ensure we save exactly the same original to out_clear
        cv2.imwrite(os.path.join(out_clear_dir, filename), img)
        
        # Generate hazy version
        hazy_img = add_synthetic_smoke(img, intensity=0.7)
        cv2.imwrite(os.path.join(out_hazy_dir, filename), hazy_img)

    print("\nSynthetic Dataset Generation Complete!")
    print(f"Ground Truth : {out_clear_dir}")
    print(f"Degraded     : {out_hazy_dir}")

if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset()
