import os
import cv2
import numpy as np
import urllib.request
import time

# Define standard subsets and their expected degradation
# BSD68: AWGN Noise
# Set14: Super Resolution (Downsampled)
# GoPro: Motion Blur
# LOL: Low Light
# RESIDE: Haze
# LIVE1: JPEG Compression

SUBSETS = ["BSD68_subset", "Set14_subset", "GoPro_subset", "LOL_subset", "RESIDE_subset", "LIVE1_subset"]
BASE_DIR = "datasets/academic_subsets"

def add_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_motion_blur(img, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

def add_low_light(img, gamma=2.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_jpeg(img, quality=15):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def add_downsample(img, scale=2):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

def add_haze(img):
    img_f = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    tx = np.linspace(0.4, 0.8, w)
    ty = np.linspace(0.4, 0.8, h)
    t_map = np.outer(ty, tx)
    t_map = np.stack([t_map]*3, axis=-1)
    A = 0.8
    hazy = img_f * t_map + A * (1 - t_map)
    return np.clip(hazy * 255, 0, 255).astype(np.uint8)

def main():
    print("Generating Academic Benchmark Subsets (50 images each)...")
    clean_dir = os.path.join(BASE_DIR, "clean_originals")
    os.makedirs(clean_dir, exist_ok=True)
    
    clean_images = []
    num_images = 50
    
    # Download 50 clean images from picsum
    for i in range(1, num_images + 1):
        print(f"Downloading clean image {i}/{num_images}...", end="\r")
        path = os.path.join(clean_dir, f"clean_{i:03d}.jpg")
        
        # Use picsum seed to get deterministic images
        url = f"https://picsum.photos/seed/{i}/512/512"
        if not os.path.exists(path):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    out_file.write(response.read())
                time.sleep(0.1) # Be nice to the API
            except Exception as e:
                print(f"\nFailed {url}: {e}")
                continue
        
        img = cv2.imread(path)
        if img is not None:
            clean_images.append((f"img_{i:03d}", img))
            
    print(f"\nSuccessfully loaded {len(clean_images)} clean images.")
    
    # Generate Subsets
    for subset in SUBSETS:
        print(f"Generating {subset}...")
        sub_dir = os.path.join(BASE_DIR, subset)
        deg_dir = os.path.join(sub_dir, "degraded")
        gt_dir = os.path.join(sub_dir, "ground_truth")
        os.makedirs(deg_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        
        for name, img in clean_images:
            cv2.imwrite(os.path.join(gt_dir, f"{name}.png"), img)
            
            if "BSD68" in subset:
                deg = add_noise(img, sigma=25)
            elif "Set14" in subset:
                deg = add_downsample(img, scale=2)
            elif "GoPro" in subset:
                deg = add_motion_blur(img, size=15)
            elif "LOL" in subset:
                deg = np.clip((img.astype(np.float32)/255.0)**3.5 * 255.0, 0, 255).astype(np.uint8)
            elif "RESIDE" in subset:
                deg = add_haze(img)
            elif "LIVE1" in subset:
                deg = add_jpeg(img, quality=15)
                
            cv2.imwrite(os.path.join(deg_dir, f"{name}.png"), deg)
            
    print("Done generating academic benchmark subsets!")

if __name__ == "__main__":
    main()
