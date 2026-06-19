import os
import glob
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

def compute_hash(filepath):
    try:
        with Image.open(filepath) as img:
            # We use dHash (Difference Hash) which compares adjacent pixels.
            # It's highly robust to contrast changes (like haze vs clear).
            hash_size = 8
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = img.getpixel((col, row))
                    pixel_right = img.getpixel((col + 1, row))
                    difference.append(pixel_left > pixel_right)
            
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                    decimal_value = 0
            return filepath, ''.join(hex_string)
    except Exception as e:
        return filepath, None

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(bin(int(s1, 16))[2:].zfill(64), bin(int(s2, 16))[2:].zfill(64)))

def main():
    dataset_root = "datasets"
    
    clear_files = glob.glob(os.path.join(dataset_root, "PG*", "clear_imgs", "*.png"))
    hazy_files = glob.glob(os.path.join(dataset_root, "PG*", "hazy_imgs", "*.png"))
    
    print(f"🔍 Found {len(clear_files)} clear images and {len(hazy_files)} hazy images.")
    print("⏳ Computing perceptual hashes (dHash) for all images to analyze scene content...")
    
    clear_hashes = {}
    hazy_hashes = {}
    
    with ThreadPool(16) as pool:
        for filepath, h in tqdm(pool.imap_unordered(compute_hash, clear_files), total=len(clear_files), desc="Clear Imgs"):
            if h: clear_hashes[filepath] = h
            
        for filepath, h in tqdm(pool.imap_unordered(compute_hash, hazy_files), total=len(hazy_files), desc="Hazy Imgs"):
            if h: hazy_hashes[filepath] = h

    print("\n⚔️ Cross-comparing structural hashes of all clear vs hazy images (approx 3M combinations)...")
    
    # Distance <= 8 means highly similar structural content
    threshold = 8 
    found_pairs = []
    
    for c_path, c_hash in tqdm(clear_hashes.items(), desc="Comparing"):
        for h_path, h_hash in hazy_hashes.items():
            dist = hamming_distance(c_hash, h_hash)
            if dist <= threshold:
                found_pairs.append((dist, c_path, h_path))
                
    found_pairs.sort(key=lambda x: x[0])
    
    print(f"\n✅ Found {len(found_pairs)} potential structural matches (distance <= {threshold})")
    
    out_file = os.path.join(dataset_root, "content_pairs.txt")
    with open(out_file, "w") as f:
        f.write(f"Found {len(found_pairs)} potential structural matches (distance <= {threshold})\n")
        f.write("Format: [Distance] Clear_Image <---> Hazy_Image\n\n")
        
        for dist, c_path, h_path in found_pairs[:50]:
            line = f"[{dist}] {c_path} <---> {h_path}"
            print(line)
            f.write(line + "\n")
            
        if len(found_pairs) > 50:
            msg = f"... and {len(found_pairs)-50} more. See {out_file} for full list."
            print(msg)
            f.write(msg + "\n")

if __name__ == "__main__":
    main()
