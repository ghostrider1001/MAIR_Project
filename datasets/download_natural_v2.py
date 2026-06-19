import os
import urllib.request

def download_file(url, dest):
    print(f"Downloading {os.path.basename(dest)}...")
    req = urllib.request.Request(
        url,
        data=None,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    with urllib.request.urlopen(req) as response:
        with open(dest, 'wb') as f:
            f.write(response.read())

def main():
    target_dir = os.path.join('datasets', 'natural_clean')
    os.makedirs(target_dir, exist_ok=True)
    
    # Stable Wikimedia Commons / Academic URLs for standard test images
    images = {
        "lenna.png": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "monalisa.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "parrots.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Parrots_macaws.jpg/400px-Parrots_macaws.jpg"
    }
    
    success_count = 0
    for filename, url in images.items():
        dest = os.path.join(target_dir, filename)
        try:
            download_file(url, dest)
            print(f"[SUCCESS] Saved to {dest}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            
    print(f"\nSuccessfully downloaded {success_count} / {len(images)} images to {target_dir}")

if __name__ == "__main__":
    main()
