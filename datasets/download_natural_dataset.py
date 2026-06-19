import os
import urllib.request

def main():
    urls = [
        ("lenna.png", "https://raw.githubusercontent.com/masteringopencv/code/master/Chapter04_Images/images/lena.png"),
        ("baboon.png", "https://raw.githubusercontent.com/masteringopencv/code/master/Chapter04_Images/images/baboon.png"),
        ("fruits.png", "https://raw.githubusercontent.com/masteringopencv/code/master/Chapter04_Images/images/fruits.png"),
    ]

    out_dir = r"datasets\natural_clean"
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading natural test images...")
    for name, url in urls:
        out_path = os.path.join(out_dir, name)
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"  [OK] Saved {name}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {name}: {e}")

if __name__ == "__main__":
    main()
