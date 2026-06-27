import os
import glob
import cv2

def add_border(img, color, thickness=6):
    """Adds a colored border to an image."""
    return cv2.copyMakeBorder(img, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=color)

def generate_pairs():
    percentages = [10, 20, 30, 40]
    
    print("Generating side-by-side pairs for LaTeX...")
    for pct in percentages:
        # Grab candidates for this percentage
        deg_candidates = glob.glob(f"datasets/synthetic_smoke_sweep/deg_{pct}_*.png")
        if not deg_candidates:
            print(f"[{pct}%] No degraded images found in datasets/synthetic_smoke_sweep.")
            continue
            
        found = False
        for deg_path in deg_candidates:
            # Extract the original filename base (e.g. TLH_10_002_0256)
            base_name = os.path.basename(deg_path).replace(f"deg_{pct}_", "").replace(".png", "")
            
            # Find the restored output image
            rest_search = glob.glob(f"outputs/*/deg_{pct}_{base_name}*.png")
            if not rest_search:
                continue
                
            rest_path = rest_search[0]
            
            # Read images
            deg_img = cv2.imread(deg_path)
            rest_img = cv2.imread(rest_path)
            
            if deg_img is None or rest_img is None:
                continue
                
            # Ensure dimensions match
            if deg_img.shape != rest_img.shape:
                rest_img = cv2.resize(rest_img, (deg_img.shape[1], deg_img.shape[0]))
                
            # Add colored borders (BGR format: Red for Before, Green for After)
            red_border = [0, 0, 255]
            green_border = [0, 255, 0]
            
            deg_bordered = add_border(deg_img, red_border)
            rest_bordered = add_border(rest_img, green_border)
            
            # Stitch them side-by-side
            stitched_img = cv2.hconcat([deg_bordered, rest_bordered])
            
            # Save the final image to the root folder for easy upload
            out_name = f"smoke_sweep_{pct}.png"
            cv2.imwrite(out_name, stitched_img)
            
            print(f"[{pct}%] Successfully generated {out_name} (using base image: {base_name})")
            found = True
            break  # We only need one pair per percentage
            
        if not found:
            print(f"[{pct}%] Could not find a matching output file for any degraded image.")

if __name__ == '__main__':
    generate_pairs()
