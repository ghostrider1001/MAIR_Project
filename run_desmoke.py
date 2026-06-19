import os
import sys
import glob

# Make sure we can import from MAIR+
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler.scheduler import run_three_stage_scheduler

def main():
    # Folder containing the hazy surgical images
    hazy_folder = r"datasets\PG002\hazy_imgs"
    output_folder = r"outputs\desmoke_results\PG002"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PNG images in the hazy folder
    hazy_images = sorted(glob.glob(os.path.join(hazy_folder, "*.png")))
    
    if not hazy_images:
        print(f"❌ No images found in {hazy_folder}")
        return
        
    print(f"🔍 Found {len(hazy_images)} hazy images. We will process the first 10 for your demonstration.")
    
    # We'll just run the first 10 images so it doesn't take hours
    for img_path in hazy_images[:10]:
        filename = os.path.basename(img_path)
        print(f"\n==========================================")
        print(f"🚀 Processing: {filename}")
        print(f"==========================================")
        
        # Run the MAIR+ brain on the image!
        # This will auto-detect the degradation, route it to the best experts, and save it.
        result = run_three_stage_scheduler(
            image_path=img_path,
            verbose=True,
            voting=True,         # Enables your voting ensemble (C12)
            use_memory=True      # Enables your CaseStore brain (C9)
        )
        
        restored_path = result.get("output_path")
        
        # Move the final output to our dedicated desmoke_results folder so it's easy to find
        if restored_path and os.path.exists(restored_path):
            new_path = os.path.join(output_folder, f"restored_{filename}")
            import shutil
            shutil.copy(restored_path, new_path)
            print(f"✅ Successfully saved to: {new_path}")
        else:
            print(f"❌ Failed to restore {filename}")

if __name__ == "__main__":
    main()
