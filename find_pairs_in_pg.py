import os
import glob

def find_matching_pairs():
    dataset_root = "datasets"
    pg_folders = glob.glob(os.path.join(dataset_root, "PG*"))
    
    total_matches = 0
    
    print("🔍 Scanning PG folders for identical filenames in hazy_imgs and clear_imgs...\n")
    
    for pg_folder in pg_folders:
        hazy_dir = os.path.join(pg_folder, "hazy_imgs")
        clear_dir = os.path.join(pg_folder, "clear_imgs")
        
        if not os.path.exists(hazy_dir) or not os.path.exists(clear_dir):
            continue
            
        hazy_files = set(os.listdir(hazy_dir))
        clear_files = set(os.listdir(clear_dir))
        
        matches = hazy_files.intersection(clear_files)
        
        if matches:
            print(f"✅ Found {len(matches)} matching pairs in {os.path.basename(pg_folder)}:")
            for match in sorted(matches)[:5]:
                print(f"   - {match}")
            if len(matches) > 5:
                print(f"   ... and {len(matches) - 5} more.")
            total_matches += len(matches)
        else:
            print(f"❌ No matching pairs found in {os.path.basename(pg_folder)}")
            
    print(f"\n📊 Total matching pairs across all folders: {total_matches}")
    
    # Save to file
    with open(os.path.join(dataset_root, "pair_analysis.txt"), "w") as f:
        f.write(f"Total matching pairs: {total_matches}\n")

if __name__ == "__main__":
    find_matching_pairs()
