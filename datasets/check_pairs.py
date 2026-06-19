import os

datasets_path = r"D:\NIt\MAIR_Project\datasets"
folders = ["PG002", "PG006", "DeSmoke-LAP dataset"]

for root, dirs, files in os.walk(datasets_path):
    if "clear" in root.lower() or "hazy" in root.lower():
        continue
    
    clear_dir = None
    hazy_dir = None
    
    for d in dirs:
        if "clear" in d.lower(): clear_dir = os.path.join(root, d)
        if "hazy" in d.lower(): hazy_dir = os.path.join(root, d)
        
    if clear_dir and hazy_dir:
        clear_files = set(os.listdir(clear_dir))
        hazy_files = set(os.listdir(hazy_dir))
        intersection = clear_files.intersection(hazy_files)
        print(f"Directory: {root}")
        print(f"  Clear files: {len(clear_files)}")
        print(f"  Hazy files: {len(hazy_files)}")
        print(f"  Intersection (pairs): {len(intersection)}")
