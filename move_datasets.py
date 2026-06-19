import os, shutil, glob

def move_files(src, dest):
    os.makedirs(dest, exist_ok=True)
    files = glob.glob(os.path.join(src, '*.*'))
    for f in files:
        shutil.move(f, os.path.join(dest, os.path.basename(f)))
    print(f'Moved {len(files)} images to {dest}')

# 1. Move BSDS100
move_files(r'datasets\BSDS100', r'datasets\academic_subsets\BSD68_subset\ground_truth')

# 2. Move Set14 (they are inside the 'original' subfolder)
move_files(r'datasets\Set14\original', r'datasets\academic_subsets\Set14_subset\ground_truth')

# 3. Move Urban100 (We can use this instead of Kodak!)
move_files(r'datasets\urban100', r'datasets\academic_subsets\Kodak_subset\ground_truth')

print('All datasets moved successfully!')
