import shutil
import os

files = [
    (r"d:\NIt\MAIR_Project\datasets\PG002\hazy_imgs\TLH_2_001_0348.png", "haze_in.png"),
    (r"d:\NIt\MAIR_Project\results\swinir_real_sr_x4\TLH_2_001_0348_SwinIR.png", "haze_out.png"),
    (r"d:\NIt\MAIR_Project\datasets\benchmark\smoke_test\degraded\TLH_2_001_0019.png", "smoke_in.png"),
    (r"d:\NIt\MAIR_Project\outputs\dehazed\TLH_2_001_0019_nafnet_dehazed.png", "smoke_out.png"),
    (r"d:\NIt\MAIR_Project\datasets\benchmark_natural\blur_test\degraded\baby.png", "natural_in.png"),
    (r"d:\NIt\MAIR_Project\outputs\deblurred\baby_deblurred.png", "natural_out.png"),
]

out_dir = r"C:\Users\aswin\.gemini\antigravity-ide\brain\57a29378-d096-4ead-a6f1-e11174c1fc9c"
os.makedirs(out_dir, exist_ok=True)

for src, name in files:
    if os.path.exists(src):
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        print(f"Copied {name}")
    else:
        print(f"Missing {src}")
