import os

file_path = r"d:\NIt\MAIR_Project\generate_pptx_v3.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Branding
content = content.replace("MAIR+ v2", "MAIR++")
content = content.replace("MAIR+", "MAIR++")
content = content.replace("MAIR++++", "MAIR++") # catch double replacement

# Update ablation metrics (around line 630-700 where ablation is discussed)
content = content.replace("Baseline + 0.293 SSIM", "Overall PSNR Gain: +2.74 dB")
content = content.replace("Baseline + 0.243 SSIM", "Overall SSIM Gain: +0.088")

# Let's save it as generate_pptx_final.py so we don't mess up the original
out_path = r"d:\NIt\MAIR_Project\generate_pptx_final.py"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated script saved as generate_pptx_final.py")
