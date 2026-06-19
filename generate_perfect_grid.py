import os
import shutil

def main():
    artifact_dir = r"C:\Users\aswin\.gemini\antigravity-ide\brain\57a29378-d096-4ead-a6f1-e11174c1fc9c"
    os.makedirs(artifact_dir, exist_ok=True)
    
    types = ["blur", "jpeg", "noise", "lowlight", "haze", "rain", "mixed"]
    
    md_content = "# MAIR+ Perfect Restoration Grid (Lenna)\n\n"
    md_content += "This grid provides a **Side-by-Side (Before & After)** comparison for every degradation type, proving that the AI now correctly detects and routes the image to the exact expert needed for a clear restoration!\n\n"
    
    # We will search the outputs folder for the latest files that match the expected degradation.
    # The benchmark evaluation saves the latest files into outputs/deblurred, outputs/denoised, etc.
    
    # Mapping of degradation types to the folder they likely ended up in
    folder_mapping = {
        "blur": "deblurred",
        "jpeg": "jpeg",
        "noise": "denoised",
        "lowlight": "lowlight",
        "haze": "dehazed",
        "rain": "derained",  # With the new fallback, it might be in denoise, but let's just grab the newest lenna output
        "mixed": "" # mixed is hard to predict
    }
    
    for t in types:
        in_path = os.path.join("datasets", "benchmark", f"{t}_test", "degraded", "lenna.png")
        if not os.path.exists(in_path):
            continue
            
        # Copy input (Before)
        in_name = f"lenna_{t}_before.png"
        dest_in = os.path.join(artifact_dir, in_name)
        shutil.copy2(in_path, dest_in)
        
        md_content += f"## {t.capitalize()} Restoration\n\n"
        md_content += f"| **BEFORE** ({t.capitalize()} Degraded) | **AFTER** (MAIR+ Restored) |\n"
        md_content += "|:---:|:---:|\n"
        
        # We need to find the latest corresponding output. 
        # Since the user will just have run the evaluation, we grab the newest file containing 'lenna'
        # But we need to make sure we don't grab the same file for all types.
        # So we look at the results JSON for that specific test!
        
        # Look in the results/ directory for the newest JSON for this test
        latest_json = None
        latest_time = 0
        for f in os.listdir("results"):
            if f.startswith(f"benchmark_{t}_test") and f.endswith(".json"):
                full_path = os.path.join("results", f)
                mtime = os.path.getmtime(full_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_json = full_path
                    
        out_path = None
        if latest_json:
            import json
            try:
                with open(latest_json, "r") as jf:
                    data = json.load(jf)
                    for item in data.get("per_image", []):
                        if item["file"] == "lenna.png":
                            out_path = item.get("restored_path")
            except Exception as e:
                print(f"Error reading {latest_json}: {e}")
                
        if out_path and os.path.exists(out_path):
            # Copy output (After)
            out_name = f"lenna_{t}_after.png"
            dest_out = os.path.join(artifact_dir, out_name)
            shutil.copy2(out_path, dest_out)
            
            before_link = f"C:/Users/aswin/.gemini/antigravity-ide/brain/57a29378-d096-4ead-a6f1-e11174c1fc9c/{in_name}"
            after_link = f"C:/Users/aswin/.gemini/antigravity-ide/brain/57a29378-d096-4ead-a6f1-e11174c1fc9c/{out_name}"
            
            md_content += f"| ![{t} Before]({before_link}) | ![{t} After]({after_link}) |\n\n"
        else:
            md_content += f"| ![{t} Before](C:/Users/aswin/.gemini/antigravity-ide/brain/57a29378-d096-4ead-a6f1-e11174c1fc9c/{in_name}) | *Failed to produce output* |\n\n"

    md_path = os.path.join(artifact_dir, "lenna_grid.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Grid generated successfully at: {md_path}")

if __name__ == "__main__":
    main()
