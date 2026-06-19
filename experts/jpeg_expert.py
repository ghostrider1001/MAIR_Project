import os
import time
import glob


def restore_jpeg(input_path):
    """
    Remove JPEG compression artifacts using the SwinIR JPEG-CAR model.
    Model: 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth (already downloaded).

    Returns:
        Path to restored output image, or None on failure.
    """
    print("\n===================================")
    print("       JPEG EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()
    print(f"[JPEG Expert] Input Path : {input_path}")

    model_path = (
        "models/SwinIr/model_zoo/"
        "006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth"
    )
    import shutil
    import tempfile
    import uuid

    # Create a temporary unique folder for just this one image
    # so SwinIR doesn't process the entire dataset directory!
    tmp_folder = os.path.join("outputs", "swinir_tmp_jpeg", str(uuid.uuid4()))
    os.makedirs(tmp_folder, exist_ok=True)
    
    # Copy the single input image there
    tmp_input_path = os.path.join(tmp_folder, os.path.basename(input_path))
    shutil.copy(input_path, tmp_input_path)

    command = (
        f"python models/SwinIr/main_test_swinir.py "
        f"--task jpeg_car --jpeg 10 "
        f"--model_path \"{model_path}\" "
        f"--folder_lq \"{tmp_folder}\""
    )

    print(f"[JPEG Expert] Model     : SwinIR JPEG-CAR q10")
    print("[JPEG Expert] Running SwinIR JPEG artifact removal...\n")

    result = os.system(command)

    # ── Failure check ──
    if result != 0:
        print("\n[JPEG Expert] SwinIR failed (non-zero exit code).")
        print("[JPEG Expert] Hint: Run 'python install_phase1_deps.py' to fix NumPy.")
        print("\n===================================")
        print("       JPEG EXPERT FINISHED")
        print("===================================\n")
        return None

    # ── Find output file ──
    output_dir = "results/swinir_jpeg_car_q10"
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    search_pattern = os.path.join(output_dir, f"{input_filename}*")
    candidates = glob.glob(search_pattern)

    if not candidates:
        print("[JPEG Expert] ERROR: No output file found.")
        return None

    output_path = candidates[0]

    # ── Stale file check ──
    if os.path.getmtime(output_path) < start_time:
        print("[JPEG Expert] Output file is stale (from a previous run).")
        return None

    elapsed = round(time.time() - start_time, 2)
    print(f"[JPEG Expert] Restored Output : {output_path}")
    print(f"[JPEG Expert] Processing Time : {elapsed}s")

    # Cleanup temp folder
    try:
        shutil.rmtree(tmp_folder)
    except Exception:
        pass

    print("\n===================================")
    print("       JPEG EXPERT FINISHED")
    print("===================================\n")

    return output_path