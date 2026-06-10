import os
import time
import glob


def restore_sr(input_path):
    """
    Run SwinIR x4 Super-Resolution on the input image.

    Returns:
        Path to restored output image, or None if restoration failed.
    """
    print("\n===================================")
    print("        SR EXPERT ACTIVATED")
    print("===================================\n")

    start_time = time.time()

    model_path = (
        "models/SwinIr/model_zoo/"
        "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    )
    folder_path = os.path.dirname(os.path.abspath(input_path))

    command = (
        f"python models/SwinIr/main_test_swinir.py "
        f"--task real_sr --scale 4 "
        f"--model_path \"{model_path}\" "
        f"--folder_lq \"{folder_path}\""
    )

    print(f"[SR Expert] Input Path  : {input_path}")
    print(f"[SR Expert] Model       : SwinIR Real-SR x4 (GAN)")
    print("[SR Expert] Running SwinIR inference...\n")

    result = os.system(command)

    # ── Failure check: non-zero exit code means SwinIR crashed ──
    if result != 0:
        print("\n[SR Expert] SwinIR failed (non-zero exit code).")
        print("[SR Expert] Hint: Run 'python install_phase1_deps.py' to fix NumPy.")
        print("\n===================================")
        print("        SR EXPERT FINISHED")
        print("===================================\n")
        return None

    # ── Find output file ──
    output_dir = "results/swinir_real_sr_x4"
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    search_pattern = os.path.join(output_dir, f"{input_filename}*")
    candidates = glob.glob(search_pattern)

    if not candidates:
        print("[SR Expert] ERROR: No output file found in results folder.")
        return None

    output_path = candidates[0]

    # ── Stale file check: output must be NEWER than this run's start ──
    file_mtime = os.path.getmtime(output_path)
    if file_mtime < start_time:
        print(f"[SR Expert] Output file is stale (from a previous run).")
        print(f"[SR Expert] File modified: {time.ctime(file_mtime)}")
        print(f"[SR Expert] Run started  : {time.ctime(start_time)}")
        print("[SR Expert] Treating as failure to avoid false quality scores.")
        return None

    elapsed = round(time.time() - start_time, 2)
    print(f"[SR Expert] Restored Output : {output_path}")
    print(f"[SR Expert] Processing Time : {elapsed}s")

    print("\n===================================")
    print("        SR EXPERT FINISHED")
    print("===================================\n")

    return output_path