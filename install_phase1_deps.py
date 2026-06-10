"""
MAIR+ Phase 1 — Dependency Fixer
Run: python install_phase1_deps.py
"""
import subprocess
import sys


def install(package):
    print(f"\n  Installing: {package}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=False,
    )
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  Status: {status}")
    return result.returncode == 0


print("\n" + "=" * 50)
print("  MAIR+ Phase 1 — Dependency Installer")
print("=" * 50)
print("""
  Root cause: NumPy 2.x breaks binary compatibility with
  torchvision and timm (compiled against NumPy 1.x API).
  Fix: downgrade to NumPy 1.26.4 (latest stable 1.x).
""")

success = install("numpy==1.26.4")

if success:
    print("\n" + "=" * 50)
    print("  SUCCESS! NumPy fixed.")
    print("  Now run: python run_pipeline.py")
    print("=" * 50 + "\n")
else:
    print("\n  FAILED — try manually:")
    print("  pip install numpy==1.26.4")
