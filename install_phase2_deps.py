"""
MAIR+ Phase 2 — Dependency Installer
Run: python install_phase2_deps.py
"""
import subprocess
import sys


def install(package, label=None):
    name = label or package
    print(f"\n  Installing: {name}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=False,
    )
    ok = result.returncode == 0
    print(f"  Status: {'OK' if ok else 'FAILED'}")
    return ok


print("\n" + "=" * 50)
print("  MAIR+ Phase 2 — Dependency Installer")
print("=" * 50)
print("""
  Installing dependencies for:
    - Restormer deblur expert (einops)
    - Progress bars (tqdm)
    - YAML config parsing (pyyaml)
""")

results = [
    install("einops",  "einops  (Restormer attention ops)"),
    install("tqdm",    "tqdm    (progress bars)"),
    install("pyyaml",  "pyyaml  (YAML config)"),
    install("natsort", "natsort (natural filename sorting)"),
]

print("\n" + "=" * 50)
if all(results):
    print("  All dependencies installed successfully!")
    print("  Next: python run_pipeline.py")
else:
    print("  Some installations failed — check errors above.")
print("=" * 50 + "\n")
