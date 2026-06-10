@echo off
:: MAIR+ v2 — Setup + Expert Comparison Runner
:: Double-click this file in Windows Explorer to:
::   1. Install cv2, scikit-image, numpy
::   2. Run the expert comparison grid

cd /d "%~dp0"
echo ============================================================
echo   MAIR+ v2 — Installing dependencies...
echo ============================================================
pip install opencv-python scikit-image numpy --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed. Trying pip3...
    pip3 install opencv-python scikit-image numpy --quiet
)

echo.
echo ============================================================
echo   Running expert comparison grid (degradation: noise)
echo ============================================================
echo.
python expert_comparison_grid.py --degradation noise

echo.
echo ============================================================
echo   Done! Output saved to:
echo   outputs\expert_comparison\expert_comparison_LATEST.png
echo ============================================================
pause
