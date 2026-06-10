@echo off
echo ============================================================
echo   MAIR+ v2 - Professor Demo Setup
echo ============================================================

cd /d "d:\NIt\MAIR_Project"

echo.
echo [1/5] Installing required packages...
python -m pip install opencv-python scikit-image einops numpy matplotlib ipykernel jupyter --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python is in PATH.
    pause
    exit /b 1
)
echo [OK] Packages installed.

echo.
echo [2/5] Running MAIR+ full demo + benchmark...
python setup_and_demo.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Demo script failed.
    pause
    exit /b 1
)

echo.
echo [3/5] Generating haze benchmark dataset...
python datasets\generate_benchmark.py --types haze --n 5
echo [OK] Haze dataset generated.

echo.
echo [4/5] Running haze benchmark (C1 + C3 demo)...
python evaluation\benchmark.py --dataset datasets\benchmark\haze_test --fast_only --max_images 3
echo [OK] Haze benchmark done.

echo.
echo [5/5] Installing Jupyter for notebook demo...
python -m pip install jupyter notebook --quiet
echo [OK] Jupyter ready.

echo.
echo ============================================================
echo   ALL DONE! 
echo ============================================================
echo.
echo   Results:
echo     - outputs\           comparison PNGs + HTML reports
echo     - results\           benchmark CSV + JSON
echo     - notebooks\MAIR_Demo.ipynb   interactive demo
echo     - CONTRIBUTIONS.md   1-page summary for professor
echo.
echo   To launch the notebook:
echo     jupyter notebook notebooks\MAIR_Demo.ipynb
echo.
pause
