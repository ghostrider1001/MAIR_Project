@echo off
echo ===================================================
echo STEP 1: Applying Mathematically Optimized Degradations
echo ===================================================
D:\NIt\MAIR_Project\venv\Scripts\python.exe apply_standard_degradations.py

echo.
echo ===================================================
echo STEP 2: Running AI Restoration & Evaluation
echo ===================================================
D:\NIt\MAIR_Project\venv\Scripts\python.exe run_academic_eval.py

echo.
echo ===================================================
echo STEP 3: Generating Final Presentation
echo ===================================================
D:\NIt\MAIR_Project\venv\Scripts\python.exe generate_academic_report.py

echo.
echo DONE! Open MAIR_Academic_Evaluation.pptx
