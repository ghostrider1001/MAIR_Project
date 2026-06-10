Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  MAIR+ — Installing required packages" -ForegroundColor Cyan
Write-Host "  Target: System Python 3.13" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Find Python
$python = "C:\Users\aswin\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
    Write-Host "[Setup] Using system 'python' from PATH" -ForegroundColor Yellow
} else {
    Write-Host "[Setup] Found Python 3.13 at: $python" -ForegroundColor Green
}

# List of packages to install
$packages = @(
    "opencv-python",
    "scikit-image",
    "einops",
    "numpy",
    "matplotlib",
    "Pillow"
)

Write-Host ""
Write-Host "[1/2] Installing core packages..." -ForegroundColor Yellow
foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -NoNewline
    & $python -m pip install $pkg --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[2/2] Installing Jupyter for notebook demo..." -ForegroundColor Yellow
& $python -m pip install jupyter notebook ipykernel --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Jupyter installed OK" -ForegroundColor Green
} else {
    Write-Host "  Jupyter install failed (optional)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[Verify] Checking installed packages..." -ForegroundColor Yellow
$checks = @("cv2", "skimage", "einops", "numpy", "matplotlib")
foreach ($mod in $checks) {
    $result = & $python -c "import $mod; print('OK')" 2>&1
    if ($result -eq "OK") {
        Write-Host "  $mod : OK" -ForegroundColor Green
    } else {
        Write-Host "  $mod : MISSING" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Done! Now run the pipeline:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  cd d:\NIt\MAIR_Project" -ForegroundColor White
Write-Host "  python run_pipeline.py --input demo_inputs\noisy_input.png" -ForegroundColor White
Write-Host ""
Write-Host "  Or run the full demo:" -ForegroundColor White
Write-Host "  python setup_and_demo.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
