# Beverly Knits ERP Server Startup Script for PowerShell

Write-Host "Beverly Knits ERP Server Launcher" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if WSL is available
$wslCheck = Get-Command wsl -ErrorAction SilentlyContinue

if ($wslCheck) {
    Write-Host "Starting server using WSL..." -ForegroundColor Green
    Write-Host "Server will be available at: http://localhost:5006" -ForegroundColor Yellow
    Write-Host "Dashboard: http://localhost:5006/consolidated" -ForegroundColor Yellow
    Write-Host ""

    # Run using WSL
    wsl bash -c "cd /mnt/c/finalee/beverly_knits_erp_v2 && chmod +x start_server.sh && ./start_server.sh"
}
else {
    Write-Host "WSL not found. Attempting to use Windows Python..." -ForegroundColor Yellow

    # Set environment variables for Windows
    $env:EFAB_SESSION = "aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B"
    $env:ENABLE_YARN_SCHEDULER = "true"
    $env:FILTER_NONPRODUCTION_YARNS = "true"

    # Check if Python is available
    $pythonCheck = Get-Command python -ErrorAction SilentlyContinue

    if ($pythonCheck) {
        Write-Host "Starting server with Windows Python..." -ForegroundColor Green
        Set-Location "C:\finalee\beverly_knits_erp_v2"
        python src\core\beverly_comprehensive_erp.py
    }
    else {
        Write-Host "ERROR: Neither WSL nor Python found!" -ForegroundColor Red
        Write-Host "Please install either:" -ForegroundColor Red
        Write-Host "  1. WSL (Windows Subsystem for Linux)" -ForegroundColor White
        Write-Host "  2. Python for Windows" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")