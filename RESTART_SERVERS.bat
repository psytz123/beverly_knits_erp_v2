@echo off
REM ============================================================================
REM Beverly Knits ERP - Force Clean Restart
REM ============================================================================

echo.
echo ============================================================================
echo Beverly Knits ERP - Forcing Clean Restart
echo ============================================================================
echo.

echo [STEP 1] Killing ALL Python processes...
taskkill /F /IM python.exe /T 2>nul
if errorlevel 1 (
    echo No Python processes found to kill
) else (
    echo Successfully killed all Python processes
)

echo.
echo [STEP 2] Waiting for processes to terminate...
timeout /t 5 /nobreak >nul

echo.
echo [STEP 3] Starting eFab API Server (port 5006)...
start "Beverly Knits - eFab API" cmd /k "cd /d C:\finalee\beverly_knits_erp_v2 && python src\api\efab_api_server.py"

echo.
echo [STEP 4] Waiting for API server to start...
timeout /t 8 /nobreak >nul

echo.
echo [STEP 5] Starting Dashboard Server (port 8080)...
start "Beverly Knits - Dashboard" cmd /k "cd /d C:\finalee\beverly_knits_erp_v2 && python web\server.py 8080"

echo.
echo [STEP 6] Waiting for dashboard server...
timeout /t 3 /nobreak >nul

echo.
echo ============================================================================
echo Services Started!
echo ============================================================================
echo.
echo eFab API Server:    http://localhost:5006/api/health
echo Dashboard:          http://localhost:8080/consolidated_dashboard_visual_preserved.html
echo.
echo Testing API server...
timeout /t 2 /nobreak >nul

curl -s http://localhost:5006/api/health

echo.
echo.
echo ============================================================================
echo Opening dashboard in browser...
echo ============================================================================
timeout /t 2 /nobreak >nul
start http://localhost:8080/consolidated_dashboard_visual_preserved.html

echo.
echo Done! Check your browser.
echo.
pause
