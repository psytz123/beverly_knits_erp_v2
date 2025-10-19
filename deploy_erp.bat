@echo off
REM Beverly Knits ERP - Complete Deployment Script
REM This script starts all required services for the ERP system

echo ============================================================
echo Beverly Knits ERP - Complete Deployment
echo ============================================================
echo.

REM Kill any existing processes on our ports
echo [1/4] Stopping old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5006"') do taskkill //F //PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000"') do taskkill //F //PID %%a 2>nul
timeout /t 2 /nobreak >nul
echo     ✓ Old processes stopped
echo.

REM Start eFab API Server (port 5006)
echo [2/4] Starting eFab API Server...
cd "%~dp0src\api"
start /B python efab_api_server.py >..\..\logs\efab_api.log 2>&1
timeout /t 5 /nobreak >nul
echo     ✓ eFab API Server starting on port 5006
echo.

REM Start Web Server (port 8000)
echo [3/4] Starting Web Dashboard Server...
cd "%~dp0web"
start /B python server.py 8000 >../logs/web_server.log 2>&1
timeout /t 3 /nobreak >nul
echo     ✓ Web Server starting on port 8000
echo.

REM Verify services
echo [4/4] Verifying services...
timeout /t 2 /nobreak >nul

netstat -ano | findstr ":5006" >nul
if %errorlevel% equ 0 (
    echo     ✓ eFab API Server: RUNNING on port 5006
) else (
    echo     ✗ eFab API Server: NOT DETECTED
)

netstat -ano | findstr ":8000" >nul
if %errorlevel% equ 0 (
    echo     ✓ Web Server: RUNNING on port 8000
) else (
    echo     ✗ Web Server: NOT DETECTED
)
echo.

echo ============================================================
echo Deployment Complete!
echo ============================================================
echo.
echo Services:
echo   - eFab API Server: http://localhost:5006
echo   - Dashboard: http://localhost:8000/consolidated_dashboard.html
echo.
echo API Endpoints:
echo   - Health: http://localhost:5006/api/health
echo   - Knit Orders: http://localhost:5006/api/knit-orders
echo   - Fabric Forecast: http://localhost:5006/api/fabric-forecast-integrated
echo   - Inventory: http://localhost:5006/api/inventory/pipeline-summary
echo.
echo Logs:
echo   - eFab API: logs\efab_api.log
echo   - Web Server: logs\web_server.log
echo.
echo Press any key to open dashboard in browser...
pause >nul
start http://localhost:8000/consolidated_dashboard.html
