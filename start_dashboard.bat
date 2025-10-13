@echo off
REM ============================================================================
REM Beverly Knits ERP Dashboard Launcher
REM ============================================================================
REM This script starts both the backend API server and the frontend dashboard
REM ============================================================================

echo.
echo ============================================================================
echo Beverly Knits ERP Dashboard - Starting Services
echo ============================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Starting Backend API Server on port 5006...
echo.

REM Ensure log directory exists
if not exist logs mkdir logs

REM Start the backend API server in a new window (production eFab proxy)
start "Beverly Knits API Server" cmd /k "python src\api\efab_api_server.py"

REM Wait a few seconds for the API server to start
timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend Dashboard Server on port 8080...
echo.

REM Start the dashboard web server in a new window
start "Beverly Knits Dashboard" cmd /k "python web\server.py 8080"

echo.
echo Validating service health...
echo.

python scripts\system_health_check.py --retries 5 --interval 2
if errorlevel 1 (
    echo ============================================================================
    echo WARNING: Health check failed. Review logs and running windows for details.
    echo Close the API and dashboard windows, resolve issues, then rerun this script.
    echo ============================================================================
    goto :after_health
)

REM Wait a moment
timeout /t 3 /nobreak >nul

echo.
echo ============================================================================
echo Services Started Successfully!
echo ============================================================================
echo.
echo Backend API:  http://localhost:5006
echo Dashboard:    http://localhost:8080/consolidated_dashboard_visual_preserved.html
echo.
echo Opening dashboard in your default browser...
echo.
echo To stop the servers, close the command windows or press Ctrl+C in each.
echo ============================================================================
echo.

REM Open the dashboard in the default browser
timeout /t 2 /nobreak >nul
start http://localhost:8080/consolidated_dashboard_visual_preserved.html

:after_health

echo.
echo Press any key to exit this launcher window...
pause >nul
