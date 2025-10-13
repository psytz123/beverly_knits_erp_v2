@echo off
REM ============================================================================
REM Beverly Knits ERP - Clean Start with eFab Data
REM ============================================================================

echo.
echo ============================================================================
echo Beverly Knits ERP Dashboard - Clean Start
echo ============================================================================
echo.

REM Kill all existing Python processes
echo Stopping any running servers...
taskkill /F /IM python.exe >nul 2>&1

REM Wait for processes to stop
timeout /t 3 /nobreak >nul

echo.
echo Starting eFab API Server (port 5006)...
start "Beverly Knits - eFab API" cmd /k "python src\api\efab_api_server.py"

REM Wait for API server to start
timeout /t 5 /nobreak >nul

echo Starting Dashboard Server (port 8080)...
start "Beverly Knits - Dashboard" cmd /k "python web\server.py 8080"

REM Wait for dashboard server
timeout /t 3 /nobreak >nul

echo.
echo ============================================================================
echo Services Started Successfully!
echo ============================================================================
echo.
echo Backend API (eFab):  http://localhost:5006
echo Dashboard:           http://localhost:8080/consolidated_dashboard_visual_preserved.html
echo.
echo Opening dashboard...
echo ============================================================================
echo.

REM Open dashboard in browser
timeout /t 2 /nobreak >nul
start http://localhost:8080/consolidated_dashboard_visual_preserved.html

echo.
echo Dashboard opened! Check your browser.
echo.
echo To stop servers, close the command windows or press Ctrl+C in each.
echo.
pause
