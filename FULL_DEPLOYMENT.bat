@echo off
REM ========================================================================
REM Beverly Knits ERP v2 - FULL SYSTEM DEPLOYMENT
REM Initializes, configures, loads data, tests, and launches all components
REM ========================================================================

echo.
echo ========================================================================
echo BEVERLY KNITS ERP v2 - FULL SYSTEM DEPLOYMENT
echo ========================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

echo [1/7] INIT: Verifying project structure...
if not exist "src\" (
    echo ERROR: src directory not found!
    exit /b 1
)
if not exist "web\" (
    echo ERROR: web directory not found!
    exit /b 1
)
if not exist ".env" (
    echo ERROR: .env file not found! Please create it from .env.example
    exit /b 1
)
echo OK: Project structure verified
echo.

echo [2/7] CONFIG: Testing Turso database connection...
python -c "from src.database.turso_client import get_turso_client; client = get_turso_client(); result = client.execute('SELECT 1 as test'); print('OK: Turso connection successful' if result else 'ERROR: Turso connection failed'); exit(0 if result else 1)"
if errorlevel 1 (
    echo ERROR: Turso connection failed. Check .env credentials
    exit /b 1
)
echo.

echo [3/7] INSTALL DEP: Checking Python dependencies...
python -c "import flask, pandas, numpy, httpx, sklearn; print('OK: All dependencies installed')"
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
    pip install httpx
)
echo.

echo [4/7] DATABASE: Initializing Turso schema...
python -c "from src.database.turso_client import get_turso_client; client = get_turso_client(); client.initialize_schema(); print('OK: Schema initialized')"
echo.

echo [5/7] DATA LOAD: Importing data to Turso...
echo.
echo NOTE: The following data import steps require data files:
echo   - eFab_Styles_20251018.xlsx for style mappings
echo   - Historical sales data (if available)
echo   - BOM and fabric specs (if available)
echo.
set /p IMPORT_DATA="Do you want to import data now? (y/n): "
if /i "%IMPORT_DATA%"=="y" (
    echo.
    echo Importing style mappings...
    set /p STYLES_FILE="Enter path to eFab_Styles Excel file (or press Enter to skip): "
    if not "%STYLES_FILE%"=="" (
        python scripts\import_style_mappings_to_turso.py --create-table --file "%STYLES_FILE%"
    )

    echo.
    echo You can import additional data later using:
    echo   - python scripts\import_sales_to_turso.py
    echo   - python scripts\import_bom_and_specs_to_turso.py
)
echo.

echo [6/7] TESTING: Running integration tests...
python scripts\test_turso_integration.py
echo.

echo [7/7] LAUNCH: Starting all components...
echo.
echo This will start:
echo   - eFab API Server (Port 5006)
echo   - Dashboard Web Server (Port 8000)
echo.
echo Press Ctrl+C to stop all services
echo.

REM Create a batch file to start API server
echo @echo off > start_api.bat
echo cd /d "%~dp0" >> start_api.bat
echo python src\api\efab_api_server.py >> start_api.bat

REM Create a batch file to start dashboard
echo @echo off > start_dash.bat
echo cd /d "%~dp0\web" >> start_dash.bat
echo python -m http.server 8000 >> start_dash.bat

echo.
echo ========================================================================
echo STARTING SERVICES...
echo ========================================================================
echo.
echo API Server will be available at: http://localhost:5006
echo Dashboard will be available at: http://localhost:8000/consolidated_dashboard.html
echo.
echo Opening new windows for each service...
echo.

REM Start API server in new window
start "Beverly Knits API Server" cmd /k "cd /d "%~dp0" && python src\api\efab_api_server.py"

REM Wait 3 seconds for API to start
timeout /t 3 /nobreak > nul

REM Start dashboard server in new window
start "Beverly Knits Dashboard" cmd /k "cd /d "%~dp0\web" && python -m http.server 8000"

REM Wait 2 seconds for dashboard to start
timeout /t 2 /nobreak > nul

echo.
echo ========================================================================
echo DEPLOYMENT COMPLETE!
echo ========================================================================
echo.
echo Services Status:
echo   - API Server: http://localhost:5006
echo   - Dashboard: http://localhost:8000/consolidated_dashboard.html
echo.
echo Next Steps:
echo   1. Open browser to: http://localhost:8000/consolidated_dashboard.html
echo   2. Verify data loads correctly
echo   3. Test forecast generation
echo   4. Review proactive production recommendations
echo.
echo To stop all services: Close the server windows or press Ctrl+C in each
echo.
echo For troubleshooting, check server logs in the command windows
echo ========================================================================
echo.

REM Open dashboard in default browser
timeout /t 3 /nobreak > nul
start http://localhost:8000/consolidated_dashboard.html

echo Press any key to exit this window (services will continue running)...
pause > nul
