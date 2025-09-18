@echo off
REM Beverly Knits ERP Server Startup Script for Windows

echo Starting Beverly Knits ERP Server...
echo.

REM Use WSL to run the Linux script
wsl bash -c "cd /mnt/c/finalee/beverly_knits_erp_v2 && ./start_server.sh"

REM Alternative: If you have Python installed on Windows
REM set EFAB_SESSION=aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B
REM set ENABLE_YARN_SCHEDULER=true
REM set FILTER_NONPRODUCTION_YARNS=true
REM python src\core\beverly_comprehensive_erp.py

pause