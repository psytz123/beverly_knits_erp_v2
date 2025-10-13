@echo off
REM AI Workspace Batch Wrapper
REM Usage: ai-workspace.bat [command] [options]

set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%bootstrap.py" %*
