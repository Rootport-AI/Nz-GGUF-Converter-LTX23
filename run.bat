@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run] .venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m converter %*
endlocal & exit /b %ERRORLEVEL%
