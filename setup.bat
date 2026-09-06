@echo off
setlocal
cd /d "%~dp0"

REM ---- Resolve base Python interpreter (3.11+) ----
REM Priority: 1) NZKONV_PYTHON env var  2) py -3.12  3) py -3.11, py -3.13
REM           4) backend-bundled CPython (read-only borrow)  5) error

set "BASE_PY="
REM Assumes this repo and Nz-LTX23-backend are cloned as sibling directories
REM (the Rootport-AI GitHub org layout). If the backend is not present here,
REM the "if not exist" check below just falls through to :no_python.
set "BACKEND_PYROOT=%~dp0..\Nz-LTX23-backend\.python"

if not defined NZKONV_PYTHON goto :try_launcher
if not exist "%NZKONV_PYTHON%" (
    echo [setup] ERROR: NZKONV_PYTHON is set but the file does not exist:
    echo [setup]   %NZKONV_PYTHON%
    pause
    exit /b 1
)
"%NZKONV_PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [setup] ERROR: NZKONV_PYTHON must point to a Python 3.11+ executable.
    "%NZKONV_PYTHON%" --version
    pause
    exit /b 1
)
set BASE_PY="%NZKONV_PYTHON%"
echo [setup] Using NZKONV_PYTHON: %NZKONV_PYTHON%
goto :found

:try_launcher
for %%V in (3.12 3.11 3.13) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "BASE_PY=py -%%V"
        echo [setup] Using py launcher: py -%%V
        goto :found
    )
)

REM ---- Fallback: CPython bundled with the backend. Borrowed READ-ONLY;
REM ---- never modify anything under the backend directory.
if not exist "%BACKEND_PYROOT%" goto :no_python
for /d %%D in ("%BACKEND_PYROOT%\cpython-3.12*") do (
    if exist "%%D\python.exe" (
        set BASE_PY="%%D\python.exe"
        echo [setup] Using backend-bundled CPython: %%D\python.exe
        echo [setup] NOTE: the venv will depend on the backend's .python directory.
        goto :found
    )
)

:no_python
echo [setup] ERROR: No suitable Python interpreter found.
echo [setup] Please install Python 3.11+ or set the NZKONV_PYTHON
echo [setup] environment variable to a Python 3.11+ executable.
pause
exit /b 1

:found
if exist ".venv\Scripts\python.exe" (
    echo [setup] .venv already exists, skipping venv creation.
    goto :install
)

echo [setup] Creating virtual environment in .venv ...
%BASE_PY% -m venv .venv
if errorlevel 1 (
    echo [setup] ERROR: failed to create virtual environment.
    pause
    exit /b 1
)

:install
echo [setup] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [setup] ERROR: failed to upgrade pip.
    pause
    exit /b 1
)

echo [setup] Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [setup] ERROR: failed to install dependencies.
    pause
    exit /b 1
)

echo [setup] Done. Virtual environment is ready in .venv
pause
endlocal & exit /b 0
