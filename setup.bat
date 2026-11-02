@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Beautiful Lyric Video Creator - Automated Environment Setup
::
:: Responsibilities:
::   1. Discover Python 3.10+ runtime (py launcher or system PATH).
::   2. Create local isolated virtual environment (.venv) if absent.
::   3. Upgrade pip and install all core dependencies from requirements.txt.
::   4. Verify external multimedia prerequisites (FFmpeg & FFprobe).
::   5. Probe hardware compute acceleration (NVIDIA CUDA / PyTorch / OpenCL).
::   6. Scaffold required workspace directories (songs, lyrics, output, logs).
:: ============================================================================

cd /d "%~dp0"
title Beautiful Lyric Video Creator - Setup Wizard

set "NO_PAUSE=0"
set "AUTO_CONFIRM=0"
if "%1"=="--no-pause" set "NO_PAUSE=1"
if "%1"=="--nopause" set "NO_PAUSE=1"
if "%1"=="-y" set "AUTO_CONFIRM=1"
if "%1"=="--yes" set "AUTO_CONFIRM=1"
if "%2"=="-y" set "AUTO_CONFIRM=1"
if "%2"=="--yes" set "AUTO_CONFIRM=1"
if defined CI (
    set "NO_PAUSE=1"
    set "AUTO_CONFIRM=1"
)

echo ============================================================================
echo      Beautiful Lyric Video Creator - Automated Environment Setup
echo ============================================================================
echo.

:: 1. Discover base Python interpreter for virtualenv creation
set "BASE_PYTHON="

where py >nul 2>nul
if !errorlevel! equ 0 (
    set "BASE_PYTHON=py -3"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "BASE_PYTHON=python"
    )
)

if not defined BASE_PYTHON (
    echo [ERROR] No Python interpreter found in system PATH.
    echo   - Please install Python 3.10+ from https://www.python.org/downloads/
    echo   - Ensure "Add python.exe to PATH" is checked during installation.
    echo.
    if "!NO_PAUSE!"=="0" pause
    exit /b 1
)

echo [1/5] Checking base Python runtime...
!BASE_PYTHON! --version
if !errorlevel! neq 0 (
    echo [ERROR] Failed to execute Python.
    if "!NO_PAUSE!"=="0" pause
    exit /b 1
)
echo.

:: 2. Create local virtual environment (.venv) if not already present
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [2/5] Creating virtual environment in '%VENV_DIR%'...
    !BASE_PYTHON! -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        if "!NO_PAUSE!"=="0" pause
        exit /b 1
    )
    echo       Virtual environment successfully created.
) else (
    echo [2/5] Virtual environment already exists at '%VENV_DIR%'.
)
echo.

:: Pre-installation plan and user confirmation prompt
echo ============================================================================
echo [PACKAGE INSTALLATION PLAN]
echo ============================================================================
echo   Virtual Environment : %VENV_DIR%
echo   Pip Tooling         : pip, setuptools, wheel
echo   Core Dependencies   : requirements.txt
echo                         - fastapi, uvicorn (Web Studio UI server)
echo                         - pydantic (schema validation)
echo                         - opencv-python, numpy (image and video processing)
echo                         - mutagen (audio metadata extraction)

set "GPU_PLAN=None (Standard OpenCL / Vectorized CPU SIMD fallback)"
where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    set "GPU_PLAN=NVIDIA GPU detected -> cupy-cuda13x (RTX CUDA RawKernels)"
)
echo   Hardware Package    : !GPU_PLAN!
echo ============================================================================
echo.

if "!AUTO_CONFIRM!"=="0" (
    set "PROCEED_PIP=Y"
    set /p "PROCEED_PIP=Proceed with installing these packages into .venv? [Y/N] (default: Y): "
    set "FIRST_CHAR=!PROCEED_PIP:~0,1!"
    if /i "!FIRST_CHAR!"=="N" (
        echo.
        echo ============================================================================
        echo [SETUP CANCELLED] Package installation aborted by user.
        echo No PIP packages were installed or modified.
        echo ============================================================================
        echo.
        if "!NO_PAUSE!"=="0" pause
        exit /b 0
    )
)
echo.

:: 3. Upgrade pip and install core dependencies
echo [3/5] Installing core dependencies from requirements.txt...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if !errorlevel! neq 0 (
    echo [WARNING] Failed to upgrade pip. Proceeding with package installation...
)

"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    if "!NO_PAUSE!"=="0" pause
    exit /b 1
)
echo       Core dependencies installed successfully.

:: 3.1 Check for NVIDIA GPU and install hardware-accelerated CuPy package
where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    echo.
    echo       NVIDIA GPU detected. Checking CuPy acceleration package...
    "%VENV_PYTHON%" -c "import cupy" >nul 2>nul
    if !errorlevel! neq 0 (
        echo       Installing cupy-cuda13x for native RTX CUDA compositing...
        "%VENV_PYTHON%" -m pip install cupy-cuda13x
        if !errorlevel! neq 0 (
            echo       [NOTICE] cupy-cuda13x was not found; trying cupy-cuda12x...
            "%VENV_PYTHON%" -m pip install cupy-cuda12x
        )
    ) else (
        echo       CuPy CUDA acceleration is already available.
    )
)
echo.

:: 4. Verify FFmpeg and FFprobe system prerequisites
echo [4/5] Verifying FFmpeg and FFprobe in system PATH...
set "FFMPEG_MISSING=0"

where ffmpeg >nul 2>nul
if !errorlevel! neq 0 set "FFMPEG_MISSING=1"

where ffprobe >nul 2>nul
if !errorlevel! neq 0 set "FFMPEG_MISSING=1"

if "!FFMPEG_MISSING!"=="1" (
    echo ============================================================================
    echo [MISSING COMPONENT] FFmpeg and/or FFprobe not found in system PATH.
    echo   - FFmpeg is mandatory for audio decoding and high-speed MP4 encoding.
    echo   - Install via Windows Package Manager:
    echo       winget install Gyan.FFmpeg
    echo   - Or download builds manually from:
    echo       https://gyan.dev/ffmpeg/builds/
    echo     and add the 'bin' folder to your Windows PATH environment variable.
    echo ============================================================================
    echo.
) else (
    echo       FFmpeg and FFprobe verified in system PATH.
    echo.
)

:: 5. Initialize directories and probe hardware compute capabilities
echo [5/5] Initializing workspace directories and probing hardware acceleration...
if not exist "%~dp0songs" mkdir "%~dp0songs"
if not exist "%~dp0lyrics" mkdir "%~dp0lyrics"
if not exist "%~dp0output" mkdir "%~dp0output"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0assets" mkdir "%~dp0assets"

"%VENV_PYTHON%" -c "from lyric_video_creator.backends import is_cupy_usable, detect_torch_device, is_opencl_usable; from lyric_video_creator.encoder import resolve_video_encoder; codec, _ = resolve_video_encoder('auto'); print(f'      - Hardware Compute : CuPy CUDA={is_cupy_usable()} | PyTorch Device={detect_torch_device()} | OpenCL={is_opencl_usable()}'); print(f'      - Video Encoder    : {codec}')"

echo.
echo ============================================================================
echo [SUCCESS] Setup completed successfully!
echo ============================================================================
echo.

if "!NO_PAUSE!"=="0" (
    echo Launching Web Studio in your browser...
    echo ^(You can also launch it at any time via run_ui.bat^)
    echo.
    call "%~dp0run_ui.bat"
)

exit /b 0

