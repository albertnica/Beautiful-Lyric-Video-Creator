@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Beautiful Lyric Video Creator - Interactive Web Studio Launcher
::
:: Behavior:
::   1. Resolves the local or system Python interpreter.
::   2. Verifies external system tools (FFmpeg & FFprobe) and notifies the user
::      clearly if anything is missing.
::   3. Inspects the `logs/` folder:
::        - If `logs/` is EMPTY: assumes first-time setup, detects user hardware
::          components (NVIDIA CUDA version vs OpenCL/CPU), runs `pip install`
::          for all required packages, verifies imports, and writes an initial
::          bootstrap log into `logs/`.
::        - If `logs/` is NOT empty: assumes a previous successful run and skips
::          `pip install` for instant startup.
::   4. Launches `main.py --ui` to start the Web Studio background server.
:: ============================================================================

cd /d "%~dp0"

:: 1. Resolve Python runtime (.venv -> venv -> system PATH python -> py launcher)
set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        where py >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=py -3"
        )
    )
)

if not defined PYTHON_EXE (
    echo ============================================================================
    echo [MISSING COMPONENT] Python 3.10+ interpreter was not found.
    echo   - Checked: .venv\Scripts\python.exe, venv\Scripts\python.exe, PATH
    echo   - Action:  Install Python 3.10+ from https://www.python.org/downloads/
    echo              and ensure "Add Python to PATH" is checked.
    echo ============================================================================
    echo.
    pause
    exit /b 1
)

:: 2. Verify FFmpeg and FFprobe availability in system PATH
where ffmpeg >nul 2>nul
if !errorlevel! neq 0 (
    echo ============================================================================
    echo [MISSING COMPONENT] FFmpeg was not found in your system PATH.
    echo   - Action: Download FFmpeg from https://gyan.dev/ffmpeg/builds/
    echo             or run: winget install Gyan.FFmpeg
    echo ============================================================================
    echo.
    pause
    exit /b 1
)

where ffprobe >nul 2>nul
if !errorlevel! neq 0 (
    echo ============================================================================
    echo [MISSING COMPONENT] FFprobe was not found in your system PATH.
    echo   - Action: Ensure the FFmpeg 'bin' directory containing ffprobe.exe
    echo             is added to your Windows PATH environment variable.
    echo ============================================================================
    echo.
    pause
    exit /b 1
)

:: 3. Ensure required workspace directories exist
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0songs" mkdir "%~dp0songs"
if not exist "%~dp0lyrics" mkdir "%~dp0lyrics"
if not exist "%~dp0output" mkdir "%~dp0output"

:: 4. Check if `logs/` directory is empty
set "LOGS_EMPTY=1"
for /f "delims=" %%F in ('dir /b /a-d "%~dp0logs\*" 2^>nul') do (
    set "LOGS_EMPTY=0"
)

if "!LOGS_EMPTY!"=="1" (
    echo ============================================================================
    echo  [FIRST-RUN BOOTSTRAP] 'logs/' folder is empty.
    echo  Detecting hardware components and installing required dependencies...
    echo ============================================================================
    echo.

    :: Install core requirements (FastAPI, Uvicorn, Pydantic V2, OpenCV, Mutagen, NumPy)
    echo [1/3] Installing core Python packages from requirements.txt...
    !PYTHON_EXE! -m pip install --upgrade pip
    !PYTHON_EXE! -m pip install -r "%~dp0requirements.txt"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Failed to install core dependencies from requirements.txt.
        echo Please check your internet connection and Python permissions.
        pause
        exit /b 1
    )

    :: Detect GPU hardware components to install matching accelerator package
    echo.
    echo [2/3] Probing GPU hardware components...
    set "HW_SUMMARY=CPU / OpenCL Fallback"
    where nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        nvidia-smi >nul 2>nul
        if !errorlevel! equ 0 (
            echo [GPU DETECTED] NVIDIA GPU found. Checking CUDA version for CuPy...
            set "CUDA_PKG=cupy-cuda13x"
            where nvcc >nul 2>nul
            if !errorlevel! equ 0 (
                for /f "tokens=*" %%C in ('nvcc --version ^| findstr /i "release"') do (
                    echo %%C | findstr /i "release 12." >nul && set "CUDA_PKG=cupy-cuda12x"
                    echo %%C | findstr /i "release 11." >nul && set "CUDA_PKG=cupy-cuda11x"
                )
            )
            echo [INSTALLING] Installing !CUDA_PKG! for NVIDIA CUDA acceleration...
            !PYTHON_EXE! -m pip install !CUDA_PKG!
            if !errorlevel! equ 0 (
                set "HW_SUMMARY=NVIDIA GPU !CUDA_PKG!"
            ) else (
                echo [WARN] !CUDA_PKG! installation failed; OpenCL/PyTorch/CPU fallback will be used.
            )
        ) else (
            echo [INFO] NVIDIA driver not active. Using OpenCL / PyTorch / CPU fallback.
        )
    ) else (
        echo [INFO] No NVIDIA GPU detected. Using OpenCL / PyTorch / CPU fallback.
    )

    :: Verify all mandatory Python modules import cleanly
    echo.
    echo [3/3] Verifying installed Python modules...
    !PYTHON_EXE! -c "import cv2, mutagen, numpy, pydantic, fastapi, uvicorn; print('[OK] All required Python modules verified successfully.')"
    if !errorlevel! neq 0 (
        echo.
        echo [MISSING COMPONENT] One or more required Python libraries failed to import.
        pause
        exit /b 1
    )

    :: Record bootstrap completion in logs/ so future runs skip installation
    (
        echo First-Run Bootstrap Completed: %DATE% %TIME%
        echo Python Interpreter: !PYTHON_EXE!
        echo Hardware Profile:   !HW_SUMMARY!
    ) > "%~dp0logs\000_bootstrap_setup.log"
    echo [BOOTSTRAP COMPLETE] Recorded setup log in logs\000_bootstrap_setup.log
    echo.
) else (
    echo [INFO] Existing logs found in 'logs/' - skipping pip install.
)

:: 5. Launch Web Studio background daemon and close launcher terminal window
echo ========================================================
echo  Beautiful Lyric Video Creator - Launching Web Studio
echo ========================================================
echo  Directory: %CD%
echo  Python:    !PYTHON_EXE!
echo ========================================================
echo.
echo Launching background server and opening Web Studio in your browser...
echo ^(This console window will close now. Use 'Stop Studio' in the browser header to halt.^)

set "PYTHONW_EXE=!PYTHON_EXE:python.exe=pythonw.exe!"
if exist "!PYTHONW_EXE!" (
    start "" "!PYTHONW_EXE!" "%~dp0main.py" --ui --foreground %*
) else (
    start "" /B "!PYTHON_EXE!" "%~dp0main.py" --ui %*
)

endlocal
exit 0
