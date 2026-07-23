@echo off
REM setup.bat - One-command project setup for Windows
REM Run this file once after cloning the repo.

REM The venv lives outside the project folder on purpose. Windows' 260-char
REM path limit truncates torch's nested license tree when site-packages sits
REM under this repo, and OneDrive would otherwise sync ~2GB of packages.
SET VENV_DIR=C:\venvs\astro

echo ============================================================
echo   Astro Power Cables Chatbot - Setup
echo ============================================================

REM 1. Create virtual environment if it does not exist
IF NOT EXIST "%VENV_DIR%\" (
    echo [1/4] Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
) ELSE (
    echo [1/4] Virtual environment already exists, skipping.
)

REM 2. Activate venv
echo [2/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM 3. Install dependencies
echo [3/4] Installing dependencies from requirements.txt...
pip install --timeout 300 -r requirements.txt

REM 4. Done
echo.
echo [4/4] Setup complete!
echo.
echo Activate the venv in your own terminal first - the activation above
echo belongs to this script and does not carry over:
echo   PowerShell : %VENV_DIR%\Scripts\Activate.ps1
echo   cmd.exe    : %VENV_DIR%\Scripts\activate.bat
echo.
echo Then run:
echo   1. Download the model    : python download_model.py
echo   2. Build the vectorstore : python build_vectorstore.py
echo   3. Run CLI chatbot       : python main.py
echo   4. Run API server        : uvicorn app.api:app --host 0.0.0.0 --port 8000
echo.
pause
