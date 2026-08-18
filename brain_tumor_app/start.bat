@echo off
:: Navigate to script directory
cd /d "%~dp0"

echo 🧠 NeuroDiagnostics AI Setup ^& Start for Windows

:: Create virtual environment if it doesn't exist
if not exist ".venv_win" (
    echo 📦 Creating Python virtual environment venv_win...
    python -m venv .venv_win
)

:: Install/Upgrade dependencies
echo 📥 Verifying dependencies...
.venv_win\Scripts\python -m pip install --upgrade pip
.venv_win\Scripts\pip install -r requirements.txt

:: Start FastAPI server
echo 🚀 Starting FastAPI server on port 5050...
.venv_win\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 5050 --reload
pause
