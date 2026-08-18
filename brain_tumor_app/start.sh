#!/bin/bash

# Navigate exactly to the script's directory so you don't run it from the wrong path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

echo "🧠 NeuroDiagnostics AI Setup for Arch Linux"

# 1. Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating a safe Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Suppress some common Arch/PyTorch warnings during install by upgrading pip first
pip install --quiet --upgrade pip

# 4. Install dependencies safely inside the venv
echo "📥 Installing & Verifying dependencies (FastAPI, PyTorch, ReportLab, etc.)..."
pip install -r requirements.txt

# 5. Start the backend server
echo "🚀 Starting FastAPI server on port 5050..."
python -m uvicorn app:app --host 0.0.0.0 --port 5050 --reload
