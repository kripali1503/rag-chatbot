@echo off
cd /d "%~dp0"
echo ========================================
echo RAG Chatbot Setup (needs Python 3.10+)
echo ========================================

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Could not create the virtual environment. Is Python installed and on PATH?
    pause
    exit /b 1
)

echo [2/3] Installing dependencies (this takes a few minutes)...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/3] Preparing .env...
if not exist .env (
    copy .env.example .env >nul
    echo Created .env from .env.example.
)

echo.
echo ========================================
echo Setup complete. Next:
echo   1. Open .env and set GLOBAL_API_KEY=your_key_here
echo      (Groq, Gemini, OpenRouter, OpenAI or Anthropic key - provider is detected automatically)
echo   2. Double-click run.bat
echo ========================================
pause
