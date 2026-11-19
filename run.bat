@echo off
cd /d "%~dp0"
echo ========================================
echo Starting RAG Chatbot...
echo ========================================

if not exist venv (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if not exist .env (
    echo No .env file found. Run setup.bat, then put your key in .env:
    echo     GLOBAL_API_KEY=your_key_here
    pause
    exit /b 1
)

echo Open http://localhost:8501 in your browser.
venv\Scripts\python.exe -m streamlit run app.py

pause
