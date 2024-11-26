@echo off
echo Starting Assignment Grader Pro...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking and installing requirements...
python -m pip install -r requirements.txt

REM Run the application
echo.
echo Starting the application...
python -m streamlit run main.py --server.address localhost --server.port 8501

pause
