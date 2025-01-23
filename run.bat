@echo off
setlocal enabledelayedexpansion

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims=." %%I in ('python -c "import sys; print(sys.version.split()[0])"') do set PYTHON_VERSION=%%I
if %PYTHON_VERSION% LSS 8 (
    echo Error: Python 3.8 or higher is required
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip and install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

:: Check required environment variables
set MISSING_VARS=0
if not defined ANTHROPIC_API_KEY (
    echo Error: ANTHROPIC_API_KEY is not set
    set MISSING_VARS=1
)
if not defined PINECONE_API_KEY (
    echo Error: PINECONE_API_KEY is not set
    set MISSING_VARS=1
)
if not defined PINECONE_ENVIRONMENT (
    echo Error: PINECONE_ENVIRONMENT is not set
    set MISSING_VARS=1
)

if %MISSING_VARS%==1 (
    echo Please set all required environment variables in the .env file
    exit /b 1
)

:: Start the Flask application
echo Starting Flask application...
python app.py

endlocal
