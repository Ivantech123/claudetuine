#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check environment variables
check_env_vars() {
    local missing_vars=()
    
    # List of required environment variables
    local required_vars=(
        "ANTHROPIC_API_KEY"
        "PINECONE_API_KEY"
        "PINECONE_ENVIRONMENT"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo "Error: The following environment variables are missing:"
        printf '%s\n' "${missing_vars[@]}"
        echo "Please set them in the .env file"
        exit 1
    fi
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        echo "Error: Python is not installed"
        exit 1
    fi
    
    # Check Python version
    PY_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if (( $(echo "$PY_VERSION < 3.8" | bc -l) )); then
        echo "Error: Python 3.8 or higher is required (current: $PY_VERSION)"
        exit 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    # Source virtual environment
    if [ "$(uname)" == "Darwin" ]; then
        source venv/bin/activate
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        source venv/bin/activate
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
        source venv/Scripts/activate
    else
        echo "Error: Unsupported operating system"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

# Main execution
echo "Starting setup..."

# Check Python version
check_python_version

# Setup virtual environment
setup_venv

# Install dependencies
install_dependencies

# Check environment variables
check_env_vars

# Start the application
echo "Starting Flask application..."
$PYTHON_CMD app.py
