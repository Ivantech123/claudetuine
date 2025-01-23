#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для красивого вывода
print_status() {
    echo -e "${YELLOW}==>${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

print_success() {
    echo -e "${GREEN}Success:${NC} $1"
}

# Проверка наличия Homebrew
check_brew() {
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Добавляем Homebrew в PATH для M1 Mac
        if [[ $(uname -m) == 'arm64' ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        print_status "Updating Homebrew..."
        brew update
    fi
}

# Проверка и установка Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_status "Installing Python..."
        brew install python@3.11
    fi
    
    # Проверяем версию Python
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
        print_error "Python 3.8 or higher is required (current: $PYTHON_VERSION)"
        print_status "Installing Python 3.11..."
        brew install python@3.11
        brew link --force python@3.11
    fi
}

# Проверка и установка pip
check_pip() {
    if ! command -v pip3 &> /dev/null; then
        print_status "Installing pip..."
        python3 -m ensurepip --upgrade
    fi
}

# Установка зависимостей системы
install_system_dependencies() {
    print_status "Installing system dependencies..."
    brew install openssl readline sqlite3 xz zlib tcl-tk
}

# Проверка переменных окружения
check_env_vars() {
    local missing_vars=()
    
    # Проверяем наличие файла .env
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << EOL
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_HOST=your_pinecone_host
VOYAGE_API_KEY=your_voyage_api_key
EOL
    fi
    
    # Загружаем переменные из .env
    set -a
    source .env
    set +a
    
    # Проверяем обязательные переменные
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        missing_vars+=("ANTHROPIC_API_KEY")
    fi
    if [ -z "$PINECONE_API_KEY" ]; then
        missing_vars+=("PINECONE_API_KEY")
    fi
    if [ -z "$PINECONE_ENVIRONMENT" ]; then
        missing_vars+=("PINECONE_ENVIRONMENT")
    fi
    if [ -z "$PINECONE_HOST" ]; then
        missing_vars+=("PINECONE_HOST")
    fi
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "The following environment variables are missing:"
        printf '%s\n' "${missing_vars[@]}"
        exit 1
    fi
}

# Настройка виртуального окружения
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
}

# Установка зависимостей Python
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi
}

# Главная функция
main() {
    print_status "Starting setup..."
    
    # Проверяем, что мы на macOS
    if [ "$(uname)" != "Darwin" ]; then
        print_error "This script is for macOS only"
        exit 1
    }
    
    # Установка и проверка всех компонентов
    check_brew
    check_python
    check_pip
    install_system_dependencies
    check_env_vars
    setup_venv
    install_python_dependencies
    
    # Запуск приложения
    print_success "Setup completed successfully!"
    print_status "Starting Flask application..."
    python3 app.py
}

# Запуск главной функции
main
