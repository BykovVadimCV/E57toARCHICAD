#!/bin/bash
# Быстрый запуск E57 → IFC конвертера

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}E57 → IFC Конвертер${NC}"
echo "================================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 не найден${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"

# Проверка виртуального окружения
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠${NC} Виртуальное окружение не найдено"
    read -p "Создать? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv .venv
        echo -e "${GREEN}✓${NC} Виртуальное окружение создано"
    fi
fi

# Активация виртуального окружения
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Виртуальное окружение активировано"
    
    # Проверка зависимостей
    if ! python -c "import numpy" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} Зависимости не установлены"
        read -p "Установить requirements-inspect.txt? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install -r requirements-inspect.txt
        fi
    fi
fi

# Запуск
echo ""
python e57_to_ifc.py
