@echo off
REM Быстрый запуск E57 -> IFC конвертера

echo E57 -^> IFC Конвертер
echo ================================

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION%

REM Проверка виртуального окружения
if not exist ".venv" (
    echo [WARNING] Виртуальное окружение не найдено
    set /p CREATE_VENV="Создать? (y/n): "
    if /i "%CREATE_VENV%"=="y" (
        python -m venv .venv
        echo [OK] Виртуальное окружение создано
    )
)

REM Активация виртуального окружения
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Виртуальное окружение активировано
    
    REM Проверка зависимостей
    python -c "import numpy" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Зависимости не установлены
        set /p INSTALL_DEPS="Установить requirements-inspect.txt? (y/n): "
        if /i "%INSTALL_DEPS%"=="y" (
            pip install -r requirements-inspect.txt
        )
    )
)

REM Запуск
echo.
python e57_to_ifc.py

pause
