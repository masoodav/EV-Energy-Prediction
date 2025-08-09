@echo off

set BUILD_DIR=%~dp0build

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
    echo Created build directory at: %BUILD_DIR%
) else (
    echo Build directory already exists at: %BUILD_DIR%
)

:: Redirect .pyc files to build directory and run main.py
set PYTHONPYCACHEPREFIX=%BUILD_DIR%
python "%~dp0main.py"
