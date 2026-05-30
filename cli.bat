@echo off
REM CLI interactive pour crispz.
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 ( set PYCMD=python ) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)

if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN
    ) else (
        set ESRGAN_DIR=%~dp0upscale_models
    )
)

%PYCMD% cli_interactive.py
endlocal
