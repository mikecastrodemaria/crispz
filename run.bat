@echo off
REM Lance crispz avec detection hardware + reco d'optimisation.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 1) Python
where py >nul 2>&1
if errorlevel 1 (
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)

REM 2) ESRGAN_DIR: priorite a la variable existante, sinon dossier sdlibs s'il existe, sinon local
if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN
    ) else (
        set ESRGAN_DIR=%~dp0upscale_models
    )
)

echo === crispz - run ===
echo ESRGAN_DIR = %ESRGAN_DIR%
echo.
echo --- Detection hardware ---
%PYCMD% _hw_check.py
echo.

echo --- Lancement de l'UI Gradio ---
echo Ouvre http://127.0.0.1:7860 dans ton navigateur
echo.
%PYCMD% app.py
endlocal
