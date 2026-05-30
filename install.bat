@echo off
REM Install standalone pour crispz - Z-Image upscaler + detailer (Windows)
REM Ne reinstalle PAS torch. Verifie l'env, installe les autres deps.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === crispz - install Windows ===
echo.

REM 1) Trouver une commande Python utilisable: py -3.10 ou py
where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Python introuvable. Installe Python 3.10+ depuis python.org.
        exit /b 1
    )
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 (
        set PYCMD=py
    ) else (
        set PYCMD=py -3.10
    )
)
echo Python: %PYCMD%
%PYCMD% --version
echo.

REM 2) Verifier torch + CUDA (NE PAS reinstaller)
%PYCMD% -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 2 (
    echo.
    echo [AVERT] PyTorch est present mais CUDA non disponible. Z-Image Turbo en CPU sera tres lent.
)
if errorlevel 1 (
    echo.
    echo [ERREUR] PyTorch introuvable. Installe d'abord ton build PyTorch + CUDA, puis relance install.bat.
    echo Exemple ^(CUDA 12.8^):
    echo   %PYCMD% -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    exit /b 1
)
echo.

REM 3) Detecter xformers casse (build pour une autre version de torch)
%PYCMD% -c "import xformers.ops" >nul 2>&1
if not errorlevel 1 (
    echo xformers OK.
) else (
    %PYCMD% -c "import xformers" >nul 2>&1
    if not errorlevel 1 (
        echo [AVERT] xformers est installe mais ne charge pas ^(DLL ou ABI torch incompatible^).
        echo         Desinstallation pour eviter qu'il bloque le chargement de diffusers.
        %PYCMD% -m pip uninstall -y xformers
    )
)
echo.

REM 4) Installer les deps du requirements.txt (diffusers depuis git, transformers, etc.)
echo Installation des dependances...
%PYCMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERREUR] echec pip install. Verifie le log ci-dessus.
    exit /b 1
)
echo.

REM 5) Verifier que ZImageImg2ImgPipeline est dispo
%PYCMD% -c "from diffusers import ZImageImg2ImgPipeline; print('ZImageImg2ImgPipeline OK')"
if errorlevel 1 (
    echo [ERREUR] diffusers ne contient pas ZImageImg2ImgPipeline. Re-essaye install depuis git+huggingface/diffusers.
    exit /b 1
)
echo.

REM 6) Dossier upscale_models
if not exist "upscale_models" mkdir upscale_models
echo Dossier upscale_models pret. Depose tes .pth dedans, ou pointe ESRGAN_DIR vers un dossier existant.
echo.
echo === Install OK. Lance run.bat ===
endlocal
