@echo off
REM Install pour crispz - Z-Image upscaler + detailer (Windows)
REM Defaut: venv .venv (--system-site-packages) qui HERITE de ton torch et isole
REM les deps. --no-venv (ou --system) pour installer sur le Python courant.
REM Ne reinstalle JAMAIS torch.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- flags ---
REM   --no-venv / --system : installer sur le Python courant
REM   --locked             : utiliser requirements-lock.txt (versions exactes
REM                          validees) au lieu des bornes de requirements.txt
set USE_VENV=1
set USE_LOCK=0
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-venv" set USE_VENV=0
if /I "%~1"=="--system" set USE_VENV=0
if /I "%~1"=="--locked" set USE_LOCK=1
shift
goto argloop
:argdone

echo === crispz - install Windows ===
echo.

REM 1) Python de base
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
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)
echo Python de base: !PYCMD!
!PYCMD! --version
echo.

REM 2) torch + CUDA (NE PAS reinstaller)
!PYCMD! -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 2 (
    echo.
    echo [AVERT] PyTorch present mais CUDA non disponible. Z-Image en CPU sera tres lent.
    goto torch_ok
)
if errorlevel 1 (
    echo.
    echo [ERREUR] PyTorch introuvable. Installe d'abord ton build PyTorch + CUDA, puis relance.
    echo Exemple ^(CUDA 12.8^): !PYCMD! -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    exit /b 1
)
:torch_ok
echo.

REM 3) xformers casse ? le neutraliser cote SYSTEME (le venv en herite)
!PYCMD! -c "import xformers.ops" >nul 2>&1
if not errorlevel 1 (
    echo xformers OK.
) else (
    !PYCMD! -c "import xformers" >nul 2>&1
    if not errorlevel 1 (
        echo [AVERT] xformers installe mais ne charge pas ^(DLL/ABI torch incompatible^). Desinstallation.
        !PYCMD! -m pip uninstall -y xformers
    )
)
echo.

REM 4) venv optionnel (defaut) avec --system-site-packages
set RUNPY=!PYCMD!
if "!USE_VENV!"=="1" (
    if not exist ".venv\Scripts\python.exe" (
        echo Creation du venv .venv ^(--system-site-packages: herite de torch^)...
        !PYCMD! -m venv --system-site-packages .venv
    )
    if exist ".venv\Scripts\python.exe" (
        .venv\Scripts\python.exe -c "import torch" >nul 2>&1
        if errorlevel 1 (
            echo [AVERT] torch non visible dans le venv -^> repli sur le Python courant.
        ) else (
            set RUNPY=.venv\Scripts\python.exe
        )
    )
) else (
    echo Mode --no-venv: install sur le Python courant.
)
echo Interpreteur d'install: !RUNPY!
echo.

REM 5) Installer les deps
set REQFILE=requirements.txt
if "!USE_LOCK!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
echo Installation des dependances depuis !REQFILE! ...

REM Pillow est installe A PART, en --no-deps. gradio 5.x declare "pillow^<12",
REM alors que les CVE Pillow atteignables ici (l'app ouvre des images fournies
REM par l'utilisateur) ne sont corrigees qu'en 12.x: resoudre les deux ensemble
REM donne ResolutionImpossible. On retire donc Pillow du fichier de deps, puis
REM on pose la version corrigee sans re-resoudre le graphe.
REM
REM ATTENDU, PAS UN BUG: pip va quand meme redescendre Pillow sous 12 pendant
REM l'etape ci-dessous, car gradio declare cette borne en dependance
REM transitive. L'etape --no-deps qui suit le remonte, et la verification de
REM version en fin de bloc garantit qu'on ne finit jamais sur une version
REM vulnerable sans le savoir.
set "REQTMP=%TEMP%\cz_req_nopillow.txt"
!RUNPY! _req_filter.py "!REQFILE!" "!REQTMP!"
!RUNPY! -m pip install -r "!REQTMP!"
if errorlevel 1 (
    echo [ERREUR] echec pip install. Verifie le log ci-dessus.
    if "!USE_LOCK!"=="0" (
        echo.
        echo   Si l'erreur est un "ResolutionImpossible", relance avec le fichier
        echo   de versions exactes deja validees:
        echo       install.bat --locked
    )
    exit /b 1
)
!RUNPY! -m pip install --no-deps --upgrade "pillow==12.3.0"
REM Verification explicite: si Pillow a ete degrade quelque part, on veut le
REM VOIR ici, pas le decouvrir dans un rapport de vulnerabilites.
!RUNPY! -c "import PIL,sys;v=PIL.__version__;ok=int(v.split('.')[0])>=12;print(('Pillow %%s OK' %% v) if ok else ('[AVERT] Pillow %%s installe (<12): encore expose aux CVE de decodage d image' %% v));sys.exit(0 if ok else 5)"
if errorlevel 5 (
    echo         Correctif: !RUNPY! -m pip install --no-deps --upgrade pillow==12.3.0
)
echo.

REM 6) Verifier ZImageImg2ImgPipeline
!RUNPY! -c "from diffusers import ZImageImg2ImgPipeline; print('ZImageImg2ImgPipeline OK')"
if errorlevel 1 (
    echo [ERREUR] diffusers ne contient pas ZImageImg2ImgPipeline.
    exit /b 1
)
echo.

REM 7) Dossier upscale_models
if not exist "upscale_models" mkdir upscale_models
echo Dossier upscale_models pret. Depose tes .pth dedans, ou pointe ESRGAN_DIR.
echo.
echo === Install OK. Lance run.bat  ^(ou run.bat --no-venv^) ===
endlocal
