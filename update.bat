@echo off
REM Update crispz (Windows): recupere les commits GitHub puis remet les
REM dependances en phase avec le lock, SANS casser l'installation existante.
REM
REM Fait, dans l'ordre:
REM   1. sauvegarde des versions installees (rollback possible)
REM   2. git pull (en refusant d'ecraser des modifications locales non commitees)
REM   3. reinstall des deps UNIQUEMENT si le fichier de deps a change
REM   4. verification que torch/CUDA et le pipeline fonctionnent encore
REM
REM Protection torch: une resolution transitive peut remplacer un build +cuXXX
REM par une roue CPU et casser le GPU. On releve la version avant/apres et on
REM alerte si elle a change.
REM
REM   --force-deps   reinstaller les deps meme si rien n'a change
REM   --no-pull      sauter le git pull (resynchroniser les deps seulement)
REM   --shared       utiliser requirements.txt au lieu du lock (venv partage)

setlocal enabledelayedexpansion
title crispz - Update
cd /d "%~dp0"

set "FORCEDEPS=0"
set "DOPULL=1"
set "ISOLATED=1"
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--force-deps" set "FORCEDEPS=1"
if /I "%~1"=="--no-pull" set "DOPULL=0"
if /I "%~1"=="--shared" set "ISOLATED=0"
shift
goto argloop
:argdone

echo === crispz - update ===
echo.

REM --- Interpreteur ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)
echo Interpreteur: !RUNPY!

REM --- 0. Etat avant: version torch + empreinte du fichier de deps ---
set REQFILE=requirements.txt
if "!ISOLATED!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
set "TORCH_BEFORE="
for /f "delims=" %%v in ('!RUNPY! -c "import torch;print(torch.__version__)" 2^>nul') do set "TORCH_BEFORE=%%v"
if defined TORCH_BEFORE (
    echo torch installe: !TORCH_BEFORE!
    !RUNPY! -m pip freeze > "%TEMP%\cz_pip_before.txt" 2>nul
    echo   ^(snapshot des versions: %TEMP%\cz_pip_before.txt^)
) else (
    echo torch non installe ^(premiere install ? lance install.bat^).
)
set "HASH_BEFORE="
if exist "!REQFILE!" for /f "delims=" %%h in ('certutil -hashfile "!REQFILE!" MD5 ^| findstr /R "^[0-9a-f][0-9a-f]*$"') do set "HASH_BEFORE=%%h"
echo.

REM --- 1. git pull ---
if "!DOPULL!"=="1" (
    where git >nul 2>&1
    if errorlevel 1 (
        echo [AVERT] git introuvable -^> pull saute. Mets a jour les fichiers a la main.
    ) else (
        REM Refuser d'ecraser du travail local non commite: on previent au lieu
        REM de perdre des modifications.
        set "DIRTY="
        for /f "delims=" %%s in ('git status --porcelain 2^>nul') do set "DIRTY=1"
        if defined DIRTY (
            echo [ATTENTION] Modifications locales non commitees:
            git status --short
            echo.
            echo   git pull risquerait un conflit. Commit / stash d'abord, ou relance
            echo   avec --no-pull pour ne resynchroniser que les dependances.
            pause & exit /b 1
        )
        echo Recuperation des commits ^(git pull^)...
        git pull --ff-only
        if errorlevel 1 (
            echo [ERREUR] git pull a echoue ^(branche divergente ?^). Resous a la main.
            pause & exit /b 1
        )
    )
    echo.
)

REM --- 2. Le fichier de deps a-t-il change ? ---
set "HASH_AFTER="
if exist "!REQFILE!" for /f "delims=" %%h in ('certutil -hashfile "!REQFILE!" MD5 ^| findstr /R "^[0-9a-f][0-9a-f]*$"') do set "HASH_AFTER=%%h"
set "NEEDDEPS=0"
if not "!HASH_BEFORE!"=="!HASH_AFTER!" set "NEEDDEPS=1"
if "!FORCEDEPS!"=="1" set "NEEDDEPS=1"
if not defined TORCH_BEFORE set "NEEDDEPS=1"

if "!NEEDDEPS!"=="0" (
    echo Dependances: !REQFILE! inchange -^> rien a reinstaller.
    echo   ^(--force-deps pour forcer^)
) else (
    echo Dependances: mise a jour depuis !REQFILE! ...
    set "REQTMP=%TEMP%\cz_req_nopillow.txt"
    !RUNPY! _req_filter.py "!REQFILE!" "!REQTMP!"
    !RUNPY! -m pip install -r "!REQTMP!"
    if not errorlevel 1 (
        REM Pillow est pose a part, en --no-deps: la borne pillow^<12 de gradio
        REM rendrait la resolution impossible. Cf. install.bat.
        !RUNPY! -m pip install --no-deps --upgrade "pillow==12.3.0" >nul 2>&1
    )
    if errorlevel 1 (
        echo [ERREUR] pip install a echoue. L'environnement peut etre incoherent.
        echo   Restauration possible: !RUNPY! -m pip install -r "%TEMP%\cz_pip_before.txt"
        pause & exit /b 1
    )
)
echo.

REM --- 3. torch a-t-il ete remplace ? (piege classique: build +cuXXX -^> CPU) ---
set "TORCH_AFTER="
for /f "delims=" %%v in ('!RUNPY! -c "import torch;print(torch.__version__)" 2^>nul') do set "TORCH_AFTER=%%v"
if defined TORCH_BEFORE if not "!TORCH_BEFORE!"=="!TORCH_AFTER!" (
    echo [ATTENTION] torch a change: !TORCH_BEFORE!  -^>  !TORCH_AFTER!
    echo    Si le suffixe +cuXXX a disparu, le GPU ne sera plus utilise.
    echo    Restauration: !RUNPY! -m pip install torch==!TORCH_BEFORE! --index-url https://download.pytorch.org/whl/cu128
    echo.
)

REM --- 4. Verifications finales ---
echo Verification de l'installation...
!RUNPY! _hw_check.py
set "HW=!errorlevel!"
echo.
if "!HW!"=="3" (
    echo [BLOQUANT] torch ne supporte plus cette carte ^(voir le correctif ci-dessus^).
    pause & exit /b 3
)
!RUNPY! -c "from diffusers import ZImageImg2ImgPipeline; print('diffusers: ZImageImg2ImgPipeline OK')"
if errorlevel 1 (
    echo [ERREUR] diffusers ne fournit plus ZImageImg2ImgPipeline.
    echo   Relance install.bat, ou restaure: !RUNPY! -m pip install -r "%TEMP%\cz_pip_before.txt"
    pause & exit /b 1
)
!RUNPY! -c "import app; print('app: imports OK')"
if errorlevel 1 (
    echo [ERREUR] l'application ne s'importe plus. Voir la trace ci-dessus.
    pause & exit /b 1
)
echo.

REM --- 5. Nouveautes de config: signaler les cles ajoutees dans l'exemple ---
if exist "preferences.json" if exist "preferences.example.json" (
    !RUNPY! -c "import json;a=json.load(open('preferences.json',encoding='utf-8'));b=json.load(open('preferences.example.json',encoding='utf-8'));n=[k for k in b if k not in a];print('Nouvelles cles de preferences disponibles: '+', '.join(n) if n else 'preferences.json a jour.')" 2>nul
    echo   ^(preferences.json n'est jamais ecrase: ajoute les cles voulues a la main^)
)
echo.

echo === Update OK ===
if exist "CHANGELOG.md" echo Nouveautes: voir CHANGELOG.md
echo Lance: run.bat  ^(ou boot_check.bat pour un diagnostic complet^)
endlocal
