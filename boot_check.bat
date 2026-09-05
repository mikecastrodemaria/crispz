@echo off
REM Boot check generique crispz.
REM
REM Diagnostique la machine AVANT de lancer l'app, quelle que soit la carte
REM (RTX 50xx / 40xx / 30xx / 20xx...), et s'arrete net si la configuration ne
REM peut pas fonctionner -- plutot que de laisser l'app planter en cours de route.
REM
REM Le check decisif est fait par _hw_check.py: il compare le sm_XX du GPU a la
REM liste d'architectures compilees dans le build torch installe. C'est ce qui
REM detecte le cas "RTX 50xx + torch non-cu128" (WinError 127 torch_cuda.dll).
REM
REM   --no-run   diagnostiquer seulement, ne pas lancer l'app
REM   --lan      ecouter sur le LAN (0.0.0.0) au lieu de 127.0.0.1
REM   --web      LAN + tunnel Cloudflare (URL publique)
REM   tout autre argument est transmis a run.bat
REM
REM ATTENTION --lan / --web: l'app n'a AUCUNE authentification et lit/ecrit des
REM chemins locaux (dossier de sortie, dossier ESRGAN). N'expose que sur un
REM reseau de confiance.

setlocal enabledelayedexpansion
title crispz - Boot Check
cd /d "%~dp0"

set "NORUN=0"
set "EXPOSE="
set "PASSTHRU="
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-run" (
    set "NORUN=1"
) else if /I "%~1"=="--lan" (
    set "EXPOSE=lan"
) else if /I "%~1"=="--web" (
    set "EXPOSE=web"
) else (
    set "PASSTHRU=!PASSTHRU! %~1"
)
shift
goto argloop
:argdone

echo ====================================================
echo    crispz - Boot Check
echo ====================================================
echo.

REM --- Interpreteur (venv prioritaire) ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)

echo [1/5] Python : !RUNPY!
!RUNPY! --version 2>nul
if errorlevel 1 (
    echo    [ERREUR] Python introuvable. Installe Python 3.10+ puis lance install.bat.
    pause & exit /b 1
)
echo.

REM --- 2. Etat du driver / de la carte (informations brutes) ---
echo [2/5] Driver NVIDIA...
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits > "%TEMP%\cz_gpu.txt" 2>nul
if errorlevel 1 (
    echo    [INFO] nvidia-smi introuvable ^(pas de GPU NVIDIA, ou drivers absents^).
) else (
    for /f "tokens=1,2,3,4,5 delims=," %%a in (%TEMP%\cz_gpu.txt) do (
        echo    Carte   : %%a
        echo    Driver  : %%b
        echo    VRAM    : %%d / %%c MB utilises   ^| Temp: %%e C
    )
    del "%TEMP%\cz_gpu.txt" >nul 2>&1
)
echo.

REM --- 3. LE check: torch supporte-t-il CETTE carte ? + recommandations ---
echo [3/5] PyTorch / GPU / reglages conseilles...
echo.
!RUNPY! _hw_check.py
set "HW=!errorlevel!"
echo.
if "!HW!"=="1" (
    echo    [ERREUR] PyTorch absent -^> lance install.bat.
    pause & exit /b 1
)
if "!HW!"=="3" (
    echo    [BLOQUANT] torch ne supporte pas cette carte ^(voir le correctif ci-dessus^).
    echo    L'app planterait a la premiere allocation CUDA. Arret.
    pause & exit /b 3
)
if "!HW!"=="2" echo    [AVERT] Mode CPU: n'utilise que la passe ESRGAN ^(denoise = 0^).

REM --- 4. Pipeline diffusers Z-Image + versions des deps critiques ---
REM L'import du pipeline ne suffit PAS: avec transformers ^< 4.51 il reussit, et
REM c'est le chargement du modele qui casse plus tard sur "module transformers has
REM no attribute Qwen3Model". On verifie donc aussi les versions minimales.
echo [4/5] diffusers / versions...
!RUNPY! -c "from diffusers import ZImageImg2ImgPipeline; print('    ZImageImg2ImgPipeline OK')" 2>nul
if errorlevel 1 echo    [ATTENTION] ZImageImg2ImgPipeline indisponible -^> lance install.bat / update.bat.
!RUNPY! -c "import app,sys;p=app.check_env();print('    versions deps OK') if not p else [sys.stderr.write('    [BLOQUANT] %%s: %%s installe, %%s+ requis\n' %% (n,g or 'absent','.'.join(map(str,m)))) for n,g,m in p] and sys.exit(4)" 2>&1
if errorlevel 4 (
    echo.
    echo    Z-Image ne pourra PAS se charger avec ces versions.
    echo    Correctif: install.bat  ^(cree un .venv qui herite de ton torch mais
    echo    isole les deps de crispz, sans toucher au Python global^).
    pause & exit /b 4
)
echo.

REM --- 5. Modeles ESRGAN ---
REM MEME resolution de ESRGAN_DIR que run.bat, sinon le diagnostic annonce
REM "0 modele" dans upscale_models alors que l'app en trouvera des dizaines
REM ailleurs -- et la variable est exportee, donc run.bat hérite du meme choix.
if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set "ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN"
    ) else (
        set "ESRGAN_DIR=%~dp0upscale_models"
    )
)
echo [5/5] Modeles ESRGAN...
REM %% : en batch un %% litteral s'ecrit double, sinon cmd mange le format Python.
!RUNPY! -c "import app;m=app.list_esrgan_models();print('    %%-4d modele(s)  %%s' %% (len(m), app.ESRGAN_DIR));[print('      - '+x) for x in m[:6]];print('      ...') if len(m)>6 else None" 2>nul
if errorlevel 1 echo    [INFO] impossible de lire la liste ^(deps manquantes ? lance install.bat^).
echo.

REM --- Optimisations CUDA (sans effet si pas de GPU NVIDIA) ---
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
REM Port fixe: evite que Gradio parte sur 7861+ quand une instance precedente
REM n'a pas encore libere le port.
if not defined GRADIO_SERVER_PORT set GRADIO_SERVER_PORT=7860

REM --- Exposition reseau (--lan / --web): Gradio lit ces variables nativement ---
set "CF_PORT=7860"
if defined EXPOSE (
    echo ----------------------------------------------------
    echo  [SECURITE] Exposition reseau demandee ^(--!EXPOSE!^).
    echo  crispz n'a AUCUNE authentification: l'UI accepte un dossier source et
    echo  un dossier de sortie arbitraires, donc lecture/ecriture sur cette
    echo  machine. N'expose que sur un reseau de confiance.
    echo ----------------------------------------------------
    set GRADIO_SERVER_NAME=0.0.0.0
    set GRADIO_SERVER_PORT=!CF_PORT!
    echo Acces LAN :
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:!CF_PORT!
    echo.
)
if /I "!EXPOSE!"=="web" (
    REM Config perso NON versionnee: CF_TUNNEL = tunnel cloudflared nomme,
    REM sinon quick tunnel ephemere.
    set "CF_TUNNEL="
    if exist "%~dp0cloudflare.local.bat" call "%~dp0cloudflare.local.bat"
    if defined CF_PORT set GRADIO_SERVER_PORT=!CF_PORT!
    where cloudflared >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] cloudflared introuvable dans le PATH.
        echo    Installe-le : winget install --id Cloudflare.cloudflared
        pause & exit /b 1
    )
    if defined CF_TUNNEL (
        echo [Cloudflare] Tunnel nomme : !CF_TUNNEL!
        start "Cloudflare Tunnel" cloudflared tunnel run !CF_TUNNEL!
    ) else (
        echo [Cloudflare] Quick tunnel ephemere: l'URL https://xxxx.trycloudflare.com
        echo              s'affiche dans la fenetre "Cloudflare Tunnel".
        start "Cloudflare Tunnel" cloudflared tunnel --url http://localhost:!CF_PORT!
    )
    echo.
)

if "!NORUN!"=="1" (
    echo ====================================================
    echo    Diagnostic termine ^(--no-run: app non lancee^).
    echo ====================================================
    endlocal & exit /b 0
)
echo ====================================================
echo    Checks OK. Lancement de crispz...
echo ====================================================
REM Petite pause. 'ping' plutot que 'timeout': timeout lit stdin et echoue avec
REM "la redirection de l'entree n'est pas prise en charge" des que le script est
REM lance depuis un pipe, une tache planifiee ou un service.
ping -n 3 127.0.0.1 >nul 2>&1
REM Le diagnostic materiel vient d'etre fait: run.bat n'a pas a le refaire.
set CZ_SKIP_HWCHECK=1
call "%~dp0run.bat" %PASSTHRU%
if /I "!EXPOSE!"=="web" (
    echo.
    echo ----------------------------------------------------
    echo  Arrete. Pense a fermer la fenetre du tunnel Cloudflare.
    echo ----------------------------------------------------
)
endlocal
