#!/usr/bin/env bash
# Install standalone pour crispz - Z-Image upscaler + detailer (Linux / macOS / WSL)
# Ne reinstalle PAS torch. Verifie l'env, installe les autres deps.

set -e
cd "$(dirname "$0")"

echo "=== crispz - install ==="
echo

# 1) Trouver Python (prefere 3.10, sinon python3)
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERREUR] Python introuvable. Installe Python 3.10+."
    exit 1
fi
echo "Python: $PYCMD"
$PYCMD --version
echo

# 2) Verifier torch + CUDA (NE PAS reinstaller)
set +e
$PYCMD -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
rc=$?
set -e
if [ $rc -eq 1 ]; then
    echo
    echo "[ERREUR] PyTorch introuvable. Installe ton build PyTorch + CUDA d'abord."
    echo "Exemple (CUDA 12.8):"
    echo "  $PYCMD -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
    exit 1
elif [ $rc -eq 2 ]; then
    echo "[AVERT] CUDA non disponible. Z-Image en CPU sera tres lent."
fi
echo

# 3) xformers casse ? le neutraliser pour eviter le DLL/ABI error au load de diffusers
if $PYCMD -c "import xformers" >/dev/null 2>&1; then
    if ! $PYCMD -c "import xformers.ops" >/dev/null 2>&1; then
        echo "[AVERT] xformers installe mais ne charge pas (ABI torch incompatible). Desinstallation."
        $PYCMD -m pip uninstall -y xformers
    else
        echo "xformers OK."
    fi
fi
echo

# 4) Installer les deps
echo "Installation des dependances..."
$PYCMD -m pip install -r requirements.txt
echo

# 5) Verifier ZImageImg2ImgPipeline
$PYCMD -c "from diffusers import ZImageImg2ImgPipeline; print('ZImageImg2ImgPipeline OK')"
echo

# 6) Dossier upscale_models
mkdir -p upscale_models
echo "Dossier upscale_models pret. Depose tes .pth dedans, ou pointe ESRGAN_DIR."
echo
echo "=== Install OK. Lance: ./run.sh ==="
