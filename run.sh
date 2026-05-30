#!/usr/bin/env bash
# Lance crispz avec detection hardware + reco d'optimisation.

set -e
cd "$(dirname "$0")"

# 1) Python
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERREUR] Python introuvable."
    exit 1
fi

# 2) ESRGAN_DIR par defaut si non defini
if [ -z "$ESRGAN_DIR" ]; then
    export ESRGAN_DIR="$(pwd)/upscale_models"
fi

echo "=== crispz - run ==="
echo "ESRGAN_DIR = $ESRGAN_DIR"
echo
echo "--- Detection hardware ---"
$PYCMD _hw_check.py
echo

echo "--- Lancement de l'UI Gradio ---"
echo "Ouvre http://127.0.0.1:7860 dans ton navigateur"
echo
$PYCMD app.py
