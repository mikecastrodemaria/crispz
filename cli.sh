#!/usr/bin/env bash
# CLI interactive pour crispz.
set -e
cd "$(dirname "$0")"

if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERREUR] Python introuvable."
    exit 1
fi

if [ -z "$ESRGAN_DIR" ]; then
    export ESRGAN_DIR="$(pwd)/upscale_models"
fi

$PYCMD cli_interactive.py
