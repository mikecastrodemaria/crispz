#!/usr/bin/env bash
# Boot check generique crispz (equivalent Unix de boot_check.bat).
#
# Diagnostique la machine AVANT de lancer l'app et s'arrete net si la
# configuration ne peut pas fonctionner, plutot que de laisser l'app planter en
# cours de route. Le check decisif est fait par _hw_check.py: il compare le
# sm_XX du GPU aux architectures compilees dans le build torch installe.
#
#   --no-run   diagnostiquer seulement, ne pas lancer l'app
#   --lan      ecouter sur le LAN (0.0.0.0) au lieu de 127.0.0.1
#   tout autre argument est transmis a run.sh
#
# ATTENTION --lan: l'app n'a AUCUNE authentification et lit/ecrit des chemins
# locaux (dossier source, dossier de sortie, dossier ESRGAN).
set -uo pipefail
cd "$(dirname "$0")"

NORUN=0; EXPOSE=0; PASSTHRU=()
for a in "$@"; do
  case "$a" in
    --no-run) NORUN=1 ;;
    --lan)    EXPOSE=1 ;;
    *)        PASSTHRU+=("$a") ;;
  esac
done

echo "===================================================="
echo "   crispz - Boot Check"
echo "===================================================="
echo

RUNPY=python3
[ -x ".venv/bin/python" ] && RUNPY=".venv/bin/python"
[ -x "env/bin/python" ] && RUNPY="env/bin/python"
[ -x ".venv/Scripts/python.exe" ] && RUNPY=".venv/Scripts/python.exe"

echo "[1/5] Python : $RUNPY"
"$RUNPY" --version || { echo "   [ERREUR] Python introuvable. Lance install.sh."; exit 1; }
echo

echo "[2/5] Driver NVIDIA..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu \
             --format=csv,noheader | sed 's/^/   /'
else
  echo "   [INFO] nvidia-smi introuvable (pas de GPU NVIDIA, ou drivers absents)."
fi
echo

echo "[3/5] PyTorch / GPU / reglages conseilles..."
echo
"$RUNPY" _hw_check.py; HW=$?
echo
case "$HW" in
  1) echo "   [ERREUR] PyTorch absent -> lance install.sh."; exit 1 ;;
  3) echo "   [BLOQUANT] torch ne supporte pas cette carte (voir le correctif ci-dessus)."
     echo "   L'app planterait a la premiere allocation CUDA. Arret."; exit 3 ;;
  2) echo "   [AVERT] Mode CPU: n'utilise que la passe ESRGAN (denoise = 0)." ;;
esac

# L'import du pipeline ne suffit PAS: avec transformers < 4.51 il reussit, et
# c'est le chargement du modele qui casse plus tard sur "module transformers has
# no attribute Qwen3Model". On verifie donc aussi les versions minimales.
echo "[4/5] diffusers / versions..."
"$RUNPY" -c "from diffusers import ZImageImg2ImgPipeline; print('    ZImageImg2ImgPipeline OK')" 2>/dev/null \
  || echo "   [ATTENTION] ZImageImg2ImgPipeline indisponible -> lance install.sh / update.sh."
if ! "$RUNPY" -c "import app,sys;p=app.check_env();print('    versions deps OK') if not p else [sys.stderr.write('    [BLOQUANT] %s: %s installe, %s+ requis\n' % (n,g or 'absent','.'.join(map(str,m)))) for n,g,m in p] and sys.exit(4)"; then
  echo
  echo "   Z-Image ne pourra PAS se charger avec ces versions."
  echo "   Correctif: ./install.sh  (cree un .venv qui herite de ton torch mais"
  echo "   isole les deps de crispz, sans toucher au Python global)."
  exit 4
fi
echo

# MEME resolution de ESRGAN_DIR que run.sh, sinon le diagnostic annonce
# "0 modele" alors que l'app en trouvera ailleurs. Exportee -> run.sh hérite.
if [ -z "${ESRGAN_DIR:-}" ]; then
  export ESRGAN_DIR="$(pwd)/upscale_models"
fi
echo "[5/5] Modeles ESRGAN..."
"$RUNPY" -c "import app;m=app.list_esrgan_models();print('    %-4d modele(s)  %s' % (len(m), app.ESRGAN_DIR));[print('      - '+x) for x in m[:6]];print('      ...') if len(m)>6 else None" 2>/dev/null \
  || echo "   [INFO] impossible de lire la liste (deps manquantes ? lance install.sh)."
echo

# Port fixe: evite que Gradio parte sur 7861+ quand une instance precedente n'a
# pas encore libere le port.
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
if [ "$EXPOSE" = "1" ]; then
  echo "----------------------------------------------------"
  echo " [SECURITE] Exposition reseau demandee (--lan)."
  echo " crispz n'a AUCUNE authentification: l'UI accepte un dossier source et"
  echo " un dossier de sortie arbitraires, donc lecture/ecriture sur cette"
  echo " machine. N'expose que sur un reseau de confiance."
  echo "----------------------------------------------------"
  export GRADIO_SERVER_NAME=0.0.0.0
  echo
fi

if [ "$NORUN" = "1" ]; then
  echo "===================================================="
  echo "   Diagnostic termine (--no-run: app non lancee)."
  echo "===================================================="
  exit 0
fi
echo "===================================================="
echo "   Checks OK. Lancement de crispz..."
echo "===================================================="
sleep 2
# Le diagnostic materiel vient d'etre fait: run.sh n'a pas a le refaire.
export CZ_SKIP_HWCHECK=1
exec ./run.sh "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
