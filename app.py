"""
crispz - Z-Image upscaler + detailer (standalone, sans ComfyUI / SwarmUI)

Pipeline en deux etages:
  1. Real-ESRGAN (charge via spandrel) -> agrandissement reel des pixels, avec tiling.
  2. Z-Image Turbo en img2img (diffusers, BF16) -> passe de raffinement a bas denoise
     qui reinjecte du detail sans changer la composition.

Pre-requis cote machine (RTX 5090, PyTorch 2.7 / CUDA 12.8 deja installes):
  pip install -r requirements.txt
  (ne pas reinstaller torch, garder ton build cu128)

Lancer:
  python app.py
"""

import os
import sys
import gc
import io
import base64
import glob
import time
import uuid
import threading
import datetime
import numpy as np
import torch
from PIL import Image
import gradio as gr

# Defauts d'UI / CLI: reglages de reference (voir README)
DEFAULT_MODEL = "4x-ClearRealityV1_Soft.safetensors"
DEFAULT_FACTOR = 2.0
DEFAULT_DENOISE = 0.30
DEFAULT_STEPS = 12
DEFAULT_TILE = 760
DEFAULT_OVERLAP = 32
# Tiling de la passe diffusion Z-Image (4K+). 0 = image entiere (defaut, pas de
# regression). >0 = decoupe en tuiles de cette taille (arrondie a un multiple de 16).
DEFAULT_REFINE_TILE = 0
DEFAULT_REFINE_OVERLAP = 64

# Garde-fou 4K: au-dela de ce cote (px), un refine "whole image" (refine_tile = Auto) est
# auto-tuile. Un whole-image 4K active le slicing (lent) et risque le spill VRAM; tuiler
# est plus rapide ET plus sur.
AUTO_REFINE_TILE_ABOVE = 1664
# Taille de la tuile de cet auto-tuilage. "auto" = choisie par _pick_refine_tile pour
# MINIMISER la surface tuilee (n tuiles x tuile^2), qui est le vrai cout de la passe.
# Mesure (RTX 5090, sortie 4096x4096, denoise 0.40, overlap 64): le cout par pixel est
# PLAT de 768 a 1024 (1.78 / 1.83 / 1.79 us/px) et ne grimpe qu'au-dela (2.41 a 1536,
# 3.00 a 2048) -> le temps suit la surface couverte, pas la taille de la tuile. Or a 1024
# la grille deborde: pas de 960 sur 4096 -> la derniere tuile est rabattue et recouvre la
# precedente sur 832px au lieu de 64, soit 1.56x la surface de l'image. A 896 le pas tombe
# juste (1.20x) -> 36.7s au lieu de 46.9s, a nombre de tuiles (25) et de coutures (8)
# IDENTIQUE. Mettre un entier ici fige la taille (ex. 1024 = comportement d'avant).
AUTO_REFINE_TILE = "auto"
AUTO_REFINE_TILE_MIN = 768      # en dessous: plus de coutures, moins de contexte par tuile
AUTO_REFINE_TILE_MAX = 1024     # au-dessus: l'attention devient superlineaire

# Prompt utilise pour le refine TUILE. Le prompt global decrit TOUTE la composition (pas
# la tuile) -> le passer a chaque tuile pousse la diffusion a recreer le sujet dans des
# tuiles qui ne sont que du fond (duplications). Par defaut on passe donc un prompt VIDE:
# chaque tuile se contente d'affiner le detail local.
#   "" (defaut) = prompt vide par tuile
#   "global" / "scene" = reutilise le prompt de la scene
#   tout autre texte = prompt generique par tuile (ex. "high detail, sharp")
REFINE_TILE_PROMPT = ""
# Plafond de denoise pour le refine TUILE (filet a fort denoise). 0 = pas de plafond.
# Le refine whole-image garde le denoise demande: une seule passe, aucune duplication.
REFINE_TILE_DENOISE_CAP = 0.40

# Choix du menu "Diffusion tile size". 0 = Auto (image entiere sous
# AUTO_REFINE_TILE_ABOVE, puis tuilage a la taille calculee).
REFINE_TILE_CHOICES = [("Auto", 0)] + [(str(t), t) for t in
                                       (512, 640, 768, 896, 1024, 1280, 1536, 2048)]
if DEFAULT_REFINE_TILE not in [v for _, v in REFINE_TILE_CHOICES]:
    REFINE_TILE_CHOICES.append((str(DEFAULT_REFINE_TILE), DEFAULT_REFINE_TILE))
    REFINE_TILE_CHOICES.sort(key=lambda c: c[1])
DEFAULT_SAVE_MODE = "display"        # display | local | alongside | custom
DEFAULT_OUTPUT_DIR = "out"
DEFAULT_OUTPUT_FORMAT = "png"        # png | webp | jpg
SUPPORTED_FORMATS = ("png", "webp", "jpg")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Sampler de la passe de raffinement. Z-Image est un modele flow-matching: son
# scheduler natif est FlowMatchEuler. unipc converge parfois en moins de steps,
# lcm est utile a tres peu de steps. Le schedule re-mappe les sigmas par-dessus.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULE = "sgm_uniform"      # sgm_uniform = natif Z-Image
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}   # sgm_uniform -> aucun flag

# Au-dela de ce cote (px) on active l'attention slicing sur la passe de diffusion
# (image entiere 2K+ -> evite le spill VRAM). En-dessous (tuiles 1024) le slicing
# est desactive: attention SDPA native = plus rapide.
DEFAULT_SLICE_ABOVE = 1664

# Presets "cas d'usage" -> reglages auto. Seules les cles presentes sont appliquees,
# le reste est laisse tel quel. Utilise par l'UI (_apply_preset) et la CLI (--preset).
PRESETS = {
    "Custom": {},
    "Photo (balanced)":    {"factor": 2.0, "denoise": 0.30, "steps": 12, "refine_tile": 0, "cpu_offload": "none"},
    "Subtle (clean-up)":   {"factor": 2.0, "denoise": 0.12, "steps": 16, "refine_tile": 0},
    "Detailed (creative)": {"factor": 2.0, "denoise": 0.40, "steps": 16},
    "Portrait (faces)":    {"factor": 2.0, "denoise": 0.22, "steps": 14},
    "4K (tiled)":          {"factor": 4.0, "denoise": 0.30, "steps": 12, "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "model"},
    "Low VRAM (8-12GB)":   {"denoise": 0.30, "steps": 12, "tile": 512, "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "sequential"},
}
# param interne -> flag CLI, pour appliquer un preset sans ecraser un flag explicite.
PRESET_FLAGMAP = {
    "factor": "--factor", "denoise": "--denoise", "steps": "--steps", "tile": "--tile",
    "overlap": "--overlap", "refine_tile": "--refine-tile", "refine_overlap": "--refine-overlap",
    "cpu_offload": "--cpu-offload",
}

# ----------------------------------------------------------------------------
# Config (persistance dans preferences.json a cote de app.py)
# Ordre de priorite pour ESRGAN_DIR et BASE_REPO:
#   1) variable d'environnement (ESRGAN_DIR / ZIMAGE_MODEL)
#   2) preferences.json
#   3) defaut: ./upscale_models  et  Tongyi-MAI/Z-Image-Turbo
# ----------------------------------------------------------------------------
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(HERE, "preferences.json")
DEFAULT_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_ESRGAN_DIR = os.path.join(HERE, "upscale_models")


def _load_prefs_raw():
    if not os.path.isfile(PREFS_PATH):
        return {}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_prefs_keys(updates):
    """Met a jour quelques cles dans preferences.json, garde le reste intact."""
    data = _load_prefs_raw()
    data.update(updates)
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


_prefs = _load_prefs_raw()
BASE_REPO = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ESRGAN_DIR = os.environ.get("ESRGAN_DIR") or _prefs.get("esrgan_dir") or DEFAULT_ESRGAN_DIR
# Override du transformer Z-Image: fichier .safetensors / .gguf single-file, ou
# repo/dossier diffusers. Vide = transformer du repo de base. Le VAE, l'encodeur
# de texte et le tokenizer viennent TOUJOURS du repo de base.
ZIMAGE_TRANSFORMER = (os.environ.get("ZIMAGE_TRANSFORMER")
                      or _prefs.get("zimage_transformer") or "").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or _prefs.get("sampler") or DEFAULT_SAMPLER).strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = DEFAULT_SAMPLER
SCHEDULE = (os.environ.get("ZIMAGE_SCHEDULE") or _prefs.get("schedule") or DEFAULT_SCHEDULE).strip().lower()
if SCHEDULE not in SCHEDULE_CHOICES:
    SCHEDULE = DEFAULT_SCHEDULE
# Sampler REELLEMENT utilise: peut differer de SAMPLER si le repli s'est
# declenche (_apply_sampler). C'est celui-la qui part dans les metadonnees.
SAMPLER_EFFECTIVE = SAMPLER
# Config natif du scheduler du modele, capture au 1er chargement -> base de
# construction des autres samplers (conserve les parametres flow/shift).
_BASE_SCHED_CONFIG = None

SLICE_ABOVE = int(_prefs.get("attention_slice_above", DEFAULT_SLICE_ABOVE))

# Motif de nommage des fichiers de sortie (meme principe que crispz-studio).
# Voir _format_filename pour la liste des placeholders. Le defaut historique de
# crispz etait "{name}_upscaled": le mettre ici en clair pour qu'il soit
# modifiable sans toucher au code.
DEFAULT_FILENAME_PATTERN = "{date}_{name}_{tag}_{w}x{h}{index}"
FILENAME_PATTERN = (os.environ.get("CRISPZ_FILENAME_PATTERN")
                    or _prefs.get("filename_pattern") or DEFAULT_FILENAME_PATTERN)

# Metadonnees embarquees dans les images sauvees (chunk PNG 'crispz' + sidecar
# .json, EXIF ImageDescription pour jpg/webp). Tracer quel modele / denoise /
# steps a produit un fichier est le minimum utile en traitement par lots.
WRITE_METADATA = bool(_prefs.get("write_metadata", True))
WRITE_SIDECAR = bool(_prefs.get("write_sidecar_json", False))

# Progression du chargement des modeles: le chargement de Z-Image peut prendre
# plusieurs minutes sur un disque lent, sans aucun signal. On rafraichit
# terminal + UI toutes les ~2 s avec le temps ecoule et la VRAM deja allouee.
LOAD_PROGRESS = bool(_prefs.get("load_progress", True))
_LOAD_TARGET_GB = float(_prefs.get("load_target_vram_gb", 13.0))
_LOAD_HEARTBEAT = float(_prefs.get("load_heartbeat_s", 2.0))
# Hook de progression UI (gradio gr.Progress). None hors UI (CLI / serveur).
_PROGRESS = None

# Caches process-wide pour ne pas recharger a chaque run
_PIPE = None
_LOADED_REPO = None  # repo associe au _PIPE actuel, sert a detecter un changement
_LOADED_OFFLOAD = None  # mode offload du _PIPE actuel, sert a detecter un changement
_LOADED_TRANSFORMER = None  # override transformer du _PIPE actuel
_ESRGAN_CACHE = {}

# Palier 2 (cohabitation VRAM, brief plugin Fooocus): offload CPU de la passe
# diffusion. none = tout en VRAM (defaut). model = decharge par sous-module
# (bon compromis). sequential = plus agressif, plus lent. N'est PAS de la quantif:
# les poids restent BF16, ils transitent juste RAM <-> GPU. Requiert accelerate.
OFFLOAD_MODE = "none"
OFFLOAD_CHOICES = ("none", "model", "sequential")

# Logs d'etape sur stderr (chargement modeles, etages, tuiles). Coupes par --quiet.
# stderr donc ne pollue pas le stdout de --print-output.
VERBOSE = True


def _log(msg):
    if VERBOSE:
        print(f"[crispz] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Progression (UI + terminal)
# ----------------------------------------------------------------------------
def _progress(frac, desc=""):
    """Pousse une progression vers l'UI si un gr.Progress a ete pose. No-op en CLI."""
    if _PROGRESS is None:
        return
    try:
        _PROGRESS(min(max(float(frac), 0.0), 1.0), desc=desc)
    except Exception:
        pass


def _fmt_load(label, elapsed, vram_gb):
    """Texte de progression de chargement (pur, testable). VRAM > 0 -> phase de
    chargement en memoire; sinon phase download / lecture disque."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """% honnete: base sur la VRAM allouee / cible une fois le chargement en memoire
    commence (plafonne a 0.95); pendant le download (VRAM ~ 0) petite barre temporelle."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Execute fn() (chargement bloquant) dans un thread et rafraichit terminal + UI
    toutes les ~2 s. Renvoie le resultat de fn (et releve son exception)."""
    if not LOAD_PROGRESS:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - re-levee dans le thread principal
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        th.join(timeout=_LOAD_HEARTBEAT)
        el = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
        line = _fmt_load(label, el, vram)
        if VERBOSE:
            sys.stderr.write("\r[crispz][load] " + line + "        ")
            sys.stderr.flush()
        _progress(_load_pct(el, vram), "Loading " + line)
        if not th.is_alive():
            break
    if VERBOSE:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def set_esrgan_dir(path):
    """Change le dossier ESRGAN. Invalide le cache (les noms peuvent collisionner entre dossiers)."""
    global ESRGAN_DIR, _ESRGAN_CACHE
    if path and path != ESRGAN_DIR:
        ESRGAN_DIR = path
        _ESRGAN_CACHE = {}


def set_zimage_model(repo_or_path):
    """Change le modele Z-Image (repo HF ou chemin local). Invalide le pipe si change."""
    global BASE_REPO, _PIPE, _LOADED_REPO
    if repo_or_path and repo_or_path != BASE_REPO:
        BASE_REPO = repo_or_path
        if _LOADED_REPO is not None and _LOADED_REPO != BASE_REPO:
            _PIPE = None
            _LOADED_REPO = None
            _log(f"pipeline invalidated (Z-Image model changed) -> will reload")


def set_zimage_transformer(path):
    """Change l'override du transformer Z-Image (single-file .safetensors/.gguf, ou
    repo/dossier diffusers). Vide = transformer du repo de base. Invalide le pipe."""
    global ZIMAGE_TRANSFORMER, _PIPE, _LOADED_TRANSFORMER
    path = (path or "").strip()
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        if _LOADED_TRANSFORMER is not None and _LOADED_TRANSFORMER != ZIMAGE_TRANSFORMER:
            _PIPE = None
            _LOADED_TRANSFORMER = None
            _log("pipeline invalidated (transformer changed) -> will reload")


def set_sampler(name):
    """Change le sampler et le re-applique au pipe en cache (pas de rechargement)."""
    global SAMPLER
    name = (name or DEFAULT_SAMPLER).strip().lower()
    if name not in SAMPLER_CHOICES:
        name = DEFAULT_SAMPLER
    if name != SAMPLER:
        SAMPLER = name
        _apply_sampler(_PIPE)
        _log(f"sampler -> {SAMPLER}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def set_schedule(name):
    """Change le schedule de sigmas et le re-applique au pipe en cache."""
    global SCHEDULE
    name = (name or DEFAULT_SCHEDULE).strip().lower()
    if name not in SCHEDULE_CHOICES:
        name = DEFAULT_SCHEDULE
    if name != SCHEDULE:
        SCHEDULE = name
        _apply_sampler(_PIPE)
        _log(f"schedule -> {SCHEDULE}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def set_offload_mode(mode):
    """Change le mode d'offload CPU de la passe diffusion. Invalide le pipe si
    change (les hooks d'offload sont poses au chargement) et libere la VRAM."""
    global OFFLOAD_MODE, _PIPE, _LOADED_OFFLOAD
    mode = mode if mode in OFFLOAD_CHOICES else "none"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        if _LOADED_OFFLOAD is not None and _LOADED_OFFLOAD != OFFLOAD_MODE:
            _PIPE = None
            _LOADED_OFFLOAD = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            _log(f"pipeline invalidated (offload -> {OFFLOAD_MODE}) -> will reload")


def free_vram():
    """Libere le pipe diffusion et rend la VRAM (palier 3: unload sur inactivite
    ou endpoint /unload). Le prochain run rechargera le pipe paresseusement."""
    global _PIPE, _LOADED_REPO, _LOADED_OFFLOAD, _LOADED_TRANSFORMER
    _PIPE = None
    _LOADED_REPO = None
    _LOADED_OFFLOAD = None
    _LOADED_TRANSFORMER = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def apply_preset_to_args(args, raw_argv):
    """Applique un preset aux champs de args qui n'ont PAS ete passes explicitement
    en CLI (un flag explicite gagne toujours sur le preset)."""
    preset = PRESETS.get(getattr(args, "preset", None) or "Custom") or {}
    raw = list(raw_argv or [])
    for key, val in preset.items():
        flag = PRESET_FLAGMAP[key]
        if not any(tok == flag or tok.startswith(flag + "=") for tok in raw):
            setattr(args, key, val)


# ----------------------------------------------------------------------------
# Etage 1 : Real-ESRGAN via spandrel
# ----------------------------------------------------------------------------
def list_esrgan_models():
    if not os.path.isdir(ESRGAN_DIR):
        return []
    return sorted(
        f for f in os.listdir(ESRGAN_DIR)
        if f.lower().endswith((".pth", ".safetensors"))
    )


def load_esrgan(model_name):
    if model_name in _ESRGAN_CACHE:
        return _ESRGAN_CACHE[model_name]
    from spandrel import ModelLoader, ImageModelDescriptor
    _log(f"loading ESRGAN model: {model_name} ...")
    path = os.path.join(ESRGAN_DIR, model_name)
    model = ModelLoader().load_from_file(path)
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"{model_name} is not a usable image SR model.")
    model = model.to(DEVICE).eval()
    _ESRGAN_CACHE[model_name] = model
    return model


def _pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _tensor_to_pil(t):
    arr = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def esrgan_upscale(img, model, tile, overlap):
    """Upscale ESRGAN avec tiling overlap-add et feather lineaire pour eviter les coutures."""
    scale = model.scale
    t = _pil_to_tensor(img)
    _, _, h, w = t.shape

    if tile <= 0 or (h <= tile and w <= tile):
        with torch.no_grad():
            out = model(t)
        return _tensor_to_pil(out)

    out_h, out_w = h * scale, w * scale
    acc = torch.zeros(1, 3, out_h, out_w, device=DEVICE)
    weight = torch.zeros(1, 1, out_h, out_w, device=DEVICE)
    step = tile - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y1, x1 = max(y2 - tile, 0), max(x2 - tile, 0)
            patch = t[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                up = model(patch)
            ph, pw = up.shape[2], up.shape[3]
            # masque feather: rampe lineaire sur la zone d'overlap
            mask = torch.ones(1, 1, ph, pw, device=DEVICE)
            f = overlap * scale
            if f > 0:
                ramp = torch.linspace(0, 1, int(f), device=DEVICE)
                if x1 > 0:
                    mask[:, :, :, :int(f)] *= ramp.view(1, 1, 1, -1)
                if x2 < w:
                    mask[:, :, :, -int(f):] *= ramp.flip(0).view(1, 1, 1, -1)
                if y1 > 0:
                    mask[:, :, :int(f), :] *= ramp.view(1, 1, -1, 1)
                if y2 < h:
                    mask[:, :, -int(f):, :] *= ramp.flip(0).view(1, 1, -1, 1)
            oy, ox = y1 * scale, x1 * scale
            acc[:, :, oy:oy + ph, ox:ox + pw] += up * mask
            weight[:, :, oy:oy + ph, ox:ox + pw] += mask

    out = acc / weight.clamp(min=1e-6)
    return _tensor_to_pil(out)


# ----------------------------------------------------------------------------
# Etage 2 : Z-Image img2img (diffusers, BF16)
# ----------------------------------------------------------------------------
# Versions minimales des deux paquets dont l'absence ne se voit qu'au CHARGEMENT
# du modele, pas a l'import: le message d'erreur natif est alors incomprehensible.
#   transformers < 4.51 -> "module transformers has no attribute Qwen3Model"
#     (Z-Image Turbo utilise un encodeur de texte Qwen3)
#   gradio < 4.44        -> composants et evenements manquants dans l'UI
MIN_VERSIONS = {"transformers": (4, 51), "gradio": (4, 44)}


def _version_tuple(s):
    """('4.57.6rc1') -> (4, 57, 6). Ignore tout suffixe non numerique."""
    out = []
    for part in str(s).split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def check_env():
    """Renvoie la liste des problemes de version bloquants (vide = tout va bien).
    Chaque entree: (paquet, version_installee_ou_None, version_minimale)."""
    problems = []
    for name, minv in MIN_VERSIONS.items():
        try:
            mod = __import__(name)
            got = _version_tuple(getattr(mod, "__version__", ""))
        except Exception:
            problems.append((name, None, minv))
            continue
        if not got or got < minv:
            problems.append((name, getattr(mod, "__version__", "?"), minv))
    return problems


def _env_problem_text(problems):
    lines = []
    for name, got, minv in problems:
        want = ".".join(str(x) for x in minv)
        lines.append(f"  - {name}: {got or 'absent'} installe, {want}+ requis")
    return "\n".join(lines)


def assert_env_ok():
    """Echoue TOT, avec un message actionnable, plutot que de laisser diffusers
    lever une AttributeError opaque au milieu du chargement du modele."""
    problems = [p for p in check_env() if p[0] == "transformers"]
    if not problems:
        return
    raise RuntimeError(
        "Environnement Python incompatible avec Z-Image:\n"
        + _env_problem_text(problems)
        + "\n\nCause: Z-Image Turbo utilise un encodeur de texte Qwen3, absent de\n"
          "transformers < 4.51 (erreur native: 'module transformers has no attribute\n"
          "Qwen3Model').\n"
          "Correctif: lance install.bat (Windows) ou ./install.sh (Unix). Ils creent\n"
          "un .venv qui herite de ton torch mais isole les deps de crispz, sans\n"
          "toucher au Python global partage avec tes autres applications."
    )


def _is_single_file(path):
    """Chemin vers un checkpoint autonome (par opposition a un repo/dossier diffusers)."""
    return bool(path) and path.lower().endswith((".safetensors", ".sft", ".ckpt", ".pt", ".gguf"))


def _load_transformer():
    """Charge UNIQUEMENT le transformer d'override (ZIMAGE_TRANSFORMER):
      - .gguf quantifie   -> from_single_file + GGUFQuantizationConfig
      - .safetensors      -> from_single_file
      - repo / dossier    -> sous-dossier 'transformer'
    Dans tous les cas l'architecture est lue depuis le repo de base, sinon
    from_single_file ne connait pas la structure du modele.
    Renvoie None si aucun override n'est configure."""
    if not ZIMAGE_TRANSFORMER:
        return None
    from diffusers import ZImageTransformer2DModel
    if _is_single_file(ZIMAGE_TRANSFORMER):
        name = os.path.basename(ZIMAGE_TRANSFORMER)
        if ZIMAGE_TRANSFORMER.lower().endswith(".gguf"):
            from diffusers import GGUFQuantizationConfig
            _log(f"loading Z-Image transformer (GGUF, quantized): {name} ...")
            return _load_monitor(
                f"transformer {name} (GGUF)",
                lambda: ZImageTransformer2DModel.from_single_file(
                    ZIMAGE_TRANSFORMER,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
                    config=BASE_REPO, subfolder="transformer", torch_dtype=DTYPE))
        _log(f"loading Z-Image transformer (single-file): {name} ...")
        return _load_monitor(
            f"transformer {name}",
            lambda: ZImageTransformer2DModel.from_single_file(
                ZIMAGE_TRANSFORMER, config=BASE_REPO, subfolder="transformer",
                torch_dtype=DTYPE))
    _log(f"loading Z-Image transformer (repo subfolder): {ZIMAGE_TRANSFORMER} ...")
    return _load_monitor(
        f"transformer {ZIMAGE_TRANSFORMER}",
        lambda: ZImageTransformer2DModel.from_pretrained(
            ZIMAGE_TRANSFORMER, subfolder="transformer", torch_dtype=DTYPE))


def load_pipe():
    global _PIPE, _LOADED_REPO, _LOADED_OFFLOAD, _LOADED_TRANSFORMER, _BASE_SCHED_CONFIG
    if (_PIPE is not None and _LOADED_REPO == BASE_REPO
            and _LOADED_OFFLOAD == OFFLOAD_MODE
            and _LOADED_TRANSFORMER == ZIMAGE_TRANSFORMER):
        _log("Z-Image pipeline: reusing cached (no reload)")
        return _PIPE
    # Verifie AVANT de telecharger / lire des gigaoctets pour rien.
    assert_env_ok()
    _log(f"loading Z-Image pipeline: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16) ... "
         "first time downloads from HF, then cached")
    _t = time.time()
    from diffusers import ZImageImg2ImgPipeline
    kwargs = {}
    transformer = _load_transformer()
    if transformer is not None:
        kwargs["transformer"] = transformer
    pipe = _load_monitor(
        f"Z-Image base {BASE_REPO}",
        lambda: ZImageImg2ImgPipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE, **kwargs))
    # Config natif du scheduler (flow-matching) -> base pour construire les autres
    # samplers sans perdre les parametres shift/flow du modele.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # Le VAE Z-Image a force_upcast=True -> encode/decode en float32. Sur Blackwell
    # (RTX 50xx: pas de tensor cores fp32) le VAE img2img devient ~50x plus lent (et
    # peut faire deborder la VRAM). On le garde en bf16 + tiling (comme ComfyUI).
    try:
        pipe.vae.config.force_upcast = False
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    # L'attention slicing n'est PLUS pose ici: il est decide par passe, selon la
    # taille reellement traitee (_set_slicing). Sur une tuile 1024 le slicing
    # coute du temps pour rien; il n'est utile qu'en image entiere 2K+.
    # Offload CPU: enable_*_cpu_offload gere lui-meme le placement device, donc
    # NE PAS faire .to(cuda) dans ce cas. Hors CUDA l'offload n'a pas de sens.
    if DEVICE == "cuda" and OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    _apply_sampler(pipe)
    _PIPE = pipe
    _LOADED_REPO = BASE_REPO
    _LOADED_OFFLOAD = OFFLOAD_MODE
    _LOADED_TRANSFORMER = ZIMAGE_TRANSFORMER
    _log(f"Z-Image pipeline ready in {time.time() - _t:.1f}s")
    return pipe


def round_to_multiple(x, m=16):
    """Alignement des dimensions. Le pipeline Z-Image exige des cotes multiples de
    vae_scale_factor * 2 = 16 (le VAE divise par 8, le transformer patchifie par 2);
    en-dessous il leve "Height/Width must be divisible by 16"."""
    return max(m, int(round(x / m) * m))


def _pick_refine_tile(w, h, overlap):
    """Tuile qui minimise la surface tuilee pour couvrir w x h (= le cout reel de la passe).

    A surface egale on garde la PLUS GRANDE tuile: moins de coutures et plus de contexte
    par tuile. Un entier dans AUTO_REFINE_TILE court-circuite le calcul (taille figee)."""
    if str(AUTO_REFINE_TILE).strip().lower() not in ("auto", "", "0"):
        try:
            return round_to_multiple(int(AUTO_REFINE_TILE), 32)
        except (TypeError, ValueError):
            _log(f"AUTO_REFINE_TILE={AUTO_REFINE_TILE!r} invalide (attendu 'auto' ou un "
                 "entier) -> calcul automatique")
    lo = max(256, AUTO_REFINE_TILE_MIN)
    hi = max(lo, AUTO_REFINE_TILE_MAX)
    ov = max(0, int(overlap))
    cands = []
    for t in range(lo, hi + 1, 32):
        step = max(16, t - ov)
        n = len(range(0, max(1, int(w)), step)) * len(range(0, max(1, int(h)), step))
        cands.append((n * t * t, -t, t))       # surface mini, puis plus grande tuile
    return min(cands)[2]


def _tile_prompt(scene_prompt):
    """Prompt a utiliser par tuile (vide par defaut, anti-duplication)."""
    if str(REFINE_TILE_PROMPT).strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return REFINE_TILE_PROMPT


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


def _set_slicing(pipe, longest_side):
    """Active/desactive l'attention slicing selon le plus grand cote a traiter.
    Appele avant CHAQUE passe de diffusion (image entiere ou tuile): sur une tuile
    de 1024 le slicing ralentit sans rien apporter, il n'est utile qu'en 2K+."""
    try:
        if int(longest_side) > SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _scheduler_accepts_sigmas(sched):
    """Le pipeline Z-Image appelle set_timesteps(..., sigmas=...). Un scheduler dont
    set_timesteps n'accepte pas `sigmas` planterait a la generation.
    NECESSAIRE MAIS PAS SUFFISANT: voir _scheduler_works."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _scheduler_works(sched):
    """Sonde FONCTIONNELLE: appelle reellement set_timesteps comme le fait le
    pipeline Z-Image, sur une copie jetable.

    Pourquoi une sonde et pas une inspection de signature: UniPC accepte bien un
    argument `sigmas`, puis calcule `flow_shift * sigmas`. Le pipeline Z-Image
    passe une LISTE Python -> 'can't multiply sequence by non-int of type float',
    au milieu de la generation, apres des minutes de chargement. La signature
    seule ne peut pas voir ca; l'essayer, si."""
    import copy
    try:
        probe = copy.deepcopy(sched)          # ne pas polluer l'etat du vrai scheduler
    except Exception:
        return True                            # pas copiable -> on ne bloque pas
    # Meme forme d'appel que retrieve_timesteps() dans le pipeline Z-Image:
    # une liste de sigmas decroissants, sur cpu.
    sigmas = [1.0 - i / 4.0 for i in range(4)]
    try:
        probe.set_timesteps(sigmas=sigmas, device="cpu")
        return True
    except Exception as e:
        _log(f"scheduler probe failed ({type(e).__name__}: {e})")
        return False


def _build_scheduler(sampler, schedule, config):
    """Construit le scheduler choisi (sampler x schedule) depuis le config natif du
    modele. schedule = remapping des sigmas (use_*_sigmas) applique par-dessus."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    kw = {}
    flag = _SCHEDULE_FLAG.get((schedule or "").lower())
    if flag:
        kw[flag] = True
    name = (sampler or DEFAULT_SAMPLER).lower()
    if name == "unipc":
        from diffusers import UniPCMultistepScheduler
        try:
            return UniPCMultistepScheduler.from_config(config, use_flow_sigmas=True, **kw)
        except Exception:
            return UniPCMultistepScheduler.from_config(config, **kw)
    if name == "lcm":
        # LCM flow-matching. Repli sur Euler si la version de diffusers ne l'expose pas.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Pose le scheduler courant (SAMPLER x SCHEDULE) sur un pipe. Verifie la
    compatibilite (signature ET sonde fonctionnelle) et retombe sur
    Euler/sgm_uniform si KO, pour ne jamais planter au moment de la generation.

    Met a jour SAMPLER_EFFECTIVE: si le repli s'est declenche, les metadonnees et
    l'UI doivent dire ce qui a REELLEMENT tourne, pas ce qui a ete demande."""
    global SAMPLER_EFFECTIVE
    SAMPLER_EFFECTIVE = SAMPLER
    if pipe is None or _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} n'accepte pas l'argument sigmas")
        if not _scheduler_works(sched):
            raise ValueError(f"{type(sched).__name__} rejette les sigmas de Z-Image "
                             "(liste Python) a l'execution")
        pipe.scheduler = sched
        _log(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e})")
        _log("  -> fallback euler/sgm_uniform (le rendu se fera avec euler)")
        SAMPLER_EFFECTIVE = "euler"
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """Passe Z-Image img2img sur l'image entiere (ou une tuile). Le slicing est pose
    selon la taille reellement traitee. L'entree est ALIGNEE /16 avant diffusion --
    le pipeline refuse une dimension non divisible par 16 -- puis le resultat est
    ramene a la taille d'origine (contrat des appelants preserve)."""
    _set_slicing(pipe, max(image.size))
    orig_size = image.size
    w = round_to_multiple(image.width)
    h = round_to_multiple(image.height)
    if (w, h) != image.size:
        _log(f"refine: input {image.size[0]}x{image.size[1]} not /16 -> resized {w}x{h}")
        image = image.resize((w, h), Image.LANCZOS)
    out = pipe(
        prompt=prompt or "",
        image=image,
        strength=float(denoise),
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        generator=_make_generator(seed),
    ).images[0]
    if out.size != orig_size:
        out = out.resize(orig_size, Image.LANCZOS)
    return out


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """Masque (th, tw, 1) a rampe lineaire sur les bords qui jouxtent une autre tuile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # Une seule tuile = image entiere -> pas de duplication possible: denoise demande.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)

    # Anti-duplication 1: prompt vide par tuile (le prompt global decrit toute la compo).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: prompt vide par tuile (anti-duplication; regle "
             "REFINE_TILE_PROMPT).")
    # Anti-duplication 2 (filet): a fort denoise chaque tuile peut encore deriver.
    denoise = float(denoise)
    if REFINE_TILE_DENOISE_CAP > 0 and denoise > REFINE_TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > plafond {REFINE_TILE_DENOISE_CAP:.2f}"
             f" -> reduit a {REFINE_TILE_DENOISE_CAP:.2f} (regle REFINE_TILE_DENOISE_CAP).")
        denoise = REFINE_TILE_DENOISE_CAP

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    i = 0
    for y in ys:
        for x in xs:
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _log(f"  tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration : process_one, save, batch, run (UI/CLI commun)
# ----------------------------------------------------------------------------
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP):
    """Pipeline complet sur une PIL Image, renvoie (refined_image, timings_dict)."""
    timings = {}
    image = image.convert("RGB")
    w0, h0 = image.size

    # Etage 1 : ESRGAN
    t0 = time.time()
    model = load_esrgan(esrgan_model)
    _log(f"stage 1/2 ESRGAN upscale: {w0}x{h0} (tile {int(tile)}) ...")
    upscaled = esrgan_upscale(image, model, int(tile), int(overlap))
    target_w = round_to_multiple(w0 * factor)
    target_h = round_to_multiple(h0 * factor)
    upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
    timings["esrgan"] = time.time() - t0
    _log(f"stage 1/2 done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")

    if denoise <= 0.001:
        timings["refine"] = 0.0
        _log("stage 2/2 skipped (denoise = 0, ESRGAN only)")
        return upscaled, timings

    # Etage 2 : Z-Image img2img (image entiere, ou tuiles si refine_tile > 0)
    t0 = time.time()
    pipe = load_pipe()
    rt = int(refine_tile)
    # Auto (rt = 0): whole image tant qu'on reste sous AUTO_REFINE_TILE_ABOVE, au-dela on
    # tuile a la taille calculee -- plus rapide qu'un whole-image slice, et sans spill.
    if rt <= 0 and max(target_w, target_h) > AUTO_REFINE_TILE_ABOVE:
        rt = _pick_refine_tile(target_w, target_h, int(refine_overlap) or 64)
        _log(f"stage 2/2 refine: {target_w}x{target_h} > {AUTO_REFINE_TILE_ABOVE}px -> "
             f"auto-tiling (tile {rt}) pour eviter le pic VRAM")
    if rt > 0:
        refined = _refine_tiled(pipe, upscaled, denoise, steps, prompt, seed,
                                rt, int(refine_overlap))
    else:
        _log(f"stage 2/2 Z-Image refine: whole image {target_w}x{target_h}, "
             f"denoise {float(denoise):.2f}, {int(steps)} steps ...")
        refined = _refine_whole(pipe, upscaled, denoise, steps, prompt, seed)
    timings["refine"] = time.time() - t0

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _log(f"stage 2/2 done in {timings['refine']:.1f}s | total "
         f"{timings['esrgan'] + timings['refine']:.1f}s")
    return refined, timings


# Chemin du dernier fichier ecrit par run(), ou None (mode display / echec).
# Sert a l'UI: renvoyer le FICHIER a gradio plutot que l'image PIL (voir
# _ui_result_path pour le pourquoi).
_LAST_SAVED = None


def _preview_path(img, output_format, meta=None):
    """Ecrit l'image dans un fichier temporaire au format demande, avec ses
    metadonnees, et renvoie le chemin. Utilise quand rien n'a ete sauve
    (save_mode=display) mais que l'UI doit quand meme proposer un telechargement
    au bon format."""
    import tempfile
    ext = (output_format or DEFAULT_OUTPUT_FORMAT).lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        ext = "png"
    d = os.path.join(tempfile.gettempdir(), "crispz_preview")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"crispz_{uuid.uuid4().hex[:12]}.{ext}")
    save_image(img, path, ext, meta=meta)
    return path


def _now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _format_filename(source_path=None, tag="upscaled", seed=None, size=None,
                     esrgan_model=None, factor=None, denoise=None, index=0):
    """Nom de fichier (sans extension) depuis FILENAME_PATTERN.

    Placeholders disponibles:
        {date}    20260728-231205
        {name}    nom du fichier source, sans extension (sinon 'image')
        {tag}     'upscaled'
        {seed}    valeur du seed, ou 'rand' si -1
        {w} {h}   dimensions de SORTIE
        {model}   modele ESRGAN, sans extension
        {factor}  facteur d'agrandissement, ex '2x'
        {denoise} denoise, ex 'd030'
        {index}   '' ou '_2', '_3'... (batch)

    Un motif invalide ne fait pas echouer un rendu de plusieurs minutes: on
    retombe sur un nom sur, et le probleme est signale une fois."""
    base = (os.path.splitext(os.path.basename(source_path))[0]
            if source_path else "image")
    seed_s = str(int(seed)) if (seed is not None and int(seed) >= 0) else "rand"
    w, h = (int(size[0]), int(size[1])) if size else (0, 0)
    fields = {
        "date": _now_stamp(),
        "name": base,
        "tag": tag or "upscaled",
        "seed": seed_s,
        "w": w,
        "h": h,
        "model": os.path.splitext(str(esrgan_model))[0] if esrgan_model else "esrgan",
        "factor": f"{float(factor):g}x" if factor is not None else "",
        "denoise": f"d{int(round(float(denoise) * 100)):03d}" if denoise is not None else "",
        "index": f"_{int(index)}" if index else "",
    }
    try:
        name = FILENAME_PATTERN.format(**fields)
    except Exception as e:
        _log(f"filename_pattern invalide ({e}); repli sur '{{name}}_{{tag}}'")
        name = f"{base}_{tag or 'upscaled'}"
    # Un motif peut produire des separateurs de chemin ou des caracteres
    # interdits par le systeme de fichiers: on ne garde que du sur.
    name = "".join(c for c in name if c.isalnum() or c in "-_.").strip("_. ")
    return name or "image"


def _unique_path(path):
    """Evite l'ecrasement silencieux: ajoute _2, _3... si le fichier existe deja.
    Sans ca, relancer sur la meme source (autres reglages, ou mode 'alongside' sur
    un dossier deja traite) remplacait le resultat precedent sans prevenir."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def build_output_path(source_path, save_mode, output_dir, output_format,
                      tag="upscaled", seed=None, size=None, esrgan_model=None,
                      factor=None, denoise=None, index=0):
    """Decide ou ecrire l'image upscale. Renvoie un chemin absolu ou None (display only).
    Le nom suit FILENAME_PATTERN (cf. _format_filename) et le chemin est rendu
    UNIQUE: on n'ecrase jamais un fichier existant.
    - display    : pas de sauvegarde (None)
    - local      : output_dir relatif au projet
    - alongside  : dans le dossier de source_path
    - custom     : output_dir tel quel (peut etre absolu)
    """
    if save_mode == "display":
        return None

    ext = output_format.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        ext = "png"

    fname = _format_filename(source_path, tag=tag, seed=seed, size=size,
                             esrgan_model=esrgan_model, factor=factor,
                             denoise=denoise, index=index) + f".{ext}"

    if save_mode == "alongside":
        if not source_path:
            raise ValueError("save_mode=alongside requires a source path (CLI or batch folder).")
        return _unique_path(os.path.join(os.path.dirname(os.path.abspath(source_path)), fname))

    if save_mode == "custom":
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
    else:  # local
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
        if not os.path.isabs(target_dir):
            target_dir = os.path.join(HERE, target_dir)

    os.makedirs(target_dir, exist_ok=True)
    return _unique_path(os.path.join(target_dir, fname))


def build_meta(source_path, esrgan_model, factor, denoise, steps, prompt, seed,
               tile, overlap, refine_tile, refine_overlap, size=None, timings=None):
    """Metadonnees de tracabilite d'un rendu. Volontairement plat et lisible:
    en traitement par lots, retrouver quel modele / denoise / steps a produit un
    fichier est le minimum utile."""
    meta = {
        "app": "crispz",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": os.path.abspath(source_path) if source_path else None,
        "esrgan_model": esrgan_model,
        "factor": float(factor),
        "denoise": float(denoise),
        "steps": int(steps),
        "prompt": prompt or "",
        "seed": int(seed),
        "esrgan_tile": int(tile),
        "esrgan_overlap": int(overlap),
        "refine_tile": int(refine_tile),
        "refine_overlap": int(refine_overlap),
        "zimage_model": BASE_REPO,
        "zimage_transformer": ZIMAGE_TRANSFORMER or None,
        "sampler": SAMPLER_EFFECTIVE,
        "sampler_requested": SAMPLER if SAMPLER != SAMPLER_EFFECTIVE else None,
        "schedule": SCHEDULE,
        "cpu_offload": OFFLOAD_MODE,
    }
    if size:
        meta["width"], meta["height"] = int(size[0]), int(size[1])
    if timings:
        meta["esrgan_s"] = round(timings.get("esrgan", 0.0), 2)
        meta["refine_s"] = round(timings.get("refine", 0.0), 2)
    return meta


def _exif_bytes(meta):
    """EXIF (ImageDescription = 0x010e) contenant le JSON des metadonnees, pour jpg/webp."""
    try:
        exif = Image.Exif()
        exif[0x010E] = json.dumps(meta, ensure_ascii=False)
        return exif.tobytes()
    except Exception:
        return None


def save_image(img, dst_path, output_format, meta=None):
    """Sauve avec le bon format Pillow. Si meta (dict) et WRITE_METADATA: embarque
    les reglages dans un chunk PNG 'crispz', ou en EXIF ImageDescription pour
    jpg/webp. Un sidecar .json est ecrit en plus si WRITE_SIDECAR."""
    fmt = output_format.lower().lstrip(".")
    meta = meta if (meta and WRITE_METADATA) else None
    if fmt in ("jpg", "jpeg"):
        kw = {"quality": 95}
        eb = _exif_bytes(meta) if meta else None
        if eb:
            kw["exif"] = eb
        img.convert("RGB").save(dst_path, "JPEG", **kw)
    elif fmt == "webp":
        kw = {"quality": 95, "method": 6}
        eb = _exif_bytes(meta) if meta else None
        if eb:
            kw["exif"] = eb
        img.save(dst_path, "WEBP", **kw)
    else:
        pnginfo = None
        if meta:
            try:
                from PIL import PngImagePlugin
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("crispz", json.dumps(meta, ensure_ascii=False))
            except Exception:
                pnginfo = None
        img.save(dst_path, "PNG", pnginfo=pnginfo)
    if meta and WRITE_SIDECAR:
        try:
            with open(dst_path + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _log(f"sidecar json failed: {e}")


def read_metadata(path):
    """Relit les metadonnees crispz d'une image (chunk PNG 'crispz', EXIF sinon).
    Renvoie un dict, ou None si l'image n'en porte pas."""
    try:
        with Image.open(path) as im:
            raw = (im.info or {}).get("crispz")
            if not raw:
                try:
                    raw = im.getexif().get(0x010E)
                except Exception:
                    raw = None
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _list_folder_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def _format_timings(t, src_path=None, dst_path=None):
    total = t.get("esrgan", 0.0) + t.get("refine", 0.0)
    parts = []
    if src_path:
        parts.append(f"Source: `{src_path}`")
    parts.append(f"ESRGAN: **{t.get('esrgan', 0.0):.1f}s**  |  Z-Image refine: **{t.get('refine', 0.0):.1f}s**  |  Total: **{total:.1f}s**")
    if dst_path:
        parts.append(f"Saved: `{dst_path}`")
    return "  \n".join(parts)


def _reset_vram_peak():
    """Remet a zero le compteur de pic VRAM avant un traitement."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _report_vram():
    """Affiche le pic VRAM du run sur stderr. No-op hors CUDA.

    Format stable et parsable: la ligne commence par '[VRAM]'.
    alloue  = pic des tensors PyTorch (max_memory_allocated).
    reserve = pic du cache allocateur PyTorch (max_memory_reserved), plus proche
              de ce que nvidia-smi voit pour ce process.
    """
    if DEVICE != "cuda":
        print("[VRAM] pas de GPU CUDA, mesure ignoree.", file=sys.stderr)
        return
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[VRAM] pic alloue: {alloc:.2f} Go | pic reserve: {reserved:.2f} Go",
          file=sys.stderr)


def run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
        tile, overlap, save_mode=DEFAULT_SAVE_MODE, output_dir=DEFAULT_OUTPUT_DIR,
        output_format=DEFAULT_OUTPUT_FORMAT, time_log_path=None, print_output=False,
        refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP):
    """Point d'entree commun UI / CLI.
    Renvoie (last_result_PIL, last_source_PIL, report_markdown).
    - Si source_folder est un dossier existant -> batch sur ses images.
    - Sinon, image est utilisee (PIL ou chemin str).
    - print_output: imprime le chemin absolu de chaque image sauvee sur stdout
      (contrat machine-parsable pour l'integration externe).
    - refine_tile > 0: passe Z-Image en tuiles (4K+, plafonne le pic VRAM).
    - refine_tile = 0 (Auto): image entiere, puis auto-tuilage au-dela de
      AUTO_REFINE_TILE_ABOVE a la taille choisie par _pick_refine_tile.
    """
    global _LAST_SAVED
    _LAST_SAVED = None
    if not esrgan_model:
        raise gr.Error(f"No ESRGAN model found in {ESRGAN_DIR}.")

    # Mode batch
    if source_folder and os.path.isdir(source_folder):
        paths = _list_folder_images(source_folder)
        if not paths:
            raise gr.Error(f"No image in {source_folder}")
        last_result = last_source = None
        lines = [f"### Batch: {len(paths)} image(s) from `{source_folder}`"]
        t_batch = time.time()
        for p in paths:
            try:
                src = Image.open(p)
                result, t = process_one(src, esrgan_model, factor, denoise, steps,
                                        prompt, seed, tile, overlap,
                                        refine_tile=refine_tile, refine_overlap=refine_overlap)
                dst = build_output_path(p, save_mode, output_dir, output_format,
                                        seed=seed, size=result.size,
                                        esrgan_model=esrgan_model, factor=factor,
                                        denoise=denoise)
                if dst:
                    save_image(result, dst, output_format,
                               meta=build_meta(p, esrgan_model, factor, denoise, steps,
                                               prompt, seed, tile, overlap, refine_tile,
                                               refine_overlap, size=result.size, timings=t))
                    _LAST_SAVED = dst
                    if print_output:
                        print(os.path.abspath(dst))
                _append_time_log(time_log_path, p, dst, t, save_mode, output_format)
                lines.append(f"- `{os.path.basename(p)}` {result.size[0]}x{result.size[1]} "
                             f"esrgan {t['esrgan']:.1f}s + refine {t['refine']:.1f}s"
                             + (f" -> `{dst}`" if dst else " (display)"))
                last_result, last_source = result, src.convert("RGB")
            except Exception as e:
                lines.append(f"- `{os.path.basename(p)}` FAILED: {e}")
        lines.append(f"**Batch total: {time.time()-t_batch:.1f}s**")
        return last_result, last_source, "  \n".join(lines)

    # Mode image unique
    if image is None:
        raise gr.Error("Load an image (or specify a source folder for batch mode).")
    if isinstance(image, str):
        source_path = image
        src_img = Image.open(source_path)
    else:
        source_path = None
        src_img = image

    result, t = process_one(src_img, esrgan_model, factor, denoise, steps,
                            prompt, seed, tile, overlap,
                            refine_tile=refine_tile, refine_overlap=refine_overlap)
    dst = None
    try:
        dst = build_output_path(source_path, save_mode, output_dir, output_format,
                                seed=seed, size=result.size,
                                esrgan_model=esrgan_model, factor=factor,
                                denoise=denoise)
    except ValueError as e:
        dst = None
        save_warning = f"  \n[WARN] {e}"
    else:
        save_warning = ""
    if dst:
        save_image(result, dst, output_format,
                   meta=build_meta(source_path, esrgan_model, factor, denoise, steps,
                                   prompt, seed, tile, overlap, refine_tile,
                                   refine_overlap, size=result.size, timings=t))
        _LAST_SAVED = dst
        if print_output:
            print(os.path.abspath(dst))
    _append_time_log(time_log_path, source_path, dst, t, save_mode, output_format)
    report = _format_timings(t, src_path=source_path, dst_path=dst) + save_warning
    return result, src_img.convert("RGB"), report


def _append_time_log(path, src, dst, t, save_mode, output_format):
    if not path:
        return
    try:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        line = (f"{ts}\t{src or ''}\t{dst or ''}\t"
                f"esrgan={t.get('esrgan', 0):.2f}s\trefine={t.get('refine', 0):.2f}s\t"
                f"mode={save_mode}\tfmt={output_format}\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[AVERT] time-log echec: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# UI Gradio
# ----------------------------------------------------------------------------
def _refresh_models(new_dir):
    """Change ESRGAN_DIR puis renvoie une mise a jour du Dropdown."""
    set_esrgan_dir(new_dir)
    models = list_esrgan_models()
    value = models[0] if models else None
    return gr.update(choices=models, value=value), f"{len(models)} model(s) found in {ESRGAN_DIR}"


def _set_filename_pattern(pattern, output_format):
    """Applique le motif de nommage et renvoie un APERCU du nom obtenu, pour que
    l'utilisateur voie ce qu'il fait avant de lancer un rendu de plusieurs minutes."""
    global FILENAME_PATTERN
    FILENAME_PATTERN = (pattern or "").strip() or DEFAULT_FILENAME_PATTERN
    ext = (output_format or DEFAULT_OUTPUT_FORMAT).lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        ext = "png"
    sample = _format_filename("photo.jpg", seed=1234, size=(1664, 2432),
                              esrgan_model="4x-UltraSharp.pth", factor=2.0,
                              denoise=0.30)
    return f"Example: `{sample}.{ext}`"


def _apply_zimage(repo, transformer):
    set_zimage_model(repo)
    set_zimage_transformer(transformer)
    extra = f" | transformer: {ZIMAGE_TRANSFORMER}" if ZIMAGE_TRANSFORMER else ""
    return f"Z-Image: {BASE_REPO}{extra} (will be (re)loaded on next run)"


def _save_paths_to_prefs(esrgan_dir, zimage_model, transformer):
    set_esrgan_dir(esrgan_dir)
    set_zimage_model(zimage_model)
    set_zimage_transformer(transformer)
    _save_prefs_keys({"esrgan_dir": ESRGAN_DIR, "zimage_model": BASE_REPO,
                      "zimage_transformer": ZIMAGE_TRANSFORMER,
                      "sampler": SAMPLER, "schedule": SCHEDULE,
                      "filename_pattern": FILENAME_PATTERN})
    return (f"Saved to {PREFS_PATH}: esrgan_dir={ESRGAN_DIR}, zimage_model={BASE_REPO}, "
            f"transformer={ZIMAGE_TRANSFORMER or '(base repo)'}, sampler={SAMPLER}/{SCHEDULE}")


# Ordre des composants mis a jour par le dropdown de presets (doit matcher l'UI).
_PRESET_UI_ORDER = ("factor", "denoise", "steps", "tile", "overlap",
                    "refine_tile", "refine_overlap", "cpu_offload")


def _apply_preset(name):
    """UI: renvoie les updates des controles pour le preset choisi (ordre _PRESET_UI_ORDER).
    Custom ou cle absente = pas de changement sur ce controle."""
    p = PRESETS.get(name, {})
    return [gr.update(value=p[k]) if k in p else gr.update() for k in _PRESET_UI_ORDER]


def _pil_to_b64_jpeg(img, max_side=1600, quality=85):
    """Reduit + encode en JPEG base64 pour embarquer en HTML sans saturer la page."""
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_compare_html(src_img, result_img):
    """Comparateur avant/apres standalone: 2 <img> superposees, slider range pilote un clip-path."""
    if src_img is None or result_img is None:
        return "<div style='padding:1em;color:#888'>No result to compare.</div>"
    src_b64 = _pil_to_b64_jpeg(src_img)
    res_b64 = _pil_to_b64_jpeg(result_img)
    uid = uuid.uuid4().hex[:8]
    return f"""
<div style="position:relative; max-width:100%; user-select:none;">
  <img src="data:image/jpeg;base64,{src_b64}" style="display:block; width:100%; height:auto;" alt="source" />
  <img id="cmp-top-{uid}" src="data:image/jpeg;base64,{res_b64}"
       style="position:absolute; top:0; left:0; display:block; width:100%; height:100%;
              clip-path: inset(0 50% 0 0); -webkit-clip-path: inset(0 50% 0 0);" alt="resultat" />
  <div id="cmp-bar-{uid}" style="position:absolute; top:0; left:50%; width:2px; height:100%;
       background:#fff; box-shadow:0 0 4px rgba(0,0,0,0.5); pointer-events:none;"></div>
  <input type="range" min="0" max="100" value="50"
         oninput="
           var v=this.value;
           document.getElementById('cmp-top-{uid}').style.clipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-top-{uid}').style.webkitClipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-bar-{uid}').style.left=v+'%';
         "
         style="position:absolute; bottom:10px; left:5%; width:90%; height:14px; cursor:ew-resize;" />
  <div style="position:absolute; top:8px; left:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">BEFORE</div>
  <div style="position:absolute; top:8px; right:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">AFTER</div>
</div>
"""


def _ui_run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
            tile, overlap, offload_mode, refine_tile, refine_overlap,
            save_mode, output_dir, output_format, progress=gr.Progress()):
    """Adaptateur UI: appelle run() et renvoie (result_image, html_slider, report_markdown).
    Le gr.Progress est pose sur le module pour que le chargement du modele (souvent
    la partie la plus longue du 1er run) affiche une progression au lieu de figer."""
    global _PROGRESS
    set_offload_mode(offload_mode)
    _PROGRESS = progress
    try:
        last_result, last_source, report = run(
            image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
            tile, overlap, save_mode=save_mode, output_dir=output_dir,
            output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
        )
    finally:
        _PROGRESS = None
    html = _make_compare_html(last_source, last_result)
    # On renvoie un CHEMIN, pas l'image PIL. gradio.image_utils.save_image
    # re-encode toute image PIL au format du composant (defaut "webp") avant de
    # la servir: le telechargement sortait donc toujours en .webp, quel que soit
    # "Output format", et le re-encodage DETRUISAIT au passage les metadonnees
    # crispz. Un chemin, lui, est servi tel quel.
    out_path = _ui_result_path(last_result, output_format, esrgan_model, factor,
                               denoise, steps, prompt, seed, tile, overlap,
                               refine_tile, refine_overlap)
    return out_path, html, report


def _ui_result_path(result, output_format, esrgan_model, factor, denoise, steps,
                    prompt, seed, tile, overlap, refine_tile, refine_overlap):
    """Chemin du fichier a servir a l'UI. Reutilise le fichier deja ecrit par
    run() quand il y en a un (zero re-encodage, vrai nom, metadonnees intactes);
    sinon (save_mode=display) ecrit un temporaire au format demande."""
    if result is None:
        return None
    if _LAST_SAVED and os.path.isfile(_LAST_SAVED):
        return _LAST_SAVED
    try:
        meta = build_meta(None, esrgan_model, factor, denoise, steps, prompt, seed,
                          tile, overlap, refine_tile, refine_overlap, size=result.size)
        return _preview_path(result, output_format, meta=meta)
    except Exception as e:
        _log(f"preview file failed ({e}); falling back to in-memory image")
        return result


def build_ui():
    models = list_esrgan_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    with gr.Blocks(title="crispz - Z-Image upscaler + detailer") as demo:
        gr.Markdown("## crispz\nReal-ESRGAN then Z-Image Turbo refinement, 100% local.")

        with gr.Accordion("Paths / models (configuration)", open=False):
            esrgan_dir_tb = gr.Textbox(value=ESRGAN_DIR, label="ESRGAN_DIR (.pth / .safetensors folder)")
            zimage_model_tb = gr.Textbox(value=BASE_REPO, label="Z-Image (HF repo or local path)")
            zimage_transformer_tb = gr.Textbox(
                value=ZIMAGE_TRANSFORMER,
                label="Z-Image transformer override (optional)",
                placeholder="empty = transformer of the base repo",
                info="Single-file .safetensors / .gguf, or a diffusers repo/folder. "
                     "VAE, text encoder and tokenizer always come from the base repo. "
                     "Useful to run a quantized transformer on a small GPU.",
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh ESRGAN list", size="sm")
                apply_zimage_btn = gr.Button("Apply Z-Image", size="sm")
                save_paths_btn = gr.Button("Save to preferences.json", size="sm", variant="primary")
            paths_status = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Source image (single mode)")
                source_folder_tb = gr.Textbox(
                    value="",
                    label="OR source folder (batch mode, takes priority if filled)",
                    placeholder="e.g. D:/images/series_a",
                )
                esrgan = gr.Dropdown(models, value=default_model, label="ESRGAN model")
                preset = gr.Dropdown(list(PRESETS), value="Custom",
                                     label="Use case (auto settings)",
                                     info="Fills the settings below. 'Custom' changes nothing.")
                factor = gr.Slider(1.0, 4.0, value=DEFAULT_FACTOR, step=0.5, label="Net upscale factor")
                denoise = gr.Slider(0.0, 0.8, value=DEFAULT_DENOISE, step=0.01,
                                    label="Denoise (strength) - 0.2-0.4 recommended")
                steps = gr.Slider(4, 30, value=DEFAULT_STEPS, step=1, label="Steps (diffusion pass)")
                prompt = gr.Textbox(label="Optional prompt", placeholder="leaving it empty works very well")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Accordion("ESRGAN tiling (VRAM)", open=False):
                    tile = gr.Slider(0, 1024, value=DEFAULT_TILE, step=8, label="Tile size (0 = disabled)")
                    overlap = gr.Slider(0, 128, value=DEFAULT_OVERLAP, step=8, label="Overlap")
                    offload = gr.Dropdown(
                        choices=list(OFFLOAD_CHOICES),
                        value="none",
                        label="CPU offload (diffusion pass)",
                        info="none=all in VRAM | model=offload per submodule (good tradeoff) | "
                             "sequential=more aggressive, slower. Lowers the VRAM peak.",
                    )
                with gr.Accordion("Z-Image tiling (4K+)", open=False):
                    refine_tile = gr.Dropdown(
                        choices=REFINE_TILE_CHOICES, value=DEFAULT_REFINE_TILE,
                        label="Diffusion tile size",
                        info="Tiles the Z-Image pass: caps the VRAM peak and enables 4K+ "
                             "without seams. Auto keeps the whole image under "
                             f"{AUTO_REFINE_TILE_ABOVE}px, then picks the tile that "
                             "minimises the diffused surface (measured 23% faster than a "
                             "fixed 1024 at 4096x4096).")
                    refine_overlap = gr.Slider(0, 256, value=DEFAULT_REFINE_OVERLAP, step=16,
                                               label="Diffusion tile overlap (feather)")
                with gr.Accordion("Sampler (diffusion pass)", open=False):
                    sampler_dd = gr.Dropdown(
                        choices=list(SAMPLER_CHOICES), value=SAMPLER, label="Sampler",
                        info="euler = native Z-Image flow-matching (default). unipc can "
                             "converge in fewer steps. lcm targets very low step counts.",
                    )
                    schedule_dd = gr.Dropdown(
                        choices=list(SCHEDULE_CHOICES), value=SCHEDULE, label="Sigma schedule",
                        info="sgm_uniform = native. beta / karras / exponential re-map the "
                             "sigmas on top. beta needs scipy.",
                    )
                    sampler_status = gr.Markdown(f"Sampler: {SAMPLER} / {SCHEDULE}")
                with gr.Accordion("Save", open=True):
                    save_mode = gr.Radio(
                        choices=["display", "local", "alongside", "custom"],
                        value=DEFAULT_SAVE_MODE,
                        label="Save mode",
                        info="display=save nothing | local=into 'output_dir' relative to the project | "
                             "alongside=same folder as the source (CLI/batch) | custom=output_dir as-is",
                    )
                    output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output folder (local/custom)")
                    output_format = gr.Dropdown(
                        choices=list(SUPPORTED_FORMATS),
                        value=DEFAULT_OUTPUT_FORMAT,
                        label="Output format",
                    )
                    filename_tb = gr.Textbox(
                        value=FILENAME_PATTERN, label="Filename pattern",
                        info="Without extension. Placeholders: {date} {name} {tag} "
                             "{seed} {w} {h} {model} {factor} {denoise} {index}. "
                             "Existing files are never overwritten (_2, _3... suffix).",
                    )
                    filename_preview = gr.Markdown("")
                btn = gr.Button("Upscale + Detail", variant="primary")
            with gr.Column():
                out_slider = gr.HTML(value="<div style='padding:1em;color:#888'>No result yet. Run an upscale.</div>",
                                     label="Before / after comparator (drag the slider)")
                # format=: filet de securite. _ui_run renvoie un CHEMIN (servi
                # tel quel), mais si un jour une image PIL repassait par ici,
                # elle serait encodee en PNG sans perte plutot qu'en webp.
                out = gr.Image(type="filepath", format="png",
                               label="Result (downloadable)")
                report = gr.Markdown(value="*No run yet.*", label="Report")

        refresh_btn.click(_refresh_models, [esrgan_dir_tb], [esrgan, paths_status])
        apply_zimage_btn.click(_apply_zimage, [zimage_model_tb, zimage_transformer_tb],
                               [paths_status])
        save_paths_btn.click(_save_paths_to_prefs,
                             [esrgan_dir_tb, zimage_model_tb, zimage_transformer_tb],
                             [paths_status])
        filename_tb.change(_set_filename_pattern, [filename_tb, output_format],
                           [filename_preview])
        output_format.change(_set_filename_pattern, [filename_tb, output_format],
                             [filename_preview])
        demo.load(_set_filename_pattern, [filename_tb, output_format], [filename_preview])
        sampler_dd.change(set_sampler, [sampler_dd], [sampler_status])
        schedule_dd.change(set_schedule, [schedule_dd], [sampler_status])
        preset.change(_apply_preset, [preset],
                      [factor, denoise, steps, tile, overlap, refine_tile, refine_overlap, offload])
        btn.click(
            _ui_run,
            inputs=[inp, source_folder_tb, esrgan, factor, denoise, steps, prompt, seed,
                    tile, overlap, offload, refine_tile, refine_overlap,
                    save_mode, output_dir, output_format],
            outputs=[out, out_slider, report],
        )
    # File d'attente OBLIGATOIRE: _ui_run utilise gr.Progress (progression du
    # chargement du modele), et Gradio refuse de demarrer si le suivi de
    # progression est demande sans queue ("Progress tracking requires queuing to
    # be enabled"). Gradio 4/5 l'active d'office, Gradio 3 non -> on l'active
    # explicitement pour couvrir toute la plage supportee (>=4.44 requis, mais
    # l'app tourne aussi sur un environnement 3.x herite).
    # Effet de bord souhaitable: les runs sont serialises, ce qui evite deux
    # passes de diffusion simultanees sur le meme GPU.
    try:
        demo.queue()
    except Exception as e:
        _log(f"queue() unavailable ({e}); progress bar will be limited")
    return demo


# ----------------------------------------------------------------------------
# Palier 3 : serveur HTTP persistant (FastAPI), load paresseux + unload sur idle
# ----------------------------------------------------------------------------
def serve_main(host="127.0.0.1", port=7861, idle_timeout=300):
    """Petit serveur HTTP. Le modele Z-Image se charge au premier /upscale et reste
    chaud (plus de rechargement entre appels -> temps stables). Apres idle_timeout
    secondes sans requete, la VRAM est rendue (utile pour cohabiter avec Fooocus).
    Endpoints: GET /health, GET /models, POST /upscale, POST /unload."""
    try:
        import threading
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as e:
        print("[serve] FastAPI/uvicorn required: pip install fastapi uvicorn", file=sys.stderr)
        print(f"[serve] detail: {e}", file=sys.stderr)
        return 1

    os.makedirs(ESRGAN_DIR, exist_ok=True)
    app = FastAPI(title="crispz")
    lock = threading.Lock()
    state = {"last": time.time()}

    class UpscaleReq(BaseModel):
        input: str
        model: str = DEFAULT_MODEL
        factor: float = DEFAULT_FACTOR
        denoise: float = DEFAULT_DENOISE
        steps: int = DEFAULT_STEPS
        prompt: str = ""
        seed: int = -1
        tile: int = DEFAULT_TILE
        overlap: int = DEFAULT_OVERLAP
        refine_tile: int = DEFAULT_REFINE_TILE
        refine_overlap: int = DEFAULT_REFINE_OVERLAP
        cpu_offload: str = "none"
        preset: str = "Custom"
        sampler: str = ""
        schedule: str = ""
        save_mode: str = "local"
        output_dir: str = DEFAULT_OUTPUT_DIR
        output_format: str = DEFAULT_OUTPUT_FORMAT

    @app.get("/health")
    def health():
        return {"status": "ok", "device": DEVICE, "pipe_loaded": _PIPE is not None,
                "offload": OFFLOAD_MODE, "idle_timeout": idle_timeout,
                "zimage_model": BASE_REPO,
                "zimage_transformer": ZIMAGE_TRANSFORMER or None,
                "sampler": SAMPLER, "schedule": SCHEDULE}

    @app.get("/models")
    def models():
        return {"esrgan_dir": ESRGAN_DIR, "models": list_esrgan_models()}

    @app.post("/unload")
    def unload():
        with lock:
            free_vram()
        return {"status": "unloaded"}

    @app.post("/upscale")
    def upscale(req: UpscaleReq):
        if not os.path.isfile(req.input):
            raise HTTPException(status_code=400, detail=f"input not found: {req.input}")
        avail = list_esrgan_models()
        if not avail:
            raise HTTPException(status_code=400, detail=f"no ESRGAN model in {ESRGAN_DIR}")
        # preset (s'il est fourni) sert de base; sinon les champs de la requete.
        p = PRESETS.get(req.preset or "Custom") or {}
        def pick(name, val):
            return p.get(name, val)
        model = req.model if req.model in avail else avail[0]
        with lock:
            state["last"] = time.time()
            set_offload_mode(pick("cpu_offload", req.cpu_offload))
            if req.sampler:
                set_sampler(req.sampler)
            if req.schedule:
                set_schedule(req.schedule)
            img = Image.open(req.input)
            result, t = process_one(
                img, model, pick("factor", req.factor), pick("denoise", req.denoise),
                pick("steps", req.steps), req.prompt, req.seed,
                pick("tile", req.tile), pick("overlap", req.overlap),
                refine_tile=pick("refine_tile", req.refine_tile),
                refine_overlap=pick("refine_overlap", req.refine_overlap),
            )
            dst = build_output_path(req.input, req.save_mode, req.output_dir,
                                    req.output_format, seed=req.seed,
                                    size=result.size, esrgan_model=model,
                                    factor=pick("factor", req.factor),
                                    denoise=pick("denoise", req.denoise))
            if dst:
                save_image(result, dst, req.output_format,
                           meta=build_meta(req.input, model, pick("factor", req.factor),
                                           pick("denoise", req.denoise), pick("steps", req.steps),
                                           req.prompt, req.seed, pick("tile", req.tile),
                                           pick("overlap", req.overlap),
                                           pick("refine_tile", req.refine_tile),
                                           pick("refine_overlap", req.refine_overlap),
                                           size=result.size, timings=t))
            state["last"] = time.time()
        return {"output": os.path.abspath(dst) if dst else None,
                "size": list(result.size),
                "esrgan_s": round(t.get("esrgan", 0.0), 2),
                "refine_s": round(t.get("refine", 0.0), 2),
                "total_s": round(t.get("esrgan", 0.0) + t.get("refine", 0.0), 2)}

    def _idle_watch():
        period = min(30, max(5, idle_timeout // 4)) if idle_timeout > 0 else 30
        while True:
            time.sleep(period)
            if idle_timeout > 0 and _PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                with lock:
                    if _PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                        free_vram()
                        print(f"[serve] model unloaded after {idle_timeout}s idle", file=sys.stderr)

    if idle_timeout and idle_timeout > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"[serve] crispz on http://{host}:{port}  (idle unload: {idle_timeout}s)", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ----------------------------------------------------------------------------
# CLI (mode batch / scripting)
# ----------------------------------------------------------------------------
def cli_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="crispz CLI. With no arguments: launches the Gradio UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (otherwise: launches the UI)")
    # Sources : fichier, glob, dossier
    parser.add_argument("-i", "--input", help="Image, glob (in/*.png) or source FOLDER for batch")
    parser.add_argument("--input-folder", help="Explicit alias for the batch folder (otherwise -i works too)")
    # Sortie
    parser.add_argument("-o", "--output",
                        help="Output file (single mode, overrides auto naming). "
                             "If a folder: equivalent to --save-mode local --output-dir <that folder>.")
    parser.add_argument("--save-mode", choices=["display", "local", "alongside", "custom"],
                        default=DEFAULT_SAVE_MODE,
                        help="display=no save | local=output_dir relative to project | "
                             "alongside=same folder as the source | custom=output_dir as-is")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output folder for --save-mode local/custom")
    parser.add_argument("--output-format", choices=list(SUPPORTED_FORMATS),
                        default=DEFAULT_OUTPUT_FORMAT, help="Output format (png/webp/jpg)")
    # Pipeline
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="ESRGAN model (file in ESRGAN_DIR). Fallback: first found.")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR, help="Net upscale factor")
    parser.add_argument("--denoise", type=float, default=DEFAULT_DENOISE, help="Z-Image strength (0 = ESRGAN only)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--prompt", default="", help="Optional prompt")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE, help="ESRGAN tile size (0 = disabled)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="ESRGAN tiling overlap")
    parser.add_argument("--refine-tile", type=int, default=DEFAULT_REFINE_TILE,
                        help="Z-Image diffusion tile size (0 = whole image). >0 tiles the "
                             "refine pass: caps VRAM and enables 4K+ without seams. Try 1024-1280.")
    parser.add_argument("--refine-overlap", type=int, default=DEFAULT_REFINE_OVERLAP,
                        help="Overlap (feather) of the Z-Image diffusion tiles")
    parser.add_argument("--cpu-offload", choices=list(OFFLOAD_CHOICES), default="none",
                        help="CPU offload of the diffusion pass (VRAM). none=all in VRAM | "
                             "model=offload per submodule (good tradeoff) | "
                             "sequential=more aggressive, slower. Requires accelerate.")
    parser.add_argument("--preset", choices=list(PRESETS), default="Custom",
                        help="Use-case preset (auto settings). Explicit flags override it.")
    parser.add_argument("--sampler", choices=list(SAMPLER_CHOICES), default=None,
                        help="Sampler of the diffusion pass. euler = native Z-Image "
                             "flow-matching; unipc can converge in fewer steps; lcm "
                             "targets very low step counts.")
    parser.add_argument("--schedule", choices=list(SCHEDULE_CHOICES), default=None,
                        help="Sigma schedule re-mapped on top of the sampler "
                             "(beta requires scipy).")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Do not embed the settings in the saved images "
                             "(PNG 'crispz' chunk / EXIF ImageDescription).")
    parser.add_argument("--sidecar-json", action="store_true",
                        help="Also write a <image>.json sidecar next to each saved image.")
    parser.add_argument("--read-metadata", metavar="IMAGE",
                        help="Print the crispz metadata of an image then exit.")
    parser.add_argument("--filename-pattern", default=None,
                        help="Output filename pattern (without extension). "
                             "Placeholders: {date} {name} {tag} {seed} {w} {h} "
                             "{model} {factor} {denoise} {index}. "
                             f"Default: {DEFAULT_FILENAME_PATTERN}")
    # Server (stage 3)
    parser.add_argument("--serve", action="store_true",
                        help="Run a persistent HTTP server (lazy model load + idle unload) "
                             "instead of the UI/one-shot. Requires fastapi + uvicorn.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (--serve)")
    parser.add_argument("--port", type=int, default=7861, help="Server port (--serve)")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Seconds of inactivity before the server frees VRAM (0 = never)")
    # Chemins config / Z-Image
    parser.add_argument("--esrgan-dir", help="Override ESRGAN_DIR for this run")
    parser.add_argument("--zimage-model", help="Override HF repo / local path for Z-Image")
    parser.add_argument("--zimage-transformer",
                        help="Override ONLY the Z-Image transformer: single-file "
                             ".safetensors / .gguf, or a diffusers repo/folder. VAE, "
                             "text encoder and tokenizer stay from the base repo.")
    parser.add_argument("--save-paths", action="store_true",
                        help="Save --esrgan-dir, --zimage-model and --zimage-transformer "
                             "to preferences.json")
    # Reports
    parser.add_argument("--list-models", action="store_true", help="List ESRGAN models then exit")
    parser.add_argument("--time-log", default=None,
                        help="If set, append the time of each run to this file (TSV)")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout verbosity")
    parser.add_argument("--report-vram", action="store_true",
                        help="Print the run VRAM peak on stderr (line '[VRAM] ...'). "
                             "Used to size coexistence with Fooocus.")
    parser.add_argument("--print-output", action="store_true",
                        help="Print ONLY the absolute output path on stdout (one per saved "
                             "image), nothing else. For external integration (Fooocus). "
                             "Implies a silent stdout; the VRAM peak stays on stderr.")
    args = parser.parse_args(argv)
    apply_preset_to_args(args, argv if argv is not None else sys.argv[1:])

    global VERBOSE, WRITE_METADATA, WRITE_SIDECAR, FILENAME_PATTERN
    VERBOSE = not args.quiet
    if args.filename_pattern:
        FILENAME_PATTERN = args.filename_pattern
    if args.no_metadata:
        WRITE_METADATA = False
    if args.sidecar_json:
        WRITE_SIDECAR = True

    # --read-metadata: pur outil de lecture, aucun modele n'est charge.
    if args.read_metadata:
        data = read_metadata(args.read_metadata)
        if data is None:
            print(f"No crispz metadata in {args.read_metadata}", file=sys.stderr)
            return 1
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    if args.esrgan_dir:
        set_esrgan_dir(args.esrgan_dir)
    if args.zimage_model:
        set_zimage_model(args.zimage_model)
    if args.zimage_transformer is not None:
        set_zimage_transformer(args.zimage_transformer)
    if args.sampler:
        set_sampler(args.sampler)
    if args.schedule:
        set_schedule(args.schedule)
    set_offload_mode(args.cpu_offload)

    if args.serve:
        return serve_main(args.host, args.port, args.idle_timeout)

    if args.save_paths:
        _save_prefs_keys({"esrgan_dir": ESRGAN_DIR, "zimage_model": BASE_REPO,
                          "zimage_transformer": ZIMAGE_TRANSFORMER,
                          "sampler": SAMPLER, "schedule": SCHEDULE})
        print(f"Saved to {PREFS_PATH}: esrgan_dir={ESRGAN_DIR}, zimage_model={BASE_REPO}, "
              f"transformer={ZIMAGE_TRANSFORMER or '(base repo)'}, sampler={SAMPLER}/{SCHEDULE}")
        if not args.input and not args.input_folder:
            return 0

    os.makedirs(ESRGAN_DIR, exist_ok=True)
    models = list_esrgan_models()

    if args.list_models:
        if not models:
            print(f"No model in {ESRGAN_DIR}")
        else:
            for m in models:
                print(m)
        return 0

    # Pas de --cli et pas d'entree -> UI
    if not args.cli and not args.input and not args.input_folder:
        build_ui().launch()
        return 0

    if not models:
        parser.error(f"No ESRGAN model in {ESRGAN_DIR}")

    model_name = args.model if args.model in models else models[0]

    if args.report_vram:
        _reset_vram_peak()

    # Resoudre les entrees : dossier > glob > fichier unique
    source_folder = args.input_folder
    if not source_folder and args.input and os.path.isdir(args.input):
        source_folder = args.input
        args.input = None

    # --output (compat) : si c'est un dossier, equivalent a --save-mode local --output-dir <dossier>
    save_mode = args.save_mode
    output_dir = args.output_dir
    explicit_output_file = None
    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(("/", "\\")):
            save_mode = "custom" if os.path.isabs(args.output) else "local"
            output_dir = args.output
        else:
            explicit_output_file = args.output
            save_mode = "custom"

    # --print-output: stdout reserve aux chemins de sortie (contrat machine).
    # Le pic VRAM, lui, reste sur stderr et n'est donc pas pollue.
    quiet = args.quiet or args.print_output

    # Mode batch dossier
    if source_folder:
        last_result, last_source, report = run(
            None, source_folder, model_name, args.factor, args.denoise, args.steps,
            args.prompt, args.seed, args.tile, args.overlap,
            save_mode=save_mode, output_dir=output_dir,
            output_format=args.output_format, time_log_path=args.time_log,
            print_output=args.print_output,
            refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
        )
        if not quiet:
            print(report)
        if args.report_vram:
            _report_vram()
        return 0

    # Mode unique : glob possible
    paths = sorted(glob.glob(args.input)) if any(c in args.input for c in "*?[") else [args.input]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        parser.error(f"No file matches {args.input}")

    # Si plusieurs fichiers via glob, on les passe un par un
    for p in paths:
        if not quiet:
            print(f"-> {p}")
        img = Image.open(p)
        # explicit_output_file ne s'applique qu'au premier fichier
        if explicit_output_file and len(paths) == 1:
            result, t = process_one(img, model_name, args.factor, args.denoise, args.steps,
                                    args.prompt, args.seed, args.tile, args.overlap,
                                    refine_tile=args.refine_tile, refine_overlap=args.refine_overlap)
            os.makedirs(os.path.dirname(os.path.abspath(explicit_output_file)) or ".", exist_ok=True)
            # -o chemin explicite: on respecte le nom demande (pas de suffixe _2),
            # c'est l'utilisateur qui a choisi la destination.
            save_image(result, explicit_output_file, args.output_format,
                       meta=build_meta(p, model_name, args.factor, args.denoise, args.steps,
                                       args.prompt, args.seed, args.tile, args.overlap,
                                       args.refine_tile, args.refine_overlap,
                                       size=result.size, timings=t))
            if args.print_output:
                print(os.path.abspath(explicit_output_file))
            _append_time_log(args.time_log, p, explicit_output_file, t, "custom", args.output_format)
            if not quiet:
                print(_format_timings(t, src_path=p, dst_path=explicit_output_file))
        else:
            # mode standard: build_output_path applique le save_mode
            last_result, last_source, report = run(
                p, None, model_name, args.factor, args.denoise, args.steps,
                args.prompt, args.seed, args.tile, args.overlap,
                save_mode=save_mode, output_dir=output_dir,
                output_format=args.output_format, time_log_path=args.time_log,
                print_output=args.print_output,
                refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
            )
            if not quiet:
                print(report)
    if args.report_vram:
        _report_vram()
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
