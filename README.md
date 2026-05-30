# crispz

> Z-Image based upscaler + detailer (hi-res fix).

Outil standalone d'upscale + denoise d'images, **100% local**, sans ComfyUI ni
SwarmUI. Deux etages:

1. **Real-ESRGAN** (charge via `spandrel`) pour l'agrandissement reel des pixels,
   avec tiling overlap-add + feather lineaire.
2. **Z-Image Turbo** en img2img (`diffusers`, BF16) pour une passe de raffinement
   a bas denoise qui reinjecte du detail sans casser la composition.

Interface Gradio + mode CLI scriptable + CLI interactif avec preferences.

---

## Installation

### Pre-requis

- Python 3.10+
- PyTorch **deja installe** avec ton build CUDA (le brief vise PyTorch 2.7+/
  CUDA 12.8). **Ne JAMAIS reinstaller torch** depuis ce projet, il s'aligne sur
  ton environnement existant.
- Un GPU NVIDIA avec >= 8 Go VRAM est conseille. RTX 5090 testee en BF16
  natif, image entiere jusqu'a 2048px sans broncher.

### Scripts d'install fournis

```bash
# Linux / macOS / WSL
./install.sh
./run.sh           # UI Gradio + detection hardware
./cli.sh           # CLI interactive avec preferences

# Windows
install.bat
run.bat
cli.bat
```

Les scripts d'install:
- verifient que Python et PyTorch sont presents (sans toucher a torch),
- desinstallent automatiquement un `xformers` casse (build pour une mauvaise
  version de torch -> DLL load error au chargement de diffusers),
- installent les autres deps depuis `requirements.txt`,
- verifient que `ZImageImg2ImgPipeline` se charge,
- creent le dossier `upscale_models/`.

Install manuel equivalent:

```bash
pip install -r requirements.txt
```

### Pieges connus d'environnement

- **`xformers` incompatible.** Si une version de `xformers` est installee mais
  buildee pour un autre torch (ex: `xformers` pour torch 2.9 alors que tu as
  torch 2.8), diffusers plante avec `DLL load failed while importing _C` au
  chargement du VAE. Solution: `pip uninstall xformers`. Le SDPA natif de
  torch 2.7+ suffit.
- **`transformers` trop ancien.** `ZImageImg2ImgPipeline` charge un encodeur
  qui importe `Dinov2WithRegistersConfig`, present depuis transformers >= 4.49.
  Le requirements pin cette borne.
- **diffusers depuis git.** Z-Image n'est dans diffusers que depuis la source
  (pas dans les releases au moment de l'ecriture), d'ou le `git+...` du
  requirements.

---

## Chemins configurables (ESRGAN_DIR + Z-Image)

Deux chemins sont configurables, avec persistance dans `preferences.json`.
Ordre de priorite a chaque lancement:

1. Variable d'environnement (`ESRGAN_DIR`, `ZIMAGE_MODEL`)
2. `preferences.json` a la racine du projet
3. Defaut: `./upscale_models` pour ESRGAN, `Tongyi-MAI/Z-Image-Turbo` pour Z-Image

Trois facons de les modifier:

- **UI Gradio**: accordion "Chemins / modeles" en haut. Boutons "Rafraichir la
  liste ESRGAN", "Appliquer Z-Image" (invalide le pipe pour recharger), et
  "Sauver dans preferences.json".
- **CLI**: `--esrgan-dir <path>`, `--zimage-model <repo_ou_path>`, `--save-paths`
  pour persister (avec ou sans `-i`).
- **CLI interactive** (`cli.sh` / `cli.bat`): premier prompt = dossier ESRGAN
  + modele Z-Image. Sauves dans `preferences.json` si tu choisis de garder.

`zimage_model` accepte aussi bien un repo HF (ex `Tongyi-MAI/Z-Image-Turbo`)
qu'un chemin local vers un dossier `diffusers` deja telecharge.

## Modeles ESRGAN

Depose au moins un `.pth` ou `.safetensors` dans `./upscale_models`, ou pointe
`ESRGAN_DIR` (env ou prefs) vers un dossier existant.

Quelques choix utiles:
- `RealESRGAN_x4plus.pth` (general)
- `4x-UltraSharp.pth` (net, polyvalent)
- `4x-ClearRealityV1_Soft.safetensors` (doux, bon sur portraits/scenes)
- `4xFaceUpDAT.pth` (portraits/visages)

`spandrel` detecte l'architecture et le facteur (x2 / x4) automatiquement.

---

## Z-Image (premier run)

Aucun fichier a fournir: au premier lancement, `diffusers` recupere depuis
Hugging Face le transformer Z-Image, le VAE et l'encodeur texte Qwen3-4B,
puis tout est cache en local. Les runs suivants sont hors-ligne.

---

## Lancement

### 1) UI Gradio (par defaut)

```bash
python app.py
```

Interface sur http://127.0.0.1:7860. L'UI inclut:

- **Slider avant/apres** (`gradio_imageslider`) qui superpose la source et le
  resultat avec un curseur a la souris. Fallback en deux images cote a cote
  si le composant n'est pas installe.
- **Rapport de temps** sous l'image: ESRGAN, raffinement Z-Image, total, chemin source, chemin de sauvegarde.
- **Section "Sauvegarde"** avec les memes modes que la CLI.
- **Mode batch**: si tu remplis "OU dossier source", l'image uploadee est ignoree et l'app traite tout le dossier.

### 2) CLI scriptable

```bash
# Une image, reglages explicites
python app.py --cli -i mon_image.jpg \
    --save-mode local --output-dir out --output-format png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32

# Batch sur un dossier entier
python app.py --cli -i ./mes_images --save-mode local --output-dir out --output-format webp

# Sauvegarde a cote de chaque source (mode "alongside")
python app.py --cli -i ./mes_images --save-mode alongside --output-format jpg

# Display only (pas de fichier ecrit), juste le timing dans stdout
python app.py --cli -i mon_image.jpg --save-mode display --denoise 0

# Avec log TSV pour suivre les temps
python app.py --cli -i ./mes_images --save-mode local --time-log runs.tsv
```

### 3) CLI interactive avec preferences

```bash
./cli.sh   # ou cli.bat sous Windows
```

Demande chaque reglage (chemins, modeles, source, pipeline, sauvegarde,
time-log) avec valeur par defaut depuis `preferences.json`. Propose
de sauver les choix en fin de session.

## Mapping UI <-> CLI <-> preferences.json

Chaque reglage de l'UI a un flag CLI et une cle de prefs:

| UI / CLI interactive | CLI flag | preferences.json | Defaut |
|---|---|---|---|
| ESRGAN_DIR | `--esrgan-dir` | `esrgan_dir` | `./upscale_models` |
| Modele Z-Image | `--zimage-model` | `zimage_model` | `Tongyi-MAI/Z-Image-Turbo` |
| Image source | `-i` (fichier ou glob) | - | - |
| Dossier source batch | `-i` (dossier) ou `--input-folder` | - | - |
| Modele ESRGAN | `-m` / `--model` | `model` | `4x-ClearRealityV1_Soft.safetensors` |
| Facteur agrandissement | `--factor` | `factor` | `2.0` |
| Denoise (strength) | `--denoise` | `denoise` | `0.30` |
| Steps diffusion | `--steps` | `steps` | `12` |
| Prompt | `--prompt` | `prompt` | `""` |
| Seed | `--seed` | `seed` | `-1` |
| Tile ESRGAN | `--tile` | `tile` | `760` |
| Overlap | `--overlap` | `overlap` | `32` |
| Mode sauvegarde | `--save-mode` | `save_mode` | `display` |
| Dossier de sortie | `--output-dir` | `output_dir` | `out` |
| Format sortie | `--output-format` | `output_format` | `png` |
| Log temps (CLI) | `--time-log <file.tsv>` | `time_log` | (vide) |
| Sauve les paths (CLI) | `--save-paths` | - | - |
| Lister modeles (CLI) | `--list-models` | - | - |
| Pic VRAM sur stderr (CLI) | `--report-vram` | - | - |
| Chemin de sortie seul (CLI) | `--print-output` | - | - |

Modes de sauvegarde:

| save_mode | Comportement |
|---|---|
| `display` | N'ecrit rien. UI rend l'image + timing. CLI imprime le rapport. |
| `local` | Ecrit dans `output_dir` resolu **relatif au projet** si non absolu. |
| `alongside` | Ecrit dans le **meme dossier que la source**. Necessite un chemin source (CLI ou batch dossier). |
| `custom` | Ecrit dans `output_dir` tel quel (typiquement un chemin absolu). |

Naming par defaut: `{nom_source}_upscaled.{png|webp|jpg}`. En CLI, `-o` accepte
un fichier (override le naming auto), un dossier (equivalent a
`--save-mode local --output-dir <dossier>`) ou est omis (utilise
`--save-mode` / `--output-dir`).

Exemple `preferences.json` complet:

```json
{
  "esrgan_dir": "D:/Github/sdlibs/models/ESRGAN",
  "zimage_model": "Tongyi-MAI/Z-Image-Turbo",
  "model": "4x-ClearRealityV1_Soft.safetensors",
  "factor": 2.0,
  "denoise": 0.30,
  "steps": 12,
  "prompt": "",
  "seed": -1,
  "tile": 760,
  "overlap": 32,
  "save_mode": "local",
  "output_dir": "out",
  "output_format": "png",
  "time_log": ""
}
```

## Rapport de temps

`run()` retourne (et imprime / log) le temps de chaque etage:

- `esrgan` : etage 1 (Real-ESRGAN + Lanczos de recadrage)
- `refine` : etage 2 (Z-Image img2img). 0s si `denoise <= 0`.
- `total`  : somme

L'UI affiche un bloc Markdown sous l'image. La CLI imprime le rapport dans
stdout (sauf `--quiet`). Avec `--time-log <fichier>`, chaque run ajoute une
ligne TSV:

```
<iso-timestamp>\t<src>\t<dst>\tesrgan=2.24s\trefine=1.87s\tmode=local\tfmt=png
```

---

## Integration externe (Fooocus, scripts)

Deux flags facilitent l'appel de crispz depuis un autre outil (process separe):

- `--print-output` : stdout ne contient QUE le chemin absolu de chaque image
  sauvee (un par ligne), rien d'autre. Le rapport lisible est supprime. C'est
  le contrat machine-parsable a utiliser pour recuperer le resultat.
- `--report-vram` : pic VRAM du run sur **stderr** (ligne `[VRAM] pic alloue:
  X.XX Go | pic reserve: Y.YY Go`). Sur stderr, donc sans polluer le stdout de
  `--print-output`. Sert a dimensionner la cohabitation VRAM (ex: avec Fooocus).

```bash
# L'appelant lit le chemin de sortie sur stdout, la VRAM sur stderr
dst=$(python app.py --cli -i in.png --save-mode local --output-dir out \
    --print-output --report-vram 2>vram.log)
echo "image upscalee: $dst"
```

`--print-output` requiert un mode de sauvegarde qui ecrit un fichier
(`local` / `alongside` / `custom`). En `display` rien n'est ecrit, donc rien
n'est imprime.

---

## Reglages utiles

| Reglage | Conseil |
|---|---|
| **Denoise (strength)** | 0.05-0.25 = subtil, reste tres proche de l'input. 0.25-0.40 = creatif, plus de detail injecte. Au-dela de ~0.40, Z-Image commence a reinventer. A fort denoise, un **prompt-caption detaille** ameliore beaucoup la coherence. |
| **Steps** | Steps reels ~= `steps * strength`. A strength 0.30, 12-16 steps donnent assez de pas de debruitage. |
| **Guidance** | Fixee a 0.0 (Z-Image Turbo). |
| **Prompt** | Optionnel. Vide marche tres bien si denoise <= 0.30. |
| **Facteur** | ESRGAN agit en x4 natif, puis Lanczos redimensionne au facteur demande. Pour un x2 net, l'image passe par un x4 brut. |
| **Tiling ESRGAN** | 0 (image entiere) sur 24+ Go VRAM. 512-768 sinon. Overlap 32 par defaut, augmente si tu vois des coutures. |

Le script `_hw_check.py` (appele par `run.sh` / `run.bat`) detecte ton GPU
et donne des recommandations basees sur la VRAM, la compute capability
(BF16 dispo a partir d'Ampere = CC 8.0), et la taille max d'image en
passe diffusion.

---

## Reglages de reference

Un point de depart fiable (source ~832x1216 -> x2), modele
`4x-ClearRealityV1_Soft.safetensors`, facteur 2, denoise 0.30, steps 12,
tile 760, overlap 32:

```bash
python app.py --cli -i mon_image.jpg -o out/mon_image_upscaled.png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32
```

---

## Limite connue / prochaine iteration

La passe Z-Image se fait en image entiere. Ideal jusqu'a ~2048px de cote,
au-dela on depasse la resolution d'entrainement et des artefacts peuvent
apparaitre.

Prochaine etape (roadmap dans `PROJECT_BRIEF.md`): tiling de la passe
diffusion facon Ultimate SD Upscale, avec feather, pour pousser en 4K+
sans coutures.
