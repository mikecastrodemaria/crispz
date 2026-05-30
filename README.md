# crispz

> Z-Image based upscaler + detailer (hi-res fix).

A standalone image upscaling + denoising tool, **100% local**, with no ComfyUI or
SwarmUI. Two stages:

1. **Real-ESRGAN** (loaded via `spandrel`) for true pixel enlargement, with
   overlap-add tiling + linear feathering.
2. **Z-Image Turbo** img2img (`diffusers`, BF16) for a low-denoise refinement pass
   that reinjects detail without breaking the composition.

Gradio UI + scriptable CLI + interactive CLI with saved preferences.

---

## Installation

### Requirements

- Python 3.10+
- PyTorch **already installed** with your CUDA build (the project targets
  PyTorch 2.7+ / CUDA 12.8). **NEVER reinstall torch** from this project; it
  aligns with your existing environment.
- An NVIDIA GPU with >= 8 GB VRAM is recommended. RTX 5090 tested in native BF16,
  whole-image up to 2048px without trouble.

### Provided install scripts

```bash
# Linux / macOS / WSL
./install.sh
./run.sh           # Gradio UI + hardware detection
./cli.sh           # interactive CLI with preferences

# Windows
install.bat
run.bat
cli.bat
```

The install scripts:
- check that Python and PyTorch are present (without touching torch),
- automatically uninstall a broken `xformers` (built for the wrong torch
  version -> DLL load error when diffusers loads),
- install the other deps from `requirements.txt`,
- verify that `ZImageImg2ImgPipeline` loads,
- create the `upscale_models/` folder.

Equivalent manual install:

```bash
pip install -r requirements.txt
```

### Known environment pitfalls

- **Incompatible `xformers`.** If an `xformers` version is installed but built
  for a different torch (e.g. `xformers` for torch 2.9 while you have torch 2.8),
  diffusers crashes with `DLL load failed while importing _C` when loading the
  VAE. Fix: `pip uninstall xformers`. The native SDPA in torch 2.7+ is enough.
- **`transformers` too old.** `ZImageImg2ImgPipeline` loads an encoder that
  imports `Dinov2WithRegistersConfig`, available since transformers >= 4.49.
  The requirements pin this lower bound.
- **diffusers from git.** Z-Image is only in diffusers from source (not in the
  releases at the time of writing), hence the `git+...` in requirements.

---

## Configurable paths (ESRGAN_DIR + Z-Image)

Two paths are configurable, persisted in `preferences.json`. Resolution order on
each launch:

1. Environment variable (`ESRGAN_DIR`, `ZIMAGE_MODEL`)
2. `preferences.json` at the project root
3. Default: `./upscale_models` for ESRGAN, `Tongyi-MAI/Z-Image-Turbo` for Z-Image

Three ways to change them:

- **Gradio UI**: "Paths / models" accordion at the top. Buttons "Refresh ESRGAN
  list", "Apply Z-Image" (invalidates the pipe so it reloads), and "Save to
  preferences.json".
- **CLI**: `--esrgan-dir <path>`, `--zimage-model <repo_or_path>`, `--save-paths`
  to persist (with or without `-i`).
- **Interactive CLI** (`cli.sh` / `cli.bat`): first prompt = ESRGAN folder +
  Z-Image model. Saved to `preferences.json` if you choose to keep them.

`zimage_model` accepts either an HF repo (e.g. `Tongyi-MAI/Z-Image-Turbo`) or a
local path to an already-downloaded `diffusers` folder.

## ESRGAN models

Drop at least one `.pth` or `.safetensors` into `./upscale_models`, or point
`ESRGAN_DIR` (env or prefs) at an existing folder.

A few useful picks:
- `RealESRGAN_x4plus.pth` (general)
- `4x-UltraSharp.pth` (sharp, versatile)
- `4x-ClearRealityV1_Soft.safetensors` (soft, good on portraits/scenes)
- `4xFaceUpDAT.pth` (portraits/faces)

`spandrel` detects the architecture and the scale (x2 / x4) automatically.

---

## Z-Image (first run)

No file to provide: on first launch, `diffusers` fetches the Z-Image transformer,
the VAE and the Qwen3-4B text encoder from Hugging Face, then everything is cached
locally. Subsequent runs are offline.

---

## Running

### 1) Gradio UI (default)

```bash
python app.py
```

UI at http://127.0.0.1:7860. It includes:

- **Before/after slider** (`gradio_imageslider`) that overlays source and result
  with a mouse cursor. Falls back to two side-by-side images if the component is
  not installed.
- **Timing report** under the image: ESRGAN, Z-Image refine, total, source path,
  save path.
- **"Save" section** with the same modes as the CLI.
- **Batch mode**: if you fill in "OR source folder", the uploaded image is ignored
  and the app processes the whole folder.

### 2) Scriptable CLI

```bash
# Single image, explicit settings
python app.py --cli -i my_image.jpg \
    --save-mode local --output-dir out --output-format png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32

# Batch over a whole folder
python app.py --cli -i ./my_images --save-mode local --output-dir out --output-format webp

# Save next to each source ("alongside" mode)
python app.py --cli -i ./my_images --save-mode alongside --output-format jpg

# Display only (no file written), just the timing on stdout
python app.py --cli -i my_image.jpg --save-mode display --denoise 0

# With a TSV log to track timings
python app.py --cli -i ./my_images --save-mode local --time-log runs.tsv
```

### 3) Interactive CLI with preferences

```bash
./cli.sh   # or cli.bat on Windows
```

Prompts for each setting (paths, models, source, pipeline, save, time-log) with a
default value from `preferences.json`. Offers to save the choices at the end of the
session.

## Mapping UI <-> CLI <-> preferences.json

Every UI setting has a CLI flag and a prefs key:

| UI / interactive CLI | CLI flag | preferences.json | Default |
|---|---|---|---|
| ESRGAN_DIR | `--esrgan-dir` | `esrgan_dir` | `./upscale_models` |
| Z-Image model | `--zimage-model` | `zimage_model` | `Tongyi-MAI/Z-Image-Turbo` |
| Source image | `-i` (file or glob) | - | - |
| Batch source folder | `-i` (folder) or `--input-folder` | - | - |
| ESRGAN model | `-m` / `--model` | `model` | `4x-ClearRealityV1_Soft.safetensors` |
| Upscale factor | `--factor` | `factor` | `2.0` |
| Denoise (strength) | `--denoise` | `denoise` | `0.30` |
| Diffusion steps | `--steps` | `steps` | `12` |
| Prompt | `--prompt` | `prompt` | `""` |
| Seed | `--seed` | `seed` | `-1` |
| ESRGAN tile | `--tile` | `tile` | `760` |
| Overlap | `--overlap` | `overlap` | `32` |
| Save mode | `--save-mode` | `save_mode` | `display` |
| Output folder | `--output-dir` | `output_dir` | `out` |
| Output format | `--output-format` | `output_format` | `png` |
| Time log (CLI) | `--time-log <file.tsv>` | `time_log` | (empty) |
| Save paths (CLI) | `--save-paths` | - | - |
| List models (CLI) | `--list-models` | - | - |
| VRAM peak on stderr (CLI) | `--report-vram` | - | - |
| Output path only (CLI) | `--print-output` | - | - |

Save modes:

| save_mode | Behavior |
|---|---|
| `display` | Writes nothing. UI renders the image + timing. CLI prints the report. |
| `local` | Writes to `output_dir`, resolved **relative to the project** if not absolute. |
| `alongside` | Writes to the **same folder as the source**. Requires a source path (CLI or batch folder). |
| `custom` | Writes to `output_dir` as-is (typically an absolute path). |

Default naming: `{source_name}_upscaled.{png|webp|jpg}`. On the CLI, `-o` accepts
a file (overrides auto naming), a folder (equivalent to
`--save-mode local --output-dir <folder>`), or is omitted (uses
`--save-mode` / `--output-dir`).

Full `preferences.json` example:

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

## Timing report

`run()` returns (and prints / logs) the time of each stage:

- `esrgan` : stage 1 (Real-ESRGAN + Lanczos resize)
- `refine` : stage 2 (Z-Image img2img). 0s if `denoise <= 0`.
- `total`  : sum

The UI shows a Markdown block under the image. The CLI prints the report on
stdout (unless `--quiet`). With `--time-log <file>`, each run appends a TSV line:

```
<iso-timestamp>\t<src>\t<dst>\tesrgan=2.24s\trefine=1.87s\tmode=local\tfmt=png
```

---

## External integration (Fooocus, scripts)

Two flags make it easy to call crispz from another tool (separate process):

- `--print-output` : stdout contains ONLY the absolute path of each saved image
  (one per line), nothing else. The human-readable report is suppressed. This is
  the machine-parsable contract for retrieving the result.
- `--report-vram` : run VRAM peak on **stderr** (line `[VRAM] pic alloue:
  X.XX Go | pic reserve: Y.YY Go`). On stderr, so it does not pollute the stdout
  of `--print-output`. Used to size VRAM coexistence (e.g. with Fooocus).

```bash
# The caller reads the output path on stdout, VRAM on stderr
dst=$(python app.py --cli -i in.png --save-mode local --output-dir out \
    --print-output --report-vram 2>vram.log)
echo "upscaled image: $dst"
```

`--print-output` requires a save mode that writes a file
(`local` / `alongside` / `custom`). In `display` nothing is written, so nothing
is printed.

---

## Useful settings

| Setting | Advice |
|---|---|
| **Denoise (strength)** | 0.05-0.25 = subtle, stays very close to the input. 0.25-0.40 = creative, more detail injected. Beyond ~0.40, Z-Image starts to reinvent. At high denoise, a **detailed caption prompt** greatly improves coherence. |
| **Steps** | Effective steps ~= `steps * strength`. At strength 0.30, 12-16 steps give enough denoising steps. |
| **Guidance** | Fixed at 0.0 (Z-Image Turbo). |
| **Prompt** | Optional. Empty works very well if denoise <= 0.30. |
| **Factor** | ESRGAN runs at native x4, then Lanczos resizes to the requested factor. For a clean x2, the image goes through a raw x4. |
| **ESRGAN tiling** | 0 (whole image) on 24+ GB VRAM. 512-768 otherwise. Overlap 32 by default, increase if you see seams. |

The `_hw_check.py` script (called by `run.sh` / `run.bat`) detects your GPU and
gives recommendations based on VRAM, compute capability (BF16 available from
Ampere = CC 8.0), and the max image size for the diffusion pass.

---

## Reference settings

A reliable starting point (source ~832x1216 -> x2), model
`4x-ClearRealityV1_Soft.safetensors`, factor 2, denoise 0.30, steps 12,
tile 760, overlap 32:

```bash
python app.py --cli -i my_image.jpg -o out/my_image_upscaled.png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32
```

---

## Known limitation / next iteration

The Z-Image pass runs on the whole image. Ideal up to ~2048px on the long side;
beyond that you exceed the training resolution and artifacts can appear.

Planned next: tiling of the diffusion pass, Ultimate SD Upscale style, with
feathering, to push to 4K+ without seams.

---

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial). See `LICENSE.txt`.
