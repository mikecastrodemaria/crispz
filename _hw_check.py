"""Detection hardware + reco reglages pour crispz.

Imprime un resume lisible. Utilise par run.bat / run.sh / boot_check.bat.

Codes de sortie (pour que les scripts .bat/.sh puissent reagir):
    0 = tout va bien
    1 = PyTorch absent
    2 = CUDA indisponible (CPU seulement) -> passe ESRGAN seule conseillee
    3 = INCOMPATIBLE: ce build torch ne supporte pas l'architecture de ce GPU
        (c'est le cas RTX 50xx + torch non-cu128 -> "WinError 127 torch_cuda.dll")
"""
import sys

# Architectures NVIDIA par compute capability. Sert a nommer le GPU et a savoir
# quel CUDA minimum il exige (Blackwell = 12.8, sinon le build par defaut suffit).
ARCHS = [
    (12, 0, "Blackwell (RTX 50xx)", "12.8"),
    (9, 0, "Hopper (H100)", "12.0"),
    (8, 9, "Ada Lovelace (RTX 40xx)", "11.8"),
    (8, 6, "Ampere (RTX 30xx)", "11.1"),
    (8, 0, "Ampere (A100)", "11.0"),
    (7, 5, "Turing (RTX 20xx / GTX 16xx)", "10.0"),
    (7, 0, "Volta", "9.0"),
    (6, 1, "Pascal (GTX 10xx)", "8.0"),
]


def arch_name(major, minor):
    for ma, mi, name, cuda in ARCHS:
        if (major, minor) >= (ma, mi):
            return name, cuda
    return f"ancienne (sm_{major}{minor})", "?"


def offload_reco(vram_gb):
    """Mode d'offload CPU conseille pour la passe de raffinement, selon la VRAM.

    Z-Image Turbo bf16 pese ~13 Go (transformer + VAE + encodeur Qwen3). Le pic
    reel depend surtout de la taille traitee: la passe img2img sur une image
    entiere en 2K coute nettement plus que la meme passe tuilee en 1024.
    'sequential' deplace chaque sous-module a chaque forward: tres lent
    (~3x le temps par step), a reserver aux petites cartes.
    """
    if vram_gb >= 20:
        return ("none", "Z-Image tient entier en VRAM, meme sur une passe 2K.")
    if vram_gb >= 12:
        return ("none", "Z-Image tient; si OOM au-dela de 1536px, tuiler la passe "
                        "de raffinement (refine_tile 1024) avant de toucher a l'offload.")
    if vram_gb >= 8:
        return ("model", "carte juste: offload par sous-module + refine_tile 1024, "
                         "tile ESRGAN 512.")
    return ("sequential", "VRAM tres limitee: refine_tile 768-1024, ou passe ESRGAN "
                          "seule (denoise = 0) qui n'a pas besoin de Z-Image du tout.")


def main():
    try:
        import torch
    except ImportError:
        print("[ERREUR] PyTorch absent.")
        return 1

    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA non disponible: Z-Image tournerait en CPU (tres lent, deconseille).")
        print("Reco: machine sans GPU NVIDIA -> passe ESRGAN seule (denoise = 0),")
        print("      qui reste utilisable meme si elle est lente.")
        return 2

    i = 0
    props = torch.cuda.get_device_properties(i)
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16 = cap[0] >= 8                     # Ampere et plus
    gen, cuda_min = arch_name(*cap)
    sm = f"sm_{cap[0]}{cap[1]}"

    print(f"GPU             : {name}")
    print(f"Architecture    : {gen}  [{sm}]")
    print(f"VRAM            : {vram_gb:.1f} Go")
    print(f"BF16 natif      : {'oui' if bf16 else 'non (Turing/Pascal, FP16 conseille)'}")

    # --- LE check qui compte: ce build torch sait-il compiler pour ce GPU ? ---
    # Un torch sans le sm_ de la carte se charge mais casse a la 1re allocation
    # CUDA ("WinError 127 ... torch_cuda.dll", ou "no kernel image is available").
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    supported = (not arch_list) or (sm in arch_list)
    print(f"Support {sm:<7}: {'oui' if supported else 'NON'}"
          f"  (build torch: {', '.join(arch_list[-4:]) if arch_list else 'inconnu'})")
    if not supported:
        print()
        print("=" * 62)
        print(f"[INCOMPATIBLE] Ce build PyTorch ne contient pas de noyaux {sm}.")
        print(f"   {gen} exige CUDA {cuda_min}+; ce torch est compile pour CUDA "
              f"{torch.version.cuda}.")
        print("   Symptome typique: 'WinError 127 ... torch_cuda.dll' ou")
        print("   'no kernel image is available for execution on the device'.")
        print("   Correctif:")
        print("     pip uninstall -y torch torchvision torchaudio")
        print(f"     pip install torch torchvision torchaudio "
              f"--index-url https://download.pytorch.org/whl/cu{cuda_min.replace('.', '')}")
        print("=" * 62)
        return 3

    # --- Recommandations (echelonnees selon la VRAM reelle) ---
    off, why = offload_reco(vram_gb)
    if vram_gb >= 20:
        tile, note = 0, "image entiere (tile = 0)"
    elif vram_gb >= 12:
        tile, note = 768, "tile 768, overlap 32"
    elif vram_gb >= 8:
        tile, note = 512, "tile 512, overlap 32"
    else:
        tile, note = 384, "tile 384, overlap 32, baisser si OOM"
    if vram_gb >= 24:
        zsize = "jusqu'a 2048px de cote en image entiere (refine_tile = 0)"
    elif vram_gb >= 12:
        zsize = "jusqu'a ~1536px en image entiere, au-dela refine_tile = 1024"
    else:
        zsize = "refine_tile = 1024 des 1024px de cote"

    print()
    print("--- Reco reglages (UI / CLI) ---")
    print(f"CPU offload     : {off}  <- {why}")
    print(f"Tiling ESRGAN   : {note}   (--tile {tile})")
    print(f"Passe Z-Image   : {zsize}")
    print(f"Dtype           : {'BF16 (defaut)' if bf16 else 'FP16 (mettre DTYPE=torch.float16 dans app.py)'}")
    print(f"Denoise         : 0.20-0.30 conservateur, 0.30-0.40 avec prompt detaille")
    print(f"Steps           : 12-16 a denoise 0.30")
    print(f"Preset conseille: {'Photo (balanced)' if vram_gb >= 12 else 'Low VRAM (8-12GB)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
