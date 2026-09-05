"""Retire Pillow d'un fichier de dependances, en ecrivant une copie filtree.

POURQUOI
    gradio 5.x declare "pillow<12", alors que les CVE de decodage d'image
    -- directement atteignables ici, l'app ouvrant des images fournies par
    l'utilisateur -- ne sont corrigees qu'en Pillow 12.x. Resoudre les deux
    ensemble donne ResolutionImpossible. On retire donc Pillow du graphe, puis
    on l'installe a part en --no-deps (cf. install.bat / install.sh).

POURQUOI PAS UN findstr / grep SUR "pillow=="
    requirements.txt contient un `pillow` NU et requirements-lock.txt un
    `pillow==12.3.0`. Un filtre sur "pillow==" laisse passer le premier. Il faut
    matcher le NOM DE PAQUET, pas un prefixe de ligne -- sans pour autant
    emporter `pillow-avif-plugin`.

CE QUE CE FILTRE NE FAIT PAS
    Il ne supprime PAS le va-et-vient Pillow 11 -> 12 pendant l'install. gradio
    declare lui-meme `pillow<12.0,>=8.0`: pip resout cette dependance
    TRANSITIVE quoi qu'il arrive et redescend Pillow sous 12, avant que l'etape
    `--no-deps pillow==12.3.0` ne le remonte. Le filtre evite seulement le
    conflit FRONTAL (un `pillow==12.3.0` direct face a la borne de gradio =
    ResolutionImpossible). La vraie protection contre une regression silencieuse
    est la verification de version en fin d'install.

Usage:  python _req_filter.py <source> <destination>
"""
import re
import sys

# Nom de paquet exact "pillow", suivi de la fin de ligne ou d'un separateur de
# specification (==, >=, <, ~=, !=, ; marker, [extra]). `pillow-avif-plugin`
# n'est PAS concerne: apres "pillow" vient un '-', absent de la classe.
_PILLOW = re.compile(r"^\s*pillow\s*($|[=<>!~;\[#])", re.IGNORECASE)


def filter_lines(lines):
    """Renvoie (lignes_gardees, lignes_retirees). Pur, testable."""
    kept, dropped = [], []
    for line in lines:
        (dropped if _PILLOW.match(line) else kept).append(line)
    return kept, dropped


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, dst = argv[1], argv[2]
    with open(src, encoding="utf-8") as f:
        kept, dropped = filter_lines(f.readlines())
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(kept)
    for line in dropped:
        print(f"  (pillow retire du graphe: {line.strip()} -> pose ensuite en --no-deps)")
    if not dropped:
        print("  (aucune ligne pillow trouvee dans le fichier de deps)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
