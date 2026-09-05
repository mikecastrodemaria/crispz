# Tests crispz

Fonctions pures uniquement : **aucun GPU, aucun modele, aucun reseau**. La suite
tourne en moins d'une seconde et sert de garde-fou sur ce qui casse
silencieusement (alignement des dimensions, ecrasement de fichiers, presets,
recomposition des tuiles, metadonnees).

## Lancer

Sans dependance supplementaire (runner integre) :

```bash
python tests/test_pure.py
```

Avec pytest, s'il est installe :

```bash
python -m pytest tests/ -q
```

Les deux executent exactement les memes tests.

## Ce qui n'est pas couvert

La generation elle-meme (ESRGAN + passe Z-Image) n'est pas testee ici : elle
demande un GPU, un modele ESRGAN dans `upscale_models/` et le telechargement de
Z-Image. Pour valider la chaine complete, utiliser :

```bash
python app.py --cli -i une_image.png --denoise 0 --save-mode local
```

`--denoise 0` fait la passe ESRGAN seule (pas de Z-Image), ce qui valide deja
l'upscale, le tiling et la sauvegarde.
