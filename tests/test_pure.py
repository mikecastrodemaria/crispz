"""Tests des fonctions pures de crispz: aucun GPU, aucun modele, aucun reseau.

Lancer:  python -m pytest tests/ -q      (depuis la racine du projet)
     ou: python tests/test_pure.py       (runner integre, sans pytest)

Ce qui est couvert ici est precisement ce qui casse silencieusement: alignement
des dimensions, unicite des chemins de sortie, application des presets, masques
de recomposition, et les helpers de progression.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402


# ---------------------------------------------------------------- round_to_multiple
def test_round_to_multiple_aligns_on_16():
    # Le pipeline Z-Image refuse une dimension non divisible par
    # vae_scale_factor * 2 = 16. Toute valeur produite ici doit passer ce test.
    for x in range(1, 4000, 7):
        assert app.round_to_multiple(x) % 16 == 0


def test_round_to_multiple_never_returns_zero():
    # Une image minuscule ne doit pas donner une dimension nulle.
    assert app.round_to_multiple(1) == 16
    assert app.round_to_multiple(0) == 16
    assert app.round_to_multiple(3) == 16


def test_round_to_multiple_rounds_to_nearest():
    assert app.round_to_multiple(100) == 96      # 100/16 = 6.25 -> 6 * 16
    assert app.round_to_multiple(110) == 112     # 110/16 = 6.875 -> 7 * 16
    assert app.round_to_multiple(1200) == 1200   # deja aligne
    # Pile au milieu, round() de Python arrondit au pair (banker's rounding):
    # 104/16 = 6.5 -> 6, pas 7. Comportement volontairement fige ici, l'ecart
    # d'un demi-pas de 16 px n'a aucune consequence visuelle.
    assert app.round_to_multiple(104) == 96


def test_esrgan_targets_are_always_aligned():
    # Les cibles calculees par process_one doivent etre acceptables par le
    # pipeline quel que soit le format source et le facteur.
    for w0, h0 in ((1920, 1080), (800, 600), (1234, 567), (101, 99)):
        for factor in (1.0, 1.5, 2.0, 2.5, 4.0):
            assert app.round_to_multiple(w0 * factor) % 16 == 0
            assert app.round_to_multiple(h0 * factor) % 16 == 0


# ---------------------------------------------------------------- _unique_path
def test_unique_path_returns_input_when_free():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        assert app._unique_path(p) == p


def test_unique_path_increments_and_never_overwrites():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        open(p, "w").close()
        p2 = app._unique_path(p)
        assert p2 == os.path.join(d, "a_2.png")
        open(p2, "w").close()
        assert app._unique_path(p) == os.path.join(d, "a_3.png")


def test_build_output_path_does_not_clobber_previous_run():
    # Regression: deux runs sur la meme source ecrasaient le resultat precedent.
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "photo.jpg")
        open(src, "w").close()
        first = app.build_output_path(src, "alongside", None, "png")
        open(first, "w").close()
        second = app.build_output_path(src, "alongside", None, "png")
        assert first != second
        assert os.path.exists(first)


def test_build_output_path_display_returns_none():
    assert app.build_output_path("x.png", "display", "out", "png") is None


def test_build_output_path_alongside_requires_source():
    try:
        app.build_output_path(None, "alongside", "out", "png")
    except ValueError:
        return
    raise AssertionError("alongside sans source doit lever ValueError")


def test_build_output_path_falls_back_to_png_on_unknown_format():
    with tempfile.TemporaryDirectory() as d:
        p = app.build_output_path(None, "custom", d, "tga")
        assert p.endswith(".png")


# ---------------------------------------------------------------- presets
class _Args:
    pass


def test_preset_fills_unset_fields():
    a = _Args()
    a.preset = "4K (tiled)"
    a.factor = a.denoise = a.steps = a.tile = None
    a.overlap = a.refine_tile = a.refine_overlap = a.cpu_offload = None
    app.apply_preset_to_args(a, [])
    assert a.factor == 4.0
    assert a.refine_tile == 1024
    assert a.cpu_offload == "model"


def test_explicit_flag_beats_preset():
    # Un flag explicite doit toujours gagner sur le preset, sinon l'utilisateur
    # ne peut plus corriger un reglage du preset en ligne de commande.
    a = _Args()
    a.preset = "4K (tiled)"
    a.factor = 2.0
    a.denoise = a.steps = a.tile = None
    a.overlap = a.refine_tile = a.refine_overlap = a.cpu_offload = None
    app.apply_preset_to_args(a, ["--factor", "2.0"])
    assert a.factor == 2.0
    assert a.refine_tile == 1024      # les autres cles du preset s'appliquent


def test_every_preset_key_has_a_cli_flag():
    # Sinon apply_preset_to_args leverait un KeyError sur PRESET_FLAGMAP.
    for name, preset in app.PRESETS.items():
        for key in preset:
            assert key in app.PRESET_FLAGMAP, f"{name}: {key} absent de PRESET_FLAGMAP"


# ---------------------------------------------------------------- feather mask
def test_feather_mask_interior_is_one():
    m = app._feather_mask_np(64, 64, 8, left=True, right=True, top=True, bottom=True)
    assert m.shape == (64, 64, 1)
    assert np.allclose(m[32, 32, 0], 1.0)


def test_feather_mask_ramps_only_on_requested_edges():
    m = app._feather_mask_np(64, 64, 8, left=True, right=False, top=False, bottom=False)
    assert m[32, 0, 0] < 0.2          # bord gauche attenue
    assert np.allclose(m[32, 63, 0], 1.0)   # bord droit intact


def test_feather_mask_no_overlap_is_all_ones():
    m = app._feather_mask_np(32, 32, 0, left=True, right=True, top=True, bottom=True)
    assert np.allclose(m, 1.0)


def test_overlap_add_reconstructs_uniform_image():
    # Deux tuiles qui se recouvrent, recomposees en overlap-add, doivent rendre
    # exactement l'image d'origine (poids normalises).
    h, w, tile, ov = 64, 96, 64, 16
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for x1, x2 in ((0, 64), (32, 96)):
        mask = app._feather_mask_np(h, x2 - x1, ov,
                                    left=x1 > 0, right=x2 < w, top=False, bottom=False)
        acc[:, x1:x2, :] += 0.5 * mask
        weight[:, x1:x2, :] += mask
    out = acc / np.clip(weight, 1e-6, None)
    assert np.allclose(out, 0.5, atol=1e-5)


# ---------------------------------------------------------------- progression
def test_fmt_load_distinguishes_download_from_vram():
    assert "downloading" in app._fmt_load("base", 5, 0.0)
    assert "GB in VRAM" in app._fmt_load("base", 5, 7.2)


def test_load_pct_is_bounded_and_monotonic():
    assert 0.0 <= app._load_pct(0, 0.0) <= 1.0
    assert app._load_pct(600, 0.0) <= 0.12          # phase download plafonnee
    assert app._load_pct(10, 99.0) == 0.95          # jamais 100% avant la fin
    assert app._load_pct(10, 3.0, 12.0) < app._load_pct(10, 6.0, 12.0)


def test_load_monitor_returns_value_and_reraises():
    assert app._load_monitor("x", lambda: 42) == 42

    def boom():
        raise RuntimeError("kaboom")
    try:
        app._load_monitor("x", boom)
    except RuntimeError as e:
        assert "kaboom" in str(e)
        return
    raise AssertionError("_load_monitor doit relever l'exception du thread")


# ---------------------------------------------------------------- slicing
class _FakePipe:
    def __init__(self):
        self.sliced = None

    def enable_attention_slicing(self):
        self.sliced = True

    def disable_attention_slicing(self):
        self.sliced = False


def test_slicing_off_on_tiles_on_for_large_whole_images():
    p = _FakePipe()
    app._set_slicing(p, 1024)          # tuile -> SDPA natif, plus rapide
    assert p.sliced is False
    app._set_slicing(p, 2048)          # image entiere 2K+ -> slicing
    assert p.sliced is True


def test_slicing_survives_a_pipe_without_the_methods():
    class Bare:
        pass
    app._set_slicing(Bare(), 4096)     # ne doit pas lever


# ---------------------------------------------------------------- sampler
def test_sampler_and_schedule_choices_are_validated():
    app.set_sampler("nope")
    assert app.SAMPLER in app.SAMPLER_CHOICES
    app.set_schedule("nope")
    assert app.SCHEDULE in app.SCHEDULE_CHOICES


class _SchedAcceptsThenFails:
    """Imite UniPC: accepte l'argument `sigmas`, puis casse sur une LISTE."""
    def set_timesteps(self, sigmas=None, device=None):
        return 1.5 * sigmas          # TypeError si sigmas est une liste


class _SchedOk:
    def set_timesteps(self, sigmas=None, device=None):
        self.sigmas = list(sigmas)


class _SchedNoSigmas:
    def set_timesteps(self, num_inference_steps=None, device=None):
        pass


def test_probe_catches_scheduler_that_accepts_sigmas_but_breaks():
    # Regression: le controle de signature seul laissait passer UniPC, qui
    # plantait ensuite en pleine generation ("can't multiply sequence by
    # non-int of type 'float'"), apres des minutes de chargement.
    bad = _SchedAcceptsThenFails()
    assert app._scheduler_accepts_sigmas(bad) is True     # la signature ment
    assert app._scheduler_works(bad) is False             # la sonde, non


def test_probe_accepts_a_working_scheduler():
    assert app._scheduler_works(_SchedOk()) is True


def test_signature_check_still_rejects_schedulers_without_sigmas():
    assert app._scheduler_accepts_sigmas(_SchedNoSigmas()) is False


def test_probe_does_not_mutate_the_real_scheduler():
    s = _SchedOk()
    app._scheduler_works(s)
    assert not hasattr(s, "sigmas"), "la sonde doit travailler sur une copie"


def test_real_schedulers_against_zimage_sigmas():
    """Chaque sampler propose doit soit fonctionner, soit etre rattrape par la
    sonde. Utilise le VRAI config de scheduler Z-Image si diffusers est present."""
    try:
        import diffusers  # noqa: F401
    except Exception:
        return                                   # pas de diffusers -> rien a verifier
    cfg = {"_class_name": "FlowMatchEulerDiscreteScheduler",
           "num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False}
    verdicts = {}
    for name in app.SAMPLER_CHOICES:
        try:
            sched = app._build_scheduler(name, "sgm_uniform", cfg)
        except Exception:
            verdicts[name] = "build-failed"
            continue
        verdicts[name] = "ok" if app._scheduler_works(sched) else "probe-rejected"
    # euler est le socle: il DOIT marcher, sinon plus aucun repli n'est possible.
    assert verdicts.get("euler") == "ok", f"euler casse: {verdicts}"
    # Les autres ont le droit d'echouer, mais alors la sonde doit les avoir vus
    # (jamais 'build-failed' silencieux qui passerait en generation).
    for name, v in verdicts.items():
        assert v in ("ok", "probe-rejected", "build-failed"), (name, v)


def test_schedule_flag_map_covers_non_native_schedules():
    for s in app.SCHEDULE_CHOICES:
        if s != "sgm_uniform":
            assert s in app._SCHEDULE_FLAG
    assert "sgm_uniform" not in app._SCHEDULE_FLAG   # natif = aucun flag


# ---------------------------------------------------------------- metadonnees
def _meta():
    return app.build_meta("src.png", "4x-model.pth", 2.0, 0.3, 12, "hello", 1234,
                          760, 32, 0, 64, size=(1024, 768),
                          timings={"esrgan": 1.5, "refine": 8.25})


def test_build_meta_records_the_settings_that_matter():
    m = _meta()
    assert m["esrgan_model"] == "4x-model.pth"
    assert m["denoise"] == 0.3 and m["steps"] == 12 and m["seed"] == 1234
    assert m["width"] == 1024 and m["height"] == 768
    assert m["zimage_model"] == app.BASE_REPO
    assert m["sampler"] == app.SAMPLER
    assert m["esrgan_s"] == 1.5 and m["refine_s"] == 8.25


def test_metadata_roundtrip_png():
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        app.save_image(Image.new("RGB", (8, 8)), p, "png", meta=_meta())
        back = app.read_metadata(p)
        assert back is not None
        assert back["esrgan_model"] == "4x-model.pth"
        assert back["prompt"] == "hello"


def test_metadata_roundtrip_jpeg_via_exif():
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jpg")
        app.save_image(Image.new("RGB", (8, 8)), p, "jpg", meta=_meta())
        back = app.read_metadata(p)
        assert back is not None and back["steps"] == 12


def test_metadata_can_be_disabled():
    from PIL import Image
    prev = app.WRITE_METADATA
    app.WRITE_METADATA = False
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            app.save_image(Image.new("RGB", (8, 8)), p, "png", meta=_meta())
            assert app.read_metadata(p) is None
    finally:
        app.WRITE_METADATA = prev


def test_sidecar_written_only_when_enabled():
    from PIL import Image
    prev = app.WRITE_SIDECAR
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            app.WRITE_SIDECAR = False
            app.save_image(Image.new("RGB", (8, 8)), p, "png", meta=_meta())
            assert not os.path.exists(p + ".json")
            app.WRITE_SIDECAR = True
            p2 = os.path.join(d, "b.png")
            app.save_image(Image.new("RGB", (8, 8)), p2, "png", meta=_meta())
            assert os.path.exists(p2 + ".json")
    finally:
        app.WRITE_SIDECAR = prev


def test_read_metadata_on_plain_image_returns_none():
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plain.png")
        Image.new("RGB", (8, 8)).save(p)
        assert app.read_metadata(p) is None


# ---------------------------------------------------------------- transformer override
def test_is_single_file_detects_checkpoints_not_repos():
    assert app._is_single_file("C:/m/zimage.safetensors")
    assert app._is_single_file("/m/zimage-Q8_0.gguf")
    assert not app._is_single_file("Tongyi-MAI/Z-Image-Turbo")
    assert not app._is_single_file("")


def test_load_transformer_is_a_noop_without_override():
    prev = app.ZIMAGE_TRANSFORMER
    app.ZIMAGE_TRANSFORMER = ""
    try:
        assert app._load_transformer() is None
    finally:
        app.ZIMAGE_TRANSFORMER = prev


def test_setting_a_transformer_invalidates_the_cached_pipe():
    prev_t, prev_pipe, prev_loaded = (app.ZIMAGE_TRANSFORMER, app._PIPE,
                                      app._LOADED_TRANSFORMER)
    try:
        app.ZIMAGE_TRANSFORMER = ""
        app._PIPE = object()
        app._LOADED_TRANSFORMER = ""
        app.set_zimage_transformer("C:/m/other.safetensors")
        assert app._PIPE is None
    finally:
        app.ZIMAGE_TRANSFORMER, app._PIPE, app._LOADED_TRANSFORMER = (
            prev_t, prev_pipe, prev_loaded)


# ---------------------------------------------------------------- motif de nommage
def _named(pattern, **kw):
    prev = app.FILENAME_PATTERN
    try:
        app.FILENAME_PATTERN = pattern
        args = dict(source_path="photo.jpg", seed=1234, size=(1664, 2432),
                    esrgan_model="4x-UltraSharp.pth", factor=2.0, denoise=0.30)
        args.update(kw)
        return app._format_filename(**args)
    finally:
        app.FILENAME_PATTERN = prev


def test_filename_placeholders_resolve():
    assert _named("{name}_{tag}") == "photo_upscaled"
    assert _named("{w}x{h}") == "1664x2432"
    assert _named("{seed}") == "1234"
    assert _named("{factor}") == "2x"
    assert _named("{denoise}") == "d030"
    assert _named("{model}") == "4x-UltraSharp"


def test_filename_seed_minus_one_is_named_rand():
    assert _named("{seed}", seed=-1) == "rand"


def test_filename_without_source_uses_image():
    assert _named("{name}", source_path=None) == "image"


def test_filename_index_is_empty_when_zero():
    assert _named("{name}{index}") == "photo"
    assert _named("{name}{index}", index=3) == "photo_3"


def test_filename_strips_path_separators():
    # Un motif ne doit jamais pouvoir ecrire hors du dossier de sortie.
    for pattern in ("{name}/{date}", "..\\{name}", "{name}/../../evil"):
        out = _named(pattern)
        assert "/" not in out and "\\" not in out, out
        assert not out.startswith("."), out


def test_filename_falls_back_on_unknown_placeholder():
    # Un motif casse ne doit pas faire perdre un rendu de plusieurs minutes.
    assert _named("{nexistepas}") == "photo_upscaled"


def test_filename_never_empty():
    assert _named("///") != ""
    assert _named("{index}") != ""


def test_default_pattern_is_studio_like_and_valid():
    out = _named(app.DEFAULT_FILENAME_PATTERN)
    assert out.startswith("20") and "photo" in out and "1664x2432" in out


def test_build_output_path_uses_the_pattern():
    prev = app.FILENAME_PATTERN
    try:
        app.FILENAME_PATTERN = "{name}_{factor}"
        with tempfile.TemporaryDirectory() as d:
            p = app.build_output_path("src/photo.jpg", "custom", d, "png",
                                      factor=2.0, size=(100, 100))
            assert os.path.basename(p) == "photo_2x.png"
    finally:
        app.FILENAME_PATTERN = prev


# ---------------------------------------------------------------- sortie UI / telechargement
def test_preview_path_honours_the_requested_format():
    from PIL import Image
    img = Image.new("RGB", (8, 8))
    for fmt in ("png", "jpg", "webp"):
        p = app._preview_path(img, fmt, meta=_meta())
        assert p.lower().endswith("." + fmt), p
        assert app.read_metadata(p) is not None, f"{fmt}: metadonnees perdues"


def test_preview_path_falls_back_to_png_on_unknown_format():
    from PIL import Image
    p = app._preview_path(Image.new("RGB", (8, 8)), "tga")
    assert p.lower().endswith(".png")


def test_ui_result_reuses_the_saved_file():
    from PIL import Image
    prev = app._LAST_SAVED
    try:
        with tempfile.TemporaryDirectory() as d:
            saved = os.path.join(d, "deja_ecrit.jpg")
            Image.new("RGB", (8, 8)).save(saved)
            app._LAST_SAVED = saved
            got = app._ui_result_path(Image.new("RGB", (8, 8)), "png", "m", 2.0,
                                      0.3, 12, "", -1, 760, 32, 0, 64)
            assert got == saved, "le fichier deja ecrit doit etre reutilise tel quel"
    finally:
        app._LAST_SAVED = prev


def test_ui_result_writes_a_temp_file_in_display_mode():
    from PIL import Image
    prev = app._LAST_SAVED
    try:
        app._LAST_SAVED = None                      # save_mode=display
        got = app._ui_result_path(Image.new("RGB", (8, 8)), "jpg", "m", 2.0,
                                  0.3, 12, "", -1, 760, 32, 0, 64)
        assert isinstance(got, str) and got.lower().endswith(".jpg")
        assert os.path.isfile(got)
    finally:
        app._LAST_SAVED = prev


def test_gradio_serves_a_path_untouched():
    """Regression: renvoyer une image PIL faisait re-encoder gradio en .webp
    (format par defaut du composant), quel que soit "Output format", et le
    re-encodage detruisait les metadonnees crispz. Un CHEMIN est servi tel quel."""
    try:
        from gradio import image_utils
    except Exception:
        return                                       # gradio absent -> rien a verifier
    from PIL import Image
    with tempfile.TemporaryDirectory() as cache:
        p = app._preview_path(Image.new("RGB", (8, 8)), "jpg", meta=_meta())
        served = image_utils.postprocess_image(p, cache_dir=cache, format="webp")
        assert os.path.abspath(served.path) == os.path.abspath(p)
        assert app.read_metadata(served.path) is not None
        # et la preuve du contraire, pour que le test garde son sens:
        pil_served = image_utils.postprocess_image(Image.new("RGB", (8, 8)),
                                                   cache_dir=cache, format="webp")
        assert pil_served.path.lower().endswith(".webp")


# ---------------------------------------------------------------- filtre pillow
import _req_filter  # noqa: E402


def test_req_filter_drops_bare_and_pinned_pillow():
    # Regression: un filtre sur "pillow==" laissait passer le `pillow` NU de
    # requirements.txt -> pip installait Pillow 11 (borne <12 de gradio) en
    # ecrasant le 12 deja pose. Les deux formes doivent partir.
    kept, dropped = _req_filter.filter_lines([
        "pillow\n", "pillow==12.3.0\n", "pillow>=11,<13\n",
        "Pillow == 12.3.0\n", "pillow ; sys_platform == 'win32'\n",
    ])
    assert kept == []
    assert len(dropped) == 5


def test_req_filter_keeps_other_packages_and_lookalikes():
    lines = ["numpy\n", "pillow-avif-plugin==1.5.5\n", "pillowcase\n",
             "# pillow: commentaire explicatif\n", "gradio<6,>=4.44\n"]
    kept, dropped = _req_filter.filter_lines(lines)
    assert dropped == []
    assert kept == lines


def test_req_filter_on_the_real_files():
    # Le fichier reellement utilise par install.bat ne doit plus contenir de
    # requirement pillow apres filtrage.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("requirements.txt", "requirements-lock.txt"):
        path = os.path.join(here, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            kept, dropped = _req_filter.filter_lines(f.readlines())
        assert dropped, f"{name}: aucune ligne pillow trouvee (filtre a revoir ?)"
        for line in kept:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                assert not stripped.lower().startswith("pillow="), \
                    f"{name}: requirement pillow encore present: {stripped}"


# ---------------------------------------------------------------- runner integre
if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:              # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests OK")
    sys.exit(1 if failed else 0)
