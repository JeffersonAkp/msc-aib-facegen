# src/ui/app_gradio_palier_d.py
"""
Palier D — StyleGAN2 (génération) : 3 modes
1) Simple (attributs) : cible Genre / Ethnie / Âge façon Palier A/B/C
   - Cherche un seed dont la race (FairFace) matche
   - Optionnel : petits nudges latents (age/gender/race) si directions .npy dispo
2) Génération (seed, truncation, taille)
3) Interpolation (entre 2 seeds)

Prérequis :
- weights/ffhq-1024.pkl  (poids StyleGAN2-ADA)
- weights/fairface_race_resnet18.pth  (classifieur race)  <-- déjà utilisé en A/B/C
- directions *.npy dans ./directions/ (facultatif mais recommandé)
"""

import os, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from src.models.stylegan2 import StyleGAN2, load_directions
from src.models.fairface_race_clf import FairFaceRaceClassifier  # ton classifieur existant
from src.data.fairface_constants import ETHNIE_CHOICES, FR_TO_ID  # mêmes libellés qu’en A/B/C

# --- chemins ---
WEIGHTS_SG = Path("weights/ffhq-1024.pkl")
WEIGHTS_RACE = Path("weights/fairface_race_resnet18.pth")
DIR_DIRECTIONS = Path("directions")  # age.npy, gender.npy, race.npy si dispo

# --- charge paresseusement ---
_g_model: Optional[StyleGAN2] = None
_g_clf: Optional[FairFaceRaceClassifier] = None
_g_dirs: Dict[str, "np.ndarray"] = {}

def _get_stylegan() -> StyleGAN2:
    global _g_model
    if _g_model is None:
        if not WEIGHTS_SG.is_file():
            raise RuntimeError(f"Poids SG introuvable : {WEIGHTS_SG}")
        _g_model = StyleGAN2(str(WEIGHTS_SG), device="cpu")  # CPU OK pour 256px
    return _g_model

def _get_race_clf() -> FairFaceRaceClassifier:
    global _g_clf
    if _g_clf is None:
        if not WEIGHTS_RACE.is_file():
            raise RuntimeError(f"Poids race introuvable : {WEIGHTS_RACE}")
        _g_clf = FairFaceRaceClassifier(str(WEIGHTS_RACE))
    return _g_clf

def _get_dirs() -> Dict[str, "np.ndarray"]:
    global _g_dirs
    if not _g_dirs:
        expected = {
            "age": "age.npy",
            "gender": "gender.npy",
            "race": "race.npy",  # proxy teint (⚠️ approximatif)
        }
        _g_dirs = load_directions(str(DIR_DIRECTIONS), expected)
    return _g_dirs

# ---------- mode SIMPLE (attributs) ----------
def _race_proxy_strength(target_fr: str) -> float:
    """
    Si 'race.npy' est dispo, on applique une petite intensité par défaut.
    Ce mapping est heuristique (proxy de teint), à manier avec prudence.
    """
    table = {
        "Blanc": -1.2,
        "Noir": +1.8,
        "Indien": +0.8,
        "Asiatique Est": -0.6,
        "Asiatique SE": +0.2,
        "Latino": +0.4,
        "Moyen-Oriental": +0.7,
    }
    return table.get(target_fr, 0.0)

def _age_to_intensity(age: int, neutral: int = 35) -> float:
    # map approx linéaire → clip [-3, 3]
    x = (int(age) - neutral) / 10.0
    return max(-3.0, min(3.0, x))

def simple_generate(
    seed_start: int,
    truncation: float,
    size: int,
    gender: str,
    ethnie: str,
    age: int,
    max_tries: int,
    thr_race: float,
    use_latent_nudges: bool,
    global_scale: float,
) -> Tuple[Image.Image, str]:
    """
    Cherche un seed dont la RACE prédit (FairFace) == cible.
    Puis applique (optionnel) des nudges latents : age/gender/race.
    """
    G = _get_stylegan()
    clf = _get_race_clf()
    dirs = _get_dirs() if use_latent_nudges else {}

    target_rid = FR_TO_ID[ethnie]

    found_img: Optional[Image.Image] = None
    found_seed: Optional[int] = None
    found_prob: float = 0.0

    # 1) seed search guidé par classifieur race
    for s in range(seed_start, seed_start + max_tries):
        img = G.generate(seed=s, truncation=truncation, out_size=size)
        p = clf.predict_proba([img]).numpy()[0]  # (7,)
        rid = int(p.argmax())
        if rid == target_rid and float(p[rid]) >= float(thr_race):
            found_img = img
            found_seed = s
            found_prob = float(p[rid])
            break
        # garde le meilleur match au cas où
        if float(p[rid]) > found_prob:
            found_img = img
            found_seed = s
            found_prob = float(p[rid])

    assert found_img is not None and found_seed is not None

    # 2) petits nudges latents (optionnels)
    if use_latent_nudges and dirs:
        coeffs: Dict[str, float] = {}

        # age : map 18..70 -> [-3,+3]
        if "age" in dirs:
            coeffs["age"] = _age_to_intensity(age)

        # gender : - masculin, + féminin (heuristique)
        if "gender" in dirs:
            coeffs["gender"] = +2.0 if gender == "Femme" else -2.0

        # race proxy (teint) : très approximatif (⚠️)
        if "race" in dirs:
            coeffs["race"] = _race_proxy_strength(ethnie)

        img2 = G.edit_with_directions(
            seed=found_seed,
            truncation=truncation,
            coeffs=coeffs,
            directions=dirs,
            out_size=size,
            scale=float(global_scale),
        )
        found_img = img2

    cap = (
        f"Mode simple — seed={found_seed} | trunc={truncation:.2f} | "
        f"Taille={size}px | Race prédite p={found_prob:.2f}\n"
        f"Nudges: {'ON' if (use_latent_nudges and dirs) else 'OFF'} | "
        f"Directions chargées: {', '.join(sorted(_get_dirs().keys())) or 'aucune'}"
    )
    return found_img, cap

# ---------- UI ----------
def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧬 Palier D — StyleGAN2 (génération, interpolation, **mode simple par attributs**)")
        with gr.Tabs():
            # ----- 1) SIMPLE -----
            with gr.Tab("Simple (attributs)"):
                with gr.Row():
                    seed0 = gr.Number(value=0, precision=0, label="Seed de départ (cherche ensuite)")
                    trunc = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Truncation ψ")
                    size = gr.Radio(choices=[256, 512, 1024], value=256, label="Taille (px)")

                with gr.Row():
                    gender = gr.Radio(["Homme", "Femme"], value="Homme", label="Genre")
                    ethnie = gr.Dropdown(choices=ETHNIE_CHOICES, value="Blanc", label="Ethnie (FairFace)")
                    age = gr.Slider(18, 70, value=35, step=1, label="Âge (approx.)")

                with gr.Row():
                    max_tries = gr.Slider(10, 200, value=40, step=5, label="Nb max de seeds testés")
                    thr_race = gr.Slider(0.5, 0.99, value=0.70, step=0.01, label="Seuil proba race (FairFace)")
                    use_nudge = gr.Checkbox(value=True, label="Affiner via directions latentes (si dispo)")
                    gscale = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Échelle globale des nudges")

                btn_simple = gr.Button("🎯 Générer (mode simple)")
                out_img_s = gr.Image(type="pil", label="Résultat")
                out_txt_s = gr.Markdown()

                btn_simple.click(
                    simple_generate,
                    inputs=[seed0, trunc, size, gender, ethnie, age, max_tries, thr_race, use_nudge, gscale],
                    outputs=[out_img_s, out_txt_s],
                )

            # ----- 2) GÉNÉRATION brute -----
            with gr.Tab("Génération"):
                seed = gr.Number(value=42, precision=0, label="Seed")
                trunc2 = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Truncation ψ")
                size2 = gr.Radio(choices=[256, 512, 1024], value=256, label="Taille (px)")
                btn_gen = gr.Button("🎲 Générer")

                out_img = gr.Image(type="pil", label="Résultat")
                out_txt = gr.Markdown()

                def ui_generate(seed, truncation, size):
                    img = _get_stylegan().generate(int(seed), float(truncation), int(size))
                    return img, f"Seed={int(seed)} | trunc={truncation:.2f} | {int(size)}px"

                btn_gen.click(ui_generate, inputs=[seed, trunc2, size2], outputs=[out_img, out_txt])

            # ----- 3) INTERPOLATION -----
            with gr.Tab("Interpolation"):
                s0 = gr.Number(value=1, precision=0, label="Seed A")
                s1 = gr.Number(value=2, precision=0, label="Seed B")
                alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="α (A→B)")
                trunc3 = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Truncation ψ")
                size3 = gr.Radio(choices=[256, 512, 1024], value=256, label="Taille (px)")
                btn_interp = gr.Button("🔀 Interpoler")

                out_img_i = gr.Image(type="pil", label="Résultat")
                out_txt_i = gr.Markdown()

                def ui_interp(sa, sb, a, truncation, size):
                    img = _get_stylegan().interpolate(int(sa), int(sb), float(a), float(truncation), int(size))
                    return img, f"Seeds=({int(sa)},{int(sb)}) | α={a:.2f} | trunc={truncation:.2f} | {int(size)}px"

                btn_interp.click(ui_interp, inputs=[s0, s1, alpha, trunc3, size3], outputs=[out_img_i, out_txt_i])

    return demo


if __name__ == "__main__":
    build_demo().launch()
