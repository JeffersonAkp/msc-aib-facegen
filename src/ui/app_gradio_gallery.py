"""
app_gradio_gallery.py
---------------------
UI Gradio (Palier A) :
- affiche UNE image (demandé vs trouvé)
- ou une GALERIE de k images
Les sous-ensembles FairFace dispos sont détectés dynamiquement.

Note : le "Mode strict" nécessite un poids local pour le classifieur :
weights/fairface_race_resnet18.pth
"""

import gradio as gr
from datasets import get_dataset_config_names

from src.data.fairface_constants import ETHNIE_CHOICES
from src.data.fairface_gallery import (
    sample_one_image,
    sample_k_images,
    sample_k_images_strict,
)

# 🔍 Détection des configs HF (avec fallback)
try:
    AVAILABLE_CONFIGS = get_dataset_config_names("HuggingFaceM4/FairFace")
    if not AVAILABLE_CONFIGS:
        AVAILABLE_CONFIGS = ["0.25", "1.25"]
except Exception:
    AVAILABLE_CONFIGS = ["0.25", "1.25"]


def ui_generate(age, gender, ethnie, subset, mode, k, strict, k_cand, thr):
    """
    Handler Gradio au clic.
    Retourne: (image unique, caption, galerie, compteur).
    """
    try:
        if subset not in AVAILABLE_CONFIGS:
            raise ValueError(
                f"Sous-ensemble '{subset}' non disponible. Options valides : {AVAILABLE_CONFIGS}"
            )

        if mode == "Une image":
            # (simple : pas de revalidation "strict" sur une seule image)
            img, meta = sample_one_image(age, gender, ethnie, subset=subset)
            caption = (
                f"**Demandé** → âge≈{age}, genre={gender}, ethnie={ethnie}  \n"
                f"**Trouvé** → âge {meta['age_range']}, genre {meta['gender']}, "
                f"ethnie {meta['race_fr']} (id={meta['race_id']})  \n"
                f"_Stratégie_: {meta['used_strategy']}, _index_: {meta['index']}"
            )
            return img, caption, None, "1 image"

        # Mode "Galerie"
        k = int(k)
        if strict:
            items = sample_k_images_strict(
                age, gender, ethnie,
                k=k, subset=subset,
                k_candidates=int(k_cand),
                min_prob=float(thr),
                deterministic=True,
            )
        else:
            items = sample_k_images(age, gender, ethnie, k=k, subset=subset)

        gallery = [
            (
                pil,
                f"{m['race_fr']} (id={m['race_id']}) | {m['gender']} | {m['age_range']} | idx={m['index']}"
            )
            for pil, m in items
        ]
        return None, "", gallery, f"{len(gallery)} images"

    except Exception as e:
        err_msg = f"**Erreur :** {str(e)}"
        return None, err_msg, None, "0 image"


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎛️ Palier A — Dataset FairFace (une image ou galerie)")

        subset = gr.Radio(
            choices=AVAILABLE_CONFIGS,
            value=AVAILABLE_CONFIGS[0],
            label="Sous-ensemble FairFace",
        )
        age = gr.Slider(18, 70, value=35, step=1, label="Âge approx.")
        gender = gr.Radio(choices=["Homme", "Femme"], value="Homme", label="Genre")

        # ✅ libellés d'ethnies issus de la source de vérité (constants)
        ethnie = gr.Dropdown(
            choices=ETHNIE_CHOICES,
            value="Blanc",  # par défaut, tu peux mettre ETHNIE_CHOICES[0]
            label="Ethnie (labels FairFace)"
        )

        with gr.Row():
            mode = gr.Radio(
                choices=["Une image", "Galerie (k images)"],
                value="Galerie (k images)",
                label="Mode d'affichage",
            )
            k = gr.Slider(3, 12, value=6, step=1, label="k (si Galerie)")

        # Options du mode strict (revalidation par classifieur)
        with gr.Row():
            strict = gr.Checkbox(
                value=False, label="Mode strict (revalider avec classifieur race)"
            )
            k_cand = gr.Slider(
                20, 120, value=60, step=10, label="k_candidats (strict)"
            )
            thr = gr.Slider(
                0.5, 0.95, value=0.7, step=0.05, label="Seuil de probabilité (strict)"
            )

        btn = gr.Button("🎲 Afficher")

        out_img = gr.Image(type="pil", label="Image unique")
        out_txt = gr.Markdown()
        out_gallery = gr.Gallery(label="Galerie", columns=4, height=800)
        out_count = gr.Markdown()

        btn.click(
            ui_generate,
            inputs=[age, gender, ethnie, subset, mode, k, strict, k_cand, thr],
            outputs=[out_img, out_txt, out_gallery, out_count],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
