"""
app_gradio_gallery.py
---------------------
UI Gradio (Palier A) qui permet :
- d'afficher UNE image (demandé vs trouvé),
- ou une GALERIE de k images.
Les sous-ensembles FairFace dispos sont détectés dynamiquement.
"""

import gradio as gr
from datasets import get_dataset_config_names

from src.data.fairface_constants import ETHNIE_CHOICES
from src.data.fairface_gallery import sample_one_image, sample_k_images

# 🔍 Détecte dynamiquement les configs HF
AVAILABLE_CONFIGS = get_dataset_config_names("HuggingFaceM4/FairFace")
if not AVAILABLE_CONFIGS:  # fallback si pas de réponse réseau
    AVAILABLE_CONFIGS = ["0.25", "1.25"]


def ui_generate(age, gender, ethnie, subset, mode, k):
    """
    Handler Gradio au clic.
    - Retourne (image unique, caption, galerie, compteur).
    - Si erreur (ex. subset invalide), on renvoie None + message clair.
    """
    try:
        if subset not in AVAILABLE_CONFIGS:
            raise ValueError(
                f"Sous-ensemble '{subset}' non disponible. Options valides : {AVAILABLE_CONFIGS}"
            )

        if mode == "Une image":
            img, meta = sample_one_image(age, gender, ethnie, subset=subset)
            caption = (
                f"**Demandé** → âge≈{age}, genre={gender}, ethnie={ethnie}  \n"
                f"**Trouvé** → âge {meta['age_range']}, genre {meta['gender']}, ethnie {meta['race']}  \n"
                f"_Stratégie_: {meta['used_strategy']}, _index_: {meta['index']}"
            )
            return img, caption, None, "1 image"
        else:
            items = sample_k_images(age, gender, ethnie, k=int(k), subset=subset)
            gallery = [
                (
                    pil,
                    f"{m['race']} (id={m['race_id']}) | {m['gender']} | {m['age_range']} | idx={m['index']}",
                )
                for pil, m in items
            ]
            return None, "", gallery, f"{len(gallery)} images"

    except Exception as e:
        # ⚠️ Si une erreur se produit, on renvoie un message explicite
        err_msg = f"**Erreur lors du chargement :** {str(e)}"
        return None, err_msg, None, "0 image"


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎛️ Palier A — Une image ou une galerie (FairFace)")

        subset = gr.Radio(
            choices=AVAILABLE_CONFIGS,
            value=AVAILABLE_CONFIGS[0],
            label="Sous-ensemble FairFace",
        )
        age = gr.Slider(18, 70, value=35, step=1, label="Âge approx.")
        gender = gr.Radio(choices=["Homme", "Femme"], value="Homme", label="Genre")
        ethnie = gr.Dropdown(
            choices=ETHNIE_CHOICES, value="Blanc", label="Ethnie (labels FairFace)"
        )

        with gr.Row():
            mode = gr.Radio(
                choices=["Une image", "Galerie (k images)"],
                value="Galerie (k images)",
                label="Mode d'affichage",
            )
            k = gr.Slider(3, 12, value=6, step=1, label="k (si Galerie)")

        btn = gr.Button("🎲 Afficher")

        out_img = gr.Image(type="pil", label="Image unique")
        out_txt = gr.Markdown()
        out_gallery = gr.Gallery(label="Galerie", columns=4, height=800)
        out_count = gr.Markdown()

        btn.click(
            ui_generate,
            inputs=[age, gender, ethnie, subset, mode, k],
            outputs=[out_img, out_txt, out_gallery, out_count],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
