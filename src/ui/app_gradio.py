"""
Interface Gradio (Palier A) :
- L'utilisateur choisit âge / genre / ethnie
- On affiche une image de FairFace correspondant "au mieux"
- On montre aussi les attributs réellement trouvés (sanity check)
"""

import gradio as gr

from src.data.fairface_constants import ETHNIE_CHOICES
from src.data.fairface_filter import sample_image_by_attributes


def ui_generate(age: int, gender: str, ethnie: str, subset: str):
    """
    Fonction appelée au clic sur le bouton.
    Retourne (image PIL, caption Markdown).
    """
    img, meta = sample_image_by_attributes(int(age), gender, ethnie, subset=subset)
    caption = (
        f"**Demandé** → âge≈{age}, genre={gender}, ethnie={ethnie}  \n"
        f"**Trouvé** → âge {meta['age_range']}, genre {meta['gender']}, ethnie {meta['race']}  \n"
        f"_Stratégie_: {meta['matched_strategy']}, _index_: {meta['index']}"
    )
    return img, caption


with gr.Blocks() as demo:
    gr.Markdown("# 🎛️ Palier A — Filtrer FairFace par attributs (aperçu)")

    # Choix du sous-ensemble (0.25 = léger pour tests)
    subset = gr.Radio(choices=["0.25", "1.0"], value="0.25", label="Sous-ensemble FairFace")

    # Contrôles utilisateur
    age = gr.Slider(18, 70, value=35, step=1, label="Âge approximatif")
    gender = gr.Radio(choices=["Homme", "Femme"], value="Homme", label="Genre")
    ethnie = gr.Dropdown(choices=ETHNIE_CHOICES, value="Blanc", label="Ethnie (labels FairFace)")

    # Bouton d'action
    btn = gr.Button("🎲 Afficher une image")

    # Sorties
    out_img = gr.Image(type="pil", label="Image issue de FairFace")
    out_txt = gr.Markdown()

    # Wiring : au clic, on appelle ui_generate
    btn.click(ui_generate, inputs=[age, gender, ethnie, subset], outputs=[out_img, out_txt])


if __name__ == "__main__":
    demo.launch()
