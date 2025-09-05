from src.data.fairface_constants import ETHNIE_CHOICES
from src.data.multiplefairface_filter import sample_image_by_attributes, sample_k_images_by_attributes

def ui_generate(age, gender, ethnie, subset, mode, k):
    if mode == "Une image":
        img, meta = sample_image_by_attributes(int(age), gender, ethnie, subset=subset)
        caption = (
            f"**Demandé** → âge≈{age}, genre={gender}, ethnie={ethnie}  \n"
            f"**Trouvé** → âge {meta['age_range']}, genre {meta['gender']}, ethnie {meta['race']}  \n"
            f"_Stratégie_: {meta['matched_strategy']}, _index_: {meta['index']}"
        )
        return img, caption, None  # image unique, pas de galerie
    else:
        items = sample_k_images_by_attributes(int(age), gender, ethnie, k=int(k), subset=subset)
        # Gradio Gallery attend une liste d’images ou [(image, caption)]
        gallery = []
        for pil, meta in items:
            cap = f"{meta['race']} (id={meta['race_id']}) | {meta['gender']} | {meta['age_range']} | idx={meta['index']}"
            gallery.append((pil, cap))
        return None, "", gallery  # pas d’image unique, mais galerie

with gr.Blocks() as demo:
    gr.Markdown("# 🎛️ Palier A — Filtrer FairFace par attributs (aperçu)")

    subset = gr.Radio(choices=["0.25", "1.0"], value="0.25", label="Sous-ensemble FairFace")
    age = gr.Slider(18, 70, value=35, step=1, label="Âge approximatif")
    gender = gr.Radio(choices=["Homme", "Femme"], value="Homme", label="Genre")
    ethnie = gr.Dropdown(choices=ETHNIE_CHOICES, value="Blanc", label="Ethnie (labels FairFace)")

    with gr.Row():
        mode = gr.Radio(choices=["Une image", "Galerie (k images)"], value="Une image", label="Mode d'affichage")
        k = gr.Slider(3, 12, value=6, step=1, label="k (si Galerie)")

    btn = gr.Button("🎲 Afficher")

    out_img = gr.Image(type="pil", label="Image unique")
    out_txt = gr.Markdown()
    out_gallery = gr.Gallery(label="Galerie", columns=3, height=400)

    btn.click(ui_generate, inputs=[age, gender, ethnie, subset, mode, k], outputs=[out_img, out_txt, out_gallery])

if __name__ == "__main__":
    demo.launch()
