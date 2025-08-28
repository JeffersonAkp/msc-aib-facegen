import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image

from src.models.baseline_dummy import generate_dummy_face

def ui_generate(age, gender, skin_tone, size):
    # 1) génère un tenseur (1, 3, H, W) dans [-1, 1]
    img = generate_dummy_face(n=1, size=size)[0]
    # 2) remap vers [0,1] pour affichage
    img = (img + 1) / 2
    # 3) convertit en image PIL
    pil = to_pil_image(img.cpu())
    return pil

with gr.Blocks() as demo:
    gr.Markdown("# Génération de visage (MVP)")
    with gr.Row():
        age = gr.Slider(18, 70, value=30, step=1, label="Âge")
        gender = gr.Radio(choices=["Homme", "Femme"], value="Homme", label="Genre")
        skin = gr.Dropdown(choices=["Clair","Moyen","Foncé"], value="Moyen", label="Teint")
        size = gr.Slider(32, 256, value=64, step=32, label="Taille")
    btn = gr.Button("Générer")
    out = gr.Image(type="pil")   # 👈 IMPORTANT : on reçoit directement une image PIL
    btn.click(ui_generate, inputs=[age, gender, skin, size], outputs=out)

if __name__ == "__main__":
    demo.launch()
