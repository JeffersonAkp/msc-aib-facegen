# src/ui/app_gradio_palier_d.py
import gradio as gr
from pathlib import Path
from src.models.stylegan2 import StyleGAN2

WEIGHTS = Path("weights/ffhq-1024.pkl")

_G = None
def _get_model():
    global _G
    if _G is None:
        if not WEIGHTS.exists():
            raise RuntimeError(f"Poids introuvable: {WEIGHTS}. Place ffhq-1024.pkl dans /weights")
        _G = StyleGAN2(str(WEIGHTS))  # device auto (cpu si pas de GPU)
    return _G

def ui_generate(seed, trunc, out_size):
    G = _get_model()
    img = G.generate(seed=int(seed), truncation=float(trunc), out_size=int(out_size))
    return img, f"Seed={int(seed)} | trunc={float(trunc):.2f} | {int(out_size)}px"

def ui_interpolate(seed_a, seed_b, alpha, trunc, out_size):
    G = _get_model()
    img = G.interpolate(seed_a=int(seed_a), seed_b=int(seed_b),
                        alpha=float(alpha), truncation=float(trunc), out_size=int(out_size))
    return img, f"A={int(seed_a)} → B={int(seed_b)} | α={float(alpha):.2f} | trunc={float(trunc):.2f} | {int(out_size)}px"

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧬 Palier D — StyleGAN2 (génération synthétique)")

        with gr.Tab("Génération"):
            seed = gr.Number(value=42, precision=0, label="Seed")
            trunc = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Truncation ψ")
            out_size = gr.Radio([256, 512, 1024], value=256, label="Taille (px)")  # 256 par défaut (CPU)
            btn = gr.Button("🎲 Générer")
            out_img = gr.Image(type="pil", label="Résultat")
            out_txt = gr.Markdown()
            btn.click(ui_generate, [seed, trunc, out_size], [out_img, out_txt])

        with gr.Tab("Interpolation"):
            seed_a = gr.Number(value=10, precision=0, label="Seed A")
            seed_b = gr.Number(value=100, precision=0, label="Seed B")
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="α (A→B)")
            trunc2 = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Truncation ψ")
            out_size2 = gr.Radio([256, 512, 1024], value=256, label="Taille (px)")
            btn2 = gr.Button("🧪 Interpoler")
            out_img2 = gr.Image(type="pil", label="Résultat")
            out_txt2 = gr.Markdown()
            btn2.click(ui_interpolate, [seed_a, seed_b, alpha, trunc2, out_size2], [out_img2, out_txt2])

    return demo

if __name__ == "__main__":
    build_demo().launch()
