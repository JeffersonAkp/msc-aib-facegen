# src/ui/app_gradio_palier_b.py
"""
Palier B — Classement + Combinaisons
- Galerie triée (par confiance race via mode strict)
- Visage moyen (sur k)
- Interpolation entre les 2 meilleures images
"""
import gradio as gr
from datasets import get_dataset_config_names

from src.data.fairface_constants import ETHNIE_CHOICES   # ✅ on n'importe que ça
from src.data.fairface_gallery import sample_k_images_strict
from src.utils.img_ops import mean_face, blend

# Configs HF disponibles (fallback si offline)
AVAILABLE_CONFIGS = get_dataset_config_names("HuggingFaceM4/FairFace") or ["0.25", "1.25"]

def ui_generate(mode, subset, age, gender, ethnie, k, k_cand, thr, alpha):
    try:
        # Le strict filtre + classe via le classifieur et renvoie des (PIL, meta)
        items = sample_k_images_strict(
            age=age,
            gender_label=gender,
            race_label=ethnie,             # ✅ libellé direct, pas d’ID
            k=int(k if mode != "Interpolation 2" else max(2, k)),
            subset=subset,
            k_candidates=int(k_cand),
            min_prob=float(thr),
            deterministic=True
        )
        if not items:
            return None, "**Aucun résultat.**", None

        pils  = [p for p, _m in items]
        metas = [m for _p, m in items]

        if mode == "Galerie triée":
            gallery = [
                (
                    p,
                    f"{m['race']} (id={m['race_id']}) | {m['gender']} | {m['age_range']} | "
                    f"score={m.get('score', 0):.2f} | idx={m['index']}"
                )
                for p, m in items
            ]
            return None, f"{len(gallery)} images triées (seuil={thr})", gallery

        if mode == "Visage moyen (k)":
            img = mean_face(pils[:k])
            cap = f"Visage moyen de {min(len(pils), k)} images  \n(seuil={thr}, k_candidats={k_cand})"
            return img, cap, None

        # Interpolation 2
        if len(pils) < 2:
            return None, "Pas assez d'images pour interpoler (2 minimum).", None
        img = blend(pils[0], pils[1], alpha=float(alpha))
        cap = (
            f"Interpolation α={alpha:.2f} entre les 2 meilleures  \n"
            f"idx0={metas[0]['index']} (score={metas[0].get('score',0):.2f}) | "
            f"idx1={metas[1]['index']} (score={metas[1].get('score',0):.2f})"
        )
        return img, cap, None

    except Exception as e:
        return None, f"**Erreur :** {e}", None


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧪 Palier B — Galerie triée, visage moyen, interpolation")

        with gr.Row():
            subset = gr.Radio(choices=AVAILABLE_CONFIGS, value=AVAILABLE_CONFIGS[0], label="Sous-ensemble")
            mode = gr.Radio(choices=["Galerie triée", "Visage moyen (k)", "Interpolation 2"],
                            value="Galerie triée", label="Mode")

        with gr.Row():
            age = gr.Slider(18, 70, value=35, step=1, label="Âge approx.")
            gender = gr.Radio(["Homme", "Femme"], value="Homme", label="Genre")
            ethnie = gr.Dropdown(choices=ETHNIE_CHOICES, value="Blanc", label="Ethnie (FairFace)")

        with gr.Row():
            k = gr.Slider(2, 16, value=8, step=1, label="k (galerie/moyenne)")
            k_cand = gr.Slider(20, 200, value=80, step=10, label="k_candidats (strict)")
            thr = gr.Slider(0.5, 0.99, value=0.7, step=0.01, label="Seuil prob. race (strict)")
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="α (interpolation)")

        btn = gr.Button("⚙️ Générer")

        out_img = gr.Image(type="pil", label="Résultat (image)")
        out_txt = gr.Markdown()
        out_gallery = gr.Gallery(label="Résultat (galerie)", columns=4, height=800)

        btn.click(
            ui_generate,
            inputs=[mode, subset, age, gender, ethnie, k, k_cand, thr, alpha],
            outputs=[out_img, out_txt, out_gallery],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
