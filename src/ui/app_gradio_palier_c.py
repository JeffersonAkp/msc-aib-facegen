# src/ui/app_gradio_palier_c.py
"""
Palier C — Alignement + Moyenne pondérée + Exports
- Aligne/recadre les visages avant calcul
- Moyenne pondérée par score (option)
- Exports: PNG visage moyen, PNG interpolation, ZIP de la galerie, run.json
"""

import gradio as gr
from datasets import get_dataset_config_names

from src.data.fairface_constants import ETHNIE_CHOICES
from src.data.fairface_gallery import sample_k_images_strict
from src.utils.img_ops import (
    align_and_crop_face, normalize_illumination,
    mean_face, mean_face_weighted, blend,
    save_png, make_zip, save_run_json, LastRun
)

# Détection des configs HF
try:
    AVAILABLE_CONFIGS = get_dataset_config_names("HuggingFaceM4/FairFace")
    if not AVAILABLE_CONFIGS:
        AVAILABLE_CONFIGS = ["0.25", "1.25"]
except Exception:
    AVAILABLE_CONFIGS = ["0.25", "1.25"]


def _prepare_images(items, align: bool, norm: bool):
    """items = [(PIL, meta), ...] -> (liste PIL traités, metas)"""
    pils, metas = [], []
    for pil, m in items:
        if align:
            pil = align_and_crop_face(pil, size=224)
        if norm:
            pil = normalize_illumination(pil)
        pils.append(pil)
        metas.append(m)
    return pils, metas


def ui_generate(mode, subset, age, gender, ethnie,
                k, k_cand, thr, alpha,
                align, normalize, weighted):
    try:
        # 1) Top-k via strict (classement par prob)
        items = sample_k_images_strict(
            age=int(age), gender_label=gender, race_label_fr=ethnie,
            k=int(k if mode != "Interpolation 2" else max(2, k)),
            subset=subset, k_candidates=int(k_cand), min_prob=float(thr),
            deterministic=True
        )
        if not items:
            return None, "**Aucun résultat.**", None, None, None, None, None

        # 2) Pré-traitements
        pils, metas = _prepare_images(items, align=align, norm=normalize)

        # 3) Sorties selon mode
        out_img, caption, gallery = None, "", None
        if mode == "Galerie triée":
            gallery = [
                (
                    p,
                    f"{m['race_fr']} (id={m['race_id']}) | {m['gender']} | {m['age_range']} | "
                    f"score={m.get('score', 0):.2f} | idx={m['index']}"
                )
                for p, m in zip(pils, metas)
            ]
            caption = f"{len(gallery)} images triées (seuil={thr})"

        elif mode == "Visage moyen (k)":
            k_int = int(k)
            scores = [m.get("score", 1.0) for m in metas[:k_int]]
            if weighted:
                out_img = mean_face_weighted(pils[:k_int], scores)
                caption = f"Visage moyen pondéré (k={k_int}) • poids=score(strict)"
            else:
                out_img = mean_face(pils[:k_int])
                caption = f"Visage moyen simple (k={k_int})"

        else:  # Interpolation 2
            if len(pils) < 2:
                return None, "Pas assez d'images pour interpoler (2 minimum).", None, None, None, None, None
            out_img = blend(pils[0], pils[1], alpha=float(alpha))
            caption = (
                f"Interpolation α={alpha:.2f} entre les 2 meilleures  \n"
                f"idx0={metas[0]['index']} (score={metas[0].get('score',0):.2f}) | "
                f"idx1={metas[1]['index']} (score={metas[1].get('score',0):.2f})"
            )

        # 4) State pour exports
        state = LastRun(
            subset=subset, mode=mode, age=int(age), gender=gender, ethnie=ethnie,
            k=int(k), k_cand=int(k_cand), thr=float(thr), alpha=float(alpha),
            indices=[m["index"] for m in metas],
            scores=[float(m.get("score", 1.0)) for m in metas],
            pils=pils,
            mean_img=out_img if mode.startswith("Visage moyen") else None,
            interp_img=out_img if mode.startswith("Interpolation") else None
        )

        # note: on retourne aussi 3 placeholders pour les fichiers (qui seront fixés par les callbacks)
        return out_img, caption, gallery, state, None, None, None

    except Exception as e:
        return None, f"**Erreur :** {e}", None, None, None, None, None


# --- Callbacks d'export (renvoient des chemins vers des fichiers) ---
def export_mean(state: LastRun):
    if not state or state.mean_img is None:
        raise gr.Error("Aucun visage moyen en mémoire.")
    path = save_png(state.mean_img, "visage_moyen.png")
    save_run_json(state.__dict__)
    return path

def export_interp(state: LastRun):
    if not state or state.interp_img is None:
        raise gr.Error("Aucune interpolation en mémoire.")
    path = save_png(state.interp_img, "interpolation.png")
    save_run_json(state.__dict__)
    return path

def export_zip(state: LastRun):
    if not state or not state.pils:
        raise gr.Error("Aucune galerie en mémoire.")
    imgs = [(f"img_{i:04d}.png", p) for i, p in enumerate(state.pils)]
    path = make_zip(imgs, "galerie.zip")
    save_run_json(state.__dict__)
    return path


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧪 Palier C — Galerie alignée, moyenne pondérée, exports")

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

        with gr.Row():
            align = gr.Checkbox(True, label="Aligner/recadrer les visages")
            normalize = gr.Checkbox(False, label="Normaliser l’illumination (léger)")
            weighted = gr.Checkbox(True, label="Moyenne pondérée par score")

        btn = gr.Button("⚙️ Générer")

        out_img = gr.Image(type="pil", label="Résultat (image)")
        out_txt = gr.Markdown()
        out_gallery = gr.Gallery(label="Résultat (galerie)", columns=4, height=800)

        # État pour exports
        run_state = gr.State()

        # --- Boutons + Files (compat toutes versions de Gradio) ---
        with gr.Row():
            btn_mean = gr.Button("⬇️ Exporter visage moyen (PNG)")
            btn_interp = gr.Button("⬇️ Exporter interpolation (PNG)")
            btn_zip = gr.Button("⬇️ Exporter galerie (ZIP)")

        mean_file = gr.File(label="visage_moyen.png", visible=False)
        interp_file = gr.File(label="interpolation.png", visible=False)
        zip_file = gr.File(label="galerie.zip", visible=False)

        # Génération principale
        btn.click(
            ui_generate,
            inputs=[mode, subset, age, gender, ethnie, k, k_cand, thr, alpha, align, normalize, weighted],
            outputs=[out_img, out_txt, out_gallery, run_state, mean_file, interp_file, zip_file],
        )

        # Exports (chaque bouton déclenche le calcul et envoie le chemin au File correspondant)
        btn_mean.click(fn=export_mean, inputs=run_state, outputs=mean_file)
        btn_interp.click(fn=export_interp, inputs=run_state, outputs=interp_file)
        btn_zip.click(fn=export_zip, inputs=run_state, outputs=zip_file)

    return demo


if __name__ == "__main__":
    build_demo().launch()
