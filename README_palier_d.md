# Palier D — StyleGAN2 (génération + édition d’attributs)

## Prérequis
- `weights/ffhq-1024.pkl` (poids StyleGAN2-ADA FFHQ).
- Dossier `weights/directions/` contenant des directions `.npy` :
  - `age.npy`, `gender.npy`, `smile.npy`, `glasses.npy`, `yaw.npy`, `race.npy` (facultatif).
  - Les vecteurs peuvent être au format **W** `(512,)` ou **W+** `(num_ws,512)`.

## Lancer l’UI
```bash
python -m src.ui.app_gradio_palier_d
