# scripts/prefetch_fairface_125.py
from src.data.fairface_gallery import load_fairface_train

# Vide le cache mémoire si besoin
load_fairface_train.cache_clear()

# Déclenche le téléchargement et la préparation en dehors de Gradio
ds = load_fairface_train("1.25")
print("OK: FairFace 1.25 chargé en mémoire. Longueur:", len(ds))
