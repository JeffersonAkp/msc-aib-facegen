# scripts/prefetch_fairface_125.py
"""
Télécharge et prépare FairFace subset 1.25 en cache local (HF datasets),
pour éviter que l'UI le fasse pendant un clic utilisateur.
À exécuter depuis la racine du projet :
    python -m scripts.prefetch_fairface_125
"""
import sys, os
sys.path.append(os.path.abspath("."))  # garantir que 'src' est importable
# scripts/prefetch_fairface_125.py
from src.data.fairface_gallery import load_fairface_train

# Vide le cache mémoire si besoin
#load_fairface_train.cache_clear()

# Déclenche le téléchargement et la préparation en dehors de Gradio
ds = load_fairface_train("1.25")
print("OK: FairFace 1.25 chargé en mémoire. Longueur:", len(ds))
