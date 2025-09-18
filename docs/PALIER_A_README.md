# Palier A — Sélection FairFace + Galerie + Mode Strict

## Objectif
- Palier A n’est pas de la génération : on **sélectionne** des images de FairFace qui correspondent aux attributs (âge, genre, ethnie).
- Le **Mode strict** revalide les images avec un **classifieur race** (7 classes FairFace) et ne garde que les plus **typiques** (proba > seuil).

## Démarrage rapide
### 1) (Optionnel) Pré-télécharger FairFace 1.25 hors UI
```bash
python -m scripts.prefetch_fairface_125
Si le réseau coupe via Xet, vous pouvez forcer HTTP classique :
setx HF_HUB_DISABLE_XET 1 puis relancer le terminal.

2) Lancer l’UI (Galerie)
bash
Copy code
python -m src.ui.app_gradio_gallery
Choisir subset 0.25 (léger) ou 1.25 (crop plus large).

Mode Galerie, choisir k (ex. 6, 12).

Si le Mode strict est décoché → sélection “standard” (selon labels).

Si Mode strict est coché → la galerie est re-filtrée par un classifieur race (voir ci-dessous).

Mode strict (comment ça marche)
On filtre par attributs demandés → on récupère k_candidates images (par ex. 60).

Un classifieur race pré-entraîné re-prédit la race sur ces images.

On garde celles dont la classe prédite == race demandée ET proba ≥ seuil (par ex. 0.7).

On trie par proba décroissante, on retourne les k meilleures.
S’il en manque, on complète avec le mode non strict (transparence).

Installer les poids du classifieur
Placer un checkpoint dans weights/fairface_race_resnet18.pth.

Sinon, entraîner rapidement notre petit classifieur local (voir plus bas).

Entraîner un classifieur race léger (5–15 min)
bash
Copy code
python -m scripts.train_race_classifier --subset 0.25 --epochs 2 --batch 64 --lr 1e-3
Sauvegarde : weights/fairface_race_resnet18.pth

Ensuite, relancer l’UI et cocher Mode strict.

Dépannage rapide
Je ne vois que 3 images : scroller ou augmenter la hauteur de la galerie ; vérifier le compteur (“X images”).

1.25 très lent la 1ère fois : normal (gros download). Faire le prefetch hors UI.

Annotations “bizarres” : c’est du bruit d’annotation (humains, subjectivité). Le Mode strict atténue cet effet pour l’affichage. Documenter en “Qualité des données / Éthique”.

