# Palier A — Sélection d’images FairFace (retrieval)

Ce palier affiche **une image** ou **une galerie de k images** issues du dataset **FairFace** selon des critères simples : âge approché, genre et *ethnie* (labels FairFace). Il ne **génère** pas d’images : il **sélectionne** des exemples pertinents.

> TL;DR
>
> * Les configurations FairFace utilisables ici sont **`0.25`** (léger) et **`1.25`** (plus complet).
> * L’UI : `python -m src.ui.app_gradio_gallery`.
> * Le **mapping** des labels (âge/genre/ethnie) est **centralisé et figé**.
> * Mode **strict** optionnel : revalidation par classifieur (ResNet18) pour une galerie plus propre.

---

## 1) Pré-requis

* Python ≥ 3.10, pip
* `pip install -r requirements.txt`
* Compte/connexion à Hugging Face (si nécessaire pour le caching public)

> **Windows** : si vous voyez l’avertissement HF *symlinks not supported*, vous pouvez soit l’ignorer, soit activer le *Developer Mode* de Windows.

---

## 2) Structure de code (Palier A)

```
src/
  data/
    fairface_constants.py   # ✅ source de vérité des mappings (âge/genre/ethnie)
    fairface_gallery.py     # 🔎 fonctions de sélection simple & stricte
  models/
    fairface_race_clf.py    # petit classifieur ResNet18 pour le mode strict
  ui/
    app_gradio_gallery.py   # 🎛️ interface Gradio Palier A
scripts/
  prefetch_fairface_125.py  # pré-télécharge la config 1.25 dans le cache HF
  train_race_classifier.py  # entraîne (ou ré-entraine) le classifieur 7 classes
weights/
  fairface_race_resnet18.pth  # (local, non versionné)
```

* **`fairface_constants.py`** = mapping **gelé** : une seule source de vérité pour les IDs/labels.
* **`fairface_gallery.py`** = sélection d’images, avec **fallbacks** si le filtre est trop strict.
* **`sample_k_images_strict`** = revalidation par classifieur (tri par confiance & seuil).

---

## 3) Datasets FairFace : quelles configs ?

Ce projet s’appuie sur le dataset **HuggingFaceM4/FairFace**. Les configurations pertinentes :

* **`0.25`** : sous-ensemble léger — idéal pour le dev rapide.
* **`1.25`** : sous-ensemble plus gros (nécessite un premier téléchargement plus long).

> ⚠️ `1.0` **n’existe pas** sur ce miroir. Si vous l’indiquez, l’UI renverra une erreur.

### (Optionnel) Pré-téléchargement

```bash
python -m scripts.prefetch_fairface_125
```

Cela “remplit” le cache local Hugging Face pour la config **1.25** afin que l’UI démarre instantanément.

---

## 4) Lancer l’UI (Palier A)

```bash
python -m src.ui.app_gradio_gallery
```

Dans la page web :

* **Sous-ensemble** : `0.25` ou `1.25`
* **Âge approx.** : slider → mappé vers les classes d’âge FairFace
* **Genre** : Homme / Femme
* **Ethnie** : valeurs issues **directement** de `ETHNIE_CHOICES` (constants gelées)
* **Mode** : `Une image` ou `Galerie (k images)`
* **k** : nombre d’images si `Galerie`
* **Mode strict** (checkbox) + paramètres :

  * `k_candidats` : combien d’images on revalide (60–120 conseillé)
  * `Seuil (thr)` : proba minimale de la classe demandée (≈ 0.7 par défaut)

---

## 5) Mode strict : revalidation par classifieur

Le mode strict compense le **bruit d’annotation** dans FairFace :

1. Filtre sur (âge/genre/ethnie).
2. Sélectionne `k_candidats` images.
3. Passe un **classifieur ResNet18**, renvoie `p(race)`.
4. Garde celles où `argmax == race_demandée` **et** `p ≥ thr`.
5. Trie par `p` décroissant et retourne les `k` meilleures (fallback si pas assez).

### Entraîner rapidement le classifieur (Windows‑friendly)

```bash
# mini run pour poids de démo
python -m scripts.train_race_classifier --subset 0.25 --epochs 1 --batch 32 \
  --num-workers 0 --train-split "train[:4000]" --val-split "validation[:1000]"
```

* Les poids sont enregistrés dans `weights/fairface_race_resnet18.pth` (non versionné).
* Dans l’UI, cochez **Mode strict** pour en profiter.

---

## 6) Tests & sanity‑checks utiles

* `tests/test_mapping.py` : vérifie que le **mapping** IDs↔labels est cohérent et figé.
* Le caption des cartes dans la galerie affiche : `race` (texte), `race_id`, `age_range`, `idx`, et **`score`** en mode strict.

---

## 7) Dépannage (FAQ rapide)

**« Je ne vois que 3 images »**
Vérifiez que le **mode** est `Galerie (k images)` et que le slider **k** est bien connecté (câblé dans `btn.click`).

**« Subset 1.0 introuvable »**
Utilisez **`0.25`** ou **`1.25`**.

**Téléchargements lents**
Faites un `prefetch` (ci‑dessus) et laissez le cache HF travailler une fois pour toutes.

**Windows + DataLoader**
Passer `--num-workers 0` à l’entraînement.

**Poids non trouvés en strict**
Vérifier l’existence de `weights/fairface_race_resnet18.pth`.

---

## 8) Git & hygiène

* Les poids dans `weights/` **ne sont pas versionnés** (`.gitignore`).
* Commit typique du palier A :

```bash
git checkout -b feat/palier-a  # si besoin

git add src/data/fairface_constants.py src/data/fairface_gallery.py src/ui/app_gradio_gallery.py
# si modifiés
git add scripts/prefetch_fairface_125.py scripts/train_race_classifier.py

git commit -m "feat(Palier A): UI galerie + mapping gelé + mode strict optionnel"
git push -u origin feat/palier-a
```

---

## 9) Limites & note éthique

* Les catégories d’« ethnie » dans FairFace proviennent d’**annotations humaines** et comportent **bruit & biais**.
* Le classifieur strict est un **filtre** pragmatique, pas une vérité absolue.

---

## 10) Checklist avant Palier B

* [ ] L’UI retourne bien des images cohérentes pour chaque combinaison (âge/genre/ethnie).
* [ ] Le **mapping** est validé (`tests/test_mapping.py`).
* [ ] `prefetch` effectué pour la config choisie (au moins une fois).
* [ ] (Optionnel) Poids du classifieur dispo pour le **mode strict**.

➡️ On peut maintenant construire le **Palier B** : ranking multi‑critères / combinaison de signaux (similarité visage, diversité, etc.).
