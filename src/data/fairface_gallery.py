"""
fairface_gallery.py
-------------------
Utilitaires pour :
- charger FairFace,
- mapper l'âge "humain" vers la classe d'âge FairFace,
- récupérer 1 ou k images qui correspondent aux attributs demandés,
avec une version "strict" (revalidation par classifieur de race).

Ce module s'aligne sur les constantes de:
    src/data/fairface_constants.py
"""

from functools import lru_cache
from typing import Optional, Tuple, List
import random

from PIL import Image
from datasets import load_dataset
import numpy as np  # utilisé dans la version stricte

from src.models.fairface_race_clf import FairFaceRaceClassifier

# Source de vérité (Âge/Genre) + conversions Race FR<->ID
from src.data.fairface_constants import (
    AGE_LABELS,
    GENDER_LABELS,
    UI_GENDER_TO_ID,
    race_id_from_fr,
    race_fr_from_id,
)

# ---------------------------------------------------------------------
# Mapping âge "humain" -> classe d'âge FairFace (0..8)
# ---------------------------------------------------------------------
def age_to_class(age: int) -> int:
    a = max(0, min(99, int(age)))
    if a <= 2:  return 0
    if a <= 9:  return 1
    if a <= 19: return 2
    if a <= 29: return 3
    if a <= 39: return 4
    if a <= 49: return 5
    if a <= 59: return 6
    if a <= 69: return 7
    return 8

# ---------------------------------------------------------------------
# Chargement dataset (cache en mémoire, clé = subset)
# Subsets valides publiquement: "0.25" et "1.25"
# ---------------------------------------------------------------------
@lru_cache(maxsize=4)
def load_fairface_train(subset: str = "0.25"):
    ds = load_dataset("HuggingFaceM4/FairFace", subset)
    return ds["train"]

# ---------------------------------------------------------------------
# Filtrage simple par indices (age_class, gender_id, race_id)
# Un critère = None => non filtré
# ---------------------------------------------------------------------
def _filter_indices(ds, age_c: Optional[int], gender_id: Optional[int], race_id: Optional[int]) -> List[int]:
    out: List[int] = []
    for i, ex in enumerate(ds):
        if age_c is not None and int(ex["age"]) != age_c:
            continue
        if gender_id is not None and int(ex["gender"]) != gender_id:
            continue
        if race_id is not None and int(ex["race"]) != race_id:
            continue
        out.append(i)
    return out

# ---------------------------------------------------------------------
# Échantillonnage d'une image
# ---------------------------------------------------------------------
def sample_one_image(age: int, gender_label: str, race_label_fr: str, subset: str = "0.25") -> Tuple[Image.Image, dict]:
    """
    Retourne (PIL.Image, meta) pour une image représentative.
    meta contient: age_class, age_range, gender, race_fr, race_id, index, used_strategy
    """
    ds = load_fairface_train(subset)
    age_c = age_to_class(age)
    gid = UI_GENDER_TO_ID.get(gender_label, None)
    # ⚠️ conversion robuste du libellé FR vers l'ID officiel HF
    rid = race_id_from_fr(race_label_fr) if race_label_fr is not None else None

    strategies = [
        (age_c, gid, rid),
        (None,  gid, rid),
        (None,  None, rid),
        (None,  gid, None),
        (age_c, None, None),
        (None,  None, None),
    ]

    for ac, g, r in strategies:
        idxs = _filter_indices(ds, ac, g, r)
        if not idxs:
            continue
        i = random.choice(idxs)
        ex = ds[i]
        race_id = int(ex["race"])
        meta = {
            "age_class": int(ex["age"]),
            "age_range": AGE_LABELS.get(int(ex["age"]), str(ex["age"])),
            "gender": GENDER_LABELS.get(int(ex["gender"]), str(ex["gender"])),
            "race_fr": race_fr_from_id(race_id),
            "race_id": race_id,
            "index": i,
            "used_strategy": (ac, g, r),
        }
        return ex["image"], meta

    # Fallback (ne devrait pas arriver)
    ex = ds[random.randrange(len(ds))]
    race_id = int(ex["race"])
    return ex["image"], {
        "age_class": int(ex["age"]),
        "age_range": AGE_LABELS.get(int(ex["age"]), str(ex["age"])),
        "gender": GENDER_LABELS.get(int(ex["gender"]), str(ex["gender"])),
        "race_fr": race_fr_from_id(race_id),
        "race_id": race_id,
        "index": None,
        "used_strategy": "any",
    }

# ---------------------------------------------------------------------
# Échantillonnage d'une galerie de k images (version "souple")
# ---------------------------------------------------------------------
def sample_k_images(age: int, gender_label: str, race_label_fr: str, k: int = 6, subset: str = "0.25"):
    """
    Retourne une liste de k éléments: [(PIL.Image, meta), ...]
    On essaie plusieurs stratégies, on s'arrête dès qu'on a au moins k images.
    """
    ds = load_fairface_train(subset)
    age_c = age_to_class(age)
    gid = UI_GENDER_TO_ID.get(gender_label, None)
    rid = race_id_from_fr(race_label_fr) if race_label_fr is not None else None

    strategies = [
        (age_c, gid, rid),
        (None,  gid, rid),
        (None,  None, rid),
        (None,  gid, None),
        (age_c, None, None),
        (None,  None, None),
    ]

    results = []
    for ac, g, r in strategies:
        idxs = _filter_indices(ds, ac, g, r)
        if not idxs:
            continue
        random.shuffle(idxs)
        for i in idxs:
            ex = ds[i]
            race_id = int(ex["race"])
            meta = {
                "age_class": int(ex["age"]),
                "age_range": AGE_LABELS.get(int(ex["age"]), str(ex["age"])),
                "gender": GENDER_LABELS.get(int(ex["gender"]), str(ex["gender"])),
                "race_fr": race_fr_from_id(race_id),
                "race_id": race_id,
                "index": i,
                "used_strategy": (ac, g, r),
            }
            results.append((ex["image"], meta))
            if len(results) >= k:
                return results
    return results  # peut être < k si dataset trop restreint

# ---------------------------------------------------------------------
# Version stricte (revalidation par classifieur race)
# ---------------------------------------------------------------------
def sample_k_images_strict(age: int, gender_label: str, race_label_fr: str,
                           k: int = 6, subset: str = "0.25",
                           k_candidates: int = 60, min_prob: float = 0.7,
                           deterministic: bool = True):
    """
    1) Filtre (age, genre, race) → candidats
    2) Prend jusqu'à k_candidates images (ordre déterministe ou shuffle)
    3) Classifie avec FairFaceRaceClassifier → proba(7)
    4) Garde celles où argmax == race_id demandé ET proba[race_id] >= min_prob
    5) Trie par proba décroissante et renvoie les k meilleures
    Fallback: si < k, complète par la version non stricte.
    """
    ds = load_fairface_train(subset)
    age_c = age_to_class(age)
    gid = UI_GENDER_TO_ID.get(gender_label, None)
    rid = race_id_from_fr(race_label_fr) if race_label_fr is not None else None

    candidates = _filter_indices(ds, age_c, gid, rid)
    if not candidates:
        return sample_k_images(age, gender_label, race_label_fr, k=k, subset=subset)

    if deterministic:
        picked_idx = sorted(candidates)[:k_candidates]
    else:
        random.shuffle(candidates)
        picked_idx = candidates[:k_candidates]

    pil_list, metas = [], []
    for i in picked_idx:
        ex = ds[i]
        race_id = int(ex["race"])
        pil_list.append(ex["image"])
        metas.append({
            "age_class": int(ex["age"]),
            "age_range": AGE_LABELS.get(int(ex["age"]), str(ex["age"])),
            "gender": GENDER_LABELS.get(int(ex["gender"]), str(ex["gender"])),
            "race_fr": race_fr_from_id(race_id),
            "race_id": race_id,
            "index": i,
            "used_strategy": (age_c, gid, rid),
        })

    clf = FairFaceRaceClassifier()          # charge weights/fairface_race_resnet18.pth
    probs = clf.predict_proba(pil_list)     # Tensor (N, 7)
    p = probs.numpy()
    argmax = p.argmax(axis=1)
    keep = (argmax == rid) & (p[:, rid] >= float(min_prob))

    kept = [(pil_list[i], metas[i], p[i, rid]) for i in range(len(p)) if keep[i]]
    kept.sort(key=lambda t: t[2], reverse=True)

    out = [(im, m) for (im, m, _) in kept[:k]]
    short = k - len(out)
    if short > 0:
        extra = sample_k_images(age, gender_label, race_label_fr, k=short, subset=subset)
        out.extend(extra)
    return out
