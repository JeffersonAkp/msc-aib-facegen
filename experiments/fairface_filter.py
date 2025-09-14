"""
Chargement de FairFace + sélection d'une image qui "colle" aux attributs demandés.
Palier A : on ne GÉNÈRE pas encore, on SÉLECTIONNE une image existante.
"""

from functools import lru_cache
from typing import Optional, Tuple, List
import random

from datasets import load_dataset
from PIL import Image

from src.data.fairface_constants import (
    AGE_LABELS,
    GENDER_LABELS,
    RACE_LABELS,
    UI_GENDER_TO_ID,
    UI_RACE_TO_ID,
)


def age_to_class(age: int) -> int:
    """
    Mappe un âge approximatif (18..70) vers la classe d'âge FairFace (0..8).
    On borne pour éviter les valeurs hors intervalle.
    """
    a = max(0, min(99, int(age)))
    if a <= 2: return 0
    if a <= 9: return 1
    if a <= 19: return 2
    if a <= 29: return 3
    if a <= 39: return 4
    if a <= 49: return 5
    if a <= 59: return 6
    if a <= 69: return 7
    return 8


@lru_cache(maxsize=1)
def load_fairface_train(subset: str = "0.25"):
    """
    Charge le dataset FairFace (split 'train') via Hugging Face.
    Le décorateur @lru_cache évite de recharger à chaque appel.
    subset: "0.25" (léger) ou "1.0" (si dispo localement).
    """
    ds = load_dataset("HuggingFaceM4/FairFace", subset)
    return ds["train"]


def _filter_indices(
    ds,
    age_c: Optional[int],
    gender_id: Optional[int],
    race_id: Optional[int],
) -> List[int]:
    """
    Renvoie les indices des exemples qui respectent les critères (optionnels).
    Si un critère est None, on ne filtre pas dessus.
    """
    idxs: List[int] = []
    for i, ex in enumerate(ds):
        ok = True
        if age_c is not None and ex["age"] != age_c:
            ok = False
        if gender_id is not None and ex["gender"] != gender_id:
            ok = False
        if race_id is not None and ex["race"] != race_id:
            ok = False
        if ok:
            idxs.append(i)
    return idxs


def sample_image_by_attributes(
    age: int,
    gender_label: str,
    race_label: str,
    subset: str = "0.25",
) -> Tuple[Image.Image, dict]:
    """
    Sélectionne UNE image du dataset qui correspond le mieux aux attributs.
    - age (int) : âge approx. (sera converti en classe d'âge FairFace)
    - gender_label (str) : "Homme" / "Femme"
    - race_label (str) : un des ETHNIE_CHOICES (voir constants)
    - subset (str) : "0.25" (léger) ou "1.0"

    Stratégie:
      1) on tente (âge + genre + ethnie)
      2) si rien → (genre + ethnie)
      3) si rien → (ethnie)
      4) si rien → (genre)
      5) si rien → (âge)
      6) si rien → (n'importe quelle image)

    Retour:
      (PIL.Image, meta: dict avec âge/genre/ethnie trouvés + infos debug)
    """
    ds = load_fairface_train(subset)

    # Convertit les entrées UI en IDs FairFace
    age_c = age_to_class(age)
    gender_id = UI_GENDER_TO_ID.get(gender_label, None)
    race_id = UI_RACE_TO_ID.get(race_label, None)

    print(
        f"[DEBUG] requested: age={age}→class={age_c}, "
        f"gender='{gender_label}'→{gender_id}, race='{race_label}'→{race_id}"
    )

    # Stratégies de fallback de + strict → + souple
    strategies = [
        (age_c, gender_id, race_id),
        (None,   gender_id, race_id),
        (None,   None,      race_id),
        (None,   gender_id, None),
        (age_c,  None,      None),
        (None,   None,      None),
    ]

    for ac, gid, rid in strategies:
        idxs = _filter_indices(ds, ac, gid, rid)
        if idxs:
            i = random.choice(idxs)        # on échantillonne aléatoirement parmi les matches
            ex = ds[i]
            meta = {
                "age_class": ex["age"],
                "age_range": AGE_LABELS.get(ex["age"], str(ex["age"])),
                "gender": GENDER_LABELS.get(ex["gender"], str(ex["gender"])),
                "race": RACE_LABELS.get(ex["race"], str(ex["race"])),
                "matched_strategy": (ac, gid, rid),
                "index": i,
            }
            print(
                f"[DEBUG] matched: index={i}, "
                f"age_class={ex['age']}({meta['age_range']}), "
                f"gender={ex['gender']}({meta['gender']}), "
                f"race={ex['race']}({meta['race']})"
            )
            return ex["image"], meta

    # Fallback ultime (ne devrait pas arriver avec FairFace)
    i = random.randrange(len(ds))
    ex = ds[i]
    return ex["image"], {
        "age_class": ex["age"],
        "age_range": AGE_LABELS.get(ex["age"], str(ex["age"])),
        "gender": GENDER_LABELS.get(ex["gender"], str(ex["gender"])),
        "race": RACE_LABELS.get(ex["race"], str(ex["race"])),
        "matched_strategy": "fallback_any",
        "index": i,
    }
