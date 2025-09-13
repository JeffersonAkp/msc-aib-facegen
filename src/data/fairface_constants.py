"""
Source de vérité pour les libellés FairFace.
NE PAS dupliquer ces mappings ailleurs dans le code.
"""

# -------------------- Âge / Genre --------------------
AGE_LABELS = {
    0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29",
    4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"
}

GENDER_LABELS = {0: "Homme", 1: "Femme"}
UI_GENDER_TO_ID = {"Homme": 0, "Femme": 1}

# -------------------- Race (ordre OFFICIEL HF) --------------------
# Ordre EXACT de HuggingFaceM4/FairFace (ne pas changer)
HF_RACE_EN = [
    "East Asian",       # 0
    "Indian",           # 1
    "Black",            # 2
    "White",            # 3
    "Middle Eastern",   # 4
    "Latino_Hispanic",  # 5
    "Southeast Asian",  # 6
]

# Libellés FR correspondants (utilisés dans l’UI)
RACE_FR_BY_EN = {
    "East Asian": "Asiatique Est",
    "Indian": "Indien",
    "Black": "Noir",
    "White": "Blanc",
    "Middle Eastern": "Moyen-Oriental",
    "Latino_Hispanic": "Latino",
    "Southeast Asian": "Asiatique SE",
}

# Tables de conversion
ID_TO_FR = {i: RACE_FR_BY_EN[name] for i, name in enumerate(HF_RACE_EN)}
FR_TO_ID = {fr: i for i, fr in ID_TO_FR.items()}

# Liste pour le dropdown UI dans l’ordre HF
ETHNIE_CHOICES = [ID_TO_FR[i] for i in range(len(HF_RACE_EN))]

def race_id_from_fr(fr: str) -> int:
    return FR_TO_ID[fr]

def race_fr_from_id(i: int) -> str:
    return ID_TO_FR[i]

# -------------------- Garde-fous --------------------
def _check_constants():
    # Ids 0..6 présents, bijection FR <-> id
    assert list(ID_TO_FR.keys()) == list(range(7)), "IDs race doivent être 0..6"
    for i, fr in ID_TO_FR.items():
        assert FR_TO_ID[fr] == i, "Bijection FR<->ID rompue"

    # Si datasets dispo, on vérifie l’ordre contre la vérité HF
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train[:1]")
        names = ds.features["race"].names
        assert names == HF_RACE_EN, (
            "Mismatch avec l’ordre HF:\n"
            f"HF : {names}\n"
            f"Ici: {HF_RACE_EN}"
        )
    except Exception:
        # hors-ligne → on ne bloque pas
        pass

_check_constants()

__all__ = [
    "AGE_LABELS", "GENDER_LABELS", "UI_GENDER_TO_ID",
    "HF_RACE_EN", "ID_TO_FR", "FR_TO_ID", "ETHNIE_CHOICES",
    "race_id_from_fr", "race_fr_from_id",
]
