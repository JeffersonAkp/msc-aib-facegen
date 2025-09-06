import sys, os
sys.path.append(os.path.abspath("."))  # garantir que 'src' est importable

from src.data.fairface_constants import RACE_LABELS, UI_RACE_TO_ID, ETHNIE_CHOICES

def main():
    # Bijectif
    for k, v in RACE_LABELS.items():
        assert UI_RACE_TO_ID[v] == k
    # Couverture UI
    assert set(ETHNIE_CHOICES) == set(RACE_LABELS.values())
    print("OK: mapping figé et cohérent.")

if __name__ == "__main__":
    main()
