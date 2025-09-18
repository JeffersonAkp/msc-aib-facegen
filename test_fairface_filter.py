from src.data.fairface_filter import sample_image_by_attributes

for race in ["Blanc", "Noir", "Indien", "Asiatique Est", "Asiatique SE", "Moyen-Oriental", "Latino"]:
    _, meta = sample_image_by_attributes(35, "Homme", race, subset="0.25")
    print("Demandé:", race, " -> Trouvé:", meta["race"], "| stratégie:", meta["matched_strategy"])