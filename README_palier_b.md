# 🧪 Palier B — Classement et Combinaisons (FairFace)

## 🎯 Objectif
Le Palier B vise à aller plus loin que le Palier A en ajoutant :
- **Galerie triée** : les images sont classées par confiance du classifieur race (FairFaceResNet18).  
- **Visage moyen (k)** : calcul d’une face moyenne à partir de *k* images sélectionnées.  
- **Interpolation 2** : génération d’un visage intermédiaire entre les 2 meilleures images.  

Ces fonctionnalités permettent de tester des combinaisons et d’évaluer la cohérence des classes FairFace.

---

## ⚙️ Fonctionnalités
1. **Galerie triée**  
   - Filtre les images selon âge, genre, ethnie.  
   - Valide avec le classifieur race local (`weights/fairface_race_resnet18.pth`).  
   - Trie par probabilité décroissante.  

2. **Visage moyen (k)**  
   - Calcule une moyenne pixel par pixel sur *k* images filtrées.  
   - Permet de visualiser une « tendance » du dataset.  

3. **Interpolation (α)**  
   - Interpolation linéaire entre les deux images les mieux classées.  
   - Paramétrée par α ∈ [0,1].  

---

## 📂 Organisation des fichiers
- `src/ui/app_gradio_palier_b.py` → Interface Gradio (UI Palier B).  
- `src/data/fairface_constants.py` → Constantes et mappings FairFace (âge, genre, ethnies).  
- `src/data/fairface_gallery.py` → Fonctions de sélection stricte/souple d’images.  
- `src/utils/img_ops.py` → Fonctions d’opérations d’images (moyenne, interpolation).  
- `weights/fairface_race_resnet18.pth` → Modèle de classification race pré-entraîné (local).  

---

## 📦 Prérequis
- Python ≥ 3.9  
- Dépendances (installées via `requirements.txt`) :  
  ```bash
  pip install -r requirements.txt


Dataset FairFace (via HuggingFace Datasets) :
téléchargé automatiquement à la 1ʳᵉ exécution.

🚀 Lancer l’UI Palier B

Depuis la racine du projet :

python -m src.ui.app_gradio_palier_b


Cela démarre une interface Gradio disponible sur :
👉 http://127.0.0.1:7860

📝 Notes

Si le mode strict est activé, le fichier de poids weights/fairface_race_resnet18.pth doit être présent localement.

Les sous-ensembles disponibles sont 0.25 (léger, rapide) et 1.25 (plus complet).

Le Palier B reste compatible avec le Palier A : les deux UIs peuvent être lancées indépendamment.

📸 Exemples d’utilisation
Galerie triée (mode strict, seuil=0.7)

Les images sont affichées par ordre décroissant de confiance du classifieur.


Visage moyen (k=8)

Construction d’un visage moyen à partir de 8 images conformes aux critères.


Interpolation (α=0.5)

Interpolation linéaire entre les 2 meilleures images (50%/50%).


✅ Conclusion

Le Palier B ajoute une dimension d’analyse qualitative en permettant :

de vérifier la confiance réelle du classifieur,

de produire des représentations synthétiques (moyenne, interpolation),

tout en restant aligné sur la source de vérité (FairFace).

Il s’agit d’un pas important vers les prochains paliers, où l’on pourra combiner génération et contrôle plus fin des attributs.