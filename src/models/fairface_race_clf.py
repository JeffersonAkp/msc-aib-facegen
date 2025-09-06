# src/models/fairface_race_clf.py
from typing import List
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

class FairFaceRaceClassifier:
    """
    Wrapper simple autour d'un resnet18 (ou autre) finetuné sur FairFace.
    - Place un checkpoint local ici: weights/fairface_race_resnet18.pth
    - 7 classes dans l'ordre officiel (0..6)
    """
    def __init__(self, ckpt_path: str = "weights/fairface_race_resnet18.pth", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 7)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Poids du classifieur introuvables: {ckpt_path}\n"
                "Télécharge un checkpoint FairFace et place-le à cet emplacement."
            )
        sd = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalisation ImageNet (très standard pour ResNet)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict_proba(self, pil_list: List[Image.Image]) -> torch.Tensor:
        """
        pil_list -> (N, 7) proba softmax (tensor CPU)
        """
        batch = torch.stack([self.tf(img.convert("RGB")) for img in pil_list], dim=0).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()
