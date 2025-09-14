"""
Entraîne un petit classifieur "race" (7 classes FairFace) pour le Mode strict.
- Backbone : ResNet18 pré-entraîné ImageNet (fine-tuning rapide).
- Sauvegarde des poids : weights/fairface_race_resnet18.pth

Exemples :
    # entraînement RAPIDE sur CPU (petit sous-ensemble)
    python -m scripts.train_race_classifier --subset 0.25 --epochs 1 --batch 32 --num-workers 0 \
        --freeze_backbone --train-split "train[:4000]" --val-split "validation[:1000]"

    # plus complet (à éviter sur CPU si pressé)
    python -m scripts.train_race_classifier --subset 0.25 --epochs 2 --batch 64 --num-workers 0
"""

import os
import argparse
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset


# ----------------------------
# Transforms (ImageNet standard)
# ----------------------------
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FairFaceTorchDataset(Dataset):
    """
    Wrapper PyTorch sur un split HF : retourne (tensor, label) par __getitem__.
    """
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex: Dict = self.ds[idx]  # ex["image"] est un PIL Image
        img = ex["image"].convert("RGB")
        x = TF(img)                  # Tensor [3,224,224]
        y = int(ex["race"])          # 0..6
        return x, y


def build_loaders(subset: str, batch_size: int, train_split: str, val_split: Optional[str],
                  num_workers: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Construit DataLoader train/val pour FairFace avec splits HF explicites.
    Exemples de splits :
      - "train" (complet) ou "train[:4000]" (sous-ensemble)
      - "validation" ou "validation[:1000]"
    """
    # Chargement direct des splits demandés
    hf_train = load_dataset("HuggingFaceM4/FairFace", subset, split=train_split)
    train_ds = FairFaceTorchDataset(hf_train)

    val_loader = None
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )

    if val_split:
        try:
            hf_val = load_dataset("HuggingFaceM4/FairFace", subset, split=val_split)
            val_ds = FairFaceTorchDataset(hf_val)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin
            )
        except Exception:
            val_loader = None

    print(f"[Data] train_split='{train_split}' → {len(train_ds)} exemples | "
          f"val_split='{val_split}' → {len(val_loader.dataset) if val_loader else 0} exemples")
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer, device: str):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model: nn.Module, loader: Optional[DataLoader], device: str):
    if loader is None:
        return {"loss": 0.0, "acc": 0.0}
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=str, default="0.25", help="FairFace subset: 0.25 ou 1.25")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Geler le backbone (fine-tune seulement la tête).")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Workers DataLoader (0 recommandé sous Windows).")
    ap.add_argument("--train-split", type=str, default="train[:4000]",
                    help="Split HF pour le train (ex: 'train' ou 'train[:4000]')")
    ap.add_argument("--val-split", type=str, default="validation[:1000]",
                    help="Split HF pour la val (ex: 'validation' ou 'validation[:1000]'; vide pour désactiver)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("weights", exist_ok=True)

    train_loader, val_loader = build_loaders(
        args.subset, args.batch, train_split=args.train_split,
        val_split=(args.val_split if args.val_split else None),
        num_workers=args.num_workers
    )

    # --- Modèle ---
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 7)

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    model.to(device)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # --- Entraînement ---
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device)
        va = evaluate(model, val_loader, device)
        print(f"[Epoch {ep}] train: loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
              f"val: loss={va['loss']:.4f} acc={va['acc']:.4f}")

        score = va["acc"] if val_loader is not None else tr["acc"]
        if score >= best_acc:
            best_acc = score
            torch.save(model.state_dict(), "weights/fairface_race_resnet18.pth")
            print(f"✔️  Saved: weights/fairface_race_resnet18.pth (acc={best_acc:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
