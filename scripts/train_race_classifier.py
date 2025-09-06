"""
Entraîne un petit classifieur race (7 classes FairFace) pour le Mode strict.
- Utilise ResNet18 pré-entraîné ImageNet, fine-tuning rapide.
- Enregistre les poids dans: weights/fairface_race_resnet18.pth

Exemple:
    python -m scripts.train_race_classifier --subset 0.25 --epochs 2 --batch 64 --lr 1e-3
"""

import os
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models, transforms
from datasets import load_dataset

# --- Transforms (ImageNet) ---
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalisation ImageNet (pour ResNet18)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def make_preprocess():
    def _pp(ex):
        img = ex["image"].convert("RGB")
        ex["pixel_values"] = TF(img)
        ex["labels"] = int(ex["race"])  # 0..6
        return ex
    return _pp

def collate(batch):
    # batch: list of dicts with keys: pixel_values (tensor 3x224x224), labels (int)
    xs = torch.stack([b["pixel_values"] for b in batch], dim=0)
    ys = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return xs, ys

def build_loaders(subset: str, batch_size: int, num_workers: int = 2):
    ds = load_dataset("HuggingFaceM4/FairFace", subset)
    train_ds = ds["train"]
    val_ds = ds.get("validation", None)

    pp = make_preprocess()
    train_ds = train_ds.with_transform(pp)
    if val_ds is not None:
        val_ds = val_ds.with_transform(pp)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader

def train_one_epoch(model, loader, opt, device):
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
def evaluate(model, loader, device):
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
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_backbone", action="store_true", help="geler le backbone (ne fine-tune que la tête)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("weights", exist_ok=True)

    train_loader, val_loader = build_loaders(args.subset, args.batch)

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

        # sauvegarde si c'est le meilleur so far (sur val si dispo, sinon train)
        score = va["acc"] if val_loader is not None else tr["acc"]
        if score >= best_acc:
            best_acc = score
            torch.save(model.state_dict(), "weights/fairface_race_resnet18.pth")
            print(f"✔️  Saved: weights/fairface_race_resnet18.pth (acc={best_acc:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()
