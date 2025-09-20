# src/models/stylegan2.py
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from PIL import Image
import os, sys

# ---- StyleGAN2-ADA imports (repo installé OU copie vendored) ----
try:
    # Cas 1 : paquet NVlabs présent sur le PYTHONPATH
    import dnnlib, legacy  # type: ignore
except Exception:
    # Cas 2 : utiliser la copie vendored dans src/third_party/stylegan2_ada
    from src.third_party.stylegan2_ada import dnnlib as _vendored_dnnlib  # type: ignore
    from src.third_party.stylegan2_ada import torch_utils as _vendored_tu  # type: ignore
    import sys as _sys
    _sys.modules["dnnlib"] = _vendored_dnnlib
    _sys.modules["torch_utils"] = _vendored_tu
    from src.third_party.stylegan2_ada import legacy  # type: ignore
# ----------------------------------------------------------------


def _to_pil(img: np.ndarray) -> Image.Image:
    img = (img * 127.5 + 128).clip(0, 255).astype(np.uint8)  # (C,H,W) in [-1,1]
    img = np.transpose(img, (1, 2, 0))  # HWC
    return Image.fromarray(img, "RGB")

class StyleGAN2:
    def __init__(self, pkl_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(pkl_path, "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].to(self.device)  # type: ignore
        self.resolution = self.G.img_resolution

    @torch.no_grad()
    def generate(self, seed: int = 0, truncation: float = 0.7, out_size: Optional[int] = 256) -> Image.Image:
        # Note: out_size=256 par défaut pour la CPU
        rng = np.random.RandomState(int(seed))
        z = torch.from_numpy(rng.randn(1, self.G.z_dim)).to(self.device)
        img = self.G(z, None, truncation_psi=float(truncation), noise_mode="const")[0].detach().cpu().numpy()
        pil = _to_pil(img)
        if out_size and out_size != self.resolution:
            pil = pil.resize((out_size, out_size), Image.BICUBIC)
        return pil

    @torch.no_grad()
    def interpolate(self, seed_a: int, seed_b: int, alpha: float = 0.5,
                    truncation: float = 0.7, out_size: Optional[int] = 256) -> Image.Image:
        alpha = float(max(0.0, min(1.0, alpha)))
        rng_a = np.random.RandomState(int(seed_a))
        rng_b = np.random.RandomState(int(seed_b))
        z_a = torch.from_numpy(rng_a.randn(1, self.G.z_dim)).to(self.device)
        z_b = torch.from_numpy(rng_b.randn(1, self.G.z_dim)).to(self.device)
        z = (1.0 - alpha) * z_a + alpha * z_b
        img = self.G(z, None, truncation_psi=float(truncation), noise_mode="const")[0].detach().cpu().numpy()
        pil = _to_pil(img)
        if out_size and out_size != self.resolution:
            pil = pil.resize((out_size, out_size), Image.BICUBIC)
        return pil
