# src/models/stylegan2.py (en-tête)
from __future__ import annotations
from typing import Optional, Dict
import os, sys
import numpy as np
import torch
from PIL import Image

# --- Import StyleGAN2-ADA ---
# 1) tentative standard (si repo installé au top-level)
try:
    import dnnlib, legacy  # type: ignore
    import torch_utils     # type: ignore
except Exception:
    # 2) fallback vers notre copie vendored
    from src.third_party.stylegan2_ada import dnnlib as _dnnlib  # type: ignore
    from src.third_party.stylegan2_ada import legacy as _legacy  # type: ignore
    from src.third_party.stylegan2_ada import torch_utils as _tu  # type: ignore

    # Publie des alias top-level pour l’unpickler (les poids attendent 'dnnlib'/'torch_utils')
    sys.modules.setdefault("dnnlib", _dnnlib)
    sys.modules.setdefault("torch_utils", _tu)

    dnnlib = _dnnlib      # rebind pour le reste du fichier
    legacy = _legacy


def _to_pil(img: np.ndarray) -> Image.Image:
    # img: (C,H,W) in [-1,1]
    img = (img * 127.5 + 128).clip(0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # HWC
    return Image.fromarray(img, "RGB")

def _ensure_dir_wplus(vec: np.ndarray, num_ws: int, w_dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 1:
        assert v.shape[0] == w_dim, f"Direction doit avoir taille {w_dim}, reçu {v.shape}"
        v = np.tile(v[None, :], (num_ws, 1))
    elif v.ndim == 2:
        assert v.shape == (num_ws, w_dim), f"Direction attendue {(num_ws, w_dim)}, reçu {v.shape}"
    else:
        raise ValueError(f"Direction invalide, ndim={v.ndim}")
    return v

class StyleGAN2:
    """Wrapper minimal : génération, interpolation, édition par directions W/W+."""
    def __init__(self, pkl_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(pkl_path, "rb") as f:
            G = legacy.load_network_pkl(f)["G_ema"]  # type: ignore
        self.G = G.to(self.device).eval()
        self.w_dim: int = self.G.w_dim
        self.num_ws: int = self.G.num_ws
        self.resolution: int = self.G.img_resolution

    @torch.no_grad()
    def w_from_seed(self, seed: int, truncation: float = 0.7) -> torch.Tensor:
        rng = np.random.RandomState(int(seed))
        z = torch.from_numpy(rng.randn(1, self.G.z_dim)).to(self.device)
        c = None
        ws = self.G.mapping(z, c, truncation_psi=float(truncation), truncation_cutoff=None)
        return ws  # (1, num_ws, w_dim)

    @torch.no_grad()
    def synthesize_from_ws(self, ws: torch.Tensor, out_size: Optional[int] = None) -> Image.Image:
        img = self.G.synthesis(ws, noise_mode="const")[0].detach().cpu().numpy()
        pil = _to_pil(img)
        if out_size and out_size != self.resolution:
            pil = pil.resize((out_size, out_size), Image.BICUBIC)
        return pil

    @torch.no_grad()
    def generate(self, seed: int = 0, truncation: float = 0.7, out_size: Optional[int] = 256) -> Image.Image:
        ws = self.w_from_seed(seed, truncation)
        return self.synthesize_from_ws(ws, out_size)

    @torch.no_grad()
    def interpolate(self, seed_a: int, seed_b: int, alpha: float = 0.5,
                    truncation: float = 0.7, out_size: Optional[int] = 256) -> Image.Image:
        alpha = float(max(0.0, min(1.0, alpha)))
        wa = self.w_from_seed(seed_a, truncation)
        wb = self.w_from_seed(seed_b, truncation)
        ws = (1.0 - alpha) * wa + alpha * wb
        return self.synthesize_from_ws(ws, out_size)

    @torch.no_grad()
    def edit_with_directions(
        self,
        seed: int,
        truncation: float,
        coeffs: Dict[str, float],
        directions: Dict[str, np.ndarray],
        out_size: Optional[int] = 256,
        scale: float = 1.0,
    ) -> Image.Image:
        ws = self.w_from_seed(seed, truncation)        # (1, num_ws, w_dim)
        base = ws.clone().cpu().numpy()[0]             # (num_ws, w_dim)
        for name, alpha in coeffs.items():
            if name not in directions or abs(alpha) < 1e-6:
                continue
            vec = _ensure_dir_wplus(directions[name], self.num_ws, self.w_dim)
            base = base + (float(alpha) * float(scale)) * vec
        ws_edit = torch.from_numpy(base[None, ...]).to(self.device)
        return self.synthesize_from_ws(ws_edit, out_size)

def load_directions(dir_path: str, expected: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Charge des fichiers .npy (W/W+ directions) si présents."""
    found: Dict[str, np.ndarray] = {}
    for key, fname in expected.items():
        p = os.path.join(dir_path, fname)
        if os.path.isfile(p):
            try:
                found[key] = np.load(p)
            except Exception:
                pass
    return found
