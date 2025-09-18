# src/utils/img_ops.py
from PIL import Image
import numpy as np

def _to_rgb(im: Image.Image, size=(256, 256)) -> Image.Image:
    return im.convert("RGB").resize(size, Image.BILINEAR)

def mean_face(pils, size=(256, 256)) -> Image.Image:
    """Moyenne pixel des images (visage 'moyen')."""
    arrs = [np.asarray(_to_rgb(p, size), dtype=np.float32) for p in pils]
    m = np.mean(arrs, axis=0)
    return Image.fromarray(np.clip(m, 0, 255).astype(np.uint8))

def blend(pil_a: Image.Image, pil_b: Image.Image, alpha: float = 0.5,
          size=(256, 256)) -> Image.Image:
    """Interpolation linéaire entre deux images (0=pil_a, 1=pil_b)."""
    a = np.asarray(_to_rgb(pil_a, size), dtype=np.float32)
    b = np.asarray(_to_rgb(pil_b, size), dtype=np.float32)
    mix = (1.0 - alpha) * a + alpha * b
    return Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
