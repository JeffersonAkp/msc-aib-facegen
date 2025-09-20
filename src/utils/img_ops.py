# src/utils/img_ops.py
"""
Opérations image pour les Paliers B/C :
- alignement/recadrage visage (MediaPipe si dispo, sinon fallback)
- moyenne (simple / pondérée)
- interpolation alpha
- exports (PNG, ZIP, run.json)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import io, json, os, zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# --- Détection visage : MediaPipe si dispo, sinon OpenCV, sinon fallback centre ---

_HAS_MEDIAPIPE = False
try:
    import mediapipe as mp  # ignoré si non dispo (ex: Python 3.12/3.13)
    _HAS_MEDIAPIPE = True
    _mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
except Exception:
    _HAS_MEDIAPIPE = False
    _mp_face = None

# OpenCV fallback
_HAS_OPENCV = False
try:
    import cv2
    _HAS_OPENCV = True
    _cv2_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    _HAS_OPENCV = False
    _cv2_cascade = None

def _detect_bbox_mediapipe(pil: Image.Image):
    if not _HAS_MEDIAPIPE or _mp_face is None:
        return None
    im = pil.convert("RGB")
    arr = np.array(im)  # H,W,3
    res = _mp_face.process(arr)
    if not res.detections:
        return None
    det = res.detections[0]
    bbox = det.location_data.relative_bounding_box
    h, w = arr.shape[0], arr.shape[1]
    x0 = max(0, int(bbox.xmin * w))
    y0 = max(0, int(bbox.ymin * h))
    x1 = min(w, int((bbox.xmin + bbox.width) * w))
    y1 = min(h, int((bbox.ymin + bbox.height) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    # petite marge
    pad_x = int(0.10 * (x1 - x0))
    pad_y = int(0.10 * (y1 - y0))
    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x); y1 = min(h, y1 + pad_y)
    return (x0, y0, x1, y1)

def _detect_bbox_opencv(pil: Image.Image):
    if not _HAS_OPENCV or _cv2_cascade is None:
        return None
    gray = np.array(pil.convert("L"))
    faces = _cv2_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # prend le plus grand visage
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    x0, y0, x1, y1 = x, y, x + w, y + h
    # marge douce
    pad_x = int(0.10 * w)
    pad_y = int(0.10 * h)
    H, W = gray.shape[:2]
    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x); y1 = min(H, y1 + pad_y)
    return (x0, y0, x1, y1)

def align_and_crop_face(pil: Image.Image, size: int = 224) -> Image.Image:
    """Essaie MediaPipe, sinon OpenCV, sinon center-crop carré."""
    bbox = _detect_bbox_mediapipe(pil)
    if bbox is None:
        bbox = _detect_bbox_opencv(pil)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        pil = pil.crop((x0, y0, x1, y1))
    else:
        # fallback centre carré
        w, h = pil.size
        s = min(w, h)
        cx, cy = w // 2, h // 2
        pil = pil.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))

    pil = pil.convert("RGB").resize((size, size), Image.BICUBIC)
    return pil


# --- Normalisation illumination (option légère) ---
def normalize_illumination(pil: Image.Image) -> Image.Image:
    # Equalize per-channel (PIL ImageOps)
    return ImageOps.equalize(pil.convert("RGB"))

# --- Moyennes & blending ---
def pil_to_float(arr_pil: List[Image.Image]) -> np.ndarray:
    stack = [np.asarray(p, dtype=np.float32) for p in arr_pil]
    return np.stack(stack, axis=0)  # (N,H,W,3)

def mean_face(pils: List[Image.Image]) -> Image.Image:
    if len(pils) == 0:
        raise ValueError("mean_face: liste vide")
    arr = pil_to_float(pils).mean(axis=0).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

def mean_face_weighted(pils: List[Image.Image], weights: List[float]) -> Image.Image:
    if len(pils) == 0:
        raise ValueError("mean_face_weighted: liste vide")
    w = np.asarray(weights, dtype=np.float32)
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    arr = pil_to_float(pils)
    m = (arr * w[:, None, None, None]).sum(axis=0).clip(0,255).astype(np.uint8)
    return Image.fromarray(m)

def blend(a: Image.Image, b: Image.Image, alpha: float = 0.5) -> Image.Image:
    alpha = float(max(0.0, min(1.0, alpha)))
    a = a.convert("RGB"); b = b.convert("RGB")
    b = b.resize(a.size, Image.BICUBIC)
    return Image.blend(a, b, alpha)

# --- Exports ---
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

def save_png(pil: Image.Image, name: str) -> str:
    path = EXPORT_DIR / name
    pil.save(path, format="PNG")
    return str(path)

def save_run_json(payload: Dict[str, Any], name: str = "run.json") -> str:
    path = EXPORT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)

def make_zip(images: List[Tuple[str, Image.Image]], zip_name: str = "galerie.zip") -> str:
    """images = [(filename, pil), ...] → crée un zip."""
    zip_path = EXPORT_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, pil in images:
            buff = io.BytesIO()
            pil.save(buff, format="PNG")
            zf.writestr(fname, buff.getvalue())
    return str(zip_path)

@dataclass
class LastRun:
    subset: str
    mode: str
    age: int
    gender: str
    ethnie: str
    k: int
    k_cand: int
    thr: float
    alpha: float
    indices: List[int]
    scores: List[float]
    # images traitées (après align/normalize)
    pils: List[Image.Image]
    mean_img: Optional[Image.Image] = None
    interp_img: Optional[Image.Image] = None
