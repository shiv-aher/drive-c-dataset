from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


class BaseDegradation:
    name = "base"

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ):
        raise NotImplementedError


@dataclass
class DegradationResult:
    image: np.ndarray
    mask: Optional[np.ndarray]
    meta: Dict[str, Any]


def to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        out = image.astype(np.float32) / 255.0
    else:
        out = image.astype(np.float32)
        if out.max() > 1.0:
            out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def ensure_depth(depth: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    if depth is None:
        return np.zeros((h, w), dtype=np.float32)

    d = depth.astype(np.float32)
    if d.shape[:2] != (h, w):
        import cv2
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    d_min = float(d.min())
    d_max = float(d.max())

    if d_max - d_min < 1e-8:
        return np.zeros((h, w), dtype=np.float32)

    d = (d - d_min) / (d_max - d_min)
    return np.clip(d, 0.0, 1.0).astype(np.float32)