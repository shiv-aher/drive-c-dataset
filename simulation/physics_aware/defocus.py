# simulation/physics_aware/defocus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01, ensure_depth


@dataclass
class DefocusConfig:
    sigma_max: float = 30.0          # maximum blur std-dev in pixels (scaled by severity)
    n_layers: int = 16               # depth bins for layered rendering
    eps: float = 1e-3               # numeric stability for inverse depth
    d_f_mode: str = "center"        # "random" or "center"
    d_f_min: float = 0.2            # focus depth range (in normalized depth units)
    d_f_max: float = 0.8
    invert_depth: bool = True      # if your depth is near=1 far=0, set True
    use_bokeh_approx: bool = False  # if True, uses larger kernel for more pronounced blur look


def _gaussian_blur_sigma(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur with sigma. OpenCV wants odd ksize; can pass (0,0) and sigmaX.
    """
    if sigma <= 1e-6:
        return img
    # For very small sigma, OpenCV still fine.
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


class DefocusDegradation(BaseDegradation):
    name = "defocus"

    def __init__(self, cfg: DefocusConfig = DefocusConfig()):
        self.cfg = cfg

    def _choose_focus_depth(self, d: np.ndarray, rng: np.random.Generator) -> float:
        if self.cfg.d_f_mode == "center":
            return float(np.median(d))
        # random within [d_f_min, d_f_max]
        return float(rng.uniform(self.cfg.d_f_min, self.cfg.d_f_max))

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        d_f: Optional[float] = None,     # optional override for deterministic tests
        return_mask: bool = True,
    ) -> DegradationResult:
        """
        Depth-aware defocus blur via CoC proxy + depth-layered Gaussian rendering.

        image: HxWx3 uint8 or float in [0,1]
        depth: HxW float in [0,1] (normalized per image)
        severity: s in [0,1] scales sigma_max
        """
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image)
        h, w = img.shape[:2]
        d = ensure_depth(depth, h, w)

        if self.cfg.invert_depth:
            d = 1.0 - d

        # Pick focus depth d_f
        if d_f is None:
            d_f = self._choose_focus_depth(d, rng)
        d_f = float(np.clip(d_f, 0.0, 1.0))

        # ---- CoC proxy ----
        # CoC(x) ∝ | 1/(d+eps) - 1/(d_f+eps) |
        inv_d = 1.0 / (d + self.cfg.eps)
        inv_df = 1.0 / (d_f + self.cfg.eps)
        coc = np.abs(inv_d - inv_df).astype(np.float32)

        # Normalize CoC to [0,1] for stable scaling
        coc_max = float(coc.max())
        if coc_max < 1e-8:
            coc_n = np.zeros_like(coc, dtype=np.float32)
        else:
            coc_n = coc / coc_max

        # ---- Sigma map ----
        sigma_max_eff = self.cfg.sigma_max * s
        sigma_map = (sigma_max_eff * coc_n).astype(np.float32)  # HxW

        # Optional: bokeh-ish approximation (makes blur "stronger" perceptually)
        if self.cfg.use_bokeh_approx:
            sigma_map = np.sqrt(sigma_map * sigma_map + 1e-6).astype(np.float32)

        # ---- Depth-layered Gaussian rendering ----
        # Bin by sigma rather than depth directly (more efficient & stable)
        n_layers = int(max(2, self.cfg.n_layers))
        # Layer edges in [0, sigma_max_eff]
        edges = np.linspace(0.0, max(sigma_max_eff, 1e-6), n_layers + 1, dtype=np.float32)

        out = np.zeros_like(img, dtype=np.float32)
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)

        # For mask/supervision: "blur strength" in [0,1]
        mask = coc_n if return_mask else None

        for li in range(n_layers):
            lo, hi = float(edges[li]), float(edges[li + 1])

            # Pixels whose sigma falls in this layer
            layer_mask = (sigma_map >= lo) & (sigma_map < hi)

            if not layer_mask.any():
                continue

            # Representative sigma for this layer (midpoint)
            sigma_layer = 0.5 * (lo + hi)

            # Blur full image once for this layer
            blurred = _gaussian_blur_sigma(img, sigma_layer)

            # Composite: only keep pixels belonging to this layer
            m = layer_mask.astype(np.float32)[..., None]  # HxWx1
            out += blurred * m
            weight_sum += m

        # Any pixels not covered (e.g., sigma==max edge): fallback to original
        # (also handles s=0 case cleanly)
        uncovered = (weight_sum < 0.5).astype(np.float32)
        out = out + img * uncovered
        weight_sum = np.maximum(weight_sum, 1.0)

        out = clamp01(out)

        meta: Dict[str, Any] = {
            "severity": s,
            "sigma_max_eff": float(sigma_max_eff),
            "d_f": float(d_f),
            "invert_depth": self.cfg.invert_depth,
            "n_layers": n_layers,
        }

        return DegradationResult(image=out, mask=mask, meta=meta)
