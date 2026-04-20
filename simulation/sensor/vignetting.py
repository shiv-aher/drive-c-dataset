# simulation/sensor/vignetting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class VignettingConfig:
    # alpha controls strength of darkening at edges
    alpha_min: float = 0.10
    alpha_max: float = 0.70

    # gamma controls curvature (how quickly it falls off)
    gamma_min: float = 1.2
    gamma_max: float = 3.5

    # optional: slight center shift (lens misalignment realism)
    center_jitter_px: float = 0.0  # set e.g. 8.0 for small random shift

    # optional: per-channel tint (some lenses vignette more in one channel)
    per_channel: bool = False


class VignettingDegradation(BaseDegradation):
    name = "vignetting"

    def __init__(self, cfg: VignettingConfig = VignettingConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,  # unused
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = True,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image).astype(np.float32)
        h, w = img.shape[:2]

        # severity -> params
        alpha = self.cfg.alpha_min + s * (self.cfg.alpha_max - self.cfg.alpha_min)
        gamma = self.cfg.gamma_min + s * (self.cfg.gamma_max - self.cfg.gamma_min)

        # optional: small random center shift
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        if self.cfg.center_jitter_px > 0:
            cx += float(rng.uniform(-self.cfg.center_jitter_px, self.cfg.center_jitter_px))
            cy += float(rng.uniform(-self.cfg.center_jitter_px, self.cfg.center_jitter_px))

        # normalized radius r in [0,1]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = (xx - cx) / max(1.0, cx)
        dy = (yy - cy) / max(1.0, cy)
        r = np.sqrt(dx * dx + dy * dy)
        r = np.clip(r, 0.0, 1.0)

        # V(r) = 1 - alpha * r^gamma
        V = 1.0 - float(alpha) * np.power(r, float(gamma))
        V = np.clip(V, 0.0, 1.0).astype(np.float32)

        # apply: I' = I * V
        if self.cfg.per_channel:
            # tiny channel variation (optional)
            scales = rng.uniform(0.98, 1.02, size=(1, 1, 3)).astype(np.float32)
            V3 = np.clip(V[..., None] * scales, 0.0, 1.0)
            out = img * V3
        else:
            out = img * V[..., None]

        out = clamp01(out)

        mask = V if return_mask else None

        meta: Dict[str, Any] = {
            "severity": s,
            "alpha": float(alpha),
            "gamma": float(gamma),
            "center": (float(cx), float(cy)),
            "per_channel": bool(self.cfg.per_channel),
        }

        return DegradationResult(image=out, mask=mask, meta=meta)
