# simulation/sensor/noise.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class NoiseConfig:
    # Base noise ranges (for images in [0,1])
    # Read noise std (constant Gaussian)
    sigma_r_min: float = 0.0015
    sigma_r_max: float = 0.0200

    # Shot noise scale (signal-dependent Gaussian):
    # std_shot(x) = sigma_s * sqrt(I(x))
    # This corresponds to Var = sigma_s^2 * I(x)
    sigma_s_min: float = 0.0020
    sigma_s_max: float = 0.0800

    # Optional: apply per-channel slightly different noise
    per_channel: bool = True

    # Optional: randomize a bit for realism
    jitter: float = 0.12  # +/- 12%


class SensorNoiseDegradation(BaseDegradation):
    name = "sensor_noise"

    def __init__(self, cfg: NoiseConfig = NoiseConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,  # unused (kept for interface)
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = False,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image).astype(np.float32)

        # severity -> noise parameters
        sigma_r = self.cfg.sigma_r_min + s * (self.cfg.sigma_r_max - self.cfg.sigma_r_min)
        sigma_s = self.cfg.sigma_s_min + s * (self.cfg.sigma_s_max - self.cfg.sigma_s_min)

        # jitter (tiny realism)
        if self.cfg.jitter > 0:
            jr = float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))
            js = float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))
            sigma_r *= jr
            sigma_s *= js

        h, w, c = img.shape

        if self.cfg.per_channel:
            # slightly different noise per channel
            sr = (sigma_r * rng.uniform(0.9, 1.1, size=(1, 1, c))).astype(np.float32)
            ss = (sigma_s * rng.uniform(0.9, 1.1, size=(1, 1, c))).astype(np.float32)
        else:
            sr = np.float32(sigma_r)
            ss = np.float32(sigma_s)

        # -------- noise model (matches slide) --------
        # read: N(0, sigma_r^2)
        n_read = rng.normal(0.0, 1.0, size=img.shape).astype(np.float32) * sr

        # shot: N(0, sigma_s^2 * I)  -> std = sigma_s * sqrt(I)
        std_shot = ss * np.sqrt(np.clip(img, 0.0, 1.0))
        n_shot = rng.normal(0.0, 1.0, size=img.shape).astype(np.float32) * std_shot

        out = clamp01(img + n_read + n_shot)

        meta: Dict[str, Any] = {
            "severity": s,
            "sigma_r": float(sigma_r),
            "sigma_s": float(sigma_s),
            "per_channel": bool(self.cfg.per_channel),
        }

        # Optional: return per-pixel noise magnitude as a "mask" for debugging
        mask = None
        if return_mask:
            mask = np.clip(np.linalg.norm((n_read + n_shot), axis=2), 0.0, 1.0).astype(np.float32)

        return DegradationResult(image=out, mask=mask, meta=meta)
