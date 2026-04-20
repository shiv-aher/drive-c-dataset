# simulation/sensor/low_light.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class LowLightConfig:
    # Gamma for darkening: I_dark = I^gamma  (gamma > 1 darkens)
    gamma_min: float = 1.3
    gamma_max: float = 3.2

    # Noise parameters (in [0,1] domain), same interpretation as your sensor_noise
    sigma_r_min: float = 0.002
    sigma_r_max: float = 0.020

    sigma_s_min: float = 0.004
    sigma_s_max: float = 0.090

    # Amplify noise variance in low light (severity controls how much)
    # We apply: sigma_r *= (1 + a_r * s), sigma_s *= (1 + a_s * s)
    amp_r: float = 2.0
    amp_s: float = 3.0

    # Optional mild desaturation (many low-light pipelines look less saturated)
    desat_min: float = 0.00
    desat_max: float = 0.25

    # Random jitter for realism
    jitter: float = 0.10  # +/-10%

    per_channel: bool = True


class LowLightDegradation(BaseDegradation):
    name = "low_light"

    def __init__(self, cfg: LowLightConfig = LowLightConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,  # unused
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = False,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image).astype(np.float32)

        # ---------------- gamma darkening ----------------
        gamma = self.cfg.gamma_min + s * (self.cfg.gamma_max - self.cfg.gamma_min)
        if self.cfg.jitter > 0:
            gamma *= float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))
        gamma = float(max(1e-6, gamma))

        I_dark = np.power(np.clip(img, 0.0, 1.0), gamma).astype(np.float32)

        # Optional: mild desaturation
        if self.cfg.desat_max > 0:
            desat = self.cfg.desat_min + s * (self.cfg.desat_max - self.cfg.desat_min)
            if self.cfg.jitter > 0:
                desat *= float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))
            desat = float(np.clip(desat, 0.0, 1.0))
            lum = (0.2126 * I_dark[..., 0] + 0.7152 * I_dark[..., 1] + 0.0722 * I_dark[..., 2])[..., None]
            I_dark = lum + (1.0 - desat) * (I_dark - lum)
            I_dark = np.clip(I_dark, 0.0, 1.0).astype(np.float32)

        # ---------------- amplified shot + read noise ----------------
        sigma_r = self.cfg.sigma_r_min + s * (self.cfg.sigma_r_max - self.cfg.sigma_r_min)
        sigma_s = self.cfg.sigma_s_min + s * (self.cfg.sigma_s_max - self.cfg.sigma_s_min)

        # amplify variance under low-light (severity-controlled)
        sigma_r *= (1.0 + self.cfg.amp_r * s)
        sigma_s *= (1.0 + self.cfg.amp_s * s)

        # jitter
        if self.cfg.jitter > 0:
            sigma_r *= float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))
            sigma_s *= float(rng.uniform(1.0 - self.cfg.jitter, 1.0 + self.cfg.jitter))

        h, w, c = I_dark.shape

        if self.cfg.per_channel:
            sr = (sigma_r * rng.uniform(0.9, 1.1, size=(1, 1, c))).astype(np.float32)
            ss = (sigma_s * rng.uniform(0.9, 1.1, size=(1, 1, c))).astype(np.float32)
        else:
            sr = np.float32(sigma_r)
            ss = np.float32(sigma_s)

        # Read noise: N(0, sigma_r^2)
        n_read = rng.normal(0.0, 1.0, size=I_dark.shape).astype(np.float32) * sr

        # Shot noise: N(0, sigma_s^2 * I_dark) => std = sigma_s * sqrt(I_dark)
        std_shot = ss * np.sqrt(np.clip(I_dark, 0.0, 1.0))
        n_shot = rng.normal(0.0, 1.0, size=I_dark.shape).astype(np.float32) * std_shot

        out = clamp01(I_dark + n_read + n_shot)

        meta: Dict[str, Any] = {
            "severity": s,
            "gamma": float(gamma),
            "sigma_r": float(sigma_r),
            "sigma_s": float(sigma_s),
            "amp_r": float(self.cfg.amp_r),
            "amp_s": float(self.cfg.amp_s),
            "desat": float(self.cfg.desat_min + s * (self.cfg.desat_max - self.cfg.desat_min)),
            "per_channel": bool(self.cfg.per_channel),
        }

        mask = None
        if return_mask:
            # Noise magnitude visualization
            noise_mag = np.linalg.norm((n_read + n_shot), axis=2)
            mask = np.clip(noise_mag / (noise_mag.max() + 1e-8), 0.0, 1.0).astype(np.float32)

        return DegradationResult(image=out, mask=mask, meta=meta)
