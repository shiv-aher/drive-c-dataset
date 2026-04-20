# simulation/sensor/exposure.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class ExposureConfig:
    # EV range. Severity maps to |Δ| within this range.
    # Example: max_abs_ev=2 means up to 4x brighter (Δ=+2) or 4x darker (Δ=-2).
    max_abs_ev: float = 2.0

    # If True, allow both under/over exposure by randomly choosing sign.
    # If False, always brighten (positive Δ) unless you override sign externally.
    random_sign: bool = True

    # Optional: small random jitter to avoid deterministic mapping
    ev_jitter: float = 0.10  # +/- 0.10 EV


class ExposureShiftDegradation(BaseDegradation):
    name = "exposure_shift"

    def __init__(self, cfg: ExposureConfig = ExposureConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,  # unused
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = False,
        ev_override: Optional[float] = None,  # if provided, use this Δ directly
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image).astype(np.float32)

        # choose Δ (EV offset)
        if ev_override is not None:
            delta = float(ev_override)
        else:
            delta = s * float(self.cfg.max_abs_ev)
            if self.cfg.random_sign:
                delta *= -1.0 if rng.uniform() < 0.5 else 1.0
            # jitter
            if self.cfg.ev_jitter > 0:
                delta += float(rng.uniform(-self.cfg.ev_jitter, self.cfg.ev_jitter))

        # exposure scaling: multiply by 2^Δ
        scale = float(2.0 ** delta)
        out = clamp01(img * scale)

        meta: Dict[str, Any] = {
            "severity": s,
            "delta_ev": float(delta),
            "scale": float(scale),
        }

        return DegradationResult(image=out, mask=None, meta=meta)
