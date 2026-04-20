# simulation/physics_aware/fog.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01, ensure_depth


@dataclass
class FogConfig:
    k_min: float = 0.3
    k_max: float = 2.2

    A_min: float = 0.85
    A_max: float = 1.0
    A_mode: str = "gray"

    gamma: float = 1.6      # controls depth curve realism
    invert_depth: bool = True   # IMPORTANT (your norm is inverted)


class FogDegradation(BaseDegradation):
    name = "fog"

    def __init__(self, cfg: FogConfig = FogConfig()):
        self.cfg = cfg

    def _sample_A(self, rng):
        a = rng.uniform(self.cfg.A_min, self.cfg.A_max)
        return np.array([a, a, a], dtype=np.float32).reshape(1,1,3)

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        return_mask: bool = True,
    ) -> DegradationResult:

        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity,0,1))
        img = to_float01(image)
        h,w = img.shape[:2]

        d = ensure_depth(depth,h,w)

        # 🔥 FIX: invert if needed
        if self.cfg.invert_depth:
            d = 1.0 - d

        # clean & shape depth
        d = clamp01(d)
        d = d ** self.cfg.gamma   # push fog farther

        # k mapping
        k = self.cfg.k_min + s*(self.cfg.k_max-self.cfg.k_min)

        A = self._sample_A(rng)

        # Beer-Lambert
        t = np.exp(-k*d).astype(np.float32)
        t3 = t[...,None]

        out = img*t3 + A*(1-t3)
        out = clamp01(out)

        mask = None
        if return_mask:
            mask = (1-t).astype(np.float32)

        meta = {
            "severity": s,
            "k": float(k),
            "gamma": self.cfg.gamma,
            "depth_stats": {
                "min": float(d.min()),
                "max": float(d.max()),
                "mean": float(d.mean())
            }
        }

        return DegradationResult(out,mask,meta)
