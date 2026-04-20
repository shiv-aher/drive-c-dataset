from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class ApplyResult:
    image: np.ndarray                                  # HxWx3 float32 [0,1]
    masks: Dict[str, Optional[np.ndarray]]             # issue -> HxW float32 [0,1] or None
    meta: Dict[str, Dict[str, Any]]                    # issue -> meta dict


class Composer:
    def __init__(self, degradations: Dict[str, BaseDegradation], seed: int = 0):
        self.degradations = degradations
        self.rng = np.random.default_rng(seed)

    def apply_plan(self, image: np.ndarray, depth: Optional[np.ndarray], plan: Dict[str, float]) -> ApplyResult:
        img = to_float01(image)
        masks: Dict[str, Optional[np.ndarray]] = {}
        meta: Dict[str, Dict[str, Any]] = {}

        # deterministic application order for reproducibility
        for issue in sorted(plan.keys()):
            if issue not in self.degradations:
                raise KeyError(f"Issue '{issue}' not registered. Available: {list(self.degradations.keys())}")

            sev = float(np.clip(plan[issue], 0.0, 1.0))
            degr = self.degradations[issue]

            out: DegradationResult = degr.apply(
                image=img,
                severity=sev,
                depth=depth,
                rng=self.rng,
            )

            img = clamp01(out.image)
            masks[issue] = None if out.mask is None else clamp01(out.mask)
            meta[issue] = out.meta or {}

        return ApplyResult(image=img, masks=masks, meta=meta)
