from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np

from simulation.registry import Taxonomy, issue_index_map
from simulation.composer import Composer


def compute_gshi(
    y_sev: np.ndarray,
    issues: List[str],
    issue_to_group: Dict[str, str],
    gshi_cfg: Dict[str, Any],
) -> float:
    """
    h = ∏ (1 - s_i)^( w_i * alpha_{g(i)} )
    """
    eps = float(gshi_cfg.get("eps", 1e-6))
    weights = gshi_cfg.get("weights", {}) or {}
    group_scale = gshi_cfg.get("group_scale", {}) or {}

    h = 1.0
    for i, name in enumerate(issues):
        s = float(y_sev[i])
        base_w = float(weights.get(name, 1.0))
        gname = issue_to_group.get(name, "")
        alpha = float(group_scale.get(gname, 1.0))
        eff_w = base_w * alpha

        term = max(1.0 - s, eps)
        h *= term ** eff_w

    return float(np.clip(h, 0.0, 1.0))


def union_mask(masks: Dict[str, Optional[np.ndarray]], H: int, W: int) -> Optional[np.ndarray]:
    valid = [m for m in masks.values() if m is not None]
    if len(valid) == 0:
        return None
    out = np.zeros((H, W), dtype=np.float32)
    for m in valid:
        if m.shape != (H, W):
            raise ValueError(f"Mask shape {m.shape} does not match image size {(H, W)}")
        out = np.maximum(out, m.astype(np.float32))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class Simulator:
    def __init__(self, taxonomy: Taxonomy, composer: Composer):
        self.taxonomy = taxonomy
        self.issues = taxonomy.issues
        self.issue_to_group = taxonomy.issue_to_group
        self.gshi_cfg = taxonomy.gshi
        self.idx = issue_index_map(self.issues)
        self.composer = composer

    def apply(self, image: np.ndarray, depth: Optional[np.ndarray], plan: Dict[str, float]) -> Dict[str, Any]:
        res = self.composer.apply_plan(image=image, depth=depth, plan=plan)
        img_out = res.image

        n = len(self.issues)
        y_pres = np.zeros((n,), dtype=np.float32)
        y_sev = np.zeros((n,), dtype=np.float32)

        for issue, sev in plan.items():
            if issue not in self.idx:
                continue
            j = self.idx[issue]
            y_pres[j] = 1.0
            y_sev[j] = float(np.clip(sev, 0.0, 1.0))

        y_health = compute_gshi(
            y_sev=y_sev,
            issues=self.issues,
            issue_to_group=self.issue_to_group,
            gshi_cfg=self.gshi_cfg,
        )

        H, W = img_out.shape[:2]
        mask_union = union_mask(res.masks, H, W)

        return {
            "image": img_out,          # HxWx3 float32 [0,1]
            "y_pres": y_pres,          # (12,)
            "y_sev": y_sev,            # (12,)
            "y_health": y_health,      # scalar
            "masks": res.masks,        # dict issue->mask or None
            "mask_union": mask_union,  # HxW float32 [0,1] or None
            "meta": res.meta,          # debug info
        }
