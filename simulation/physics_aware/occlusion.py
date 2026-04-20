# simulation/physics_aware/occlusion.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class OcclusionConfig:
    # base blob count bounds (we will ALSO use coverage-based stopping)
    n_blobs_min: int = 1
    n_blobs_max: int = 10          # increased (so severity can cover more area)

    # blob radius relative to min(H,W)
    radius_min: float = 0.07
    radius_max: float = 0.40       # increased (so s=1 can occlude large regions)

    # distortion strength k in (u',v') = (u,v) * (1 + k(1-r^2))
    k_min: float = 0.25
    k_max: float = 1.10

    # alpha mask softness
    feather: float = 0.18
    alpha_min: float = 0.45        # slightly stronger baseline
    alpha_max: float = 1.00        # allow fully opaque-ish blend at s=1

    # optional intensity change inside blob
    add_specular: bool = True
    specular_strength: float = 0.06

    # blur inside blob
    blur_inside: bool = True
    blur_sigma_min: float = 0.0
    blur_sigma_max: float = 3.0

    # if True, prefer blobs near top half
    top_bias: float = 0.4

    # ✅ NEW: force severity -> visible occluded AREA
    coverage_min: float = 0.06     # target coverage at s=0 (if any blobs)
    coverage_max: float = 0.55     # target coverage at s=1
    coverage_thresh: float = 0.10  # pixels with alpha >= thresh count as "occluded"
    max_trials: int = 24           # safety cap to avoid infinite loops


def _soft_circle_alpha(u: np.ndarray, v: np.ndarray, feather: float) -> np.ndarray:
    r = np.sqrt(u*u + v*v).astype(np.float32)
    edge0 = 1.0 - feather
    edge1 = 1.0
    alpha = np.zeros_like(r, dtype=np.float32)
    inside = r <= edge0
    band = (r > edge0) & (r <= edge1)
    alpha[inside] = 1.0
    t = (r[band] - edge0) / max(edge1 - edge0, 1e-6)
    alpha[band] = 1.0 - (t*t*(3.0 - 2.0*t))
    return alpha


class LensOcclusionDegradation(BaseDegradation):
    name = "occlusion"

    def __init__(self, cfg: OcclusionConfig = OcclusionConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = True,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image)
        h, w = img.shape[:2]
        out = img.copy()

        min_hw = float(min(h, w))

        # Base coordinate grids
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)  # HxW

        # Accumulated mask
        acc_mask = np.zeros((h, w), dtype=np.float32)
        meta_blobs: List[Dict[str, Any]] = []

        # Target coverage increases with severity (use a mild nonlinearity)
        s_area = s**1.15
        target_cov = self.cfg.coverage_min + s_area * (self.cfg.coverage_max - self.cfg.coverage_min)
        target_cov = float(np.clip(target_cov, 0.0, 0.95))

        # We’ll add blobs until we reach target coverage or run out of trials
        trials = 0
        while trials < self.cfg.max_trials:
            trials += 1

            # Stop if we've already hit target coverage (only meaningful for s>0)
            if s > 1e-6:
                occ = float((acc_mask >= self.cfg.coverage_thresh).mean())
                if occ >= target_cov:
                    break

            # Also stop if we already have a lot of blobs
            if len(meta_blobs) >= self.cfg.n_blobs_max and s > 1e-6:
                break

            # Sample radius: grows strongly with severity (more visual difference)
            R = self.cfg.radius_min + (s**1.25) * (self.cfg.radius_max - self.cfg.radius_min)
            R *= float(rng.uniform(0.85, 1.15))
            R_px = max(8.0, R * min_hw)

            # Sample center (top bias)
            cx = float(rng.uniform(0.12 * w, 0.88 * w))
            if self.cfg.top_bias > 0 and rng.uniform() < self.cfg.top_bias:
                cy = float(rng.uniform(0.08 * h, 0.55 * h))
            else:
                cy = float(rng.uniform(0.12 * h, 0.88 * h))

            # Distortion strength
            k = self.cfg.k_min + s * (self.cfg.k_max - self.cfg.k_min)
            k *= float(rng.uniform(0.85, 1.15))
            k = float(np.clip(k, 0.0, 2.0))

            # Alpha strength (stronger at high severity)
            alpha_strength = self.cfg.alpha_min + (s**1.10) * (self.cfg.alpha_max - self.cfg.alpha_min)
            alpha_strength *= float(rng.uniform(0.90, 1.05))
            alpha_strength = float(np.clip(alpha_strength, 0.0, 1.0))

            # Normalized coordinates for this blob
            u = (X - cx) / R_px
            v = (Y - cy) / R_px
            r2 = u*u + v*v

            inside = r2 <= 1.0
            if not np.any(inside):
                continue

            # Distortion
            scale = (1.0 + k * (1.0 - r2)).astype(np.float32)
            up = (u * scale).astype(np.float32)
            vp = (v * scale).astype(np.float32)

            map_x = (cx + up * R_px).astype(np.float32)
            map_y = (cy + vp * R_px).astype(np.float32)

            map_x_full = X.copy()
            map_y_full = Y.copy()
            map_x_full[inside] = map_x[inside]
            map_y_full[inside] = map_y[inside]

            warped = cv2.remap(
                (out * 255.0).astype(np.uint8),
                map_x_full,
                map_y_full,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            ).astype(np.float32) / 255.0

            # Blur inside blob
            if self.cfg.blur_inside:
                sigma = self.cfg.blur_sigma_min + (s**1.10) * (self.cfg.blur_sigma_max - self.cfg.blur_sigma_min)
                sigma *= float(rng.uniform(0.85, 1.15))
                if sigma > 1e-3:
                    warped = cv2.GaussianBlur(
                        warped, (0, 0),
                        sigmaX=float(sigma), sigmaY=float(sigma),
                        borderType=cv2.BORDER_REFLECT
                    )

            # Soft alpha mask
            alpha = _soft_circle_alpha(u, v, feather=self.cfg.feather) * alpha_strength
            alpha3 = alpha[..., None].astype(np.float32)

            # Specular highlight
            if self.cfg.add_specular and self.cfg.specular_strength > 0:
                highlight = (np.clip(1.0 - r2, 0.0, 1.0) ** 2).astype(np.float32)
                highlight3 = (highlight[..., None] * float(self.cfg.specular_strength) * (0.5 + 0.5*s)).astype(np.float32)
                warped = clamp01(warped + highlight3)

            out = clamp01(out * (1.0 - alpha3) + warped * alpha3)
            acc_mask = np.maximum(acc_mask, alpha.astype(np.float32))

            meta_blobs.append({
                "cx": cx, "cy": cy, "R_px": R_px, "k": k,
                "alpha_strength": alpha_strength,
            })

            # For s==0, we should not keep adding blobs
            if s <= 1e-6:
                break

        meta: Dict[str, Any] = {
            "severity": s,
            "n_blobs": len(meta_blobs),
            "target_coverage": target_cov,
            "coverage_reached": float((acc_mask >= self.cfg.coverage_thresh).mean()),
            "blobs": meta_blobs,
        }
        mask = acc_mask if return_mask else None
        return DegradationResult(image=out, mask=mask, meta=meta)
