# simulation/digital/motion_blur.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class MotionBlurConfig:
    # kernel size range (odd sizes). final size scales with severity.
    k_min: int = 7
    k_max: int = 41

    # trajectory sampling
    n_points_min: int = 8
    n_points_max: int = 40

    # how "curvy" the trajectory is (random acceleration)
    accel_min: float = 0.0
    accel_max: float = 0.8

    # optional: small gaussian smooth of PSF (reduces jaggy kernel)
    psf_smooth_sigma_min: float = 0.0
    psf_smooth_sigma_max: float = 1.2

    # optional: prefer near-linear (ego-motion-like) blur
    linear_prob: float = 0.55

    # apply same kernel per-channel
    border: int = cv2.BORDER_REFLECT


def _odd(n: int) -> int:
    return int(n) if int(n) % 2 == 1 else int(n) + 1


def _sample_trajectory(
    rng: np.random.Generator,
    n_points: int,
    accel: float,
    linear: bool,
) -> np.ndarray:
    """
    Returns trajectory points in continuous kernel coordinates (x, y),
    centered roughly around (0,0). Units are arbitrary; later normalized into kernel size.
    """
    if linear:
        # line with slight jitter
        theta = float(rng.uniform(0, 2 * np.pi))
        step = float(rng.uniform(0.8, 1.3))
        dx, dy = step * np.cos(theta), step * np.sin(theta)
        pts = []
        x, y = 0.0, 0.0
        for _ in range(n_points):
            # tiny jitter
            jx = float(rng.normal(0, 0.05))
            jy = float(rng.normal(0, 0.05))
            pts.append((x + jx, y + jy))
            x += dx
            y += dy
        return np.array(pts, dtype=np.float32)

    # random-walk with acceleration (curvy trajectory)
    pts = []
    x, y = 0.0, 0.0
    vx, vy = float(rng.normal(0, 1.0)), float(rng.normal(0, 1.0))
    for _ in range(n_points):
        pts.append((x, y))
        ax = float(rng.normal(0, accel))
        ay = float(rng.normal(0, accel))
        vx += ax
        vy += ay
        # normalize velocity to avoid exploding
        vnorm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
        vx, vy = vx / vnorm, vy / vnorm
        x += vx
        y += vy
    return np.array(pts, dtype=np.float32)


def _trajectory_to_psf(pts: np.ndarray, ksize: int) -> np.ndarray:
    """
    Rasterize polyline trajectory into a kernel image.
    """
    k = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2

    # normalize pts extent to fit within kernel
    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(pts_centered)) + 1e-8
    # scale so it roughly spans half the kernel
    scale = (0.45 * (ksize - 1)) / max_abs
    pts_scaled = pts_centered * scale

    # draw polyline into kernel
    for i in range(len(pts_scaled) - 1):
        x1, y1 = pts_scaled[i]
        x2, y2 = pts_scaled[i + 1]
        p1 = (int(round(c + x1)), int(round(c + y1)))
        p2 = (int(round(c + x2)), int(round(c + y2)))
        cv2.line(k, p1, p2, color=1.0, thickness=1, lineType=cv2.LINE_AA)

    # normalize to sum=1 (PSF)
    s = float(k.sum())
    if s <= 1e-8:
        k[c, c] = 1.0
        s = 1.0
    k /= s
    return k


class MotionBlurDegradation(BaseDegradation):
    name = "motion_blur"

    def __init__(self, cfg: MotionBlurConfig = MotionBlurConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,   # unused (kept for interface consistency)
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = False,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image)

        # kernel size (odd)
        ksize = int(round(self.cfg.k_min + s * (self.cfg.k_max - self.cfg.k_min)))
        ksize = _odd(max(self.cfg.k_min, min(self.cfg.k_max, ksize)))

        # trajectory points
        n_points = int(round(self.cfg.n_points_min + s * (self.cfg.n_points_max - self.cfg.n_points_min)))
        n_points = max(2, n_points)

        accel = self.cfg.accel_min + s * (self.cfg.accel_max - self.cfg.accel_min)
        accel *= float(rng.uniform(0.9, 1.1))
        accel = float(max(0.0, accel))

        linear = bool(rng.uniform() < self.cfg.linear_prob)

        pts = _sample_trajectory(rng, n_points=n_points, accel=accel, linear=linear)
        K = _trajectory_to_psf(pts, ksize=ksize)

        # optional smooth (still sums to 1)
        sigma = self.cfg.psf_smooth_sigma_min + s * (self.cfg.psf_smooth_sigma_max - self.cfg.psf_smooth_sigma_min)
        sigma *= float(rng.uniform(0.9, 1.1))
        sigma = float(max(0.0, sigma))
        if sigma > 1e-6:
            K = cv2.GaussianBlur(K, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
            K /= float(K.sum() + 1e-8)

        # Convolution per channel: I' = I ⊗ K
        out = np.empty_like(img, dtype=np.float32)
        for c in range(3):
            out[..., c] = cv2.filter2D(img[..., c], -1, K, borderType=self.cfg.border)

        out = clamp01(out)

        meta: Dict[str, Any] = {
            "severity": s,
            "ksize": int(ksize),
            "n_points": int(n_points),
            "accel": float(accel),
            "linear": bool(linear),
            "psf_smooth_sigma": float(sigma),
        }

        # Optional: return K as "mask" for debugging/visualization
        mask = None
        if return_mask:
            mask = K.astype(np.float32)

        return DegradationResult(image=out, mask=mask, meta=meta)
