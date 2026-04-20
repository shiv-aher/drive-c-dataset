# simulation/physics_aware/glare.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class GlareConfig:
    # Additive intensity scaling (alpha)
    alpha_min: float = 0.10
    alpha_max: float = 0.55

    # Bloom params
    bloom_thresh: float = 0.90          # threshold in [0,1] for "saturated" regions
    bloom_sigma_min: float = 6.0        # blur for bloom (pixels)
    bloom_sigma_max: float = 25.0

    # Procedural flare pattern params (F(x))
    n_ghosts_min: int = 2
    n_ghosts_max: int = 7
    ghost_sigma_min: float = 8.0
    ghost_sigma_max: float = 40.0
    ghost_gain_min: float = 0.10
    ghost_gain_max: float = 0.50

    # Radial halo around the bright source
    halo_sigma_min: float = 25.0
    halo_sigma_max: float = 110.0
    halo_gain_min: float = 0.05
    halo_gain_max: float = 0.25

    # Streaks (lens scattering lines)
    streak_gain_min: float = 0.03
    streak_gain_max: float = 0.18
    streak_len_min: int = 25
    streak_len_max: int = 140
    streak_width_min: float = 1.5
    streak_width_max: float = 6.0
    n_streaks_min: int = 2
    n_streaks_max: int = 5

    # Source selection: pick brightest region as flare center
    source_percentile: float = 99.7     # choose among top percentile bright pixels
    soften_source_sigma: float = 2.0    # smooth luminance before argmax

    # If True, use per-channel flare tinting (slight warm/cool)
    color_tint: bool = True
    tint_strength: float = 0.18         # 0=no tint

    # Optional: keep glare mask (useful for supervision)
    return_mask: bool = True


def _luminance(rgb01: np.ndarray) -> np.ndarray:
    # ITU-R BT.709 luminance
    return (0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]).astype(np.float32)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


def _draw_streak_layer(h: int, w: int, cx: float, cy: float, angle: float,
                       length: int, width: float) -> np.ndarray:
    """
    Returns a grayscale streak image in [0,1] centered at (cx,cy).
    """
    streak = np.zeros((h, w), dtype=np.float32)

    # line endpoints
    dx = (length / 2.0) * np.cos(angle)
    dy = (length / 2.0) * np.sin(angle)
    x1, y1 = int(round(cx - dx)), int(round(cy - dy))
    x2, y2 = int(round(cx + dx)), int(round(cy + dy))

    # Draw line (anti-aliased)
    cv2.line(streak, (x1, y1), (x2, y2), color=1.0, thickness=max(1, int(round(width))), lineType=cv2.LINE_AA)
    # soften
    streak = _gauss(streak, sigma=max(1.0, width * 0.8))
    return streak


class GlareDegradation(BaseDegradation):
    name = "glare"

    def __init__(self, cfg: GlareConfig = GlareConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,   # glare is mostly image-plane
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: Optional[bool] = None,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()
        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image)
        h, w = img.shape[:2]

        if return_mask is None:
            return_mask = self.cfg.return_mask

        # ---------------- Choose source position ----------------
        lum = _luminance(img)
        if self.cfg.soften_source_sigma > 0:
            lum_s = _gauss(lum, self.cfg.soften_source_sigma)
        else:
            lum_s = lum

        # pick a bright pixel among top percentile (more robust than pure argmax)
        thr = np.percentile(lum_s, self.cfg.source_percentile)
        ys, xs = np.where(lum_s >= thr)
        if len(xs) == 0:
            # no very bright regions; fall back to max
            y0, x0 = np.unravel_index(np.argmax(lum_s), lum_s.shape)
            cx, cy = float(x0), float(y0)
        else:
            idx = int(rng.integers(0, len(xs)))
            cx, cy = float(xs[idx]), float(ys[idx])

        # ---------------- Severity -> params ----------------
        alpha = self.cfg.alpha_min + s * (self.cfg.alpha_max - self.cfg.alpha_min)
        alpha *= float(rng.uniform(0.9, 1.1))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        bloom_sigma = self.cfg.bloom_sigma_min + s * (self.cfg.bloom_sigma_max - self.cfg.bloom_sigma_min)
        bloom_sigma *= float(rng.uniform(0.9, 1.1))

        # ---------------- Bloom: blur saturated regions + re-add ----------------
        sat_mask = (lum >= self.cfg.bloom_thresh).astype(np.float32)  # HxW
        # Extract bright component
        bright = img * sat_mask[..., None]
        bloom = _gauss(bright, bloom_sigma)

        # ---------------- Flare overlay F(x) ----------------
        F = np.zeros_like(img, dtype=np.float32)

        # (a) Halo around bright source
        halo_sigma = self.cfg.halo_sigma_min + s * (self.cfg.halo_sigma_max - self.cfg.halo_sigma_min)
        halo_gain = self.cfg.halo_gain_min + s * (self.cfg.halo_gain_max - self.cfg.halo_gain_min)
        halo_sigma *= float(rng.uniform(0.9, 1.1))
        halo_gain *= float(rng.uniform(0.9, 1.1))

        # Create an impulse at (cx,cy), blur to halo
        impulse = np.zeros((h, w), dtype=np.float32)
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < w and 0 <= iy < h:
            impulse[iy, ix] = 1.0
        halo = _gauss(impulse, halo_sigma)
        halo = (halo / (halo.max() + 1e-8)).astype(np.float32)
        F += halo_gain * halo[..., None]

        # (b) Ghosts: along line through image center (classic lens ghosting)
        n_ghosts = int(np.round(self.cfg.n_ghosts_min + s * (self.cfg.n_ghosts_max - self.cfg.n_ghosts_min)))
        n_ghosts = max(self.cfg.n_ghosts_min, min(self.cfg.n_ghosts_max, n_ghosts))

        cx0, cy0 = w / 2.0, h / 2.0
        vx, vy = cx0 - cx, cy0 - cy  # direction from source to center
        for gi in range(n_ghosts):
            t = (gi + 1) / (n_ghosts + 1)  # between (0,1)
            gx = cx + t * vx
            gy = cy + t * vy

            gsigma = self.cfg.ghost_sigma_min + s * (self.cfg.ghost_sigma_max - self.cfg.ghost_sigma_min)
            gsigma *= float(rng.uniform(0.75, 1.25))
            ggain = self.cfg.ghost_gain_min + s * (self.cfg.ghost_gain_max - self.cfg.ghost_gain_min)
            ggain *= float(rng.uniform(0.75, 1.25))

            imp = np.zeros((h, w), dtype=np.float32)
            ix, iy = int(round(gx)), int(round(gy))
            if 0 <= ix < w and 0 <= iy < h:
                imp[iy, ix] = 1.0
            ghost = _gauss(imp, gsigma)
            ghost = (ghost / (ghost.max() + 1e-8)).astype(np.float32)

            # Slightly modulate ghost by source brightness to be data-driven
            src_gain = float(np.clip(lum_s[int(round(cy)) % h, int(round(cx)) % w], 0.0, 1.0))
            F += (ggain * src_gain) * ghost[..., None]

        # (c) Streaks: anisotropic lines through source, blurred
        n_streaks = int(np.round(self.cfg.n_streaks_min + s * (self.cfg.n_streaks_max - self.cfg.n_streaks_min)))
        n_streaks = max(self.cfg.n_streaks_min, min(self.cfg.n_streaks_max, n_streaks))

        base_angle = float(rng.uniform(0, np.pi))
        for si in range(n_streaks):
            ang = base_angle + (si * np.pi / max(n_streaks, 1))
            length = int(self.cfg.streak_len_min + s * (self.cfg.streak_len_max - self.cfg.streak_len_min))
            length = int(length * float(rng.uniform(0.9, 1.1)))
            width = self.cfg.streak_width_min + s * (self.cfg.streak_width_max - self.cfg.streak_width_min)
            width *= float(rng.uniform(0.9, 1.1))
            sgain = self.cfg.streak_gain_min + s * (self.cfg.streak_gain_max - self.cfg.streak_gain_min)
            sgain *= float(rng.uniform(0.9, 1.1))

            streak = _draw_streak_layer(h, w, cx, cy, ang, length=length, width=width)
            # modulate by source brightness
            src_gain = float(np.clip(lum_s[int(round(cy)) % h, int(round(cx)) % w], 0.0, 1.0))
            F += (sgain * src_gain) * streak[..., None]

        # Optional tint: warm/cool lens flare
        if self.cfg.color_tint and self.cfg.tint_strength > 0:
            t = float(self.cfg.tint_strength) * s
            # slight warm tint (R up, B down) or random
            if rng.uniform() < 0.5:
                tint = np.array([1.0 + t, 1.0, 1.0 - t], dtype=np.float32)
            else:
                tint = np.array([1.0 - t, 1.0, 1.0 + t], dtype=np.float32)
            F = F * tint.reshape(1, 1, 3)

        # Normalize flare pattern (keep it bounded)
        F = F / (F.max() + 1e-8)
        F = clamp01(F)

        # Combine F with bloom into a single flare field
        flare_field = clamp01(F + bloom)

        # ---------------- Additive blending ----------------
        out = clamp01(img + alpha * flare_field)

        # Optional mask: where flare energy is high
        mask = None
        if return_mask:
            mask = np.clip(_luminance(flare_field), 0.0, 1.0).astype(np.float32)

        meta: Dict[str, Any] = {
            "severity": s,
            "alpha": alpha,
            "bloom_sigma": float(bloom_sigma),
            "source_xy": [float(cx), float(cy)],
            "n_ghosts": n_ghosts,
            "n_streaks": n_streaks,
        }

        return DegradationResult(image=out, mask=mask, meta=meta)
