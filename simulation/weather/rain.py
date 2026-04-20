# simulation/weather/rain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01, ensure_depth


@dataclass
class RainConfig:
    # streak count scales with severity
    n_min: int = 200
    n_max: int = 1800

    # streak geometry
    length_min: int = 12
    length_max: int = 55
    width_min: int = 1
    width_max: int = 2

    # direction (degrees), near vertical with some tilt
    angle_mean_deg: float = 80.0     # 90 = vertical
    angle_jitter_deg: float = 12.0

    # rain intensity (compositing weight)
    alpha_min: float = 0.10
    alpha_max: float = 0.55

    # streak brightness layer (R(x))
    streak_value_min: float = 0.55
    streak_value_max: float = 1.00

    # blur streaks slightly (looks more natural)
    blur_sigma_min: float = 0.3
    blur_sigma_max: float = 1.2

    # depth foreground bias exponent eta
    eta_min: float = 0.5
    eta_max: float = 2.5

    # if your depth is near=1 far=0, set True so (1-d) means "far"
    invert_depth: bool = False

    # optional: color tint (rain slightly whitish)
    rgb_tint: bool = True

    # ---- realism upgrades ----
    # break up streaks so they aren't perfect lines
    enable_dropout: bool = True
    dropout_prob: float = 0.35          # probability a streak becomes "broken"
    dropout_keep_min: float = 0.55      # keep ratio for broken streaks
    dropout_keep_max: float = 0.95

    # multi-layer rain (far + near)
    enable_multilayer: bool = True
    # fraction of streaks assigned to near layer (thicker/longer/brighter)
    near_layer_frac: float = 0.30
    # extra scaling for near layer
    near_length_mul: float = 1.35
    near_width_add: float = 1.0
    near_value_mul: float = 1.15
    near_blur_mul: float = 1.10

    # rainy tone mapping: make scene less sunny (dim/low-contrast/desaturate)
    apply_rainy_tone: bool = True
    dim_min: float = 0.10
    dim_max: float = 0.45
    contrast_min: float = 0.05
    contrast_max: float = 0.25
    desat_min: float = 0.05
    desat_max: float = 0.25
    cool_tint_min: float = 0.02
    cool_tint_max: float = 0.10


def _apply_rain_tone(img01: np.ndarray, s: float, rng: np.random.Generator, cfg: RainConfig) -> np.ndarray:
    """Cheap but high-impact: dim exposure, reduce contrast, slight desaturation, subtle cool tint."""
    s = float(np.clip(s, 0.0, 1.0))
    out = img01.astype(np.float32)

    # 1) exposure dimming
    dim = 1.0 - (cfg.dim_min + s * (cfg.dim_max - cfg.dim_min)) * float(rng.uniform(0.85, 1.15))
    out *= dim

    # 2) contrast reduction (mix toward mean)
    mean = out.mean(axis=(0, 1), keepdims=True)
    cdrop = (cfg.contrast_min + s * (cfg.contrast_max - cfg.contrast_min)) * float(rng.uniform(0.85, 1.15))
    contrast = 1.0 - cdrop
    out = mean + contrast * (out - mean)

    # 3) slight desaturation
    lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
    sdrop = (cfg.desat_min + s * (cfg.desat_max - cfg.desat_min)) * float(rng.uniform(0.85, 1.15))
    sat = 1.0 - sdrop
    out = lum + sat * (out - lum)

    # 4) subtle cool tint (rainy/cloudy)
    tint = (cfg.cool_tint_min + s * (cfg.cool_tint_max - cfg.cool_tint_min)) * float(rng.uniform(0.85, 1.15))
    out[..., 2] = np.clip(out[..., 2] * (1.0 + tint), 0.0, 1.0)         # blue up
    out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.5 * tint), 0.0, 1.0)   # red down slightly

    return np.clip(out, 0.0, 1.0).astype(np.float32)


class RainDegradation(BaseDegradation):
    name = "rain"

    def __init__(self, cfg: RainConfig = RainConfig()):
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
        d = ensure_depth(depth, h, w)

        if self.cfg.invert_depth:
            d = 1.0 - d

        # ---------------- severity -> params ----------------
        n_streaks = int(np.round(self.cfg.n_min + s * (self.cfg.n_max - self.cfg.n_min)))
        n_streaks = max(1, n_streaks)

        alpha = self.cfg.alpha_min + s * (self.cfg.alpha_max - self.cfg.alpha_min)
        alpha *= float(rng.uniform(0.9, 1.1))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        eta = self.cfg.eta_min + s * (self.cfg.eta_max - self.cfg.eta_min)
        eta *= float(rng.uniform(0.9, 1.1))
        eta = float(max(0.0, eta))

        blur_sigma = self.cfg.blur_sigma_min + s * (self.cfg.blur_sigma_max - self.cfg.blur_sigma_min)
        blur_sigma *= float(rng.uniform(0.9, 1.1))
        blur_sigma = float(max(0.0, blur_sigma))

        # Angle (in radians)
        ang_deg = self.cfg.angle_mean_deg + float(rng.uniform(-self.cfg.angle_jitter_deg, self.cfg.angle_jitter_deg))
        ang = np.deg2rad(ang_deg)
        cos_a, sin_a = np.cos(ang), np.sin(ang)

        # ---------------- build rain layer R(x) ----------------
        R = np.zeros((h, w), dtype=np.float32)

        # Multilayer split
        if self.cfg.enable_multilayer:
            n_near = int(round(n_streaks * float(np.clip(self.cfg.near_layer_frac, 0.0, 1.0))))
            n_far = max(0, n_streaks - n_near)
        else:
            n_near, n_far = 0, n_streaks

        def draw_streaks(n: int, *, near: bool):
            nonlocal R
            for _ in range(n):
                # random center
                x0 = float(rng.uniform(0, w - 1))
                y0 = float(rng.uniform(0, h - 1))

                # base length/width
                L = self.cfg.length_min + s * (self.cfg.length_max - self.cfg.length_min)
                L *= float(rng.uniform(0.85, 1.15))
                if near:
                    L *= float(self.cfg.near_length_mul)

                width = self.cfg.width_min + s * (self.cfg.width_max - self.cfg.width_min)
                width *= float(rng.uniform(0.85, 1.15))
                if near:
                    width += float(self.cfg.near_width_add)
                width_i = int(max(1, round(width)))

                # endpoints
                dx = 0.5 * L * cos_a
                dy = 0.5 * L * sin_a
                x1, y1 = int(round(x0 - dx)), int(round(y0 - dy))
                x2, y2 = int(round(x0 + dx)), int(round(y0 + dy))

                # intensity
                val = self.cfg.streak_value_min + s * (self.cfg.streak_value_max - self.cfg.streak_value_min)
                val *= float(rng.uniform(0.8, 1.2))
                if near:
                    val *= float(self.cfg.near_value_mul)
                val = float(np.clip(val, 0.0, 1.0))

                # draw into temporary mask, then optionally break it up
                tmp = np.zeros((h, w), dtype=np.float32)
                cv2.line(tmp, (x1, y1), (x2, y2), color=1.0, thickness=width_i, lineType=cv2.LINE_AA)

                tmp *= val

                # streak breakup / dropout
                if self.cfg.enable_dropout and (rng.uniform() < self.cfg.dropout_prob):
                    keep = float(rng.uniform(self.cfg.dropout_keep_min, self.cfg.dropout_keep_max))
                    noise = (rng.uniform(size=tmp.shape).astype(np.float32) < keep).astype(np.float32)
                    tmp *= noise

                R = np.maximum(R, tmp)

        # far layer (thin, faint)
        draw_streaks(n_far, near=False)
        # near layer (thicker, longer)
        if n_near > 0:
            draw_streaks(n_near, near=True)

        # Optional blur to soften lines (slightly stronger for near look)
        sigma_eff = blur_sigma * (self.cfg.near_blur_mul if (self.cfg.enable_multilayer and n_near > 0) else 1.0)
        if sigma_eff > 1e-6:
            R = cv2.GaussianBlur(R, (0, 0), sigmaX=float(sigma_eff), sigmaY=float(sigma_eff), borderType=cv2.BORDER_REFLECT)

        # ---------------- depth foreground bias ----------------
        # R <- R * (1 - d)^eta
        depth_weight = np.power(np.clip(1.0 - d, 0.0, 1.0), eta).astype(np.float32)
        R_biased = (R * depth_weight).astype(np.float32)

        # Convert rain layer to RGB
        if self.cfg.rgb_tint:
            R_rgb = np.stack([R_biased, R_biased, R_biased], axis=-1)
        else:
            R_rgb = np.repeat(R_biased[..., None], 3, axis=-1)

        # ---------------- composite (exact slide form) ----------------
        # I' = I*(1-alpha) + (I + R)*alpha
        out = img * (1.0 - alpha) + (img + R_rgb) * alpha
        out = clamp01(out)

        # NEW: rainy tone mapping to make scene less sunny
        if self.cfg.apply_rainy_tone and s > 1e-6:
            out = _apply_rain_tone(out, s, rng, self.cfg)
            out = clamp01(out)

        # Mask (where rain exists) for optional supervision/debugging
        mask = None
        if return_mask:
            mask = np.clip(R_biased, 0.0, 1.0).astype(np.float32)

        meta: Dict[str, Any] = {
            "severity": s,
            "alpha": alpha,
            "eta": eta,
            "n_streaks": n_streaks,
            "n_far": int(n_far),
            "n_near": int(n_near),
            "angle_deg": float(ang_deg),
            "blur_sigma": float(sigma_eff),
            "invert_depth": self.cfg.invert_depth,
            "multilayer": bool(self.cfg.enable_multilayer),
            "rainy_tone": bool(self.cfg.apply_rainy_tone),
            "dropout": bool(self.cfg.enable_dropout),
        }

        return DegradationResult(image=out, mask=mask, meta=meta)
