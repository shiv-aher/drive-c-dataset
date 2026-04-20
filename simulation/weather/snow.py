# simulation/weather/snow.py  (v2: multi-layer + wind + tone)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01, ensure_depth


@dataclass
class SnowConfig:
    # total flakes scales with severity
    n_min: int = 400
    n_max: int = 6000

    # layer fractions (sum <= 1; remainder goes to mid)
    far_frac: float = 0.55     # many tiny flakes
    near_frac: float = 0.10    # few big bokeh flakes
    # mid_frac = 1 - far - near

    # FAR layer: tiny flakes (pixel-ish)
    far_r_min: float = 0.35
    far_r_max: float = 1.40
    far_val_min: float = 0.25
    far_val_max: float = 0.75

    # MID layer: small soft flakes
    mid_r_min: float = 0.8
    mid_r_max: float = 3.0
    mid_val_min: float = 0.35
    mid_val_max: float = 0.90

    # NEAR layer: few large out-of-focus bokeh flakes (soft)
    near_r_min: float = 3.0
    near_r_max: float = 14.0
    near_val_min: float = 0.20
    near_val_max: float = 0.65

    # feather controls soft edges of circles
    feather_far: float = 0.10
    feather_mid: float = 0.55
    feather_near: float = 0.85

    # depth bias controls how much flakes get stronger when "near"
    # weight w = (1 - d)^eta  (assuming d: far=1, near=0)
    eta_min: float = 0.6
    eta_max: float = 2.4

    # additive blending: I' = I + alpha * S
    alpha_min: float = 0.08
    alpha_max: float = 0.40

    # if your depth is near=1 far=0, set True
    invert_depth: bool = False

    # mild layer blur for softness (post)
    layer_blur_sigma_min: float = 0.2
    layer_blur_sigma_max: float = 1.2

    # WIND / motion blur on snow layer
    enable_wind_blur: bool = True
    wind_angle_mean_deg: float = 75.0     # direction of streaking (close to vertical)
    wind_angle_jitter_deg: float = 25.0
    wind_len_min: int = 1
    wind_len_max: int = 7

    # "snowy" tone mapping: less sunny
    apply_snowy_tone: bool = True
    dim_min: float = 0.05
    dim_max: float = 0.35
    contrast_min: float = 0.05
    contrast_max: float = 0.30
    desat_min: float = 0.05
    desat_max: float = 0.30
    cool_tint_min: float = 0.02
    cool_tint_max: float = 0.10

    # Optional: return snow mask
    return_mask: bool = True


def _draw_soft_circle(layer: np.ndarray, cx: float, cy: float, r: float, val: float, feather: float):
    """Feathered disk using smoothstep falloff in a small ROI."""
    h, w = layer.shape
    r = float(max(0.2, r))
    feather = float(np.clip(feather, 0.0, 0.95))

    x0 = int(max(0, np.floor(cx - r - 2)))
    x1 = int(min(w - 1, np.ceil(cx + r + 2)))
    y0 = int(max(0, np.floor(cy - r - 2)))
    y1 = int(min(h - 1, np.ceil(cy + r + 2)))
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    inner = (1.0 - feather) * r
    outer = r

    a = np.zeros_like(dist, dtype=np.float32)
    inside = dist <= inner
    band = (dist > inner) & (dist <= outer)
    a[inside] = 1.0
    if np.any(band):
        t = (dist[band] - inner) / max(outer - inner, 1e-6)
        a[band] = 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep

    patch = (val * a).astype(np.float32)
    layer[y0:y1 + 1, x0:x1 + 1] = np.maximum(layer[y0:y1 + 1, x0:x1 + 1], patch)


def _directional_blur_kernel(length: int, angle_rad: float) -> np.ndarray:
    """Line kernel rotated by angle."""
    length = int(max(1, length))
    ksize = length * 2 + 1
    k = np.zeros((ksize, ksize), dtype=np.float32)
    center = length
    x1 = int(center - length * np.cos(angle_rad))
    y1 = int(center - length * np.sin(angle_rad))
    x2 = int(center + length * np.cos(angle_rad))
    y2 = int(center + length * np.sin(angle_rad))
    cv2.line(k, (x1, y1), (x2, y2), color=1.0, thickness=1, lineType=cv2.LINE_AA)
    k /= (k.sum() + 1e-8)
    return k


def _apply_snowy_tone(img01: np.ndarray, s: float, rng: np.random.Generator, cfg: SnowConfig) -> np.ndarray:
    """Overcast/snow lighting: dim, lower contrast, desaturate, slight cool tint."""
    s = float(np.clip(s, 0.0, 1.0))
    out = img01.astype(np.float32)

    # dim
    dim = 1.0 - (cfg.dim_min + s * (cfg.dim_max - cfg.dim_min)) * float(rng.uniform(0.85, 1.15))
    out *= dim

    # contrast reduction
    mean = out.mean(axis=(0, 1), keepdims=True)
    cdrop = (cfg.contrast_min + s * (cfg.contrast_max - cfg.contrast_min)) * float(rng.uniform(0.85, 1.15))
    contrast = 1.0 - cdrop
    out = mean + contrast * (out - mean)

    # desaturation
    lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
    sdrop = (cfg.desat_min + s * (cfg.desat_max - cfg.desat_min)) * float(rng.uniform(0.85, 1.15))
    sat = 1.0 - sdrop
    out = lum + sat * (out - lum)

    # cool tint
    tint = (cfg.cool_tint_min + s * (cfg.cool_tint_max - cfg.cool_tint_min)) * float(rng.uniform(0.85, 1.15))
    out[..., 2] = np.clip(out[..., 2] * (1.0 + tint), 0.0, 1.0)
    out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.5 * tint), 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


class SnowDegradation(BaseDegradation):
    name = "snow"

    def __init__(self, cfg: SnowConfig = SnowConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: Optional[bool] = None,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))
        img = to_float01(image)
        h, w = img.shape[:2]
        d = ensure_depth(depth, h, w)

        # Ensure convention: d=far(1), near(0) for (1-d)=near
        if self.cfg.invert_depth:
            d = 1.0 - d

        if return_mask is None:
            return_mask = self.cfg.return_mask

        # params from severity
        n_total = int(np.round(self.cfg.n_min + s * (self.cfg.n_max - self.cfg.n_min)))
        n_total = max(1, n_total)

        alpha = self.cfg.alpha_min + s * (self.cfg.alpha_max - self.cfg.alpha_min)
        alpha *= float(rng.uniform(0.9, 1.1))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # ---- OPTIONAL CVPR POLISH: soften extreme snow slightly ----
        # At s≈1.0, reduce opacity so the scene is still faintly visible and doesn't
        # look like a complete sensor blackout.
        if s > 0.95:
            alpha *= 0.9  # reduce by 10%

        eta = self.cfg.eta_min + s * (self.cfg.eta_max - self.cfg.eta_min)
        eta *= float(rng.uniform(0.9, 1.1))
        eta = float(max(0.0, eta))

        # layer counts
        n_far = int(round(n_total * float(np.clip(self.cfg.far_frac, 0.0, 1.0))))
        n_near = int(round(n_total * float(np.clip(self.cfg.near_frac, 0.0, 1.0))))
        n_mid = max(0, n_total - n_far - n_near)

        S = np.zeros((h, w), dtype=np.float32)

        def sample_depth_weight(x: float, y: float) -> float:
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            dv = float(np.clip(d[yi, xi], 0.0, 1.0))
            return float(np.power(1.0 - dv, eta))  # near -> larger weight

        # FAR layer: emphasize far (small) but still allow some near contribution
        for _ in range(n_far):
            x = float(rng.uniform(0, w - 1))
            y = float(rng.uniform(0, h - 1))
            w_near = sample_depth_weight(x, y)

            r = (self.cfg.far_r_min + rng.uniform() * (self.cfg.far_r_max - self.cfg.far_r_min))
            # far flakes should NOT become huge when near; keep small
            r *= (0.9 + 0.3 * (1.0 - w_near))

            val = (self.cfg.far_val_min + rng.uniform() * (self.cfg.far_val_max - self.cfg.far_val_min))
            # slightly dimmer when near to avoid “bokeh blobs everywhere”
            val *= (0.7 + 0.5 * (1.0 - w_near))
            val = float(np.clip(val, 0.0, 1.0))

            _draw_soft_circle(S, x, y, r=r, val=val, feather=self.cfg.feather_far)

        # MID layer: depth-biased size/intensity (near stronger)
        for _ in range(n_mid):
            x = float(rng.uniform(0, w - 1))
            y = float(rng.uniform(0, h - 1))
            w_near = sample_depth_weight(x, y)

            r = (self.cfg.mid_r_min + rng.uniform() * (self.cfg.mid_r_max - self.cfg.mid_r_min))
            r *= (1.0 + 0.9 * w_near)

            val = (self.cfg.mid_val_min + rng.uniform() * (self.cfg.mid_val_max - self.cfg.mid_val_min))
            val *= (0.8 + 0.8 * w_near)
            val = float(np.clip(val, 0.0, 1.0))

            _draw_soft_circle(S, x, y, r=r, val=val, feather=self.cfg.feather_mid)

        # NEAR layer: few large bokeh flakes concentrated toward near depths
        for _ in range(n_near):
            # sample until we hit a near region (biased)
            for _tries in range(12):
                x = float(rng.uniform(0, w - 1))
                y = float(rng.uniform(0, h - 1))
                w_near = sample_depth_weight(x, y)
                if w_near > rng.uniform(0.15, 0.65):
                    break

            r = (self.cfg.near_r_min + rng.uniform() * (self.cfg.near_r_max - self.cfg.near_r_min))
            r *= (1.0 + 1.4 * w_near)

            val = (self.cfg.near_val_min + rng.uniform() * (self.cfg.near_val_max - self.cfg.near_val_min))
            # near bokeh is usually translucent, not fully white
            val *= (0.6 + 0.6 * w_near)
            val = float(np.clip(val, 0.0, 1.0))

            _draw_soft_circle(S, x, y, r=r, val=val, feather=self.cfg.feather_near)

        # global softness
        layer_blur = self.cfg.layer_blur_sigma_min + s * (self.cfg.layer_blur_sigma_max - self.cfg.layer_blur_sigma_min)
        layer_blur *= float(rng.uniform(0.9, 1.1))
        if layer_blur > 1e-6:
            S = cv2.GaussianBlur(S, (0, 0), sigmaX=float(layer_blur), sigmaY=float(layer_blur), borderType=cv2.BORDER_REFLECT)

        # wind / motion blur (very effective for realism)
        wind_len = int(round(self.cfg.wind_len_min + s * (self.cfg.wind_len_max - self.cfg.wind_len_min)))
        if self.cfg.enable_wind_blur and wind_len > 1:
            ang_deg = self.cfg.wind_angle_mean_deg + float(rng.uniform(-self.cfg.wind_angle_jitter_deg, self.cfg.wind_angle_jitter_deg))
            k = _directional_blur_kernel(wind_len, np.deg2rad(ang_deg))
            S = cv2.filter2D(S, -1, k, borderType=cv2.BORDER_REFLECT)

        # additive blend: I' = I + alpha * S
        S_rgb = np.repeat(S[..., None], 3, axis=-1)
        out = clamp01(img + alpha * S_rgb)

        # snowy lighting (less sunny)
        if self.cfg.apply_snowy_tone and s > 1e-6:
            out = _apply_snowy_tone(out, s, rng, self.cfg)
            out = clamp01(out)

        mask = None
        if return_mask:
            mask = np.clip(S, 0.0, 1.0).astype(np.float32)

        meta: Dict[str, Any] = {
            "severity": s,
            "alpha": float(alpha),
            "eta": eta,
            "n_total": n_total,
            "n_far": n_far,
            "n_mid": n_mid,
            "n_near": n_near,
            "layer_blur_sigma": float(layer_blur),
            "invert_depth": self.cfg.invert_depth,
            "wind_blur": bool(self.cfg.enable_wind_blur),
        }

        return DegradationResult(image=out, mask=mask, meta=meta)
