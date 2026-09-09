#!/usr/bin/env python3
"""
Generate depth-aware rain corruptions for Drive-C.

Key updates:
- darkens bright sunny sky into rainy overcast look
- depth-aware rain + haze
- GPU support
- kernel caching
- per-image shared random maps
- manifest writing
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import csv
import hashlib
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# =========================================================
# USER SETTINGS
# =========================================================
CLEAN_ROOT = Path("data/processed/drive_c_clean")
DEPTH_ROOT = Path("data/processed/depth_maps")
OUTPUT_ROOT = Path("data/processed/drive_c_corruptions/rain")
MANIFEST_PATH = Path("data/metadata/rain_manifest.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Set to None for all videos, or e.g. ["video_015", "video_016"]
ONLY_VIDEOS: Optional[List[str]] = None

SKIP_EXISTING = True

# Severity values in [0,1]
SEVERITY_LEVELS: Dict[str, float] = {
    "s1": 0.08,
    "s2": 0.18,
    "s3": 0.35,
    "s4": 0.55,
    "s5": 0.75,
}

GLOBAL_SEED = 12345

# GPU
PREFER_GPU = True

# Depth convention
# False: near=0, far=1
# True : near=1, far=0
INVERT_DEPTH = False

# Optional JPEG output
FORCE_JPG_OUTPUT = False
JPG_QUALITY = 95

# ---------------- Rain streak settings ----------------
DENSITY_MIN = 0.0010
DENSITY_MAX = 0.0080

LENGTH_MIN = 10
LENGTH_MAX = 40
WIDTH_MIN = 1
WIDTH_MAX = 2

ANGLE_MEAN_DEG = 80.0
ANGLE_JITTER_DEG = 10.0
LAYER_ANGLE_EXTRA_JITTER_DEG = 8.0

ALPHA_MIN = 0.08
ALPHA_MAX = 0.28

STREAK_VALUE_MIN = 0.45
STREAK_VALUE_MAX = 1.00

ETA_MIN = 0.6
ETA_MAX = 1.8
DEPTH_SEPARATION_EXP = 1.6

ENABLE_MULTILAYER = True
NEAR_LAYER_FRAC = 0.35
NEAR_LENGTH_MUL = 1.30
NEAR_WIDTH_ADD = 1
NEAR_VALUE_MUL = 1.20

RAIN_BLUR_SIGMA_MIN = 0.25
RAIN_BLUR_SIGMA_MAX = 0.85

# ---------------- Rain haze / mist ----------------
APPLY_RAIN_HAZE = True
RAIN_HAZE_MIN = 0.03
RAIN_HAZE_MAX = 0.18
RAIN_HAZE_AIRLIGHT = 0.82
RAIN_HAZE_DEPTH_EXP = 1.35

# ---------------- Rainy tone mapping ----------------
APPLY_RAINY_TONE = True
DIM_MIN = 0.08
DIM_MAX = 0.24
CONTRAST_MIN = 0.04
CONTRAST_MAX = 0.14
DESAT_MIN = 0.03
DESAT_MAX = 0.12
COOL_TINT_MIN = 0.01
COOL_TINT_MAX = 0.05

EXTRA_DESAT_MIN = 0.00
EXTRA_DESAT_MAX = 0.06

# ---------------- Overcast sky normalization ----------------
# This is the main fix for sunny/bright sky scenes.
APPLY_OVERCAST_SKY = True

# bright-pixel threshold for likely sky / sun-lit sky
SKY_BRIGHT_THRESH = 0.68

# stronger effect at top of image
SKY_TOP_WEIGHT_POWER = 1.8

# reduce brightness in bright sky
SKY_DIM_MIN = 0.08
SKY_DIM_MAX = 0.30

# reduce contrast in bright sky
SKY_CONTRAST_MIN = 0.04
SKY_CONTRAST_MAX = 0.18

# reduce saturation in bright sky
SKY_DESAT_MIN = 0.08
SKY_DESAT_MAX = 0.30

# slight cool gray shift
SKY_COOL_SHIFT_MIN = 0.01
SKY_COOL_SHIFT_MAX = 0.04

# blur sky mask for smooth transition
SKY_MASK_BLUR_SIGMA = 12.0

# ---------------- Spatial variability ----------------
ENABLE_PATCHINESS = True
PATCHINESS_SIGMA_MIN = 18.0
PATCHINESS_SIGMA_MAX = 40.0
PATCHINESS_STRENGTH_MIN = 0.08
PATCHINESS_STRENGTH_MAX = 0.28

ENABLE_STREAK_BRIGHTNESS_RANDOMNESS = False
STREAK_BRIGHTNESS_SIGMA_MIN = 3.0
STREAK_BRIGHTNESS_SIGMA_MAX = 8.0
STREAK_BRIGHTNESS_STRENGTH_MIN = 0.08
STREAK_BRIGHTNESS_STRENGTH_MAX = 0.18

# ---------------- Optional visibility blur ----------------
APPLY_VISIBILITY_BLUR = False
VISIBILITY_BLUR_SIGMA_MIN = 0.15
VISIBILITY_BLUR_SIGMA_MAX = 0.70
VISIBILITY_BLUR_ONLY_FOR_SEVERITIES_AT_LEAST = 0.35
# =========================================================


GAUSSIAN_KERNEL_CACHE: Dict[Tuple[int, float, str], torch.Tensor] = {}
MOTION_KERNEL_CACHE: Dict[Tuple[int, int, float, str], torch.Tensor] = {}


def list_video_dirs(root: Path) -> List[Path]:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if ONLY_VIDEOS is not None:
        allowed = set(ONLY_VIDEOS)
        dirs = [d for d in dirs if d.name in allowed]
    return dirs


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])


def to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def ensure_depth(depth: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    if depth is None:
        return np.zeros((h, w), dtype=np.float32)
    d = depth.astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min = float(d.min())
    d_max = float(d.max())
    if d_max - d_min < 1e-8:
        return np.zeros((h, w), dtype=np.float32)
    d = (d - d_min) / (d_max - d_min)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def stable_seed_from_path(path_str: str, severity_name: str) -> int:
    key = f"{path_str}|{severity_name}|{GLOBAL_SEED}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def stable_seed_from_image(path_str: str) -> int:
    key = f"{path_str}|shared_maps|{GLOBAL_SEED}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def pick_device() -> torch.device:
    if PREFER_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_torch_image01(img01: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img01).to(device=device, dtype=torch.float32)
    return t.permute(2, 0, 1).contiguous()


def to_torch_depth(depth01: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(depth01).to(device=device, dtype=torch.float32)


def to_numpy_image01(img_chw: torch.Tensor) -> np.ndarray:
    out = img_chw.detach().permute(1, 2, 0).contiguous().cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_numpy_mask(mask_hw: torch.Tensor) -> np.ndarray:
    out = mask_hw.detach().cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    sigma_key = round(float(sigma), 3)
    key = (int(kernel_size), sigma_key, str(device))
    if key in GAUSSIAN_KERNEL_CACHE:
        return GAUSSIAN_KERNEL_CACHE[key]

    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    GAUSSIAN_KERNEL_CACHE[key] = kernel
    return kernel


def motion_kernel2d(length: int, width: int, angle_deg: float, device: torch.device) -> torch.Tensor:
    angle_key = round(float(angle_deg), 1)
    key = (int(length), int(width), angle_key, str(device))
    if key in MOTION_KERNEL_CACHE:
        return MOTION_KERNEL_CACHE[key]

    length = max(3, int(length))
    width = max(1, int(width))

    k = max(9, int(math.ceil(length * 1.5)))
    if k % 2 == 0:
        k += 1

    canvas = np.zeros((k, k), dtype=np.float32)
    cx = (k - 1) / 2.0
    cy = (k - 1) / 2.0

    ang = math.radians(angle_deg)
    dx = 0.5 * length * math.cos(ang)
    dy = 0.5 * length * math.sin(ang)

    x1, y1 = int(round(cx - dx)), int(round(cy - dy))
    x2, y2 = int(round(cx + dx)), int(round(cy + dy))

    cv2.line(canvas, (x1, y1), (x2, y2), color=1.0, thickness=width, lineType=cv2.LINE_AA)

    if canvas.max() > 0:
        canvas /= canvas.sum()

    kernel = torch.from_numpy(canvas).to(device=device, dtype=torch.float32)
    MOTION_KERNEL_CACHE[key] = kernel
    return kernel


def conv2d_same(x_hw: torch.Tensor, kernel_hw: torch.Tensor) -> torch.Tensor:
    kh, kw = int(kernel_hw.shape[0]), int(kernel_hw.shape[1])
    x = x_hw[None, None, ...]
    k = kernel_hw[None, None, ...]
    out = F.conv2d(x, k, padding=(kh // 2, kw // 2))
    return out[0, 0]


def maybe_blur_hw(x_hw: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 1e-6:
        return x_hw
    ksize = max(3, int(math.ceil(sigma * 6)))
    if ksize % 2 == 0:
        ksize += 1
    kernel = gaussian_kernel2d(ksize, sigma, device)
    return conv2d_same(x_hw, kernel)


def build_shared_maps(h: int, w: int, device: torch.device, image_seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(image_seed))
    return {
        "patch_base": torch.rand((h, w), device=device, dtype=torch.float32, generator=gen),
        "bright_base": torch.rand((h, w), device=device, dtype=torch.float32, generator=gen),
    }


def make_patchiness_map(shared_maps: Dict[str, torch.Tensor], s: float, device: torch.device) -> torch.Tensor:
    sigma = PATCHINESS_SIGMA_MIN + s * (PATCHINESS_SIGMA_MAX - PATCHINESS_SIGMA_MIN)
    strength = PATCHINESS_STRENGTH_MIN + s * (PATCHINESS_STRENGTH_MAX - PATCHINESS_STRENGTH_MIN)

    base = maybe_blur_hw(shared_maps["patch_base"], sigma=sigma, device=device)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    patch = 1.0 + strength * (2.0 * base - 1.0)
    return torch.clamp(patch, 0.5, 1.5)


def make_streak_brightness_map(shared_maps: Dict[str, torch.Tensor], s: float, device: torch.device) -> torch.Tensor:
    sigma = STREAK_BRIGHTNESS_SIGMA_MIN + s * (STREAK_BRIGHTNESS_SIGMA_MAX - STREAK_BRIGHTNESS_SIGMA_MIN)
    strength = STREAK_BRIGHTNESS_STRENGTH_MIN + s * (STREAK_BRIGHTNESS_STRENGTH_MAX - STREAK_BRIGHTNESS_STRENGTH_MIN)

    base = maybe_blur_hw(shared_maps["bright_base"], sigma=sigma, device=device)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    mod = 1.0 + strength * (2.0 * base - 1.0)
    return torch.clamp(mod, 0.6, 1.4)


def apply_overcast_sky_torch(
    img_chw: torch.Tensor,
    s: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Make sunny / bright sky look rainy-overcast.
    Works by detecting bright top-region pixels and darkening/desaturating them.
    """
    c, h, w = img_chw.shape
    out = img_chw.clone()

    gray = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]

    y = torch.linspace(0.0, 1.0, h, device=device).view(h, 1).expand(h, w)
    top_weight = torch.pow(1.0 - y, SKY_TOP_WEIGHT_POWER)

    bright_mask = torch.clamp((gray - SKY_BRIGHT_THRESH) / max(1e-6, (1.0 - SKY_BRIGHT_THRESH)), 0.0, 1.0)
    sky_mask = bright_mask * top_weight
    sky_mask = maybe_blur_hw(sky_mask, sigma=SKY_MASK_BLUR_SIGMA, device=device)
    sky_mask = torch.clamp(sky_mask, 0.0, 1.0)

    dim_amt = SKY_DIM_MIN + s * (SKY_DIM_MAX - SKY_DIM_MIN)
    contrast_drop = SKY_CONTRAST_MIN + s * (SKY_CONTRAST_MAX - SKY_CONTRAST_MIN)
    desat_amt = SKY_DESAT_MIN + s * (SKY_DESAT_MAX - SKY_DESAT_MIN)
    cool_amt = SKY_COOL_SHIFT_MIN + s * (SKY_COOL_SHIFT_MAX - SKY_COOL_SHIFT_MIN)

    # darken
    darkened = out * (1.0 - dim_amt * sky_mask.unsqueeze(0))

    # reduce contrast around mean
    mean = darkened.mean(dim=(1, 2), keepdim=True)
    darkened = mean + (1.0 - contrast_drop * sky_mask.unsqueeze(0)) * (darkened - mean)

    # desaturate
    lum = 0.2126 * darkened[0:1] + 0.7152 * darkened[1:2] + 0.0722 * darkened[2:3]
    darkened = darkened * (1.0 - desat_amt * sky_mask.unsqueeze(0)) + lum * (desat_amt * sky_mask.unsqueeze(0))

    # slight cool gray shift
    darkened[2] = torch.clamp(darkened[2] * (1.0 + cool_amt * sky_mask), 0.0, 1.0)
    darkened[0] = torch.clamp(darkened[0] * (1.0 - 0.4 * cool_amt * sky_mask), 0.0, 1.0)

    return torch.clamp(darkened, 0.0, 1.0), float(sky_mask.mean().item())


def make_sparse_seed_map(
    h: int,
    w: int,
    density: float,
    value: float,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    density = float(np.clip(density, 0.0, 1.0))
    probs = torch.full((h, w), density, device=device, dtype=torch.float32)
    seeds = torch.bernoulli(probs)

    gain = value * float(rng.uniform(0.88, 1.12))
    noise = torch.rand((h, w), device=device, dtype=torch.float32)
    seeds = seeds * (0.7 + 0.3 * noise) * gain
    return seeds


def build_rain_layer(
    h: int,
    w: int,
    *,
    density: float,
    length: int,
    width: int,
    angle_deg: float,
    value: float,
    blur_sigma: float,
    rng: np.random.Generator,
    device: torch.device,
    severity: float,
    shared_maps: Dict[str, torch.Tensor],
) -> torch.Tensor:
    seeds = make_sparse_seed_map(h, w, density, value, rng, device)

    if ENABLE_PATCHINESS:
        seeds = seeds * make_patchiness_map(shared_maps, severity, device)

    if ENABLE_STREAK_BRIGHTNESS_RANDOMNESS:
        seeds = seeds * make_streak_brightness_map(shared_maps, severity, device)

    kernel = motion_kernel2d(length=length, width=width, angle_deg=angle_deg, device=device)
    rain = conv2d_same(seeds, kernel)
    rain = maybe_blur_hw(rain, sigma=blur_sigma, device=device)

    mx = rain.max()
    if float(mx) > 1e-8:
        rain = rain / mx

    return torch.clamp(rain, 0.0, 1.0)


def apply_rain_tone_torch(img_chw: torch.Tensor, s: float, rng: np.random.Generator) -> torch.Tensor:
    out = img_chw

    dim_amt = DIM_MIN + s * (DIM_MAX - DIM_MIN)
    dim_amt *= float(rng.uniform(0.9, 1.1))
    out = out * (1.0 - dim_amt)

    mean = out.mean(dim=(1, 2), keepdim=True)
    cdrop = CONTRAST_MIN + s * (CONTRAST_MAX - CONTRAST_MIN)
    cdrop *= float(rng.uniform(0.9, 1.1))
    out = mean + (1.0 - cdrop) * (out - mean)

    lum = 0.2126 * out[0:1] + 0.7152 * out[1:2] + 0.0722 * out[2:3]
    sdrop = DESAT_MIN + s * (DESAT_MAX - DESAT_MIN)
    out = lum + (1.0 - sdrop) * (out - lum)

    tint = COOL_TINT_MIN + s * (COOL_TINT_MAX - COOL_TINT_MIN)
    out[2] = torch.clamp(out[2] * (1.0 + tint), 0.0, 1.0)
    out[0] = torch.clamp(out[0] * (1.0 - 0.5 * tint), 0.0, 1.0)

    return torch.clamp(out, 0.0, 1.0)


def apply_extra_desat_torch(img_chw: torch.Tensor, s: float) -> torch.Tensor:
    amt = EXTRA_DESAT_MIN + s * (EXTRA_DESAT_MAX - EXTRA_DESAT_MIN)
    if amt <= 1e-8:
        return img_chw
    lum = 0.2126 * img_chw[0:1] + 0.7152 * img_chw[1:2] + 0.0722 * img_chw[2:3]
    out = img_chw * (1.0 - amt) + lum * amt
    return torch.clamp(out, 0.0, 1.0)


def apply_rain_haze_torch(img_chw: torch.Tensor, d_t: torch.Tensor, s: float) -> Tuple[torch.Tensor, float]:
    haze_strength = RAIN_HAZE_MIN + s * (RAIN_HAZE_MAX - RAIN_HAZE_MIN)
    depth_haze = torch.pow(torch.clamp(d_t, 0.0, 1.0), RAIN_HAZE_DEPTH_EXP)
    haze_map = haze_strength * depth_haze

    air = torch.full_like(img_chw, float(RAIN_HAZE_AIRLIGHT))
    out = img_chw * (1.0 - haze_map.unsqueeze(0)) + air * haze_map.unsqueeze(0)
    return torch.clamp(out, 0.0, 1.0), float(haze_strength)


def apply_visibility_blur_torch(img_chw: torch.Tensor, s: float, device: torch.device) -> Tuple[torch.Tensor, float]:
    sigma = VISIBILITY_BLUR_SIGMA_MIN + s * (VISIBILITY_BLUR_SIGMA_MAX - VISIBILITY_BLUR_SIGMA_MIN)
    if sigma <= 1e-6:
        return img_chw, 0.0

    ksize = max(3, int(math.ceil(sigma * 6)))
    if ksize % 2 == 0:
        ksize += 1

    kernel = gaussian_kernel2d(ksize, sigma, device)
    out = torch.stack([conv2d_same(img_chw[c], kernel) for c in range(3)], dim=0)
    return torch.clamp(out, 0.0, 1.0), float(sigma)


def apply_rain(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    device: torch.device,
    shared_maps: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    s = float(np.clip(severity, 0.0, 1.0))

    img = to_float01(image_rgb)
    h, w = img.shape[:2]
    d = ensure_depth(depth, h, w)

    if INVERT_DEPTH:
        d = 1.0 - d

    d = np.clip(d, 0.0, 1.0).astype(np.float32)

    img_t = to_torch_image01(img, device=device)
    d_t = to_torch_depth(d, device=device)

    overcast_mask_mean = 0.0
    if APPLY_OVERCAST_SKY and s > 1e-6:
        img_t, overcast_mask_mean = apply_overcast_sky_torch(img_t, s, device)

    density = DENSITY_MIN + s * (DENSITY_MAX - DENSITY_MIN)
    density *= float(rng.uniform(0.92, 1.08))
    density = float(max(1e-6, density))

    alpha = ALPHA_MIN + s * (ALPHA_MAX - ALPHA_MIN)
    alpha *= float(rng.uniform(0.92, 1.08))
    alpha = float(np.clip(alpha, 0.0, 1.0))

    eta = ETA_MIN + s * (ETA_MAX - ETA_MIN)
    eta *= float(rng.uniform(0.92, 1.08))
    eta = float(max(0.0, eta))

    blur_sigma = RAIN_BLUR_SIGMA_MIN + s * (RAIN_BLUR_SIGMA_MAX - RAIN_BLUR_SIGMA_MIN)
    blur_sigma *= float(rng.uniform(0.92, 1.08))
    blur_sigma = float(max(0.0, blur_sigma))

    base_angle = ANGLE_MEAN_DEG + float(rng.uniform(-ANGLE_JITTER_DEG, ANGLE_JITTER_DEG))
    far_angle = base_angle + float(rng.uniform(-LAYER_ANGLE_EXTRA_JITTER_DEG, LAYER_ANGLE_EXTRA_JITTER_DEG))
    near_angle = base_angle + float(rng.uniform(-LAYER_ANGLE_EXTRA_JITTER_DEG, LAYER_ANGLE_EXTRA_JITTER_DEG))

    base_length = LENGTH_MIN + s * (LENGTH_MAX - LENGTH_MIN)
    base_width = WIDTH_MIN + s * (WIDTH_MAX - WIDTH_MIN)
    base_value = STREAK_VALUE_MIN + s * (STREAK_VALUE_MAX - STREAK_VALUE_MIN)

    near_frac = float(np.clip(NEAR_LAYER_FRAC, 0.0, 1.0)) if ENABLE_MULTILAYER else 0.0
    density_near = density * near_frac
    density_far = density * (1.0 - near_frac)

    rain_far = build_rain_layer(
        h, w,
        density=density_far,
        length=int(round(base_length)),
        width=int(round(base_width)),
        angle_deg=far_angle,
        value=float(np.clip(base_value, 0.0, 1.0)),
        blur_sigma=blur_sigma,
        rng=rng,
        device=device,
        severity=s,
        shared_maps=shared_maps,
    )

    if density_near > 1e-7:
        rain_near = build_rain_layer(
            h, w,
            density=density_near,
            length=int(round(base_length * NEAR_LENGTH_MUL)),
            width=int(round(base_width + NEAR_WIDTH_ADD)),
            angle_deg=near_angle,
            value=float(np.clip(base_value * NEAR_VALUE_MUL, 0.0, 1.0)),
            blur_sigma=blur_sigma * 1.05,
            rng=rng,
            device=device,
            severity=s,
            shared_maps=shared_maps,
        )
        R = torch.maximum(rain_far, rain_near)
    else:
        R = rain_far

    # far field emphasis
    depth_weight = torch.pow(torch.clamp(1.0 - d_t, 0.0, 1.0), eta)
    depth_weight = torch.pow(depth_weight, DEPTH_SEPARATION_EXP)
    R_biased = torch.clamp(R * depth_weight, 0.0, 1.0)

    # reduce streak visibility in very bright sky after overcast transform
    gray_now = 0.2126 * img_t[0] + 0.7152 * img_t[1] + 0.0722 * img_t[2]
    sky_reduce = torch.clamp((gray_now - 0.70) / 0.30, 0.0, 1.0)
    sky_reduce = maybe_blur_hw(sky_reduce, sigma=10.0, device=device)
    R_biased = R_biased * (1.0 - 0.45 * sky_reduce)

    R_rgb = torch.stack([
        R_biased * 0.97,
        R_biased * 1.00,
        R_biased * 1.03,
    ], dim=0)

    out = torch.clamp(img_t + alpha * R_rgb, 0.0, 1.0)

    if APPLY_RAINY_TONE and s > 1e-6:
        out = apply_rain_tone_torch(out, s, rng)

    haze_strength_used = 0.0
    if APPLY_RAIN_HAZE and s > 1e-6:
        out, haze_strength_used = apply_rain_haze_torch(out, d_t, s)

    out = apply_extra_desat_torch(out, s)

    visibility_blur_sigma = 0.0
    do_visibility_blur = APPLY_VISIBILITY_BLUR and s >= VISIBILITY_BLUR_ONLY_FOR_SEVERITIES_AT_LEAST
    if do_visibility_blur:
        out, visibility_blur_sigma = apply_visibility_blur_torch(out, s, device)

    out = torch.clamp(out, 0.0, 1.0)
    mask = to_numpy_mask(R_biased)

    meta: Dict[str, Any] = {
        "severity": s,
        "alpha": alpha,
        "eta": eta,
        "depth_separation_exp": float(DEPTH_SEPARATION_EXP),
        "density": density,
        "density_far": density_far,
        "density_near": density_near,
        "base_angle_deg": float(base_angle),
        "far_angle_deg": float(far_angle),
        "near_angle_deg": float(near_angle),
        "blur_sigma": float(blur_sigma),
        "invert_depth": bool(INVERT_DEPTH),
        "multilayer": bool(ENABLE_MULTILAYER),
        "rainy_tone": bool(APPLY_RAINY_TONE),
        "apply_rain_haze": bool(APPLY_RAIN_HAZE),
        "rain_haze_strength": float(haze_strength_used),
        "apply_overcast_sky": bool(APPLY_OVERCAST_SKY),
        "overcast_mask_mean": float(overcast_mask_mean),
        "extra_desat": float(EXTRA_DESAT_MIN + s * (EXTRA_DESAT_MAX - EXTRA_DESAT_MIN)),
        "patchiness": bool(ENABLE_PATCHINESS),
        "brightness_randomness": bool(ENABLE_STREAK_BRIGHTNESS_RANDOMNESS),
        "apply_visibility_blur": bool(do_visibility_blur),
        "visibility_blur_sigma": float(visibility_blur_sigma),
        "device": str(device),
    }

    return to_numpy_image01(out), mask, meta


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def init_manifest(manifest_path: Path) -> None:
    ensure_parent(manifest_path)
    if not manifest_path.exists():
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_id",
                "frame_name",
                "clean_path",
                "depth_path",
                "severity_name",
                "severity_value",
                "output_path",
                "alpha",
                "eta",
                "depth_separation_exp",
                "density",
                "density_far",
                "density_near",
                "base_angle_deg",
                "far_angle_deg",
                "near_angle_deg",
                "blur_sigma",
                "invert_depth",
                "multilayer",
                "rainy_tone",
                "apply_rain_haze",
                "rain_haze_strength",
                "apply_overcast_sky",
                "overcast_mask_mean",
                "extra_desat",
                "patchiness",
                "brightness_randomness",
                "apply_visibility_blur",
                "visibility_blur_sigma",
                "device",
                "status",
            ])


def append_manifest(
    manifest_path: Path,
    video_id: str,
    frame_name: str,
    clean_path: Path,
    depth_path: Path,
    severity_name: str,
    severity_value: float,
    output_path: Path,
    meta: dict,
    status: str,
) -> None:
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            video_id,
            frame_name,
            str(clean_path),
            str(depth_path),
            severity_name,
            severity_value,
            str(output_path),
            meta.get("alpha") if meta else None,
            meta.get("eta") if meta else None,
            meta.get("depth_separation_exp") if meta else None,
            meta.get("density") if meta else None,
            meta.get("density_far") if meta else None,
            meta.get("density_near") if meta else None,
            meta.get("base_angle_deg") if meta else None,
            meta.get("far_angle_deg") if meta else None,
            meta.get("near_angle_deg") if meta else None,
            meta.get("blur_sigma") if meta else None,
            meta.get("invert_depth") if meta else None,
            meta.get("multilayer") if meta else None,
            meta.get("rainy_tone") if meta else None,
            meta.get("apply_rain_haze") if meta else None,
            meta.get("rain_haze_strength") if meta else None,
            meta.get("apply_overcast_sky") if meta else None,
            meta.get("overcast_mask_mean") if meta else None,
            meta.get("extra_desat") if meta else None,
            meta.get("patchiness") if meta else None,
            meta.get("brightness_randomness") if meta else None,
            meta.get("apply_visibility_blur") if meta else None,
            meta.get("visibility_blur_sigma") if meta else None,
            meta.get("device") if meta else None,
            status,
        ])


def make_output_path(out_dir: Path, img_path: Path) -> Path:
    if not FORCE_JPG_OUTPUT:
        return out_dir / img_path.name
    return out_dir / f"{img_path.stem}.jpg"


def save_image(path: Path, rgb01: np.ndarray) -> bool:
    rgb01 = np.nan_to_num(rgb01, nan=0.0, posinf=1.0, neginf=0.0)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    bgr_u8 = cv2.cvtColor((rgb01 * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return cv2.imwrite(str(path), bgr_u8, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    return cv2.imwrite(str(path), bgr_u8)


def process_video(video_dir: Path, device: torch.device) -> tuple[int, int]:
    video_id = video_dir.name
    images = list_images(video_dir)

    total = 0
    saved = 0

    for img_path in images:
        total += 1

        depth_path = DEPTH_ROOT / video_id / f"{img_path.stem}.npy"
        if not depth_path.exists():
            print(f"[WARN] Missing depth map: {depth_path}")
            for severity_name, severity_value in SEVERITY_LEVELS.items():
                out_path = make_output_path(OUTPUT_ROOT / severity_name / video_id, img_path)
                append_manifest(
                    MANIFEST_PATH,
                    video_id,
                    img_path.name,
                    img_path,
                    depth_path,
                    severity_name,
                    severity_value,
                    out_path,
                    meta={},
                    status="missing_depth",
                )
            continue

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        h, w = image_rgb.shape[:2]
        image_seed = stable_seed_from_image(str(img_path))
        shared_maps = build_shared_maps(h, w, device, image_seed)

        for severity_name, severity_value in SEVERITY_LEVELS.items():
            out_dir = OUTPUT_ROOT / severity_name / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = make_output_path(out_dir, img_path)

            if SKIP_EXISTING and out_path.exists():
                append_manifest(
                    MANIFEST_PATH,
                    video_id,
                    img_path.name,
                    img_path,
                    depth_path,
                    severity_name,
                    severity_value,
                    out_path,
                    meta={},
                    status="skipped_existing",
                )
                saved += 1
                continue

            seed = stable_seed_from_path(str(img_path), severity_name)
            rng = np.random.default_rng(seed)

            rain_rgb, rain_mask, meta = apply_rain(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
                device=device,
                shared_maps=shared_maps,
            )

            ok = save_image(out_path, rain_rgb)

            if ok:
                append_manifest(
                    MANIFEST_PATH,
                    video_id,
                    img_path.name,
                    img_path,
                    depth_path,
                    severity_name,
                    severity_value,
                    out_path,
                    meta=meta,
                    status="ok",
                )
                saved += 1
            else:
                append_manifest(
                    MANIFEST_PATH,
                    video_id,
                    img_path.name,
                    img_path,
                    depth_path,
                    severity_name,
                    severity_value,
                    out_path,
                    meta=meta,
                    status="write_failed",
                )

    return total, saved


def main() -> None:
    if not CLEAN_ROOT.exists():
        raise FileNotFoundError(f"Clean root not found: {CLEAN_ROOT}")
    if not DEPTH_ROOT.exists():
        raise FileNotFoundError(f"Depth root not found: {DEPTH_ROOT}")

    init_manifest(MANIFEST_PATH)

    video_dirs = list_video_dirs(CLEAN_ROOT)
    if not video_dirs:
        print("[WARN] No video folders found.")
        return

    device = pick_device()

    print("[INFO] Videos to process:")
    print([v.name for v in video_dirs])

    print(f"[INFO] Rain severities: {SEVERITY_LEVELS}")
    print(
        f"[INFO] device={device}, "
        f"INVERT_DEPTH={INVERT_DEPTH}, "
        f"APPLY_OVERCAST_SKY={APPLY_OVERCAST_SKY}, "
        f"SKY_BRIGHT_THRESH={SKY_BRIGHT_THRESH}, "
        f"ALPHA_MIN={ALPHA_MIN}, ALPHA_MAX={ALPHA_MAX}, "
        f"RAIN_HAZE_MIN={RAIN_HAZE_MIN}, RAIN_HAZE_MAX={RAIN_HAZE_MAX}, "
        f"ENABLE_PATCHINESS={ENABLE_PATCHINESS}, "
        f"APPLY_VISIBILITY_BLUR={APPLY_VISIBILITY_BLUR}"
    )

    total_images = 0
    total_saved = 0

    for video_dir in video_dirs:
        print(f"\n[INFO] Processing {video_dir.name}")
        n_total, n_saved = process_video(video_dir, device=device)
        total_images += n_total
        total_saved += n_saved
        print(f"[INFO] {video_dir.name}: base_images={n_total}, outputs_saved={n_saved}")

    print("\n[DONE] Rain generation complete.")
    print(f"[SUMMARY] Base images processed: {total_images}")
    print(f"[SUMMARY] Corrupted outputs saved: {total_saved}")


if __name__ == "__main__":
    main()