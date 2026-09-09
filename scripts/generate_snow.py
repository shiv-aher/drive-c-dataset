#!/usr/bin/env python3
"""
Generate depth-aware snow corruptions for Drive-C.

Updated version:
- keeps same severity scale as fog/rain
- stronger mid/high snow via nonlinear severity mapping
- mixed flake sizes
- overcast sky normalization
- snow haze
- GPU support
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
OUTPUT_ROOT = Path("data/processed/drive_c_corruptions/snow")
MANIFEST_PATH = Path("data/metadata/snow_manifest.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

ONLY_VIDEOS: Optional[List[str]] = None
SKIP_EXISTING = True

# Keep SAME severity scale across corruptions
SEVERITY_LEVELS: Dict[str, float] = {
    "s1": 0.08,
    "s2": 0.18,
    "s3": 0.35,
    "s4": 0.55,
    "s5": 0.75,
}

GLOBAL_SEED = 12345

PREFER_GPU = True
INVERT_DEPTH = False

FORCE_JPG_OUTPUT = False
JPG_QUALITY = 95

# ---------------- Snow strength mapping ----------------
# IMPORTANT:
# This keeps severity labels the same, but makes visual snow grow faster.
SEVERITY_GAMMA = 1.8

# ---------------- Snow particle density ----------------
DENSITY_MIN = 0.0010
DENSITY_MAX = 0.0200

# ---------------- Brightness / opacity ----------------
ALPHA_MIN = 0.10
ALPHA_MAX = 0.55

SNOW_VALUE_MIN = 0.78
SNOW_VALUE_MAX = 1.00

# ---------------- Small / medium / large flake shares ----------------
# Large flakes grow with severity.
MEDIUM_FRACTION_FIXED = 0.30
LARGE_FRACTION_MIN = 0.10
LARGE_FRACTION_MAX = 0.35

# ---------------- Flake blur sigma ranges ----------------
SMALL_SIGMA_MIN = 0.6
SMALL_SIGMA_MAX = 1.5

MEDIUM_SIGMA_MIN = 1.4
MEDIUM_SIGMA_MAX = 3.0

LARGE_SIGMA_MIN = 2.5
LARGE_SIGMA_MAX = 7.0

# ---------------- Depth weighting ----------------
ETA_MIN = 0.7
ETA_MAX = 1.8
DEPTH_SEPARATION_EXP = 1.4

# ---------------- Snow haze ----------------
APPLY_SNOW_HAZE = True
SNOW_HAZE_MIN = 0.02
SNOW_HAZE_MAX = 0.38
SNOW_HAZE_AIRLIGHT = 0.88
SNOW_HAZE_DEPTH_EXP = 1.3

# ---------------- Tone mapping ----------------
APPLY_SNOWY_TONE = True
DIM_MIN = 0.04
DIM_MAX = 0.20
CONTRAST_MIN = 0.03
CONTRAST_MAX = 0.14
DESAT_MIN = 0.02
DESAT_MAX = 0.10
COOL_TINT_MIN = 0.00
COOL_TINT_MAX = 0.03

# ---------------- Overcast sky normalization ----------------
APPLY_OVERCAST_SKY = True
SKY_BRIGHT_THRESH = 0.68
SKY_TOP_WEIGHT_POWER = 1.8
SKY_DIM_MIN = 0.06
SKY_DIM_MAX = 0.24
SKY_CONTRAST_MIN = 0.03
SKY_CONTRAST_MAX = 0.14
SKY_DESAT_MIN = 0.05
SKY_DESAT_MAX = 0.22
SKY_COOL_SHIFT_MIN = 0.00
SKY_COOL_SHIFT_MAX = 0.03
SKY_MASK_BLUR_SIGMA = 12.0

# ---------------- Spatial variability ----------------
ENABLE_PATCHINESS = True
PATCHINESS_SIGMA_MIN = 18.0
PATCHINESS_SIGMA_MAX = 45.0
PATCHINESS_STRENGTH_MIN = 0.08
PATCHINESS_STRENGTH_MAX = 0.30

# ---------------- Optional visibility blur ----------------
APPLY_VISIBILITY_BLUR = False
VISIBILITY_BLUR_SIGMA_MIN = 0.10
VISIBILITY_BLUR_SIGMA_MAX = 0.60
VISIBILITY_BLUR_ONLY_FOR_SEVERITIES_AT_LEAST = 0.35
# =========================================================


GAUSSIAN_KERNEL_CACHE: Dict[Tuple[int, float, str], torch.Tensor] = {}


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
        "small_base": torch.rand((h, w), device=device, dtype=torch.float32, generator=gen),
        "medium_base": torch.rand((h, w), device=device, dtype=torch.float32, generator=gen),
        "large_base": torch.rand((h, w), device=device, dtype=torch.float32, generator=gen),
    }


def make_patchiness_map(shared_maps: Dict[str, torch.Tensor], severity_scaled: float, device: torch.device) -> torch.Tensor:
    sigma = PATCHINESS_SIGMA_MIN + severity_scaled * (PATCHINESS_SIGMA_MAX - PATCHINESS_SIGMA_MIN)
    strength = PATCHINESS_STRENGTH_MIN + severity_scaled * (PATCHINESS_STRENGTH_MAX - PATCHINESS_STRENGTH_MIN)

    base = maybe_blur_hw(shared_maps["patch_base"], sigma=sigma, device=device)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    patch = 1.0 + strength * (2.0 * base - 1.0)
    return torch.clamp(patch, 0.5, 1.5)


def apply_overcast_sky_torch(
    img_chw: torch.Tensor,
    severity_scaled: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    out = img_chw.clone()
    _, h, w = out.shape

    gray = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    y = torch.linspace(0.0, 1.0, h, device=device).view(h, 1).expand(h, w)
    top_weight = torch.pow(1.0 - y, SKY_TOP_WEIGHT_POWER)

    bright_mask = torch.clamp((gray - SKY_BRIGHT_THRESH) / max(1e-6, (1.0 - SKY_BRIGHT_THRESH)), 0.0, 1.0)
    sky_mask = bright_mask * top_weight
    sky_mask = maybe_blur_hw(sky_mask, sigma=SKY_MASK_BLUR_SIGMA, device=device)
    sky_mask = torch.clamp(sky_mask, 0.0, 1.0)

    dim_amt = SKY_DIM_MIN + severity_scaled * (SKY_DIM_MAX - SKY_DIM_MIN)
    contrast_drop = SKY_CONTRAST_MIN + severity_scaled * (SKY_CONTRAST_MAX - SKY_CONTRAST_MIN)
    desat_amt = SKY_DESAT_MIN + severity_scaled * (SKY_DESAT_MAX - SKY_DESAT_MIN)
    cool_amt = SKY_COOL_SHIFT_MIN + severity_scaled * (SKY_COOL_SHIFT_MAX - SKY_COOL_SHIFT_MIN)

    out = out * (1.0 - dim_amt * sky_mask.unsqueeze(0))

    mean = out.mean(dim=(1, 2), keepdim=True)
    out = mean + (1.0 - contrast_drop * sky_mask.unsqueeze(0)) * (out - mean)

    lum = 0.2126 * out[0:1] + 0.7152 * out[1:2] + 0.0722 * out[2:3]
    out = out * (1.0 - desat_amt * sky_mask.unsqueeze(0)) + lum * (desat_amt * sky_mask.unsqueeze(0))

    out[2] = torch.clamp(out[2] * (1.0 + cool_amt * sky_mask), 0.0, 1.0)
    out[0] = torch.clamp(out[0] * (1.0 - 0.4 * cool_amt * sky_mask), 0.0, 1.0)

    return torch.clamp(out, 0.0, 1.0), float(sky_mask.mean().item())


def make_flake_seed(base: torch.Tensor, density: float) -> torch.Tensor:
    probs = torch.full_like(base, density)
    return torch.bernoulli(probs)


def build_snow_component(
    base_random: torch.Tensor,
    density: float,
    sigma: float,
    value: float,
    device: torch.device,
) -> torch.Tensor:
    seeds = make_flake_seed(base_random, density)
    seeds = seeds * value
    snow = maybe_blur_hw(seeds, sigma=sigma, device=device)
    mx = snow.max()
    if float(mx) > 1e-8:
        snow = snow / mx
    return torch.clamp(snow, 0.0, 1.0)


def apply_snowy_tone_torch(img_chw: torch.Tensor, severity_scaled: float, rng: np.random.Generator) -> torch.Tensor:
    out = img_chw

    dim_amt = DIM_MIN + severity_scaled * (DIM_MAX - DIM_MIN)
    dim_amt *= float(rng.uniform(0.9, 1.1))
    out = out * (1.0 - dim_amt)

    mean = out.mean(dim=(1, 2), keepdim=True)
    cdrop = CONTRAST_MIN + severity_scaled * (CONTRAST_MAX - CONTRAST_MIN)
    out = mean + (1.0 - cdrop) * (out - mean)

    lum = 0.2126 * out[0:1] + 0.7152 * out[1:2] + 0.0722 * out[2:3]
    sdrop = DESAT_MIN + severity_scaled * (DESAT_MAX - DESAT_MIN)
    out = lum + (1.0 - sdrop) * (out - lum)

    tint = COOL_TINT_MIN + severity_scaled * (COOL_TINT_MAX - COOL_TINT_MIN)
    out[2] = torch.clamp(out[2] * (1.0 + tint), 0.0, 1.0)

    return torch.clamp(out, 0.0, 1.0)


def apply_snow_haze_torch(img_chw: torch.Tensor, d_t: torch.Tensor, severity_scaled: float) -> Tuple[torch.Tensor, float]:
    haze_strength = SNOW_HAZE_MIN + severity_scaled * (SNOW_HAZE_MAX - SNOW_HAZE_MIN)
    depth_haze = torch.pow(torch.clamp(d_t, 0.0, 1.0), SNOW_HAZE_DEPTH_EXP)
    haze_map = haze_strength * depth_haze

    air = torch.full_like(img_chw, float(SNOW_HAZE_AIRLIGHT))
    out = img_chw * (1.0 - haze_map.unsqueeze(0)) + air * haze_map.unsqueeze(0)
    return torch.clamp(out, 0.0, 1.0), float(haze_strength)


def apply_visibility_blur_torch(img_chw: torch.Tensor, severity_scaled: float, device: torch.device) -> Tuple[torch.Tensor, float]:
    sigma = VISIBILITY_BLUR_SIGMA_MIN + severity_scaled * (VISIBILITY_BLUR_SIGMA_MAX - VISIBILITY_BLUR_SIGMA_MIN)
    if sigma <= 1e-6:
        return img_chw, 0.0

    ksize = max(3, int(math.ceil(sigma * 6)))
    if ksize % 2 == 0:
        ksize += 1
    kernel = gaussian_kernel2d(ksize, sigma, device)
    out = torch.stack([conv2d_same(img_chw[c], kernel) for c in range(3)], dim=0)
    return torch.clamp(out, 0.0, 1.0), float(sigma)


def apply_snow(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    device: torch.device,
    shared_maps: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    severity = float(np.clip(severity, 0.0, 1.0))
    severity_scaled = float(severity ** SEVERITY_GAMMA)

    img = to_float01(image_rgb)
    h, w = img.shape[:2]
    d = ensure_depth(depth, h, w)

    if INVERT_DEPTH:
        d = 1.0 - d
    d = np.clip(d, 0.0, 1.0).astype(np.float32)

    img_t = to_torch_image01(img, device=device)
    d_t = to_torch_depth(d, device=device)

    overcast_mask_mean = 0.0
    if APPLY_OVERCAST_SKY and severity > 1e-6:
        img_t, overcast_mask_mean = apply_overcast_sky_torch(img_t, severity_scaled, device)

    density = DENSITY_MIN + severity_scaled * (DENSITY_MAX - DENSITY_MIN)
    density *= float(rng.uniform(0.95, 1.05))
    density = float(max(1e-6, density))

    alpha = ALPHA_MIN + severity_scaled * (ALPHA_MAX - ALPHA_MIN)
    alpha *= float(rng.uniform(0.95, 1.05))
    alpha = float(np.clip(alpha, 0.0, 1.0))

    eta = ETA_MIN + severity_scaled * (ETA_MAX - ETA_MIN)
    eta *= float(rng.uniform(0.95, 1.05))
    eta = float(max(0.0, eta))

    snow_value = SNOW_VALUE_MIN + severity_scaled * (SNOW_VALUE_MAX - SNOW_VALUE_MIN)

    large_fraction = LARGE_FRACTION_MIN + severity_scaled * (LARGE_FRACTION_MAX - LARGE_FRACTION_MIN)
    medium_fraction = MEDIUM_FRACTION_FIXED
    small_fraction = max(0.05, 1.0 - medium_fraction - large_fraction)

    frac_sum = small_fraction + medium_fraction + large_fraction
    small_fraction /= frac_sum
    medium_fraction /= frac_sum
    large_fraction /= frac_sum

    small_density = density * small_fraction
    medium_density = density * medium_fraction
    large_density = density * large_fraction

    small_sigma = SMALL_SIGMA_MIN + severity_scaled * (SMALL_SIGMA_MAX - SMALL_SIGMA_MIN)
    medium_sigma = MEDIUM_SIGMA_MIN + severity_scaled * (MEDIUM_SIGMA_MAX - MEDIUM_SIGMA_MIN)
    large_sigma = LARGE_SIGMA_MIN + severity_scaled * (LARGE_SIGMA_MAX - LARGE_SIGMA_MIN)

    small = build_snow_component(
        shared_maps["small_base"], small_density, small_sigma, snow_value * 0.85, device
    )
    medium = build_snow_component(
        shared_maps["medium_base"], medium_density, medium_sigma, snow_value * 0.95, device
    )
    large = build_snow_component(
        shared_maps["large_base"], large_density, large_sigma, snow_value * 1.00, device
    )

    # combine components
    snow = torch.maximum(torch.maximum(small, medium), large)

    if ENABLE_PATCHINESS:
        snow = snow * make_patchiness_map(shared_maps, severity_scaled, device)

    # depth-aware far-field emphasis
    depth_weight = torch.pow(torch.clamp(1.0 - d_t, 0.0, 1.0), eta)
    depth_weight = torch.pow(depth_weight, DEPTH_SEPARATION_EXP)

    # keep some snow visible in near field too
    depth_weight = 0.6 + 0.4 * depth_weight

    snow_biased = torch.clamp(snow * depth_weight, 0.0, 1.0)

    snow_rgb = torch.stack([
        snow_biased * 0.98,
        snow_biased * 1.00,
        snow_biased * 1.00,
    ], dim=0)

    out = torch.clamp(
        img_t * (1.0 - alpha * snow_biased.unsqueeze(0)) + snow_rgb * alpha,
        0.0,
        1.0,
    )

    if APPLY_SNOWY_TONE and severity > 1e-6:
        out = apply_snowy_tone_torch(out, severity_scaled, rng)

    haze_strength_used = 0.0
    if APPLY_SNOW_HAZE and severity > 1e-6:
        out, haze_strength_used = apply_snow_haze_torch(out, d_t, severity_scaled)

    visibility_blur_sigma = 0.0
    do_visibility_blur = APPLY_VISIBILITY_BLUR and severity >= VISIBILITY_BLUR_ONLY_FOR_SEVERITIES_AT_LEAST
    if do_visibility_blur:
        out, visibility_blur_sigma = apply_visibility_blur_torch(out, severity_scaled, device)

    out = torch.clamp(out, 0.0, 1.0)

    meta: Dict[str, Any] = {
        "severity": severity,
        "severity_scaled": severity_scaled,
        "alpha": alpha,
        "eta": eta,
        "depth_separation_exp": float(DEPTH_SEPARATION_EXP),
        "density": density,
        "small_fraction": small_fraction,
        "medium_fraction": medium_fraction,
        "large_fraction": large_fraction,
        "small_density": small_density,
        "medium_density": medium_density,
        "large_density": large_density,
        "small_sigma": float(small_sigma),
        "medium_sigma": float(medium_sigma),
        "large_sigma": float(large_sigma),
        "invert_depth": bool(INVERT_DEPTH),
        "apply_snow_haze": bool(APPLY_SNOW_HAZE),
        "snow_haze_strength": float(haze_strength_used),
        "apply_overcast_sky": bool(APPLY_OVERCAST_SKY),
        "overcast_mask_mean": float(overcast_mask_mean),
        "patchiness": bool(ENABLE_PATCHINESS),
        "apply_visibility_blur": bool(do_visibility_blur),
        "visibility_blur_sigma": float(visibility_blur_sigma),
        "device": str(device),
    }

    return to_numpy_image01(out), to_numpy_image01(snow_rgb), meta


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
                "severity_scaled",
                "alpha",
                "eta",
                "depth_separation_exp",
                "density",
                "small_fraction",
                "medium_fraction",
                "large_fraction",
                "small_density",
                "medium_density",
                "large_density",
                "small_sigma",
                "medium_sigma",
                "large_sigma",
                "invert_depth",
                "apply_snow_haze",
                "snow_haze_strength",
                "apply_overcast_sky",
                "overcast_mask_mean",
                "patchiness",
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
            meta.get("severity_scaled") if meta else None,
            meta.get("alpha") if meta else None,
            meta.get("eta") if meta else None,
            meta.get("depth_separation_exp") if meta else None,
            meta.get("density") if meta else None,
            meta.get("small_fraction") if meta else None,
            meta.get("medium_fraction") if meta else None,
            meta.get("large_fraction") if meta else None,
            meta.get("small_density") if meta else None,
            meta.get("medium_density") if meta else None,
            meta.get("large_density") if meta else None,
            meta.get("small_sigma") if meta else None,
            meta.get("medium_sigma") if meta else None,
            meta.get("large_sigma") if meta else None,
            meta.get("invert_depth") if meta else None,
            meta.get("apply_snow_haze") if meta else None,
            meta.get("snow_haze_strength") if meta else None,
            meta.get("apply_overcast_sky") if meta else None,
            meta.get("overcast_mask_mean") if meta else None,
            meta.get("patchiness") if meta else None,
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

            snow_rgb, snow_mask, meta = apply_snow(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
                device=device,
                shared_maps=shared_maps,
            )

            ok = save_image(out_path, snow_rgb)

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

    print(f"[INFO] Snow severities: {SEVERITY_LEVELS}")
    print(
        f"[INFO] device={device}, "
        f"INVERT_DEPTH={INVERT_DEPTH}, "
        f"SEVERITY_GAMMA={SEVERITY_GAMMA}, "
        f"APPLY_OVERCAST_SKY={APPLY_OVERCAST_SKY}, "
        f"APPLY_SNOW_HAZE={APPLY_SNOW_HAZE}, "
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

    print("\n[DONE] Snow generation complete.")
    print(f"[SUMMARY] Base images processed: {total_images}")
    print(f"[SUMMARY] Corrupted outputs saved: {total_saved}")


if __name__ == "__main__":
    main()