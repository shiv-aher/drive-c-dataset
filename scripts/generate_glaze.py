#!/usr/bin/env python3
"""
Generate glare / flare corruptions for Drive-C.

Features:
- same severity scale as other corruptions
- bright-source-driven bloom
- optional directional streak flare
- optional lens ghost blobs
- mild global washout
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
DEPTH_ROOT = Path("data/processed/depth_maps")   # kept for consistency, not required
OUTPUT_ROOT = Path("data/processed/drive_c_corruptions/glare")
MANIFEST_PATH = Path("data/metadata/glare_manifest.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

ONLY_VIDEOS: Optional[List[str]] = None
SKIP_EXISTING = True

SEVERITY_LEVELS: Dict[str, float] = {
    "s1": 0.08,
    "s2": 0.18,
    "s3": 0.35,
    "s4": 0.55,
    "s5": 0.75,
}

GLOBAL_SEED = 12345

PREFER_GPU = True

FORCE_JPG_OUTPUT = False
JPG_QUALITY = 95

# Nonlinear scaling like snow
SEVERITY_GAMMA = 1.5

# ---------------- Bright-region detection ----------------
BRIGHT_THRESH_MIN = 0.82
BRIGHT_THRESH_MAX = 0.65

TOP_BIAS_STRENGTH_MIN = 0.15
TOP_BIAS_STRENGTH_MAX = 0.45

# ---------------- Bloom ----------------
BLOOM_STRENGTH_MIN = 0.08
BLOOM_STRENGTH_MAX = 0.65

BLOOM_SIGMA1_MIN = 8.0
BLOOM_SIGMA1_MAX = 28.0

BLOOM_SIGMA2_MIN = 20.0
BLOOM_SIGMA2_MAX = 110.0

# ---------------- Directional flare streak ----------------
ENABLE_STREAK = True
STREAK_STRENGTH_MIN = 0.03
STREAK_STRENGTH_MAX = 0.20

STREAK_LENGTH_MIN = 25
STREAK_LENGTH_MAX = 160

STREAK_WIDTH_MIN = 3
STREAK_WIDTH_MAX = 12

ANGLE_MEAN_DEG = 0.0
ANGLE_JITTER_DEG = 18.0

# ---------------- Lens ghosts ----------------
ENABLE_GHOSTS = True
GHOST_COUNT_MIN = 1
GHOST_COUNT_MAX = 5

GHOST_STRENGTH_MIN = 0.015
GHOST_STRENGTH_MAX = 0.05

GHOST_SIZE_MIN = 18
GHOST_SIZE_MAX = 120

# ---------------- Global washout ----------------
WASHOUT_MIN = 0.01
WASHOUT_MAX = 0.04

CONTRAST_DROP_MIN = 0.01
CONTRAST_DROP_MAX = 0.12

DESAT_MIN = 0.00
DESAT_MAX = 0.08

# Slight warm tint for glare
WARM_TINT_MIN = 0.00
WARM_TINT_MAX = 0.04

# ---------------- Vignette reduction (center brighten) ----------------
CENTER_GLOW_MIN = 0.00
CENTER_GLOW_MAX = 0.08

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


def stable_seed_from_path(path_str: str, severity_name: str) -> int:
    key = f"{path_str}|{severity_name}|{GLOBAL_SEED}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def pick_device() -> torch.device:
    if PREFER_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_torch_image01(img01: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img01).to(device=device, dtype=torch.float32)
    return t.permute(2, 0, 1).contiguous()


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


def make_top_bias(h: int, w: int, strength: float, device: torch.device) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, h, device=device).view(h, 1).expand(h, w)
    top = torch.pow(1.0 - y, 1.6)
    return 1.0 + strength * top


def build_bright_mask(img_t: torch.Tensor, severity_scaled: float, device: torch.device) -> Tuple[torch.Tensor, float]:
    gray = 0.2126 * img_t[0] + 0.7152 * img_t[1] + 0.0722 * img_t[2]

    thresh = BRIGHT_THRESH_MIN + severity_scaled * (BRIGHT_THRESH_MAX - BRIGHT_THRESH_MIN)
    top_bias_strength = TOP_BIAS_STRENGTH_MIN + severity_scaled * (TOP_BIAS_STRENGTH_MAX - TOP_BIAS_STRENGTH_MIN)

    top_bias = make_top_bias(gray.shape[0], gray.shape[1], top_bias_strength, device)
    boosted = gray * top_bias

    mask = torch.clamp((boosted - thresh) / max(1e-6, 1.0 - thresh), 0.0, 1.0)
    mask = maybe_blur_hw(mask, sigma=5.0, device=device)
    return torch.clamp(mask, 0.0, 1.0), float(thresh)


def build_bloom(mask: torch.Tensor, severity_scaled: float, device: torch.device) -> torch.Tensor:
    sigma1 = BLOOM_SIGMA1_MIN + severity_scaled * (BLOOM_SIGMA1_MAX - BLOOM_SIGMA1_MIN)
    sigma2 = BLOOM_SIGMA2_MIN + severity_scaled * (BLOOM_SIGMA2_MAX - BLOOM_SIGMA2_MIN)

    b1 = maybe_blur_hw(mask, sigma1, device)
    b2 = maybe_blur_hw(mask, sigma2, device)

    bloom = 0.65 * b1 + 0.35 * b2
    mx = bloom.max()
    if float(mx) > 1e-8:
        bloom = bloom / mx
    return torch.clamp(bloom, 0.0, 1.0)


def build_streak(mask: torch.Tensor, severity_scaled: float, rng: np.random.Generator, device: torch.device) -> Tuple[torch.Tensor, float]:
    if not ENABLE_STREAK:
        return torch.zeros_like(mask), 0.0

    strength = STREAK_STRENGTH_MIN + severity_scaled * (STREAK_STRENGTH_MAX - STREAK_STRENGTH_MIN)
    length = int(round(STREAK_LENGTH_MIN + severity_scaled * (STREAK_LENGTH_MAX - STREAK_LENGTH_MIN)))
    width = int(round(STREAK_WIDTH_MIN + severity_scaled * (STREAK_WIDTH_MAX - STREAK_WIDTH_MIN)))
    angle = ANGLE_MEAN_DEG + float(rng.uniform(-ANGLE_JITTER_DEG, ANGLE_JITTER_DEG))

    kernel = motion_kernel2d(length, width, angle, device)
    streak = conv2d_same(mask, kernel)
    mx = streak.max()
    if float(mx) > 1e-8:
        streak = streak / mx
    streak = streak * strength
    return torch.clamp(streak, 0.0, 1.0), float(angle)


def build_ghosts(mask: torch.Tensor, severity_scaled: float, rng: np.random.Generator, device: torch.device) -> Tuple[torch.Tensor, int]:
    if not ENABLE_GHOSTS:
        return torch.zeros_like(mask), 0

    h, w = mask.shape
    ghost_map = torch.zeros_like(mask)

    count = int(round(GHOST_COUNT_MIN + severity_scaled * (GHOST_COUNT_MAX - GHOST_COUNT_MIN)))
    count = max(0, count)

    ys, xs = torch.where(mask > 0.35)
    if ys.numel() == 0:
        return ghost_map, 0

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    for _ in range(count):
        idx = int(rng.integers(0, ys.numel()))
        sy = float(ys[idx].item())
        sx = float(xs[idx].item())

        # reflect source across center with slight randomness
        gx = cx - (sx - cx) * float(rng.uniform(0.6, 1.15))
        gy = cy - (sy - cy) * float(rng.uniform(0.6, 1.15))

        gx += float(rng.uniform(-0.08 * w, 0.08 * w))
        gy += float(rng.uniform(-0.08 * h, 0.08 * h))

        size = GHOST_SIZE_MIN + severity_scaled * (GHOST_SIZE_MAX - GHOST_SIZE_MIN)
        size *= float(rng.uniform(0.7, 1.3))
        strength = GHOST_STRENGTH_MIN + severity_scaled * (GHOST_STRENGTH_MAX - GHOST_STRENGTH_MIN)
        strength *= float(rng.uniform(0.7, 1.3))

        yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
        xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)

        rr = ((xx - gx) ** 2 + (yy - gy) ** 2) / max(1e-6, (size * size))
        blob = torch.exp(-rr * 2.0)

        # ring-like look
        ring = torch.exp(-((torch.sqrt(rr + 1e-8) - 0.65) ** 2) / 0.08)
        blob = 0.55 * blob + 0.45 * ring

        ghost_map = torch.maximum(ghost_map, blob * strength)

    return torch.clamp(ghost_map, 0.0, 1.0), count


def apply_global_glare_tone(img_t: torch.Tensor, severity_scaled: float) -> torch.Tensor:
    out = img_t.clone()

    washout = WASHOUT_MIN + severity_scaled * (WASHOUT_MAX - WASHOUT_MIN)
    contrast_drop = CONTRAST_DROP_MIN + severity_scaled * (CONTRAST_DROP_MAX - CONTRAST_DROP_MIN)
    desat = DESAT_MIN + severity_scaled * (DESAT_MAX - DESAT_MIN)
    warm = WARM_TINT_MIN + severity_scaled * (WARM_TINT_MAX - WARM_TINT_MIN)
    center_glow = CENTER_GLOW_MIN + severity_scaled * (CENTER_GLOW_MAX - CENTER_GLOW_MIN)

    # washout towards white
    out = out * (1.0 - washout) + washout

    # contrast reduction
    mean = out.mean(dim=(1, 2), keepdim=True)
    out = mean + (1.0 - contrast_drop) * (out - mean)

    # mild desaturation
    lum = 0.2126 * out[0:1] + 0.7152 * out[1:2] + 0.0722 * out[2:3]
    out = out * (1.0 - desat) + lum * desat

    # warm tint
    out[0] = torch.clamp(out[0] * (1.0 + warm), 0.0, 1.0)
    out[1] = torch.clamp(out[1] * (1.0 + 0.5 * warm), 0.0, 1.0)

    # center glow
    if center_glow > 1e-8:
        _, h, w = out.shape
        yy = torch.linspace(-1.0, 1.0, h, device=out.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=out.device).view(1, w).expand(h, w)
        rr = xx * xx + yy * yy
        center = torch.exp(-rr / 0.45)
        out = torch.clamp(out + center_glow * center.unsqueeze(0), 0.0, 1.0)

    return torch.clamp(out, 0.0, 1.0)


def apply_glare(
    image_rgb: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    severity = float(np.clip(severity, 0.0, 1.0))
    severity_scaled = float(severity ** SEVERITY_GAMMA)

    img = to_float01(image_rgb)
    img_t = to_torch_image01(img, device)

    bright_mask, bright_thresh = build_bright_mask(img_t, severity_scaled, device)

    bloom_strength = BLOOM_STRENGTH_MIN + severity_scaled * (BLOOM_STRENGTH_MAX - BLOOM_STRENGTH_MIN)
    bloom = build_bloom(bright_mask, severity_scaled, device) * bloom_strength

    streak, streak_angle = build_streak(bright_mask, severity_scaled, rng, device)
    ghosts, ghost_count = build_ghosts(bright_mask, severity_scaled, rng, device)

    flare_map = torch.clamp(bloom + streak + ghosts, 0.0, 1.0)
    flare_map = flare_map ** 0.8

    h, w = flare_map.shape
    y = torch.linspace(0, 1, h, device=device).view(h, 1)
    downward_bleed = torch.exp(-3 * y)
    flare_map = flare_map * (1 + 0.5 * downward_bleed)
    flare_map = torch.clamp(flare_map, 0.0, 1.0)

    # colorize slightly warm
    flare_rgb = torch.stack([
        flare_map * 1.00,
        flare_map * 0.97,
        flare_map * 0.92,
    ], dim=0)

    out = torch.clamp(img_t + flare_rgb, 0.0, 1.0)
    out = apply_global_glare_tone(out, severity_scaled)

    meta: Dict[str, Any] = {
        "severity": severity,
        "severity_scaled": severity_scaled,
        "bright_thresh": bright_thresh,
        "bloom_strength": float(bloom_strength),
        "streak_enabled": bool(ENABLE_STREAK),
        "streak_angle_deg": float(streak_angle),
        "ghosts_enabled": bool(ENABLE_GHOSTS),
        "ghost_count": int(ghost_count),
        "flare_mean": float(flare_map.mean().item()),
        "device": str(device),
    }

    return to_numpy_image01(out), meta

    # colorize slightly warm
    flare_rgb = torch.stack([
        flare_map * 1.00,
        flare_map * 0.97,
        flare_map * 0.92,
    ], dim=0)

    out = torch.clamp(img_t + flare_rgb, 0.0, 1.0)
    out = apply_global_glare_tone(out, severity_scaled)

    meta: Dict[str, Any] = {
        "severity": severity,
        "severity_scaled": severity_scaled,
        "bright_thresh": bright_thresh,
        "bloom_strength": float(bloom_strength),
        "streak_enabled": bool(ENABLE_STREAK),
        "streak_angle_deg": float(streak_angle),
        "ghosts_enabled": bool(ENABLE_GHOSTS),
        "ghost_count": int(ghost_count),
        "flare_mean": float(flare_map.mean().item()),
        "device": str(device),
    }

    return to_numpy_image01(out), meta


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
                "severity_name",
                "severity_value",
                "output_path",
                "severity_scaled",
                "bright_thresh",
                "bloom_strength",
                "streak_enabled",
                "streak_angle_deg",
                "ghosts_enabled",
                "ghost_count",
                "flare_mean",
                "device",
                "status",
            ])


def append_manifest(
    manifest_path: Path,
    video_id: str,
    frame_name: str,
    clean_path: Path,
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
            severity_name,
            severity_value,
            str(output_path),
            meta.get("severity_scaled") if meta else None,
            meta.get("bright_thresh") if meta else None,
            meta.get("bloom_strength") if meta else None,
            meta.get("streak_enabled") if meta else None,
            meta.get("streak_angle_deg") if meta else None,
            meta.get("ghosts_enabled") if meta else None,
            meta.get("ghost_count") if meta else None,
            meta.get("flare_mean") if meta else None,
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

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

            glare_rgb, meta = apply_glare(
                image_rgb=image_rgb,
                severity=severity_value,
                rng=rng,
                device=device,
            )

            ok = save_image(out_path, glare_rgb)

            if ok:
                append_manifest(
                    MANIFEST_PATH,
                    video_id,
                    img_path.name,
                    img_path,
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

    init_manifest(MANIFEST_PATH)

    video_dirs = list_video_dirs(CLEAN_ROOT)
    if not video_dirs:
        print("[WARN] No video folders found.")
        return

    device = pick_device()

    print("[INFO] Videos to process:")
    print([v.name for v in video_dirs])

    print(f"[INFO] Glare severities: {SEVERITY_LEVELS}")
    print(
        f"[INFO] device={device}, "
        f"SEVERITY_GAMMA={SEVERITY_GAMMA}, "
        f"ENABLE_STREAK={ENABLE_STREAK}, "
        f"ENABLE_GHOSTS={ENABLE_GHOSTS}"
    )

    total_images = 0
    total_saved = 0

    for video_dir in video_dirs:
        print(f"\n[INFO] Processing {video_dir.name}")
        n_total, n_saved = process_video(video_dir, device=device)
        total_images += n_total
        total_saved += n_saved
        print(f"[INFO] {video_dir.name}: base_images={n_total}, outputs_saved={n_saved}")

    print("\n[DONE] Glare generation complete.")
    print(f"[SUMMARY] Base images processed: {total_images}")
    print(f"[SUMMARY] Corrupted outputs saved: {total_saved}")


if __name__ == "__main__":
    main()