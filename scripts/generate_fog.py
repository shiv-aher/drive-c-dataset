#!/usr/bin/env python3
"""
Generate depth-aware fog corruptions for Drive-C.

Input:
    data/processed/drive_c_clean/
        video_001/
        ...
    data/processed/depth_maps/
        video_001/
            frame_000001.npy
            ...

Output:
    data/processed/drive_c_corruptions/fog/
        s1/
            video_001/
        s2/
            video_001/
        s3/
            video_001/

Also writes:
    data/metadata/fog_manifest.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
import csv
import hashlib

import cv2
import numpy as np

# =========================================================
# USER SETTINGS
# =========================================================
CLEAN_ROOT = Path("data/processed/drive_c_clean")
DEPTH_ROOT = Path("data/processed/depth_maps")
OUTPUT_ROOT = Path("data/processed/drive_c_corruptions/fog")
MANIFEST_PATH = Path("data/metadata/fog_manifest.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Set to None for all videos, or e.g. ["video_015", "video_016"]
ONLY_VIDEOS: Optional[List[str]] = None

SKIP_EXISTING = True

# Severity definitions
# These are normalized severity values in [0,1]
SEVERITY_LEVELS: Dict[str, float] = {
    "s1": 0.08,
    "s2": 0.18,
    "s3": 0.35,
    "s4": 0.55,
    "s5": 0.75,
}

# Fog parameters
K_MIN = 0.8
K_MAX = 2.0
A_MIN = 0.40
A_MAX = 0.55
GAMMA = 0.7
INVERT_DEPTH = True

# Realism controls
DEPTH_SHIFT = 0.30          # keep near field clearer
DISTANCE_EXP = 1.5          # nonlinear buildup: weak mid, stronger far
T_MIN = 0.25                # prevent full white-out
CONTRAST_SCALE = 0.95       # mild contrast reduction
DARKEN_FACTOR = 0.88        # overcast / no-sun look
BLUR_KERNEL = 5             # set to 0 to disable; must be odd if > 0

# Optional reproducible randomness
GLOBAL_SEED = 12345
# =========================================================


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


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def stable_seed_from_path(path_str: str, severity_name: str) -> int:
    key = f"{path_str}|{severity_name}|{GLOBAL_SEED}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def sample_atmospheric_light(rng: np.random.Generator) -> np.ndarray:
    """
    Slightly cool gray fog/airlight.
    """
    a = rng.uniform(A_MIN, A_MAX)
    A = np.array([0.95 * a, 1.00 * a, 1.05 * a], dtype=np.float32)
    return A.reshape(1, 1, 3)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d_min = float(d.min())
    d_max = float(d.max())

    if d_max - d_min < 1e-8:
        return np.zeros_like(d, dtype=np.float32)

    d = (d - d_min) / (d_max - d_min)
    d = np.clip(d, 0.0, 1.0)
    return d.astype(np.float32)


def apply_fog(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Beer-Lambert fog with realism refinements:
        I(x) = J(x) * t(x) + A * (1 - t(x))
        t(x) = exp(-k * d_eff(x))

    Refinements:
    - inverse-depth option
    - depth shift so foreground stays clearer
    - nonlinear distance buildup
    - transmission floor to avoid complete washout
    - mild contrast reduction
    - overcast darkening
    - optional blur
    """
    s = float(np.clip(severity, 0.0, 1.0))

    img = to_float01(image_rgb)
    d = normalize_depth(depth)

    if INVERT_DEPTH:
        d = 1.0 - d

    d = clamp01(d)
    d_shaped = np.power(d, GAMMA).astype(np.float32)

    # Keep near region clearer, push fog more into mid/far distance
    d_eff = np.clip(d_shaped - DEPTH_SHIFT, 0.0, 1.0).astype(np.float32)
    d_eff = np.power(d_eff, DISTANCE_EXP).astype(np.float32)

    # Severity-dependent extinction coefficient
    k = K_MIN + s * (K_MAX - K_MIN)

    # Atmospheric light
    A = sample_atmospheric_light(rng)

    # Transmission
    t = np.exp(-k * d_eff).astype(np.float32)
    t = np.clip(t, T_MIN, 1.0)
    t3 = t[..., None]

    # Fog synthesis
    out = img * t3 + A * (1.0 - t3)

    # Mild contrast reduction
    mean = float(out.mean())
    out = (out - mean) * CONTRAST_SCALE + mean

    # Overcast / no direct sun look
    out = out * DARKEN_FACTOR

    out = clamp01(out)

    # Slight blur improves realism for fog
    if BLUR_KERNEL and BLUR_KERNEL >= 3 and BLUR_KERNEL % 2 == 1:
        out = cv2.GaussianBlur(out, (BLUR_KERNEL, BLUR_KERNEL), 0)
        out = clamp01(out)

    fog_mask = (1.0 - t).astype(np.float32)

    meta = {
        "severity": s,
        "k": float(k),
        "gamma": float(GAMMA),
        "depth_shift": float(DEPTH_SHIFT),
        "distance_exp": float(DISTANCE_EXP),
        "t_min": float(T_MIN),
        "contrast_scale": float(CONTRAST_SCALE),
        "darken_factor": float(DARKEN_FACTOR),
        "blur_kernel": int(BLUR_KERNEL),
        "A": [float(A[0, 0, 0]), float(A[0, 0, 1]), float(A[0, 0, 2])],
        "depth_min": float(d_shaped.min()),
        "depth_max": float(d_shaped.max()),
    }

    return out, fog_mask, meta


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
                "k",
                "gamma",
                "depth_shift",
                "distance_exp",
                "t_min",
                "contrast_scale",
                "darken_factor",
                "blur_kernel",
                "A_r",
                "A_g",
                "A_b",
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
        A = meta.get("A", [None, None, None]) if meta else [None, None, None]
        writer.writerow([
            video_id,
            frame_name,
            str(clean_path),
            str(depth_path),
            severity_name,
            severity_value,
            str(output_path),
            meta.get("k") if meta else None,
            meta.get("gamma") if meta else None,
            meta.get("depth_shift") if meta else None,
            meta.get("distance_exp") if meta else None,
            meta.get("t_min") if meta else None,
            meta.get("contrast_scale") if meta else None,
            meta.get("darken_factor") if meta else None,
            meta.get("blur_kernel") if meta else None,
            A[0],
            A[1],
            A[2],
            status,
        ])


def process_video(video_dir: Path) -> tuple[int, int]:
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
                out_path = OUTPUT_ROOT / severity_name / video_id / img_path.name
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

        for severity_name, severity_value in SEVERITY_LEVELS.items():
            out_dir = OUTPUT_ROOT / severity_name / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / img_path.name

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

            fog_rgb, fog_mask, meta = apply_fog(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
            )

            fog_bgr_u8 = cv2.cvtColor(
                (fog_rgb * 255.0 + 0.5).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
            ok = cv2.imwrite(str(out_path), fog_bgr_u8)

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

    print("[INFO] Videos to process:")
    print([v.name for v in video_dirs])

    print(f"[INFO] Fog severities: {SEVERITY_LEVELS}")
    print(
        f"[INFO] INVERT_DEPTH={INVERT_DEPTH}, "
        f"GAMMA={GAMMA}, "
        f"K_MIN={K_MIN}, K_MAX={K_MAX}, "
        f"A_MIN={A_MIN}, A_MAX={A_MAX}, "
        f"DEPTH_SHIFT={DEPTH_SHIFT}, "
        f"DISTANCE_EXP={DISTANCE_EXP}, "
        f"T_MIN={T_MIN}, "
        f"CONTRAST_SCALE={CONTRAST_SCALE}, "
        f"DARKEN_FACTOR={DARKEN_FACTOR}, "
        f"BLUR_KERNEL={BLUR_KERNEL}"
    )

    total_images = 0
    total_saved = 0

    for video_dir in video_dirs:
        print(f"\n[INFO] Processing {video_dir.name}")
        n_total, n_saved = process_video(video_dir)
        total_images += n_total
        total_saved += n_saved
        print(f"[INFO] {video_dir.name}: base_images={n_total}, outputs_saved={n_saved}")

    print("\n[DONE] Fog generation complete.")
    print(f"[SUMMARY] Base images processed: {total_images}")
    print(f"[SUMMARY] Corrupted outputs saved: {total_saved}")


if __name__ == "__main__":
    main()