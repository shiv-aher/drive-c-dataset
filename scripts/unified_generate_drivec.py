#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # .../drive-c
DATASET_ROOT = PROJECT_ROOT / "DRIVE-C-Core"

# Make repo imports work regardless of where script is launched from
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ============================================================
# DATA PATHS
# ============================================================

CLEAN_FRAMES_ROOT = PROJECT_ROOT / "data/processed/drivec_core_frames"
DEPTH_ROOT = PROJECT_ROOT / "data/processed/drivec_core_depth"

CLEAN_CLIPS_DIR = DATASET_ROOT / "clean_clips"
CORRUPTED_CLIPS_DIR = DATASET_ROOT / "corrupted"
TMP_ROOT = DATASET_ROOT / "_tmp_build"

SCENARIO_METADATA_CSV = DATASET_ROOT / "scenario_metadata.csv"
SAMPLES_METADATA_CSV = DATASET_ROOT / "samples_metadata.csv"

# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OVERWRITE_EXISTING = False
DEFAULT_GSHI = -1.0

# ============================================================
# IMPORT GENERATORS
# ============================================================
# These imports assume your current repo layout is already working
# for the newer generators.

from generate_fog import apply_fog  # type: ignore
from generate_rain import (  # type: ignore
    apply_rain,
    build_shared_maps as build_rain_shared_maps,
    pick_device as pick_rain_device,
)
from generate_snow import (  # type: ignore
    apply_snow,
    build_shared_maps as build_snow_shared_maps,
    pick_device as pick_snow_device,
)
from generate_glaze import apply_glare, pick_device as pick_glare_device  # type: ignore

from simulation.digital.motion_blur import MotionBlurDegradation  # type: ignore
from simulation.digital.compression import CompressionArtifactsDegradation  # type: ignore
from simulation.physics_aware.defocus import DefocusDegradation  # type: ignore
from simulation.physics_aware.occlusion import LensOcclusionDegradation  # type: ignore
from simulation.sensor.noise import SensorNoiseDegradation  # type: ignore
from simulation.sensor.low_light import LowLightDegradation  # type: ignore
from simulation.sensor.exposure import ExposureShiftDegradation  # type: ignore

# ============================================================
# CONFIG
# ============================================================

SEVERITY_MAP: Dict[str, float] = {
    "s1": 0.08,
    "s2": 0.18,
    "s3": 0.35,
    "s4": 0.55,
    "s5": 0.75,
}

CORRUPTIONS = [
    "fog",
    "rain",
    "snow",
    "glare_flare",
    "motion_blur",
    "defocus_blur",
    "lens_occlusion",
    "sensor_noise",
    "low_light",
    "overexposure",
    "underexposure",
    "jpeg_compression",
]

DEV_SCENARIOS = {"S01", "S02", "S03", "S04", "S05"}
TEST_SCENARIOS = {"S06", "S07", "S08", "S09", "S10"}

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ScenarioRow:
    scenario_id: str
    source_video_id: str
    clip_start_frame: int
    clip_end_frame: int
    num_frames: int
    fps: int
    orig_resolution: str
    output_resolution: str
    weather: str
    time_of_day: str
    capture_date: str
    camera_model: str
    scene_type: str
    traffic_level: str
    notes: str
    output_clip_path: str


@dataclass
class SampleRow:
    sample_id: str
    scenario_id: str
    source_video_id: str
    split: str
    clip_type: str
    corruption_type: str
    severity_level: int
    severity_name: str
    severity_value: float
    is_clean: int
    gshi: float
    weather: str
    time_of_day: str
    scene_type: str
    traffic_level: str
    fps: int
    resolution: str
    num_frames: int
    output_path: str
    extra_json: str


# ============================================================
# HELPERS
# ============================================================

def stable_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_pngs(folder: Path) -> List[Path]:
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {folder}")
    return frames


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing depth map: {path}")
    return np.load(path).astype(np.float32)


def resize_rgb(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)


def resize_depth(depth: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def uint8_rgb_to_float01(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def write_rgb01_png(path: Path, rgb01: np.ndarray) -> None:
    ensure_dir(path.parent)
    rgb01 = np.nan_to_num(rgb01, nan=0.0, posinf=1.0, neginf=0.0)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    bgr = cv2.cvtColor((rgb01 * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def frames_to_mp4(frame_paths: List[Path], output_path: Path, fps: int) -> None:
    ensure_dir(output_path.parent)

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    try:
        for fp in frame_paths:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read frame: {fp}")
            writer.write(img)
    finally:
        writer.release()


def scenario_split(scenario_id: str) -> str:
    if scenario_id in DEV_SCENARIOS:
        return "dev"
    if scenario_id in TEST_SCENARIOS:
        return "test"
    return "unknown"


def load_scenario_metadata(csv_path: Path) -> Dict[str, ScenarioRow]:
    rows: Dict[str, ScenarioRow] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = ScenarioRow(
                scenario_id=r["scenario_id"],
                source_video_id=r["source_video_id"],
                clip_start_frame=int(r["clip_start_frame"]),
                clip_end_frame=int(r["clip_end_frame"]),
                num_frames=int(r["num_frames"]),
                fps=int(r["fps"]),
                orig_resolution=r["orig_resolution"],
                output_resolution=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
                weather=r["weather"],
                time_of_day=r["time_of_day"],
                capture_date=r["capture_date"],
                camera_model=r["camera_model"],
                scene_type=r["scene_type"],
                traffic_level=r["traffic_level"],
                notes=r["notes"],
                output_clip_path=r["output_clip_path"],
            )
            rows[row.scenario_id] = row
    return rows


# ============================================================
# UNIFIED WRAPPER
# ============================================================

class UnifiedGenerator:
    def __init__(self) -> None:
        self.motion_blur = MotionBlurDegradation()
        self.defocus = DefocusDegradation()
        self.occlusion = LensOcclusionDegradation()
        self.noise = SensorNoiseDegradation()
        self.low_light = LowLightDegradation()
        self.exposure = ExposureShiftDegradation()
        self.compression = CompressionArtifactsDegradation()

        self.rain_device = pick_rain_device()
        self.snow_device = pick_snow_device()
        self.glare_device = pick_glare_device()

    def apply_one(
        self,
        corruption: str,
        image_rgb: np.ndarray,
        depth: Optional[np.ndarray],
        severity_value: float,
        rng: np.random.Generator,
        clip_shared: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if corruption == "fog":
            out, mask, meta = apply_fog(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
            )
            return out, {"mask_mean": float(mask.mean()), **meta}

        if corruption == "rain":
            out, mask, meta = apply_rain(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
                device=self.rain_device,
                shared_maps=clip_shared["rain_shared_maps"],
            )
            return out, {"mask_mean": float(mask.mean()), **meta}

        if corruption == "snow":
            out, mask, meta = apply_snow(
                image_rgb=image_rgb,
                depth=depth,
                severity=severity_value,
                rng=rng,
                device=self.snow_device,
                shared_maps=clip_shared["snow_shared_maps"],
            )
            return out, {"mask_mean": float(mask.mean()), **meta}

        if corruption == "glare_flare":
            out, meta = apply_glare(
                image_rgb=image_rgb,
                severity=severity_value,
                rng=rng,
                device=self.glare_device,
            )
            return out, meta

        if corruption == "motion_blur":
            res = self.motion_blur.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
            )
            return res.image, res.meta

        if corruption == "defocus_blur":
            res = self.defocus.apply(
                image=image_rgb,
                severity=severity_value,
                depth=depth,
                rng=rng,
                return_mask=True,
            )
            return res.image, res.meta

        if corruption == "lens_occlusion":
            res = self.occlusion.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=True,
            )
            return res.image, res.meta

        if corruption == "sensor_noise":
            res = self.noise.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
            )
            return res.image, res.meta

        if corruption == "low_light":
            res = self.low_light.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
            )
            return res.image, res.meta

        if corruption == "overexposure":
            ev = abs(2.0 * severity_value)
            res = self.exposure.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
                ev_override=ev,
            )
            return res.image, res.meta

        if corruption == "underexposure":
            ev = -abs(2.0 * severity_value)
            res = self.exposure.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
                ev_override=ev,
            )
            return res.image, res.meta

        if corruption == "jpeg_compression":
            res = self.compression.apply(
                image=image_rgb,
                severity=severity_value,
                depth=None,
                rng=rng,
                return_mask=False,
            )
            return res.image, res.meta

        raise ValueError(f"Unknown corruption: {corruption}")


# ============================================================
# SHARED MAPS
# ============================================================

def build_clip_shared_maps(
    scenario_id: str,
    severity_name: str,
    frame_shape: Tuple[int, int, int],
    gen: UnifiedGenerator,
) -> dict:
    h, w = frame_shape[:2]
    image_seed = stable_seed("shared_maps", scenario_id, severity_name)

    return {
        "rain_shared_maps": build_rain_shared_maps(
            h=h,
            w=w,
            device=gen.rain_device,
            image_seed=image_seed,
        ),
        "snow_shared_maps": build_snow_shared_maps(
            h=h,
            w=w,
            device=gen.snow_device,
            image_seed=image_seed,
        ),
    }


# ============================================================
# CLEAN CLIP BUILD
# ============================================================

def build_clean_clip_from_resized_frames(
    scenario_id: str,
    clean_rgbs: List[np.ndarray],
    fps: int,
) -> Path:
    out_path = CLEAN_CLIPS_DIR / f"{scenario_id}_clean.mp4"

    if out_path.exists() and not OVERWRITE_EXISTING:
        return out_path

    tmp_dir = TMP_ROOT / "clean" / scenario_id
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    temp_paths: List[Path] = []
    for idx, rgb in enumerate(clean_rgbs):
        fp = tmp_dir / f"frame_{idx:06d}.png"
        write_rgb01_png(fp, uint8_rgb_to_float01(rgb))
        temp_paths.append(fp)

    frames_to_mp4(temp_paths, out_path, fps=fps)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path


# ============================================================
# CORE PROCESSING
# ============================================================

def process_scenario(
    scenario: ScenarioRow,
    gen: UnifiedGenerator,
    selected_corruptions: List[str],
    selected_severity_names: List[str],
    sample_rows: List[SampleRow],
) -> None:
    scenario_id = scenario.scenario_id
    split = scenario_split(scenario_id)
    fps = scenario.fps

    frames_dir = CLEAN_FRAMES_ROOT / scenario_id
    depth_dir = DEPTH_ROOT / scenario_id

    frame_paths = list_pngs(frames_dir)

    # Read + resize BEFORE generation so all outputs are 1280x720
    clean_rgbs: List[np.ndarray] = []
    clean_depths: List[np.ndarray] = []

    for fp in frame_paths:
        rgb = read_rgb(fp)
        depth = read_depth(depth_dir / f"{fp.stem}.npy")

        rgb = resize_rgb(rgb, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        depth = resize_depth(depth, OUTPUT_WIDTH, OUTPUT_HEIGHT)

        clean_rgbs.append(rgb)
        clean_depths.append(depth)

    clean_clip_path = build_clean_clip_from_resized_frames(
        scenario_id=scenario_id,
        clean_rgbs=clean_rgbs,
        fps=fps,
    )

    # Add clean row
    sample_rows.append(
        SampleRow(
            sample_id=f"{scenario_id}_clean",
            scenario_id=scenario_id,
            source_video_id=scenario.source_video_id,
            split=split,
            clip_type="clean",
            corruption_type="clean",
            severity_level=0,
            severity_name="s0",
            severity_value=0.0,
            is_clean=1,
            gshi=1.0,
            weather=scenario.weather,
            time_of_day=scenario.time_of_day,
            scene_type=scenario.scene_type,
            traffic_level=scenario.traffic_level,
            fps=fps,
            resolution=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
            num_frames=scenario.num_frames,
            output_path=str(clean_clip_path.relative_to(DATASET_ROOT)),
            extra_json="{}",
        )
    )

    frame_shape = clean_rgbs[0].shape

    for corruption in selected_corruptions:
        for severity_name in selected_severity_names:
            severity_value = SEVERITY_MAP[severity_name]
            severity_level = int(severity_name[1:])

            out_mp4 = CORRUPTED_CLIPS_DIR / corruption / severity_name / f"{scenario_id}_{corruption}_{severity_name}.mp4"

            if out_mp4.exists() and not OVERWRITE_EXISTING:
                print(f"[SKIP] {out_mp4}")
                continue

            tmp_dir = TMP_ROOT / corruption / severity_name / scenario_id
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            ensure_dir(tmp_dir)

            clip_shared = build_clip_shared_maps(
                scenario_id=scenario_id,
                severity_name=severity_name,
                frame_shape=frame_shape,
                gen=gen,
            )

            first_meta = {}

            for idx, (rgb, depth) in enumerate(zip(clean_rgbs, clean_depths)):
                rng = np.random.default_rng(
                    stable_seed(scenario_id, corruption, severity_name, idx)
                )

                need_depth = corruption in {"fog", "rain", "snow", "defocus_blur"}

                out_rgb01, meta = gen.apply_one(
                    corruption=corruption,
                    image_rgb=rgb,
                    depth=depth if need_depth else None,
                    severity_value=severity_value,
                    rng=rng,
                    clip_shared=clip_shared,
                )

                if idx == 0:
                    first_meta = meta

                out_fp = tmp_dir / f"frame_{idx:06d}.png"
                write_rgb01_png(out_fp, out_rgb01)

            temp_frame_paths = list_pngs(tmp_dir)
            frames_to_mp4(temp_frame_paths, out_mp4, fps=fps)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            sample_rows.append(
                SampleRow(
                    sample_id=f"{scenario_id}_{corruption}_{severity_name}",
                    scenario_id=scenario_id,
                    source_video_id=scenario.source_video_id,
                    split=split,
                    clip_type="corrupted",
                    corruption_type=corruption,
                    severity_level=severity_level,
                    severity_name=severity_name,
                    severity_value=severity_value,
                    is_clean=0,
                    gshi=DEFAULT_GSHI,
                    weather=scenario.weather,
                    time_of_day=scenario.time_of_day,
                    scene_type=scenario.scene_type,
                    traffic_level=scenario.traffic_level,
                    fps=fps,
                    resolution=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
                    num_frames=scenario.num_frames,
                    output_path=str(out_mp4.relative_to(DATASET_ROOT)),
                    extra_json=json.dumps(first_meta, separators=(",", ":")),
                )
            )

            print(f"[OK] {scenario_id} {corruption} {severity_name}")


# ============================================================
# CSV WRITE
# ============================================================

def write_samples_csv(rows: List[SampleRow], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "scenario_id",
                "source_video_id",
                "split",
                "clip_type",
                "corruption_type",
                "severity_level",
                "severity_name",
                "severity_value",
                "is_clean",
                "gshi",
                "weather",
                "time_of_day",
                "scene_type",
                "traffic_level",
                "fps",
                "resolution",
                "num_frames",
                "output_path",
                "extra_json",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario IDs to run, e.g. S01 S02",
    )
    parser.add_argument(
        "--severity-names",
        nargs="*",
        default=None,
        help="Severity names to run, e.g. s3 or s1 s2 s3",
    )
    parser.add_argument(
        "--corruptions",
        nargs="*",
        default=None,
        help="Corruptions to run, e.g. fog rain snow",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    scenarios = load_scenario_metadata(SCENARIO_METADATA_CSV)
    scenario_ids = sorted(scenarios.keys()) if args.scenarios is None else args.scenarios
    selected_severity_names = list(SEVERITY_MAP.keys()) if args.severity_names is None else args.severity_names
    selected_corruptions = CORRUPTIONS if args.corruptions is None else args.corruptions

    for sid in scenario_ids:
        if sid not in scenarios:
            raise ValueError(f"Unknown scenario: {sid}")

    for sev in selected_severity_names:
        if sev not in SEVERITY_MAP:
            raise ValueError(f"Unknown severity: {sev}")

    for corr in selected_corruptions:
        if corr not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {corr}")

    gen = UnifiedGenerator()
    sample_rows: List[SampleRow] = []

    for sid in scenario_ids:
        print(f"\n[INFO] Processing {sid}")
        process_scenario(
            scenario=scenarios[sid],
            gen=gen,
            selected_corruptions=selected_corruptions,
            selected_severity_names=selected_severity_names,
            sample_rows=sample_rows,
        )

    write_samples_csv(sample_rows, SAMPLES_METADATA_CSV)
    print(f"\n[DONE] Wrote {SAMPLES_METADATA_CSV}")


if __name__ == "__main__":
    main()