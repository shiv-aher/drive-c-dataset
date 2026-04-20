#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path("/home/sa/shiva/projects/drive-c")

ANON_ROOT = PROJECT_ROOT / "data/processed/anonymized"
CLIP_SELECTION_CSV = PROJECT_ROOT / "DRIVE-C-Core/clip_selection.csv"
OUT_FRAMES_ROOT = PROJECT_ROOT / "data/processed/drivec_core_frames"


@dataclass
class ClipSelection:
    scenario_id: str
    source_video_id: str
    clip_start_frame: int
    num_frames: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_clip_selection(csv_path: Path) -> List[ClipSelection]:
    rows: List[ClipSelection] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ClipSelection(
                    scenario_id=r["scenario_id"].strip(),
                    source_video_id=r["source_video_id"].strip(),
                    clip_start_frame=int(r["clip_start_frame"]),
                    num_frames=int(r["num_frames"]),
                )
            )
    return rows


def list_source_frames(video_dir: Path) -> List[Path]:
    frames = sorted(video_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    return frames


def copy_subset(selection: ClipSelection) -> None:
    src_frames_dir = ANON_ROOT / selection.source_video_id
    dst_frames_dir = OUT_FRAMES_ROOT / selection.scenario_id

    ensure_dir(dst_frames_dir)

    source_frames = list_source_frames(src_frames_dir)

    start = selection.clip_start_frame
    end = start + selection.num_frames

    if end > len(source_frames):
        raise ValueError(
            f"{selection.source_video_id}: requested frames [{start}, {end}) "
            f"but only {len(source_frames)} frames exist."
        )

    selected = source_frames[start:end]

    for new_idx, src_frame_path in enumerate(selected):
        dst_frame_path = dst_frames_dir / f"frame_{new_idx:06d}.png"
        shutil.copy2(src_frame_path, dst_frame_path)

    print(
        f"[OK] {selection.scenario_id} <- {selection.source_video_id} "
        f"frames {start}..{end-1} ({selection.num_frames} frames)"
    )


def main() -> None:
    selections = load_clip_selection(CLIP_SELECTION_CSV)
    ensure_dir(OUT_FRAMES_ROOT)

    for sel in selections:
        copy_subset(sel)

    print("\nDone.")
    print(f"Frames: {OUT_FRAMES_ROOT}")


if __name__ == "__main__":
    main()