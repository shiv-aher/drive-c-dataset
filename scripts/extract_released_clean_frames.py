#!/usr/bin/env python3
"""Extract canonical PNG frames from the ten released anonymized clean clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    clips = sorted((args.dataset_root / "clean_clips").glob("S??_clean.mp4"))
    if len(clips) != 10:
        raise SystemExit(f"Expected 10 canonical clean clips; found {len(clips)}")

    for clip in clips:
        scenario = clip.stem.removesuffix("_clean")
        outdir = args.output_root / scenario
        outdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip}")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            target = outdir / f"frame_{count:06d}.png"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {target}")
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f"Cannot write {target}")
            count += 1
        cap.release()
        if count != 128:
            raise RuntimeError(f"{clip.name}: expected 128 frames, extracted {count}")
        print(f"[OK] {scenario}: {count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
