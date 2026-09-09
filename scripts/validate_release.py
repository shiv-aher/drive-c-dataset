#!/usr/bin/env python3
"""Validate the public DRIVE-C dataset against the frozen release claims."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def result(name: str, passed: bool, detail: str) -> tuple[str, bool, str]:
    return name, passed, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path(__file__).parents[1] / "docs" / "release_claims.json",
    )
    parser.add_argument("--skip-video-probe", action="store_true")
    args = parser.parse_args()

    expected = json.loads(args.claims.read_text())["expected"]
    root = args.dataset_root.resolve()
    metadata = root / "final_metadata.csv"
    checks: list[tuple[str, bool, str]] = []

    if not metadata.is_file():
        print(f"CRITICAL: missing {metadata}")
        return 2

    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    checks.append(result("Metadata rows", len(rows) == expected["metadata_rows"], str(len(rows))))
    checks.append(result("Metadata fields", len(fields) == expected["metadata_fields"], str(len(fields))))

    mp4s = sorted(root.rglob("*.mp4"))
    checks.append(result("Filesystem clips", len(mp4s) == expected["clips_total"], str(len(mp4s))))

    counts = Counter(row.get("clip_type", "") for row in rows)
    checks.append(result("Clean metadata", counts["clean"] == expected["clips_clean"], str(counts["clean"])))
    checks.append(result("Corrupted metadata", counts["corrupted"] == expected["clips_corrupted"], str(counts["corrupted"])))

    scenarios = {row.get("scenario_id") for row in rows if row.get("scenario_id")}
    corruptions = {
        row.get("corruption_type") for row in rows
        if row.get("clip_type") == "corrupted" and row.get("corruption_type")
    }
    severities = {
        row.get("severity_level") for row in rows if row.get("clip_type") == "corrupted"
    }
    checks.append(result("Scenarios", len(scenarios) == expected["scenarios"], str(len(scenarios))))
    checks.append(result("Corruptions", len(corruptions) == expected["corruption_types"], str(len(corruptions))))
    checks.append(result("Severity levels", len(severities) == expected["severity_levels_corrupted"], str(sorted(severities))))

    splits = Counter(row.get("split", "") for row in rows)
    checks.append(result("Dev rows", splits["dev"] == expected["dev_rows"], str(splits["dev"])))
    checks.append(result("Test rows", splits["test"] == expected["test_rows"], str(splits["test"])))

    missing_paths = []
    duplicate_paths = []
    seen_paths: Counter[str] = Counter()
    json_errors = []
    json_fields = [name for name in fields if name.endswith("_json")]
    for row in rows:
        rel = row.get("output_path", "")
        seen_paths[rel] += 1
        if not rel or not (root / rel).is_file():
            missing_paths.append(row.get("sample_id", rel))
        for field in json_fields:
            value = row.get(field, "")
            if value:
                try:
                    json.loads(value)
                except json.JSONDecodeError:
                    json_errors.append(f"{row.get('sample_id')}:{field}")
    duplicate_paths = [path for path, count in seen_paths.items() if path and count > 1]
    checks.append(result("Metadata paths", not missing_paths, f"missing={len(missing_paths)}"))
    checks.append(result("Unique metadata paths", not duplicate_paths, f"duplicates={len(duplicate_paths)}"))
    checks.append(result("JSON fields", not json_errors, f"parse_errors={len(json_errors)}"))

    if not args.skip_video_probe:
        try:
            import cv2
        except ImportError as exc:
            print("CRITICAL: OpenCV is required unless --skip-video-probe is used")
            raise SystemExit(2) from exc
        bad_videos = []
        for path in mp4s:
            cap = cv2.VideoCapture(str(path))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            valid = (
                frames == expected["frames_per_clip"]
                and width == expected["width"]
                and height == expected["height"]
                and abs(fps - expected["fps"]) <= expected["fps_absolute_tolerance"]
            )
            if not valid:
                bad_videos.append(f"{path.relative_to(root)}:{frames},{width}x{height},{fps}")
        checks.append(result("Video properties", not bad_videos, f"failures={len(bad_videos)}"))

    print("DRIVE-C RELEASE AUDIT")
    print("=====================")
    for name, passed, detail in checks:
        print(f"{name:28} {'PASS' if passed else 'FAIL':4}  {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print(f"\nCritical failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
