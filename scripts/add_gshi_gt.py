#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # .../drive-c
DATASET_ROOT = PROJECT_ROOT / "DRIVE-C-Core"

# Make repo imports work
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.gshi_utils import load_gshi_config, compute_gshi  # type: ignore


DEFAULT_INPUT_CSV = DATASET_ROOT / "samples_metadata.csv"
DEFAULT_OUTPUT_CSV = DATASET_ROOT / "samples_metadata_with_gshi.csv"

# Update this if your taxonomy/GSHI YAML lives elsewhere.
# This should be the YAML that defines groups + gshi weights/scales.
DEFAULT_TAXONOMY_YAML = PROJECT_ROOT / "configs" / "taxonomy" / "camera_issues.yaml"


# Map dataset corruption names -> taxonomy issue names used by GSHI config
DATASET_TO_TAXONOMY = {
    "clean": None,
    "fog": "haze_fog",
    "rain": "rain",
    "snow": "snow",
    "glare_flare": "glare_flare",
    "motion_blur": "motion_blur",
    "defocus_blur": "defocus_blur",
    "lens_occlusion": "lens_occlusion",
    "sensor_noise": "sensor_noise",
    "low_light": "low_light",
    "overexposure": "exposure_shift",
    "underexposure": "exposure_shift",
    "jpeg_compression": "compression",
}


def build_empty_severity_dict(cfg) -> Dict[str, float]:
    """
    Create a severity dict with all issues from the GSHI config initialized to 0.
    """
    return {issue: 0.0 for issue in cfg.issue_to_group.keys()}


def compute_clip_gshi_gt(
    corruption_type: str,
    severity_value: float,
    cfg,
    clean_value: float = 1.0,
) -> float:
    """
    Compute GSHI ground truth for a single-corruption clip.

    For clean clips, returns clean_value directly.
    For corrupted clips, fills a full severity dict with zeros except the active issue.
    """
    if corruption_type == "clean":
        return float(clean_value)

    if corruption_type not in DATASET_TO_TAXONOMY:
        raise KeyError(f"Unknown corruption_type '{corruption_type}'")

    issue_name = DATASET_TO_TAXONOMY[corruption_type]
    if issue_name is None:
        return float(clean_value)

    sev_dict = build_empty_severity_dict(cfg)

    if issue_name not in sev_dict:
        raise KeyError(
            f"Mapped issue '{issue_name}' not found in GSHI config. "
            f"Available issues: {sorted(sev_dict.keys())}"
        )

    sev_dict[issue_name] = float(max(0.0, min(1.0, severity_value)))
    return float(compute_gshi(sev_dict, cfg))


def update_extra_json(extra_json: str, gshi_gt: float) -> str:
    """
    Preserve existing extra_json and add gshi_gt_input info if possible.
    """
    if extra_json is None or extra_json == "":
        payload = {}
    else:
        try:
            payload = json.loads(extra_json)
            if not isinstance(payload, dict):
                payload = {"raw_extra_json": payload}
        except Exception:
            payload = {"raw_extra_json": extra_json}

    payload["gshi_gt"] = float(gshi_gt)
    return json.dumps(payload, separators=(",", ":"))


def process_rows(
    rows: List[Dict[str, str]],
    cfg,
    clean_value: float,
    overwrite_extra_json: bool,
) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []

    for row in rows:
        corruption_type = row["corruption_type"].strip()
        severity_value = float(row["severity_value"])

        gshi_gt = compute_clip_gshi_gt(
            corruption_type=corruption_type,
            severity_value=severity_value,
            cfg=cfg,
            clean_value=clean_value,
        )

        new_row = dict(row)
        new_row["gshi_gt"] = f"{gshi_gt:.6f}"

        # Preserve existing gshi column if present; do not overwrite it.
        # You can later use gshi_pred there if you want.
        if overwrite_extra_json:
            new_row["extra_json"] = update_extra_json(new_row.get("extra_json", ""), gshi_gt)

        out_rows.append(new_row)

    return out_rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    # Insert gshi_gt after gshi if possible
    fieldnames = list(rows[0].keys())
    if "gshi_gt" not in fieldnames:
        if "gshi" in fieldnames:
            idx = fieldnames.index("gshi") + 1
            fieldnames.insert(idx, "gshi_gt")
        else:
            fieldnames.append("gshi_gt")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to samples_metadata.csv",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write enriched CSV",
    )
    ap.add_argument(
        "--taxonomy-yaml",
        type=Path,
        default=DEFAULT_TAXONOMY_YAML,
        help="Path to taxonomy/GSHI YAML",
    )
    ap.add_argument(
        "--clean-value",
        type=float,
        default=1.0,
        help="GSHI value to assign to clean clips. Default: 1.0",
    )
    ap.add_argument(
        "--overwrite-extra-json",
        action="store_true",
        help="Also inject gshi_gt into extra_json",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.taxonomy_yaml.exists():
        raise FileNotFoundError(f"Missing taxonomy YAML: {args.taxonomy_yaml}")

    cfg = load_gshi_config(str(args.taxonomy_yaml))
    rows = read_csv(args.input_csv)
    out_rows = process_rows(
        rows=rows,
        cfg=cfg,
        clean_value=float(args.clean_value),
        overwrite_extra_json=bool(args.overwrite_extra_json),
    )
    write_csv(args.output_csv, out_rows)

    print(f"[DONE] Wrote: {args.output_csv}")
    print(f"[INFO] Rows: {len(out_rows)}")
    print(f"[INFO] Taxonomy YAML: {args.taxonomy_yaml}")


if __name__ == "__main__":
    main()