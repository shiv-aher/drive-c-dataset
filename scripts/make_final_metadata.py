#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATASET_ROOT = PROJECT_ROOT / "DRIVE-C-Core"

DEFAULT_GT_CSV = DATASET_ROOT / "samples_metadata_with_gshi.csv"
DEFAULT_PRED_CSV = DATASET_ROOT / "samples_metadata_with_gshi_pred.csv"
DEFAULT_OUT_CSV = DATASET_ROOT / "final_metadata.csv"


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def index_by_sample_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = row.get("sample_id", "").strip()
        if not sid:
            raise ValueError("Found row without sample_id")
        if sid in out:
            raise ValueError(f"Duplicate sample_id found: {sid}")
        out[sid] = row
    return out


def ordered_union(base_fields: List[str], extra_fields: List[str]) -> List[str]:
    out = list(base_fields)
    for f in extra_fields:
        if f not in out:
            out.append(f)
    return out


def merge_rows(
    gt_rows: List[Dict[str, str]],
    pred_index: Dict[str, Dict[str, str]],
    pred_fields: List[str],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []

    pred_only_cols = [
        "gshi_pred",
        "pred_top1_issue",
        "pred_top1_prob",
        "pred_presence_json",
        "pred_severity_json",
        "pred_present_thresh_json",
    ]

    for gt in gt_rows:
        sid = gt["sample_id"].strip()
        row = dict(gt)

        pred = pred_index.get(sid)
        if pred is not None:
            for col in pred_only_cols:
                if col in pred:
                    row[col] = pred[col]
        else:
            for col in pred_only_cols:
                if col not in row:
                    row[col] = ""

        merged.append(row)

    return merged


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    ap.add_argument("--pred-csv", type=Path, default=DEFAULT_PRED_CSV)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = ap.parse_args()

    if not args.gt_csv.exists():
        raise FileNotFoundError(f"Missing GT CSV: {args.gt_csv}")
    if not args.pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {args.pred_csv}")

    gt_rows, gt_fields = read_csv_rows(args.gt_csv)
    pred_rows, pred_fields = read_csv_rows(args.pred_csv)

    pred_index = index_by_sample_id(pred_rows)

    merged_rows = merge_rows(
        gt_rows=gt_rows,
        pred_index=pred_index,
        pred_fields=pred_fields,
    )

    final_fields = ordered_union(
        gt_fields,
        [
            "gshi_pred",
            "pred_top1_issue",
            "pred_top1_prob",
            "pred_presence_json",
            "pred_severity_json",
            "pred_present_thresh_json",
        ],
    )

    write_csv_rows(args.out_csv, merged_rows, final_fields)

    print(f"[DONE] Wrote: {args.out_csv}")
    print(f"[INFO] Rows: {len(merged_rows)}")


if __name__ == "__main__":
    main()