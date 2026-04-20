#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATASET_ROOT = PROJECT_ROOT / "DRIVE-C-Core"

DEFAULT_INPUT_CSV = DATASET_ROOT / "final_metadata.csv"
DEFAULT_REPORT_TXT = DATASET_ROOT / "gshi_pred_sanity_report.txt"
DEFAULT_PER_CORR_CSV = DATASET_ROOT / "gshi_pred_per_corruption_stats.csv"


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    mx = mean(x)
    my = mean(y)

    num = 0.0
    dx2 = 0.0
    dy2 = 0.0

    for a, b in zip(x, y):
        da = a - mx
        db = b - my
        num += da * db
        dx2 += da * da
        dy2 += db * db

    denom = math.sqrt(dx2 * dy2)
    if denom <= 1e-12:
        return float("nan")
    return num / denom


def rankdata(vals: List[float]) -> List[float]:
    """
    Average-rank ties, 1-based ranks.
    """
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            orig_idx = indexed[k][0]
            ranks[orig_idx] = avg_rank

        i = j + 1

    return ranks


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def severity_monotonic_fraction(
    rows: List[Dict[str, str]],
) -> float:
    """
    For each (scenario_id, corruption_type), check whether predicted health is
    non-increasing from s1 to s5. Return fraction that satisfy monotonicity.
    """
    groups: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)

    for r in rows:
        if r["clip_type"] != "corrupted":
            continue
        sid = r["scenario_id"]
        corr = r["corruption_type"]
        sev = int(r["severity_level"])
        pred = safe_float(r["gshi_pred"])
        groups[(sid, corr)][sev] = pred

    ok = 0
    total = 0

    for _, sev_map in groups.items():
        if not all(s in sev_map for s in [1, 2, 3, 4, 5]):
            continue
        total += 1
        vals = [sev_map[s] for s in [1, 2, 3, 4, 5]]
        mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        if mono:
            ok += 1

    if total == 0:
        return float("nan")
    return ok / total


def per_corruption_stats(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["clip_type"] == "corrupted":
            grouped[r["corruption_type"]].append(r)

    out = []

    for corr in sorted(grouped.keys()):
        corr_rows = grouped[corr]

        gt = [safe_float(r["gshi_gt"]) for r in corr_rows]
        pred = [safe_float(r["gshi_pred"]) for r in corr_rows]

        by_sev: Dict[int, List[float]] = defaultdict(list)
        for r in corr_rows:
            by_sev[int(r["severity_level"])].append(safe_float(r["gshi_pred"]))

        sev_means = {s: mean(by_sev[s]) for s in sorted(by_sev.keys()) if by_sev[s]}

        mono = (
            1
            if all(
                sev_means[s] >= sev_means[s + 1]
                for s in [1, 2, 3, 4]
                if s in sev_means and (s + 1) in sev_means
            )
            else 0
        )

        out.append(
            {
                "corruption_type": corr,
                "n_rows": str(len(corr_rows)),
                "gshi_gt_mean": f"{mean(gt):.6f}",
                "gshi_pred_mean": f"{mean(pred):.6f}",
                "pearson_gt_pred": f"{pearson_corr(gt, pred):.6f}",
                "spearman_gt_pred": f"{spearman_corr(gt, pred):.6f}",
                "pred_mean_s1": f"{sev_means.get(1, float('nan')):.6f}",
                "pred_mean_s2": f"{sev_means.get(2, float('nan')):.6f}",
                "pred_mean_s3": f"{sev_means.get(3, float('nan')):.6f}",
                "pred_mean_s4": f"{sev_means.get(4, float('nan')):.6f}",
                "pred_mean_s5": f"{sev_means.get(5, float('nan')):.6f}",
                "mean_monotonic_s1_to_s5": str(mono),
            }
        )

    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_report(rows: List[Dict[str, str]], per_corr: List[Dict[str, str]]) -> str:
    clean_rows = [r for r in rows if r["clip_type"] == "clean"]
    corrupt_rows = [r for r in rows if r["clip_type"] == "corrupted"]

    gt_all = [safe_float(r["gshi_gt"]) for r in rows]
    pred_all = [safe_float(r["gshi_pred"]) for r in rows]

    clean_pred = [safe_float(r["gshi_pred"]) for r in clean_rows]
    corrupt_pred = [safe_float(r["gshi_pred"]) for r in corrupt_rows]

    lines: List[str] = []
    lines.append("DRIVE-C Core v1 — gshi_pred sanity report")
    lines.append("")
    lines.append(f"Total rows: {len(rows)}")
    lines.append(f"Clean rows: {len(clean_rows)}")
    lines.append(f"Corrupted rows: {len(corrupt_rows)}")
    lines.append("")
    lines.append(f"Overall Pearson(gshi_gt, gshi_pred):  {pearson_corr(gt_all, pred_all):.6f}")
    lines.append(f"Overall Spearman(gshi_gt, gshi_pred): {spearman_corr(gt_all, pred_all):.6f}")
    lines.append("")
    lines.append(f"Mean gshi_pred on clean clips:      {mean(clean_pred):.6f}")
    lines.append(f"Mean gshi_pred on corrupted clips:  {mean(corrupt_pred):.6f}")
    lines.append("")
    lines.append(
        f"Fraction of (scenario, corruption) groups with monotonic non-increasing gshi_pred from s1->s5: "
        f"{severity_monotonic_fraction(rows):.6f}"
    )
    lines.append("")
    lines.append("Per-corruption summary:")
    for r in per_corr:
        lines.append(
            f"- {r['corruption_type']}: "
            f"pearson={r['pearson_gt_pred']}, "
            f"spearman={r['spearman_gt_pred']}, "
            f"s1={r['pred_mean_s1']}, s3={r['pred_mean_s3']}, s5={r['pred_mean_s5']}, "
            f"monotonic={r['mean_monotonic_s1_to_s5']}"
        )

    worst_clean = sorted(clean_rows, key=lambda r: safe_float(r["gshi_pred"]))[:5]
    lines.append("")
    lines.append("Lowest-predicted clean clips:")
    for r in worst_clean:
        lines.append(
            f"- {r['sample_id']}: gshi_pred={safe_float(r['gshi_pred']):.6f}, "
            f"top1={r.get('pred_top1_issue','')}, top1_prob={r.get('pred_top1_prob','')}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    ap.add_argument("--report-txt", type=Path, default=DEFAULT_REPORT_TXT)
    ap.add_argument("--per-corr-csv", type=Path, default=DEFAULT_PER_CORR_CSV)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")

    rows = read_rows(args.input_csv)
    per_corr = per_corruption_stats(rows)

    report = build_report(rows, per_corr)

    args.report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.write_text(report, encoding="utf-8")
    write_csv(args.per_corr_csv, per_corr)

    print(f"[DONE] Wrote report: {args.report_txt}")
    print(f"[DONE] Wrote per-corruption stats: {args.per_corr_csv}")
    print("")
    print(report)


if __name__ == "__main__":
    main()