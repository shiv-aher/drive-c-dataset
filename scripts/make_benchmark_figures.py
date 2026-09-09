#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
DATASET_ROOT = Path(os.environ.get("DRIVE_C_DATASET_ROOT", PROJECT_ROOT / "dataset"))

DEFAULT_FINAL_METADATA = DATASET_ROOT / "final_metadata.csv"
DEFAULT_PER_CORR_STATS = DATASET_ROOT / "gshi_pred_per_corruption_stats.csv"
DEFAULT_OUTDIR = DATASET_ROOT / "figures"


CORRUPTION_ORDER = [
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

DISPLAY_NAMES = {
    "fog": "fog",
    "rain": "rain",
    "snow": "snow",
    "glare_flare": "glare flare",
    "motion_blur": "motion blur",
    "defocus_blur": "defocus blur",
    "lens_occlusion": "lens occlusion",
    "sensor_noise": "sensor noise",
    "low_light": "low light",
    "overexposure": "overexposure",
    "underexposure": "underexposure",
    "jpeg_compression": "JPEG compression",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(vals: List[float]) -> List[float]:
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].tolist()
    y = y[mask].tolist()

    if len(x) < 2:
        return float("nan")

    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def build_fig1_gshi_gt_vs_severity(
    rows: List[Dict[str, str]],
    out_path_png: Path,
    out_path_pdf: Path,
) -> None:
    sev_levels = [1, 2, 3, 4, 5]
    grouped: Dict[str, Dict[int, List[float]]] = {
        corr: {s: [] for s in sev_levels} for corr in CORRUPTION_ORDER
    }

    for r in rows:
        if r["clip_type"] != "corrupted":
            continue
        corr = r["corruption_type"]
        sev = int(r["severity_level"])
        if corr in grouped and sev in grouped[corr]:
            grouped[corr][sev].append(safe_float(r["gshi_gt"]))

    colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.tick_params(axis='both', labelsize=14)

    for i, corr in enumerate(CORRUPTION_ORDER):
        means = []
        stds = []
        for s in sev_levels:
            vals = np.asarray(grouped[corr][s], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            stds.append(np.std(vals) if len(vals) > 0 else np.nan)

        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        color = colors[i % len(colors)]

        ax.plot(
            sev_levels,
            means,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            label=DISPLAY_NAMES[corr],
        )
        ax.fill_between(
            sev_levels,
            np.clip(means - stds, 0.0, 1.0),
            np.clip(means + stds, 0.0, 1.0),
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Severity level", fontsize=18)
    ax.set_ylabel("Mean severity-derived reference GSHI", fontsize=18)
    ax.set_title("Reference GSHI vs severity", fontsize=18)
    ax.set_xticks(sev_levels)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)

    ax.legend(
        fontsize=15,
        ncol=2,
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def build_fig2_gt_vs_pred_scatter(
    rows: List[Dict[str, str]],
    out_path_png: Path,
    out_path_pdf: Path,
) -> None:
    colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.tick_params(axis='both', labelsize=14)

    all_x = []
    all_y = []

    # corrupted points
    for i, corr in enumerate(CORRUPTION_ORDER):
        xs = []
        ys = []
        for r in rows:
            if r["clip_type"] != "corrupted":
                continue
            if r["corruption_type"] != corr:
                continue
            x = safe_float(r["gshi_gt"])
            y = safe_float(r["gshi_pred"])
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)

        if xs:
            ax.scatter(
                xs,
                ys,
                s=28,
                alpha=0.72,
                color=colors[i % len(colors)],
                label=DISPLAY_NAMES[corr],
            )
            all_x.extend(xs)
            all_y.extend(ys)

    # clean points
    clean_x = []
    clean_y = []
    for r in rows:
        if r["clip_type"] == "clean":
            x = safe_float(r["gshi_gt"])
            y = safe_float(r["gshi_pred"])
            if np.isfinite(x) and np.isfinite(y):
                clean_x.append(x)
                clean_y.append(y)

    if clean_x:
        ax.scatter(
            clean_x,
            clean_y,
            marker="x",
            s=140,
            linewidths=2.2,
            color="black",
            label="clean",
        )
        all_x.extend(clean_x)
        all_y.extend(clean_y)

    # identity line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")

    # regression line
    xx = np.asarray(all_x, dtype=np.float64)
    yy = np.asarray(all_y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]

    if len(xx) >= 2:
        m, b = np.polyfit(xx, yy, 1)
        xfit = np.linspace(0, 1, 200)
        yfit = m * xfit + b
        ax.plot(xfit, yfit, linewidth=2.0, color="black", alpha=0.8)

    pr = pearson_corr(all_x, all_y)
    sr = spearman_corr(all_x, all_y)
    ax.text(
        0.65, 0.10,   # bottom-right area
        f"Pearson = {pr:.3f}\nSpearman = {sr:.3f}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel("gshi_gt", fontsize=18)
    ax.set_ylabel("gshi_pred", fontsize=18)
    ax.set_title("Reference vs predicted GSHI", fontsize=18)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)

    ax.legend(
    fontsize=16,
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    frameon=True,
)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def build_table1_rows(stats_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out_rows = []

    for r in stats_rows:
        s1 = safe_float(r.get("pred_mean_s1", "nan"))
        s3 = safe_float(r.get("pred_mean_s3", "nan"))
        s5 = safe_float(r.get("pred_mean_s5", "nan"))
        delta = s1 - s5 if np.isfinite(s1) and np.isfinite(s5) else float("nan")

        out_rows.append(
            {
                "corruption_type": r["corruption_type"],
                "display_name": DISPLAY_NAMES.get(r["corruption_type"], r["corruption_type"]),
                "pearson": safe_float(r.get("pearson_gt_pred", "nan")),
                "spearman": safe_float(r.get("spearman_gt_pred", "nan")),
                "monotonic_yes_no": "yes" if r.get("mean_monotonic_s1_to_s5", "0") == "1" else "no",
                "mean_pred_s1": s1,
                "mean_pred_s3": s3,
                "mean_pred_s5": s5,
                "delta_s1_to_s5": delta,
            }
        )

    # sort by spearman descending
    out_rows.sort(key=lambda x: (-(x["spearman"]) if np.isfinite(x["spearman"]) else float("inf")))
    return out_rows


def build_table1_csv(stats_rows: List[Dict[str, str]], out_csv: Path) -> None:
    rows = build_table1_rows(stats_rows)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corruption_type",
                "display_name",
                "pearson",
                "spearman",
                "monotonic_yes_no",
                "mean_pred_s1",
                "mean_pred_s3",
                "mean_pred_s5",
                "delta_s1_to_s5",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corruption_type": row["corruption_type"],
                    "display_name": row["display_name"],
                    "pearson": f"{row['pearson']:.6f}" if np.isfinite(row["pearson"]) else "",
                    "spearman": f"{row['spearman']:.6f}" if np.isfinite(row["spearman"]) else "",
                    "monotonic_yes_no": row["monotonic_yes_no"],
                    "mean_pred_s1": f"{row['mean_pred_s1']:.6f}" if np.isfinite(row["mean_pred_s1"]) else "",
                    "mean_pred_s3": f"{row['mean_pred_s3']:.6f}" if np.isfinite(row["mean_pred_s3"]) else "",
                    "mean_pred_s5": f"{row['mean_pred_s5']:.6f}" if np.isfinite(row["mean_pred_s5"]) else "",
                    "delta_s1_to_s5": f"{row['delta_s1_to_s5']:.6f}" if np.isfinite(row["delta_s1_to_s5"]) else "",
                }
            )


def build_table1_markdown(stats_rows: List[Dict[str, str]], out_md: Path) -> None:
    rows = build_table1_rows(stats_rows)

    lines = []
    lines.append("| corruption_type | pearson | spearman | monotonic | mean_pred_s1 | mean_pred_s3 | mean_pred_s5 | delta_s1_to_s5 |")
    lines.append("|---|---:|---:|:---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['display_name']} | "
            f"{row['pearson']:.6f} | "
            f"{row['spearman']:.6f} | "
            f"{row['monotonic_yes_no']} | "
            f"{row['mean_pred_s1']:.6f} | "
            f"{row['mean_pred_s3']:.6f} | "
            f"{row['mean_pred_s5']:.6f} | "
            f"{row['delta_s1_to_s5']:.6f} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-metadata", type=Path, default=DEFAULT_FINAL_METADATA)
    ap.add_argument("--per-corr-stats", type=Path, default=DEFAULT_PER_CORR_STATS)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if not args.final_metadata.exists():
        raise FileNotFoundError(f"Missing final metadata: {args.final_metadata}")
    if not args.per_corr_stats.exists():
        raise FileNotFoundError(f"Missing per-corruption stats: {args.per_corr_stats}")

    ensure_dir(args.outdir)

    rows = read_csv_rows(args.final_metadata)
    stats_rows = read_csv_rows(args.per_corr_stats)

    fig1_png = args.outdir / "figure1_gshi_gt_vs_severity.png"
    fig1_pdf = args.outdir / "figure1_gshi_gt_vs_severity.pdf"

    fig2_png = args.outdir / "figure2_gshi_gt_vs_gshi_pred_scatter.png"
    fig2_pdf = args.outdir / "figure2_gshi_gt_vs_gshi_pred_scatter.pdf"

    table1_csv = args.outdir / "table1_per_corruption_summary.csv"
    table1_md = args.outdir / "table1_per_corruption_summary.md"

    build_fig1_gshi_gt_vs_severity(rows, fig1_png, fig1_pdf)
    build_fig2_gt_vs_pred_scatter(rows, fig2_png, fig2_pdf)
    build_table1_csv(stats_rows, table1_csv)
    build_table1_markdown(stats_rows, table1_md)

    print(f"[DONE] Figure 1 PNG: {fig1_png}")
    print(f"[DONE] Figure 1 PDF: {fig1_pdf}")
    print(f"[DONE] Figure 2 PNG: {fig2_png}")
    print(f"[DONE] Figure 2 PDF: {fig2_pdf}")
    print(f"[DONE] Table 1 CSV: {table1_csv}")
    print(f"[DONE] Table 1 MD:  {table1_md}")


if __name__ == "__main__":
    main()
