#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # .../drive-c
DATASET_ROOT = PROJECT_ROOT / "DRIVE-C-Core"

# Add repo roots BEFORE imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATASET_ROOT))

from simulation.registry import load_taxonomy  # type: ignore
from src.models.perception_health_net import PerceptionHealthNet, ModelConfig  # type: ignore


DEFAULT_INPUT_CSV = DATASET_ROOT / "samples_metadata_with_gshi.csv"
DEFAULT_OUTPUT_CSV = DATASET_ROOT / "samples_metadata_with_gshi_pred.csv"
DEFAULT_TAXONOMY_YAML = PROJECT_ROOT / "configs" / "taxonomy" / "camera_issues.yaml"

# Match your training / inference defaults
DEFAULT_H = 384
DEFAULT_W = 1280
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_FRAMES = 8
DEFAULT_PRESENCE_THRESH = 0.25


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resize_direct(img_rgb: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)


def resize_with_pad(img_rgb: np.ndarray, H: int, W: int, pad_value=(0, 0, 0)) -> np.ndarray:
    h, w, _ = img_rgb.shape
    scale = min(W / w, H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    pad_w = W - nw
    pad_h = H - nh

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return img_padded


def center_crop_resize(img_rgb: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w, _ = img_rgb.shape
    target_ar = W / H
    ar = w / h

    if ar > target_ar:
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        crop = img_rgb[:, x0:x0 + new_w, :]
    else:
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        crop = img_rgb[y0:y0 + new_h, :, :]

    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def preprocess_frame(img_rgb: np.ndarray, H: int, W: int, mode: str) -> np.ndarray:
    if mode == "pad":
        out = resize_with_pad(img_rgb, H, W)
    elif mode == "crop":
        out = center_crop_resize(img_rgb, H, W)
    else:
        out = resize_direct(img_rgb, H, W)

    x = out.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x


def sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if num_frames >= total_frames:
        return list(range(total_frames))

    idxs = np.linspace(0, total_frames - 1, num=num_frames)
    idxs = np.round(idxs).astype(int).tolist()
    # de-dup while preserving order
    out = []
    seen = set()
    for i in idxs:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def read_sampled_frames(video_path: Path, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, num_frames)
    frames: List[np.ndarray] = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return frames


def load_model(
    ckpt_path: Path,
    device: torch.device,
    H: int,
    W: int,
    pretrained_backbone: bool,
) -> PerceptionHealthNet:
    model = PerceptionHealthNet(
        ModelConfig(num_classes=12, pixel_head=True),
        pretrained=pretrained_backbone,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # warm-up to instantiate lazy pixel head
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, H, W, device=device)
        _ = model(dummy)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_clip(
    model: PerceptionHealthNet,
    device: torch.device,
    issues: List[str],
    video_path: Path,
    H: int,
    W: int,
    preprocess: str,
    num_frames: int,
    batch_size: int,
    presence_thresh: float,
) -> Dict[str, str]:
    frames = read_sampled_frames(video_path, num_frames=num_frames)
    x_list = [preprocess_frame(fr, H, W, preprocess) for fr in frames]
    x_np = np.stack(x_list, axis=0)  # N,C,H,W
    x = torch.from_numpy(x_np).to(device)

    logits_chunks = []
    sev_chunks = []
    health_chunks = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        out = model(xb)

        logits_chunks.append(out["logits_pres"].detach().cpu())
        sev_chunks.append(out["pred_sev"].detach().cpu())
        health_chunks.append(out["pred_health"].detach().cpu())

    logits = torch.cat(logits_chunks, dim=0)      # N,12
    pred_sev = torch.cat(sev_chunks, dim=0)       # N,12
    pred_health = torch.cat(health_chunks, dim=0) # N,1

    # Aggregate across sampled frames
    probs = torch.sigmoid(logits)                 # N,12
    probs_mean = probs.mean(dim=0)                # 12
    sev_mean = pred_sev.mean(dim=0)               # 12
    health_mean = float(pred_health.mean().item())

    top1_idx = int(torch.argmax(probs_mean).item())
    top1_issue = issues[top1_idx]
    top1_prob = float(probs_mean[top1_idx].item())

    pred_presence = {issues[i]: round(float(probs_mean[i].item()), 6) for i in range(len(issues))}
    pred_severity = {issues[i]: round(float(sev_mean[i].item()), 6) for i in range(len(issues))}

    pred_present_thresh = [
        (issues[i], float(probs_mean[i].item()))
        for i in range(len(issues))
        if float(probs_mean[i].item()) >= presence_thresh
    ]
    pred_present_thresh.sort(key=lambda x: -x[1])

    pred_present_thresh_json = json.dumps(
        [{"issue": name, "prob": round(prob, 6)} for name, prob in pred_present_thresh],
        separators=(",", ":"),
    )

    return {
        "gshi_pred": f"{health_mean:.6f}",
        "pred_top1_issue": top1_issue,
        "pred_top1_prob": f"{top1_prob:.6f}",
        "pred_presence_json": json.dumps(pred_presence, separators=(",", ":")),
        "pred_severity_json": json.dumps(pred_severity, separators=(",", ":")),
        "pred_present_thresh_json": pred_present_thresh_json,
    }


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_prediction_columns(fieldnames: List[str]) -> List[str]:
    extras = [
        "gshi_pred",
        "pred_top1_issue",
        "pred_top1_prob",
        "pred_presence_json",
        "pred_severity_json",
        "pred_present_thresh_json",
    ]
    out = list(fieldnames)
    for col in extras:
        if col not in out:
            out.append(col)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to trained model checkpoint .pth")
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input metadata CSV")
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV")
    ap.add_argument("--taxonomy-yaml", type=Path, default=DEFAULT_TAXONOMY_YAML, help="Taxonomy YAML")
    ap.add_argument("--H", type=int, default=DEFAULT_H, help="Model input height")
    ap.add_argument("--W", type=int, default=DEFAULT_W, help="Model input width")
    ap.add_argument("--preprocess", type=str, default="resize", choices=["resize", "pad", "crop"])
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Frames sampled per clip")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")
    ap.add_argument("--presence-thresh", type=float, default=DEFAULT_PRESENCE_THRESH)
    ap.add_argument("--pretrained-backbone", action="store_true", help="Use pretrained backbone init before loading ckpt")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows for testing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.taxonomy_yaml.exists():
        raise FileNotFoundError(f"Missing taxonomy YAML: {args.taxonomy_yaml}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taxonomy = load_taxonomy(str(args.taxonomy_yaml))
    issues = taxonomy.issues

    model = load_model(
        ckpt_path=args.ckpt,
        device=device,
        H=args.H,
        W=args.W,
        pretrained_backbone=args.pretrained_backbone,
    )

    rows, fieldnames = read_csv_rows(args.input_csv)
    fieldnames = add_prediction_columns(fieldnames)

    if args.limit is not None:
        rows_to_process = rows[:args.limit]
        rows_rest = rows[args.limit:]
    else:
        rows_to_process = rows
        rows_rest = []

    updated_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(rows_to_process, start=1):
        rel_path = row["output_path"].strip()
        video_path = DATASET_ROOT / rel_path
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video referenced by CSV: {video_path}")

        pred = predict_clip(
            model=model,
            device=device,
            issues=issues,
            video_path=video_path,
            H=args.H,
            W=args.W,
            preprocess=args.preprocess,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
            presence_thresh=args.presence_thresh,
        )

        new_row = dict(row)
        new_row.update(pred)
        updated_rows.append(new_row)

        if idx % 25 == 0 or idx == len(rows_to_process):
            print(f"[INFO] Processed {idx}/{len(rows_to_process)} clips")

    updated_rows.extend(rows_rest)
    write_csv_rows(args.output_csv, updated_rows, fieldnames)

    print(f"[DONE] Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()