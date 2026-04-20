#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


PROJECT_ROOT = Path("/home/sa/shiva/projects/drive-c")

INPUT_ROOT = PROJECT_ROOT / "data/processed/drivec_core_frames"
OUTPUT_ROOT = PROJECT_ROOT / "data/processed/drivec_core_depth"

MODEL_ID = "Intel/dpt-beit-large-512"
SAVE_VISUALIZATION = False
SKIP_EXISTING = True
USE_AMP = True


def list_scenario_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(folder: Path) -> List[Path]:
    return sorted(folder.glob("frame_*.png"))


def normalize_for_visualization(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min = float(depth.min())
    d_max = float(depth.max())

    if d_max - d_min < 1e-8:
        vis = np.zeros_like(depth, dtype=np.uint8)
    else:
        vis = (255.0 * (depth - d_min) / (d_max - d_min)).astype(np.uint8)

    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)


def load_model(device: torch.device):
    print(f"[INFO] Loading model: {MODEL_ID}")
    processor = DPTImageProcessor.from_pretrained(MODEL_ID)
    model = DPTForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


def predict_depth(image_bgr, processor, model, device, use_amp):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if device.type == "cuda" and use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
        else:
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.detach().cpu().numpy().astype(np.float32)
    return depth


def process_scenario_dir(scenario_dir: Path, out_dir: Path, processor, model, device, use_amp):
    images = list_images(scenario_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(images)
    done = 0

    for img_path in images:
        stem = img_path.stem
        npy_path = out_dir / f"{stem}.npy"
        vis_path = out_dir / f"{stem}_vis.png"

        if SKIP_EXISTING and npy_path.exists() and (not SAVE_VISUALIZATION or vis_path.exists()):
            done += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        depth = predict_depth(
            image_bgr=image,
            processor=processor,
            model=model,
            device=device,
            use_amp=use_amp,
        )

        np.save(npy_path, depth)

        if SAVE_VISUALIZATION:
            vis = normalize_for_visualization(depth)
            cv2.imwrite(str(vis_path), vis)

        done += 1
        if done % 20 == 0 or done == total:
            print(f"[{scenario_dir.name}] {done}/{total}")

    return total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    processor, model = load_model(device)

    scenario_dirs = list_scenario_dirs(INPUT_ROOT)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario dirs found in {INPUT_ROOT}")

    t0 = time.time()

    for scenario_dir in scenario_dirs:
        out_dir = OUTPUT_ROOT / scenario_dir.name
        print(f"\n[INFO] Processing {scenario_dir.name}")
        process_scenario_dir(
            scenario_dir=scenario_dir,
            out_dir=out_dir,
            processor=processor,
            model=model,
            device=device,
            use_amp=USE_AMP,
        )

    dt = time.time() - t0
    print(f"\n[DONE] Depth generation complete in {dt/60:.1f} minutes")
    print(f"[OUT] {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()