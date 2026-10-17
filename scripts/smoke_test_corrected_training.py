#!/usr/bin/env python3
"""Run one real KITTI optimizer step through the corrected four-head model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_perception_health_net import pixel_loss, seed_everything, severity_loss
from simulation.config import build_simulator_and_policy
from src.data.kitti_health_dataset import (
    KittiHealthDataset,
    KittiPaths,
    default_collate_health,
)
from src.models.perception_health_net import ModelConfig, PerceptionHealthNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti-root",
        type=Path,
        default=Path("/home/sa/shiva/projects/oshm-uncertainty/data/kitti"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator, policy = build_simulator_and_policy(
        taxonomy_yaml=str(REPO / "configs/taxonomy/camera_issues.yaml"),
        policy_yaml=str(REPO / "configs/training/mixture_policy.yaml"),
        seed=0,
    )
    paths = KittiPaths(
        images_dir=str(args.kitti_root / "images/data_object_image_2/training/image_2"),
        depth_dir=str(args.kitti_root / "depth/midas/norm"),
        split_file=str(args.kitti_root / "splits/train.txt"),
    )
    dataset = KittiHealthDataset(
        paths=paths,
        simulator=simulator,
        policy=policy,
        resize_hw=(384, 1280),
        cache_plans=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate_health,
    )
    batch = next(iter(loader))

    model = PerceptionHealthNet(
        ModelConfig(num_classes=12, pixel_head=True), pretrained=True
    ).to(device)
    model.initialize_pixel_head((384, 1280), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_pixel_parameters = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("_pixel_head.") and id(parameter) not in optimizer_parameter_ids
    ]
    if missing_pixel_parameters:
        raise RuntimeError(f"Pixel parameters absent from optimizer: {missing_pixel_parameters}")

    x = batch["image"].to(device)
    y_pres = batch["y_pres"].to(device)
    y_sev = batch["y_sev"].to(device)
    y_health = batch["y_health"].to(device)
    masks = batch["mask_union"].to(device)
    has_mask = batch["has_mask"].to(device)
    issues = simulator.taxonomy.issues
    mask_eligible = {"haze_fog", "rain", "snow", "lens_occlusion", "glare_flare"}
    pixel_indices = [index for index, name in enumerate(issues) if name in mask_eligible]

    model.train()
    output = model(x)
    loss_presence = nn.functional.binary_cross_entropy_with_logits(
        output["logits_pres"], y_pres
    )
    presence = (y_pres > 0.5).float()
    negative_mask = 1.0 - presence
    negative_regularizer = (
        output["pred_sev"].abs() * negative_mask
    ).sum() / (negative_mask.sum() + 1e-6)
    loss_severity = severity_loss(output["pred_sev"], y_sev, y_pres) + 0.02 * negative_regularizer
    loss_health = nn.functional.smooth_l1_loss(output["pred_health"], y_health)
    loss_pixel = pixel_loss(
        output["pred_pix"], masks, has_mask, y_pres, y_sev, pixel_indices
    )
    total = loss_presence + loss_severity + 0.5 * loss_health + 0.25 * loss_pixel
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()

    pixel_gradient = model._pixel_head.out.weight.grad
    report = {
        "status": "PASS",
        "device": str(device),
        "batch_size": int(x.shape[0]),
        "input_shape": list(x.shape),
        "has_mask_count": int(has_mask.sum().item()),
        "health_target_min": float(y_health.min().item()),
        "health_target_max": float(y_health.max().item()),
        "loss_presence": float(loss_presence.item()),
        "loss_severity": float(loss_severity.item()),
        "loss_health": float(loss_health.item()),
        "loss_pixel": float(loss_pixel.item()),
        "loss_total": float(total.item()),
        "pixel_head_gradient_nonzero": bool(
            pixel_gradient is not None and torch.count_nonzero(pixel_gradient).item() > 0
        ),
        "pixel_parameters_missing_from_optimizer": missing_pixel_parameters,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
