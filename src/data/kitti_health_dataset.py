from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import random


@dataclass(frozen=True)
class KittiPaths:
    images_dir: str
    depth_dir: str
    split_file: str  # text file with relative image paths (one per line)


def _read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # uint8 HWC


def _load_depth(path: str) -> np.ndarray:
    # Expecting saved normalized depth in [0,1].
    # Support .npy (recommended) or 16-bit PNG.
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
        return np.clip(d, 0.0, 1.0)

    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Could not read depth: {path}")
    if d.dtype == np.uint16:
        d = d.astype(np.float32) / 65535.0
    else:
        d = d.astype(np.float32)
        if d.max() > 1.5:
            d = d / 255.0
    return np.clip(d, 0.0, 1.0)


def default_collate_health(batch):

    out = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["y_pres"] = torch.stack([b["y_pres"] for b in batch], dim=0)
    out["y_sev"] = torch.stack([b["y_sev"] for b in batch], dim=0)
    out["y_health"] = torch.stack([b["y_health"] for b in batch], dim=0)

    # Build mask_union tensor even if some are missing
    has_mask = torch.tensor([b.get("mask_union") is not None for b in batch], dtype=torch.bool)
    out["has_mask"] = has_mask

    # Determine shape from images
    B, _, H, W = out["image"].shape
    masks = []
    for b in batch:
        m = b.get("mask_union")
        if m is None:
            masks.append(torch.zeros((1, H, W), dtype=torch.float32))
        else:
            masks.append(m)  # already (1,H,W)
    out["mask_union"] = torch.stack(masks, dim=0)  # (B,1,H,W)

    return out



class KittiHealthDataset(Dataset):
    def __init__(self, paths, simulator, policy, resize_hw=None,
                 fixed_plans=None, cache_plans=False, seed=0):
        self.paths = paths
        self.simulator = simulator
        self.policy = policy
        self.resize_hw = resize_hw
        self.rel_paths = _read_lines(paths.split_file)

        self.fixed_plans = fixed_plans
        self.cache_plans = cache_plans
        self.seed = seed

        if self.fixed_plans is None and self.cache_plans:
            # Pre-sample one plan per item deterministically
            import copy
            rng_state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            self.fixed_plans = [copy.deepcopy(self.policy.sample_plan()) for _ in range(len(self.rel_paths))]

            # restore RNG state (so training randomness is unaffected)
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.random.set_rng_state(rng_state[2])

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self.rel_paths[idx]

        # split file contains e.g. "000000.png"
        stem = os.path.splitext(rel)[0]  # "000000"

        img_path = os.path.join(self.paths.images_dir, f"{stem}.png")

        depth_candidates = [
            os.path.join(self.paths.depth_dir, f"{stem}_depth_norm.npy"),  # ✅ your actual files
            os.path.join(self.paths.depth_dir, f"{stem}.npy"),             # optional fallback
            os.path.join(self.paths.depth_dir, f"{stem}.png"),             # legacy fallback
        ]

        depth_path = None
        for p in depth_candidates:
            if os.path.exists(p):
                depth_path = p
                break

        if depth_path is None:
            raise FileNotFoundError(
                f"Missing depth for {stem}. Tried:\n" + "\n".join(depth_candidates)
            )


        img = _load_image(img_path)
        depth = _load_depth(depth_path)

        if self.resize_hw is not None:
            H, W = self.resize_hw
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        plan = self.fixed_plans[idx] if self.fixed_plans is not None else self.policy.sample_plan()
        sample = self.simulator.apply(image=img, depth=depth, plan=plan)


        # Convert to tensors
        x = torch.from_numpy(sample["image"]).permute(2, 0, 1).float()
        y_pres = torch.from_numpy(sample["y_pres"]).float()
        y_sev = torch.from_numpy(sample["y_sev"]).float()
        y_health = torch.tensor([sample["y_health"]], dtype=torch.float32)

        out = {
            "image": x,
            "y_pres": y_pres,
            "y_sev": y_sev,
            "y_health": y_health,
        }

        if sample.get("mask_union") is not None:
            mu = torch.from_numpy(sample["mask_union"]).unsqueeze(0).float()
            out["mask_union"] = mu
        else:
            out["mask_union"] = None

        return out

