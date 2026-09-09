from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import numpy as np

from simulation.config import build_simulator_and_policy
from src.data.kitti_health_dataset import (
    KittiPaths,
    KittiHealthDataset,
    default_collate_health,
)
from src.models.perception_health_net import PerceptionHealthNet, ModelConfig


# -------------------------
# Metrics / Eval
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    all_pres_probs = []
    all_pres_gt = []
    all_sev_pred = []
    all_sev_gt = []
    all_health_pred = []
    all_health_gt = []
    all_has_mask = []
    all_pix_pred = []
    all_pix_gt = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_pres = batch["y_pres"].to(device, non_blocking=True)
        y_sev = batch["y_sev"].to(device, non_blocking=True)
        y_health = batch["y_health"].to(device, non_blocking=True)

        mask_union = batch.get("mask_union", None)
        has_mask = batch.get("has_mask", None)

        if mask_union is not None:
            mask_union = mask_union.to(device, non_blocking=True)

        if has_mask is None:
            has_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
        else:
            has_mask = has_mask.to(device, non_blocking=True)

        out = model(x)

        # Presence probabilities
        pres_prob = torch.sigmoid(out["logits_pres"])
        # Safety: avoid silent NaN propagation
        if not torch.isfinite(pres_prob).all():
            pres_prob = torch.nan_to_num(pres_prob, nan=0.0, posinf=1.0, neginf=0.0)

        all_pres_probs.append(pres_prob.detach().cpu())
        all_pres_gt.append(y_pres.detach().cpu())
        all_sev_pred.append(out["pred_sev"].detach().cpu())
        all_sev_gt.append(y_sev.detach().cpu())
        all_health_pred.append(out["pred_health"].detach().cpu())
        all_health_gt.append(y_health.detach().cpu())

        # Pixel outputs (store only if GT mask exists for this batch)
        if ("pred_pix" in out) and (out["pred_pix"] is not None) and (mask_union is not None):
            all_has_mask.append(has_mask.detach().cpu())
            all_pix_pred.append(out["pred_pix"].detach().cpu())
            all_pix_gt.append(mask_union.detach().cpu())

    pres_probs = torch.cat(all_pres_probs, dim=0)   # (N,12)
    pres_gt = torch.cat(all_pres_gt, dim=0)         # (N,12)
    sev_pred = torch.cat(all_sev_pred, dim=0)       # (N,12)
    sev_gt = torch.cat(all_sev_gt, dim=0)           # (N,12)
    h_pred = torch.cat(all_health_pred, dim=0)      # (N,1)
    h_gt = torch.cat(all_health_gt, dim=0)          # (N,1)

    # -------------------------
    # Presence metric: robust macro AUPRC (+ micro AUPRC)
    # -------------------------
    macro_auprc = float("nan")
    micro_auprc = float("nan")
    valid_classes = 0
    skipped_classes = 0

    try:
        from sklearn.metrics import average_precision_score

        ap = []
        pos_counts = pres_gt.sum(dim=0)  # (12,)
        N = pres_gt.shape[0]

        print("[VAL] positives per class:", pos_counts.tolist(), "N=", N)

        for c in range(pres_gt.shape[1]):
            pos = int(pos_counts[c].item())
            neg = N - pos
            # PR-AUC undefined if class has no positives or no negatives
            if pos == 0 or neg == 0:
                skipped_classes += 1
                continue
            ap_c = average_precision_score(pres_gt[:, c].numpy(), pres_probs[:, c].numpy())
            ap.append(ap_c)
            valid_classes += 1

        if valid_classes > 0:
            macro_auprc = float(np.mean(ap))
        else:
            macro_auprc = float("nan")

        # Micro AUPRC is usually stable even when macro has missing classes
        micro_auprc = float(
            average_precision_score(
                pres_gt.numpy().reshape(-1),
                pres_probs.numpy().reshape(-1),
            )
        )

    except Exception as e:
        print("[VAL] AUPRC computation failed:", repr(e))
        macro_auprc = float("nan")
        micro_auprc = float("nan")
        valid_classes = 0
        skipped_classes = 0

    # Severity MAE only where present
    sev_mae = (torch.abs(sev_pred - sev_gt) * pres_gt).sum() / (pres_gt.sum() + 1e-6)
    sev_mae = float(sev_mae.item())

    # Health MAE
    health_mae = float(torch.mean(torch.abs(h_pred - h_gt)).item())

    # Pixel BCE on masked samples (if any)
    # IMPORTANT: we evaluate BCE in the same "space" as training.
    # Your training pixel_loss uses BCEWithLogits, so we do the same here.
    if len(all_has_mask) > 0:
        has_mask_all = torch.cat(all_has_mask, dim=0)  # (N,)
        pix_pred = torch.cat(all_pix_pred, dim=0)      # (N,1,H,W) logits
        pix_gt = torch.cat(all_pix_gt, dim=0)          # (N,1,H,W) {0,1}

        if has_mask_all.sum() > 0:
            bce = nn.functional.binary_cross_entropy_with_logits(pix_pred, pix_gt, reduction="none")
            bce = bce.mean(dim=(1, 2, 3))  # per-sample
            pix_bce = float((bce * has_mask_all.float()).sum().item() / has_mask_all.sum().item())
            mask_rate = float(has_mask_all.float().mean().item())
        else:
            pix_bce = float("nan")
            mask_rate = 0.0
    else:
        pix_bce = float("nan")
        mask_rate = 0.0

    return {
        "macro_auprc": macro_auprc,
        "micro_auprc": micro_auprc,
        "auprc_valid_classes": float(valid_classes),
        "auprc_skipped_classes": float(skipped_classes),
        "sev_mae": sev_mae,
        "health_mae": health_mae,
        "pix_bce": pix_bce,
        "mask_rate": mask_rate,
    }


# -------------------------
# Loss helpers
# -------------------------
def severity_loss(pred: torch.Tensor, target: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B,12)
    presence: (B,12)
    Only penalize severity where degradation is present.
    """
    l1 = torch.abs(pred - target)
    return (l1 * presence).sum() / (presence.sum() + 1e-6)


def pixel_loss(
    pred_pix: torch.Tensor,
    gt_pix: Optional[torch.Tensor],
    has_mask: torch.Tensor,
    y_pres: torch.Tensor,
    y_sev: torch.Tensor,
    pix_relevant_idx: List[int],
) -> torch.Tensor:
    """
    pred_pix, gt_pix: (B,1,H,W)
    has_mask: (B,)
    y_pres, y_sev: (B,12)
    pix_relevant_idx: indices for degradations that can produce masks
    """
    # No GT masks available in this batch
    if gt_pix is None:
        return pred_pix.sum() * 0.0

    # If no samples in batch have a mask, skip pixel loss
    if has_mask.sum() == 0:
        return pred_pix.sum() * 0.0

    # Presence/severity gating for mask-relevant degradations
    pres_w = y_pres[:, pix_relevant_idx].max(dim=1).values            # (B,)
    sev_w  = y_sev[:,  pix_relevant_idx].max(dim=1).values            # (B,)
    sev_w = torch.clamp(sev_w, 0.0, 1.0)  # no floor; allow true zero

    # BCE per-sample (logits)
    bce = nn.functional.binary_cross_entropy_with_logits(pred_pix, gt_pix, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))                                     # (B,)

    # Final weights: must have mask, and class must be present (GT), scaled by severity
    # tiny epsilon avoids zero-weight division instability when sev_w=0 for present masks
    w = has_mask.float() * pres_w * (sev_w + 1e-3)                    # (B,)

    # If weights are all zero (e.g., masks exist but GT pres says none), skip safely
    if w.sum() <= 1e-6:
        return pred_pix.sum() * 0.0

    return (bce * w).sum() / (w.sum() + 1e-6)


# -------------------------
# Train
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 4e-5
    epochs: int = 22
    img_hw: tuple = (384, 1280)
    num_workers: int = 4
    weight_decay: float = 1e-4

    # resume
    resume_ckpt = "output/checkpoints/epoch_019.pth"


    # logging / output
    out_dir: str = "output"
    ckpt_dir: str = "output/checkpoints"
    log_csv: str = "output/train_log.csv"

    # loss weights
    w_cls: float = 1.0
    w_sev: float = 1.0
    w_health: float = 0.5
    w_pix: float = 0.25

    # dataset root
    root: str = "data/kitti"

    train_images_dir: str = "images/data_object_image_2/training/image_2"
    train_depth_dir: str = "depth/midas/norm"

    val_images_dir = "images/data_object_image_2/testing/image_2"
    val_depth_dir  = "depth/midas/norm/validation/norm"

    train_split: str = "splits/train.txt"
    val_split: str = "splits/val.txt"

    # mask-eligible degradation names
    mask_eligible_names: Set[str] = None

    def __post_init__(self):
        if self.mask_eligible_names is None:
            self.mask_eligible_names = {"haze_fog", "rain", "snow", "lens_occlusion", "glare_flare"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may slightly slow)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    cfg = TrainConfig()
    seed_everything(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------
    # Simulator + Policy
    # -------------------------
    simulator, policy = build_simulator_and_policy(
        taxonomy_yaml="configs/taxonomy/camera_issues.yaml",
        policy_yaml="configs/training/mixture_policy.yaml",
        seed=0,
    )

    # Build pixel-relevant indices from taxonomy order
    issues = simulator.taxonomy.issues  # list[str]
    pix_relevant_idx = [i for i, name in enumerate(issues) if name in cfg.mask_eligible_names]
    if len(pix_relevant_idx) == 0:
        raise RuntimeError(
            "pix_relevant_idx is empty. Check taxonomy issue names vs mask_eligible_names "
            f"(issues={issues}, mask_eligible={sorted(list(cfg.mask_eligible_names))})."
        )
    print("Mask-eligible issues:", [issues[i] for i in pix_relevant_idx])

    # -------------------------
    # Datasets
    # -------------------------
    train_paths = KittiPaths(
        images_dir=os.path.join(cfg.root, cfg.train_images_dir),
        depth_dir=os.path.join(cfg.root, cfg.train_depth_dir),
        split_file=os.path.join(cfg.root, cfg.train_split),
    )
    val_paths = KittiPaths(
        images_dir=os.path.join(cfg.root, cfg.val_images_dir),
        depth_dir=os.path.join(cfg.root, cfg.val_depth_dir),
        split_file=os.path.join(cfg.root, cfg.val_split),
    )

    train_ds = KittiHealthDataset(
        paths=train_paths, simulator=simulator, policy=policy, resize_hw=cfg.img_hw, cache_plans=False
    )
    val_ds = KittiHealthDataset(
        paths=val_paths, simulator=simulator, policy=policy, resize_hw=cfg.img_hw, cache_plans=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=default_collate_health,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=default_collate_health,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    # -------------------------
    # Model
    # -------------------------
    model = PerceptionHealthNet(
        ModelConfig(num_classes=12, pixel_head=True),
        pretrained=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce_logits = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 0
    if cfg.resume_ckpt is not None and os.path.exists(cfg.resume_ckpt):
        ckpt = torch.load(cfg.resume_ckpt, map_location="cpu")

        # Warm-up to instantiate lazy heads (if any)
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.img_hw[0], cfg.img_hw[1], device=device)
            _ = model(dummy)

        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])

        # FORCE LR from config after resume (checkpoint restores old LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
        print(f"[RESUME] Forced LR to {optimizer.param_groups[0]['lr']:.2e}")

        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {cfg.resume_ckpt} → starting at epoch {start_epoch}")

    # -------------------------
    # Logging setup
    # -------------------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if not os.path.exists(cfg.log_csv):
        with open(cfg.log_csv, "w") as f:
            f.write(
                "epoch,train_loss,"
                "val_macro_auprc,val_micro_auprc,val_auprc_valid_classes,val_auprc_skipped_classes,"
                "val_sev_mae,val_health_mae,val_pix_bce,val_mask_rate\n"
            )

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y_pres = batch["y_pres"].to(device, non_blocking=True)
            y_sev = batch["y_sev"].to(device, non_blocking=True)
            y_health = batch["y_health"].to(device, non_blocking=True)

            mask_union = batch.get("mask_union", None)
            if mask_union is not None:
                mask_union = mask_union.to(device, non_blocking=True)

            has_mask = batch.get("has_mask", None)
            if has_mask is None:
                has_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
            else:
                has_mask = has_mask.to(device, non_blocking=True)

            out = model(x)

            # push severity toward 0 when not present (light regularizer)
            presence = (y_pres > 0.5).float()
            neg_mask = 1.0 - presence
            neg_reg = (out["pred_sev"].abs() * neg_mask).sum() / (neg_mask.sum() + 1e-6)

            loss_pres = bce_logits(out["logits_pres"], y_pres)
            loss_sev = severity_loss(out["pred_sev"], y_sev, y_pres) + 0.02 * neg_reg
            loss_health = huber(out["pred_health"], y_health)
            loss_pix = pixel_loss(out["pred_pix"], mask_union, has_mask, y_pres, y_sev, pix_relevant_idx)

            # Ramp pixel loss in slowly (first 3 epochs)
            pix_w = cfg.w_pix * min(1.0, epoch / 3.0)

            loss = (
                cfg.w_cls * loss_pres
                + cfg.w_sev * loss_sev
                + cfg.w_health * loss_health
                + pix_w * loss_pix
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

            if step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"... lr={lr:.2e}")
                mask_rate = float(has_mask.float().mean().item())
                print(
                    f"[Epoch {epoch:02d} | Step {step:04d}] "
                    f"loss={loss.item():.4f} "
                    f"(cls={loss_pres.item():.3f}, "
                    f"sev={loss_sev.item():.3f}, "
                    f"health={loss_health.item():.3f}, "
                    f"pix={loss_pix.item():.3f}, "
                    f"mask_rate={mask_rate:.2f})"
                )

        train_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} avg loss: {train_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")

        # Validation
        metrics = evaluate(model, val_loader, device)
        print(
            f"[VAL] macroAUPRC={metrics['macro_auprc']:.3f} "
            f"(micro={metrics['micro_auprc']:.3f}, "
            f"valid={int(metrics['auprc_valid_classes'])}/12, "
            f"skipped={int(metrics['auprc_skipped_classes'])}) "
            f"sev_MAE={metrics['sev_mae']:.3f} "
            f"health_MAE={metrics['health_mae']:.3f} "
            f"pix_BCE={metrics['pix_bce']:.3f} "
            f"mask_rate={metrics['mask_rate']:.2f}"
        )

        # CSV log
        with open(cfg.log_csv, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},"
                f"{metrics['macro_auprc']},{metrics['micro_auprc']},"
                f"{metrics['auprc_valid_classes']},{metrics['auprc_skipped_classes']},"
                f"{metrics['sev_mae']},{metrics['health_mae']},{metrics['pix_bce']},{metrics['mask_rate']}\n"
            )


if __name__ == "__main__":
    main()
