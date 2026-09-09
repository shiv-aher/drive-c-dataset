from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 12
    dropout: float = 0.2
    pixel_head: bool = True
    pixel_feature_level: str = "mid"  # "mid" or "late"


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PixelHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.block1 = ConvBNAct(in_ch, 128)
        self.block2 = ConvBNAct(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.block1(feat)
        x = self.block2(x)
        x = self.out(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class PerceptionHealthNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig(), pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes

        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b2(weights=weights)

        self.features = backbone.features
        feat_dim = backbone.classifier[1].in_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=cfg.dropout)

        # Heads
        self.head_pres = nn.Linear(feat_dim, self.num_classes)  # logits
        self.head_sev = nn.Linear(feat_dim, self.num_classes)   # sigmoid -> [0,1]
        self.head_health = nn.Linear(feat_dim, 1)               # sigmoid -> [0,1]

        # Pixel head: lazy init because channel dims depend on tap point
        self._tap_index = 4 if cfg.pixel_feature_level == "mid" else -2
        self._pixel_head: Optional[PixelHead] = None

    def _run_features(self, x: torch.Tensor):
        tap = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == self._tap_index:
                tap = x
        if tap is None:
            tap = x
        return tap, x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        tap, feat = self._run_features(x)

        pooled = self.pool(feat).view(B, -1)
        pooled = self.dropout(pooled)

        logits_pres = self.head_pres(pooled)
        pred_sev = torch.sigmoid(self.head_sev(pooled))
        pred_health = torch.sigmoid(self.head_health(pooled))

        out: Dict[str, torch.Tensor] = {
            "logits_pres": logits_pres,
            "pred_sev": pred_sev,
            "pred_health": pred_health,
        }

        if self.cfg.pixel_head:
            if self._pixel_head is None:
                self._pixel_head = PixelHead(in_ch=tap.shape[1]).to(tap.device)
            out["pred_pix"] = self._pixel_head(tap, out_hw=(H, W))

        return out
