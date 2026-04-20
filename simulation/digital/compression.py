# simulation/digital/compression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from simulation.base import BaseDegradation, DegradationResult, to_float01, clamp01


@dataclass
class CompressionConfig:
    # Map severity -> JPEG quality Q (lower = worse).
    # Severity 0 -> Q_high, Severity 1 -> Q_low.
    q_high: int = 95
    q_low: int = 10

    # Optional: add a tiny random jitter to Q for realism
    q_jitter: int = 3  # +/- 3

    # JPEG chroma subsampling: OpenCV uses default (typically 4:2:0) internally.
    # Keep as-is for realism and simplicity.


class CompressionArtifactsDegradation(BaseDegradation):
    name = "compression_artifacts"

    def __init__(self, cfg: CompressionConfig = CompressionConfig()):
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        severity: float,
        depth: Optional[np.ndarray] = None,  # unused
        rng: Optional[np.random.Generator] = None,
        *,
        return_mask: bool = False,
    ) -> DegradationResult:
        if rng is None:
            rng = np.random.default_rng()

        s = float(np.clip(severity, 0.0, 1.0))

        # severity -> quality
        q = int(round(self.cfg.q_high - s * (self.cfg.q_high - self.cfg.q_low)))
        if self.cfg.q_jitter > 0:
            q += int(rng.integers(-self.cfg.q_jitter, self.cfg.q_jitter + 1))
        q = int(np.clip(q, 1, 100))

        # OpenCV JPEG expects uint8 BGR
        img01 = to_float01(image)
        img_u8 = (img01 * 255.0 + 0.5).astype(np.uint8)

        # handle both RGB and BGR inputs gracefully:
        # assume incoming is RGB (your pipeline), convert to BGR for encode
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        # encode -> decode round-trip
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            # fail-safe: return original
            out = img01.astype(np.float32)
            meta: Dict[str, Any] = {"severity": s, "quality": q, "encode_ok": False}
            return DegradationResult(image=out, mask=None, meta=meta)

        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        # back to RGB float01
        out_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_rgb = clamp01(out_rgb)

        meta: Dict[str, Any] = {
            "severity": s,
            "quality": q,
            "encode_ok": True,
        }

        # optional "artifact mask" (difference magnitude) for debugging/supervision
        mask = None
        if return_mask:
            diff = np.abs(out_rgb - img01).mean(axis=2)  # [H,W]
            mask = np.clip(diff / (diff.max() + 1e-8), 0.0, 1.0).astype(np.float32)

        return DegradationResult(image=out_rgb, mask=mask, meta=meta)
