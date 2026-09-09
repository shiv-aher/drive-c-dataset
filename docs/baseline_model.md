# PerceptionHealthNet provenance freeze (WP1)

This document records the recovered baseline artifacts and execution settings.
WP2 will convert these facts into the manuscript description and assess any
remaining scientific limitations.

## Origin and training data

PerceptionHealthNet is a locally developed multi-head model. The recovered
training program uses KITTI RGB images and generated depth maps with on-the-fly
synthetic degradations. DRIVE-C clips are not referenced by the training code,
so the evidence currently supports describing DRIVE-C as an external evaluation
set for this checkpoint.

The recovered split files contain 5,985 training images and 1,496 validation
images. Their SHA-256 values are frozen in the RC1 provenance manifest.

## Architecture

- Backbone: torchvision EfficientNet-B2 initialized with
  `EfficientNet_B2_Weights.IMAGENET1K_V1` during training.
- Input: RGB, 384 x 1280.
- Dropout: 0.2.
- Presence head: 12 logits.
- Severity head: 12 sigmoid outputs.
- Health head: one sigmoid output.
- Pixel head: two convolution/batch-normalization/SiLU blocks followed by a
  one-channel sigmoid map; feature tap index 4 (`mid`).

The model vocabulary follows `camera_issues.yaml` and includes vignetting.
DRIVE-C's generated corruption set has no vignetting category and separately
contains overexposure and underexposure, both mapped to `exposure_shift`.

## Recovered training configuration

| Setting | Value |
|---|---:|
| Seed | 0 |
| Epochs | 22 (0–21) |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | `4e-5` |
| Weight decay | `1e-4` |
| Workers | 4 |
| Classification loss weight | 1.0 |
| Severity loss weight | 1.0 |
| Health loss weight | 0.5 |
| Pixel loss weight | 0.25 |
| Presence loss | BCE with logits |
| Severity/health loss | Smooth L1 |
| Deterministic cuDNN | Enabled |

The recovered script resumes from `epoch_019.pth` when present and forces the
configured learning rate after loading optimizer state. The final checkpoint is
a dictionary containing `epoch = 21`, 526 model tensors, and one optimizer
parameter group.

## Frozen inference settings

- Checkpoint SHA-256:
  `c210d9a4f207584687583d1ab5b96a99e12b2f3dbb2727464c6c039e44fb8c0b`.
- Checkpoint loading: `pretrained_backbone=False`, followed by strict loading of
  the checkpoint model state.
- Frames per clip: 8.
- Sampling: rounded, uniformly spaced indices from frame 0 through the final
  frame; duplicates removed while preserving order.
- Preprocessing: direct resize to 384 x 1280, RGB conversion, division by 255,
  and CHW layout; no normalization beyond `[0,1]` scaling.
- Batch size: 8 frames.
- Presence threshold: 0.25.
- Clip aggregation: arithmetic mean of frame probabilities, predicted
  severities, and predicted health.

## Reproduction evidence

Full inference was rerun over all 610 RC1 clips using the recovered checkpoint.
Comparison with `final_metadata.csv` produced:

- missing/extra sample IDs: 0;
- top-class mismatches: 0;
- JSON-output mismatches: 0;
- maximum scalar absolute error: 0;
- status: PASS.

The regenerated sanity report and per-corruption CSV are byte-identical to the
released copies under `dataset/`.

## Historical-provenance limitations

- Record the precise training host/GPU and original software environment.
- Resolve the historical training log, which contains appended/restarted epoch
  sequences and should not be presented as one clean uninterrupted run.
- The released checkpoint retains optimizer state to preserve the recovered
  artifact exactly; inference loads only its model state.
