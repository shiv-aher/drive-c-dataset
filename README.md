# DRIVE-C

Current corrected Version 3 code release: `v1.1.1` (October 24, 2026).
Version `v1.1.1` synchronizes the public documentation and embedded metadata
with DRIVE-C Core dataset Version 1.1. Tag `v1.1.0` contains the same corrected
model and training path but predates this metadata synchronization. The earlier
`v1.0.1` tag remains historical provenance for the Version 2 baseline.

DRIVE-C is a compact controlled testbed for evaluating degradation awareness,
configured health-index estimation, and camera-prediction consistency. It contains 10
original forward-facing scenarios and 600 controlled corrupted variants: 12
corruption types, five severity levels, and 10 scenarios, plus 10 clean clips.

- Dataset version DOI: <https://doi.org/10.5281/zenodo.23622197>
- Dataset concept DOI: <https://doi.org/10.5281/zenodo.19656443>

## Reproducibility boundary

This corrected release supports validation and regeneration of corruptions and
severity-derived GSHI labels from released anonymized clean material and frozen
generation inputs. The original unprocessed recordings are not distributed
because of privacy considerations. Acquisition, source-video clip selection,
and initial anonymization are documented provenance steps, not a publicly
rerunnable pipeline.

`gshi_gt` is a deterministic severity-derived reference index. It is not an
independent physical sensor measurement or a measurement of downstream
perception reliability. `gshi_pred` is a PerceptionHealthNet model output.

## Release contents

```text
drive-c-dataset/
  checkpoints/        selected corrected baseline checkpoint
  configs/            taxonomy and generation/training policy
  dataset/            metadata, reports, and dataset download guide
  provenance/         frozen training provenance
  scripts/            generation, labeling, inference, and analysis
  simulation/         corruption and GSHI implementation
  src/                baseline model and training data loader
  requirements.txt
```

The downloaded dataset archive has this structure:

```text
drivec_core_v1_1/
  clean_clips/        10 clips, S01_clean.mp4 ... S10_clean.mp4
  corrupted/          600 clips grouped by corruption and severity
  final_metadata.csv  610 rows, 27 fields
  scenario_metadata.csv
```

## Environment

The corrected training run used Python 3.10.20, PyTorch 2.10.0, torchvision
0.25.0, and CUDA 12.8. Its complete package snapshot is
`provenance/corrected_training_pip_freeze.txt`; the exact training protocol is
recorded with the run outputs. The existing `provenance/environment.yaml` is
retained as historical provenance for the recovered Version 2 training run and
must not be interpreted as the corrected-run environment. The versions in
`requirements.txt` define the lean reproduction environment. A CPU
installation may be used for inference but will be slower.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export DRIVE_C_DATASET_ROOT=/absolute/path/to/drivec_core_v1_1
```

All commands below are run from the repository root.

## Prepare released clean inputs for corruption regeneration

```bash
python scripts/extract_released_clean_frames.py \
  --dataset-root "$DRIVE_C_DATASET_ROOT" \
  --output-root work/clean_frames

python scripts/generate_drivec_core_depth.py \
  --input-root work/clean_frames \
  --output-root work/depth
```

The depth model identifier is frozen by the depth-generation command. The
original pre-anonymization recordings are not required for this public stage.

## Recompute reference GSHI

```bash
python scripts/add_gshi_gt.py \
  --input-csv "$DRIVE_C_DATASET_ROOT/samples_metadata.csv" \
  --output-csv "$DRIVE_C_DATASET_ROOT/samples_metadata_with_gshi.csv" \
  --taxonomy-yaml configs/taxonomy/camera_issues.yaml
```

The implementation is in `simulation/gshi_utils.py`; all weights and numerical
settings are in `configs/taxonomy/camera_issues.yaml`.

## Reproduce baseline predictions

PerceptionHealthNet uses an ImageNet-initialized EfficientNet-B2 backbone with
presence, severity, health, and pixel heads. The checkpoint was trained on
KITTI images with on-the-fly synthetic degradations; DRIVE-C was not its
training dataset. The Version 3 training path corrects three audited defects:
the pixel head now emits logits for BCE-with-logits, is materialized before
optimizer construction, and the health target applies the configured GSHI
beta/floor/ceiling. Gradient and optimizer-coverage tests, training code, and
frozen provenance are included.

The selected checkpoint is epoch 7, chosen only by minimum held-out KITTI
health MAE (0.106707), not by DRIVE-C results. Its SHA-256 is
`fcfb0ea54d6329bae2756f5fd32c39be860a90b04967dfde97f7b213c316836c`.

Inference samples eight uniformly spaced frames from each 128-frame clip,
resizes them directly to 384 x 1280, and averages frame-level outputs.

```bash
python scripts/add_gshi_pred.py \
  --ckpt checkpoints/best_health_mae.pth \
  --input-csv "$DRIVE_C_DATASET_ROOT/samples_metadata_with_gshi.csv" \
  --output-csv "$DRIVE_C_DATASET_ROOT/samples_metadata_with_gshi_pred.csv" \
  --taxonomy-yaml configs/taxonomy/camera_issues.yaml \
  --H 384 --W 1280 --preprocess resize --num-frames 8 --batch-size 8 \
  --presence-thresh 0.25
```

The model vocabulary includes `vignetting`, whereas DRIVE-C does not generate
vignetting clips. A vignetting prediction is therefore a model-vocabulary
output, not a DRIVE-C corruption label.

## Reproduce metadata and analysis

```bash
python scripts/make_final_metadata.py \
  --gt-csv "$DRIVE_C_DATASET_ROOT/samples_metadata_with_gshi.csv" \
  --pred-csv "$DRIVE_C_DATASET_ROOT/samples_metadata_with_gshi_pred.csv" \
  --out-csv "$DRIVE_C_DATASET_ROOT/final_metadata.csv"

python scripts/analyze_gshi_pred.py \
  --input-csv "$DRIVE_C_DATASET_ROOT/final_metadata.csv" \
  --report-txt "$DRIVE_C_DATASET_ROOT/gshi_pred_sanity_report.txt" \
  --per-corr-csv "$DRIVE_C_DATASET_ROOT/gshi_pred_per_corruption_stats.csv"

python scripts/make_benchmark_figures.py \
  --final-metadata "$DRIVE_C_DATASET_ROOT/final_metadata.csv" \
  --per-corr-stats "$DRIVE_C_DATASET_ROOT/gshi_pred_per_corruption_stats.csv" \
  --outdir "$DRIVE_C_DATASET_ROOT/figures"
```

For the corrected checkpoint, the reported 0.508 monotonicity fraction is 61
of 120 complete scenario-corruption sequences. The 9-of-12 count is computed
after averaging predictions across scenarios for each corruption and severity.
These quantities therefore use different aggregation rules. Across all 610
clips, the corrected raw predictions have Pearson `r=0.382`, Spearman
`rho=0.376`, MAE `0.299`, and RMSE `0.365`; the substantial clean-clip MAE
(`0.570`) should be retained when reporting the baseline.

## Validate the corrected training path

```bash
pytest -q tests/test_pixel_head_training.py tests/test_gshi_training_target.py
python scripts/smoke_test_corrected_training.py
```

The pixel head is auxiliary and does not feed the scalar health prediction at
inference. Its target is the union of eligible simulator masks for fog, rain,
snow, glare/flare, and lens occlusion; samples without an eligible mask do not
contribute pixel loss.

## Licenses

- Code: MIT License.
- Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0).

Please cite the dataset version DOI and associated descriptor paper.
