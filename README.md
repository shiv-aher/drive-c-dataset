# DRIVE-C

DRIVE-C is a compact controlled testbed for evaluating degradation awareness,
sensor-health estimation, and camera-perception robustness. It contains 10
original forward-facing scenarios and 600 controlled corrupted variants: 12
corruption types, five severity levels, and 10 scenarios, plus 10 clean clips.

Dataset version DOI: <https://doi.org/10.5281/zenodo.19656444>
Dataset concept DOI: <https://doi.org/10.5281/zenodo.19656443>

## Reproducibility boundary

The public release supports validation and regeneration of corruptions and
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
  checkpoints/        frozen baseline checkpoint
  configs/            taxonomy and generation/training policy
  dataset/            metadata, reports, and dataset download guide
  provenance/         frozen training provenance
  scripts/            generation, labeling, inference, and analysis
  simulation/         corruption and GSHI implementation
  src/                baseline model and training data loader
  docs/               metadata, schemas, model, GSHI, and validation records
  RELEASE_MANIFEST.json
  SHA256SUMS.txt
  requirements.txt
```

The downloaded dataset archive has this structure:

```text
drive-c-core-v1/
  clean_clips/        10 clips, S01_clean.mp4 ... S10_clean.mp4
  corrupted/          600 clips grouped by corruption and severity
  final_metadata.csv  610 rows, 27 fields
  scenario_metadata.csv
```

## Environment

The release was audited with Python 3.10 and the versions pinned in
`requirements.txt`. GPU inference was tested with PyTorch 2.10.0, torchvision
0.25.0, and CUDA 12.8. A CPU installation may be used but will be slower.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export DRIVE_C_DATASET_ROOT=/absolute/path/to/drive-c-core-v1
```

All commands below are run from the repository root.

## Frozen documentation

- [Metadata dictionary](docs/metadata_dictionary.md)
- [`extra_json` schemas](docs/extra_json_schema.md)
- [GSHI definition and weights](docs/gshi_definition.md)
- [PerceptionHealthNet provenance](docs/baseline_model.md)
- [Regeneration boundary and audit](docs/regeneration.md)
- [Release validation](docs/release_validation.md)

## Regenerate corruptions from the released clean clips

Keep the downloaded release separate from generated outputs. The following
variables make every input and output path explicit:

```bash
export DRIVE_C_RELEASE_ROOT=/absolute/path/to/drive-c-core-v1
export DRIVE_C_CLEAN_FRAMES_ROOT="$PWD/work/clean_frames"
export DRIVE_C_DEPTH_ROOT="$PWD/work/depth"

python scripts/extract_released_clean_frames.py \
  --dataset-root "$DRIVE_C_RELEASE_ROOT" \
  --output-root "$DRIVE_C_CLEAN_FRAMES_ROOT"

python scripts/generate_drivec_core_depth.py \
  --input-root "$DRIVE_C_CLEAN_FRAMES_ROOT" \
  --output-root "$DRIVE_C_DEPTH_ROOT"
```

The depth model identifier is frozen by the depth-generation command. The
original pre-anonymization recordings are not required for this public stage.

For a one-scenario, one-corruption smoke test, use a new output directory and
copy the released scenario metadata required by the generator:

```bash
export DRIVE_C_DATASET_ROOT="$PWD/work/regenerated-smoke"
mkdir -p "$DRIVE_C_DATASET_ROOT"
cp "$DRIVE_C_RELEASE_ROOT/scenario_metadata.csv" "$DRIVE_C_DATASET_ROOT/"

python scripts/unified_generate_drivec.py \
  --scenarios S01 \
  --corruptions motion_blur \
  --severity-names s3
```

For full-dataset regeneration, use a different empty output directory so that
the generated metadata cannot omit outputs skipped from an earlier smoke test:

```bash
export DRIVE_C_DATASET_ROOT="$PWD/work/regenerated-full"
mkdir -p "$DRIVE_C_DATASET_ROOT"
cp "$DRIVE_C_RELEASE_ROOT/scenario_metadata.csv" "$DRIVE_C_DATASET_ROOT/"

python scripts/unified_generate_drivec.py
```

The generator reads clean frames from `DRIVE_C_CLEAN_FRAMES_ROOT`, depth maps
from `DRIVE_C_DEPTH_ROOT`, and writes clips and `samples_metadata.csv` beneath
`DRIVE_C_DATASET_ROOT`. Start each regeneration run with an empty output
directory.

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
training dataset. Training code and frozen provenance are included.

Inference samples eight uniformly spaced frames from each 128-frame clip,
resizes them directly to 384 x 1280, and averages frame-level outputs.

```bash
python scripts/add_gshi_pred.py \
  --ckpt checkpoints/epoch_021_best.pth \
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

The reported 0.475 monotonicity fraction is computed over 120
complete scenario-corruption five-level sequences (57 of 120 pass). The
6-of-12 count is computed after averaging
predictions across scenarios for each corruption and severity. These quantities
therefore use different aggregation rules.

## Validate a downloaded release

```bash
python scripts/validate_release.py "$DRIVE_C_DATASET_ROOT"
sha256sum --check SHA256SUMS.txt
```

The validator checks the 610/10/600 counts, 27 metadata fields, scenario,
corruption, severity and split counts, relative paths, JSON parsing, and video
properties. `SHA256SUMS.txt` covers the code repository contents; dataset-video
checksums are distributed with the dataset archive.

## Licenses

- Code: MIT License.
- Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0).

Please cite the dataset version DOI and associated descriptor paper.
