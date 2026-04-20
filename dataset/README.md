# DRIVE-C Core v1
### A Controlled Corruption Dataset for Autonomous Driving

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

DRIVE-C is a video clip dataset for benchmarking visual perception robustness in autonomous driving systems. It pairs real-world forward-facing driving footage with 12 types of synthetic degradation, each applied at 5 severity levels, providing a controlled and reproducible benchmark for degradation-aware modeling and sensor health estimation.

Download full dataset from:
https://doi.org/10.5281/zenodo.19656444

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Scenarios | 10 (S01–S10) |
| Corruption types | 12 |
| Severity levels | 5 (s1–s5) + clean (s0) |
| Total clips | 610 |
| Clean clips | 10 |
| Corrupted clips | 600 |
| Frames per clip | 128 (~4.3 s at 30 fps) |
| Resolution | 1280×720 (HD) |
| Native resolution | 3840×2160 (4K UHD) |
| Frame rate | 30 fps |
| Format | MP4 (H.264) |
| Dev / test split | 305 / 305 clips |
| Capture dates | April 1 and April 5, 2026 |
| Capture location | Michigan, United States |

---

## Capture Scenarios

| ID | Scene Type | Weather | Time of Day | Traffic |
|---|---|---|---|---|
| S01 | Urban | Cloudy | Day | Moderate |
| S02 | Freeway | Cloudy | Day | Moderate |
| S03 | Urban | Sunny | Day | Moderate |
| S04 | Rural | Cloudy | Day | Moderate |
| S05 | Parking lot | Sunny | Day | Moderate |
| S06 | Urban | Cloudy | Night | Moderate |
| S07 | Urban | Cloudy | Night | Moderate |
| S08 | Urban | Cloudy | Night | None |
| S09 | Urban | Sunny | Day | Low |
| S10 | Rural | Cloudy | Day | Moderate |

---

## Corruption Types

Corruptions are organized into three degradation families:

**Weather**
- `fog` — Homogeneous scattering reducing contrast and depth cues
- `rain` — Streak-based precipitation with motion-dependent angle
- `snow` — Particle-based accumulation with brightness variation
- `glare_flare` — Specular highlight blooming from direct or reflected light

**Optical**
- `motion_blur` — Linear kernel blur along the direction of vehicle motion
- `defocus_blur` — Radially symmetric blur simulating lens focus error
- `lens_occlusion` — Partial obstruction from water droplets or debris
- `low_light` — Luminance reduction with amplified sensor noise

**Sensor**
- `sensor_noise` — Gaussian additive noise at the pixel level
- `overexposure` — Global gain increase causing highlight clipping
- `underexposure` — Global gain reduction causing shadow crushing
- `jpeg_compression` — Block-artifact compression at decreasing quality factors

Each corruption is applied at five severity levels using a continuous normalized severity value:

| Level | Name | Severity value |
|---|---|---|
| 0 | s0 | 0.00 (clean) |
| 1 | s1 | 0.08 |
| 2 | s2 | 0.18 |
| 3 | s3 | 0.35 |
| 4 | s4 | 0.55 |
| 5 | s5 | 0.75 |

---

## Directory Structure

```
drive-c-core-v1/
├── clean_clips/
│   ├── S01_clean.mp4
│   ├── S02_clean.mp4
│   └── ... (10 clips total)
├── corrupted/
│   ├── fog/
│   │   ├── s1/   S01_fog_s1.mp4 ... S10_fog_s1.mp4
│   │   ├── s2/   S01_fog_s2.mp4 ... S10_fog_s2.mp4
│   │   ├── s3/
│   │   ├── s4/
│   │   └── s5/
│   ├── rain/
│   │   └── s1/ ... s5/
│   ├── snow/
│   ├── glare_flare/
│   ├── motion_blur/
│   ├── defocus_blur/
│   ├── lens_occlusion/
│   ├── low_light/
│   ├── sensor_noise/
│   ├── overexposure/
│   ├── underexposure/
│   └── jpeg_compression/
├── final_metadata.csv
├── scenario_metadata.csv
├── README.md
└── LICENSE
```

**Filename convention:** `[scenario_id]_[corruption_type]_[severity_name].mp4`  
Example: `S03_motion_blur_s4.mp4`

---

## Metadata Files

### `final_metadata.csv` — primary metadata (610 rows, 27 fields)

One row per clip. Key fields:

| Field | Type | Description |
|---|---|---|
| `sample_id` | String | Unique clip identifier, e.g. `S01_fog_s3` |
| `scenario_id` | String | Scenario identifier (S01–S10) |
| `source_video_id` | String | Source video identifier for traceability |
| `split` | String | Benchmark split: `dev` or `test` |
| `clip_type` | String | `clean` or `corrupted` |
| `corruption_type` | String | Corruption name, e.g. `fog`, `motion_blur` |
| `severity_level` | Integer | Integer severity (0 = clean, 1–5 = corrupted) |
| `severity_value` | Float | Continuous severity used by the generator (0.08–0.75) |
| `weather` | String | Scene weather: `sunny` or `cloudy` |
| `time_of_day` | String | Lighting regime: `day` or `night` |
| `scene_type` | String | Scene label: `urban`, `rural`, `freeway`, etc. |
| `fps` | Integer | Frame rate (30 for all clips) |
| `resolution` | String | Clip resolution (`1280x720` for all clips) |
| `num_frames` | Integer | Number of frames (128 for all clips) |
| `output_path` | String | Relative path to the MP4 file |
| `extra_json` | JSON | Per-clip generation parameters for reproducibility |
| `gshi_gt` | Float | Ground-truth GSHI (see below) |
| `gshi_pred` | Float | Baseline model (PerceptionHealthNet) predicted GSHI |
| `pred_top1_issue` | String | Highest-probability predicted degradation class |
| `pred_top1_prob` | Float | Probability of the top predicted class |

### `scenario_metadata.csv` — source video details (10 rows)

Per-scenario fields including source video ID, clip frame range, capture date, native resolution, and scene annotations.

---

## Global Sensor Health Index (GSHI)

The **Global Sensor Health Index (GSHI)** is a scalar metric in [0, 1] quantifying perceptual reliability under degradation.

- **1.0** → clean, unimpaired image
- **0.0** → severe degradation

Ground-truth GSHI (`gshi_gt`) is computed deterministically from the applied severity and per-corruption taxonomy weights. It decreases monotonically from s1 to s5 for all 12 corruption types.

The `gshi_pred` field contains predictions from **PerceptionHealthNet**, a baseline model included to provide a starting benchmark. Overall Pearson r = 0.34 across all 610 clips. The model performs well on underexposure (r = 0.77) and motion blur (r = 0.73), but struggles with sensor noise (r = −0.16), JPEG compression (r = 0.09), and defocus blur (r = 0.12) — establishing these as open research challenges.

---

## Benchmark Splits

Clips are partitioned into `dev` and `test` splits of 305 clips each.

- **dev** (S01–S05): includes the freeway and parking-lot scenarios; daytime only
- **test** (S06–S10): includes all three nighttime scenarios; urban and rural

> **Note:** Split assignment is at the scenario level, not the clip level. The dev and test sets therefore differ in scene type and lighting distribution, not just sample identity.

---

## Usage

### Load metadata

```python
import pandas as pd

df = pd.read_csv("final_metadata.csv")
print(df.shape)   # (610, 27)
print(df["corruption_type"].value_counts())
```

### Access a specific clip

```python
row = df[df["sample_id"] == "S03_motion_blur_s4"].iloc[0]
clip_path = row["output_path"]   # e.g. corrupted/motion_blur/s4/S03_motion_blur_s4.mp4
severity  = row["severity_value"]  # 0.55
gshi      = row["gshi_gt"]         # ground-truth health score
```

### Load all clips for one corruption type

```python
fog_clips = df[df["corruption_type"] == "fog"].copy()
fog_clips = fog_clips.sort_values(["scenario_id", "severity_level"])
```

### Compare clean vs. corrupted GSHI

```python
clean     = df[df["clip_type"] == "clean"]["gshi_gt"].mean()
corrupted = df[df["clip_type"] == "corrupted"]["gshi_gt"].mean()
print(f"Mean GSHI — clean: {clean:.3f}, corrupted: {corrupted:.3f}")
```

### Load a video clip with OpenCV

```python
import cv2

cap = cv2.VideoCapture("corrupted/fog/s3/S01_fog_s3.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame is (720, 1280, 3) BGR
cap.release()
```

---

## Caveats

- **Scene type imbalance:** 6 urban, 2 rural, 1 freeway, 1 parking lot. Urban scenarios are overrepresented.
- **Nighttime coverage:** All three nighttime scenarios (S06–S08) are urban and cloudy. There are no nighttime rural, freeway, or clear-sky scenarios.
- **Single-corruption design:** Each corrupted clip contains exactly one active corruption type. Compound degradations are not included but can be generated using the provided scripts.
- **Baseline false positives on clean clips:** PerceptionHealthNet assigns low `gshi_pred` to some clean nighttime clips (e.g. S08: 0.149, S07: 0.181). Validate thresholds against the clean clip distribution before using `gshi_pred` for quality filtering.

---

## Source Code

The processing pipeline, corruption generator, and baseline model are available at:

**https://github.com/shiv-aher/drive-c-dataset**

The repository includes scripts for clip extraction, anonymization, corruption generation, GSHI computation, and metadata validation. All per-clip generation parameters are stored in the `extra_json` field of `final_metadata.csv`, enabling any individual clip to be regenerated independently.

---

## Citation

If you use DRIVE-C in your research, please cite:

```bibtex
@misc{aher2026drivec,
  author       = {Aher, Shiva},
  title        = {DRIVE-C: A Controlled Corruption Dataset for Autonomous Driving},
  year         = {2026},
  note         = {Under review at IEEE Data Descriptions},
  howpublished = {\url{https://github.com/shiv-aher/drive-c}}
}
```

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the material for any purpose, provided appropriate credit is given.

Full license text: https://creativecommons.org/licenses/by/4.0/

---

## Acknowledgements

Special thanks to Manisha A. for supporting real-world data collection, including coordinating capture sessions and assisting with camera setup.

---

## Contact

For questions or issues, please open an issue on the GitHub repository.