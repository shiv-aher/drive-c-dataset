# DRIVE-C metadata dictionary (draft freeze)

Source audited: local release `final_metadata.csv`, 610 rows and 27 columns.
Types and constraints must be validated before final release.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Unique clip identifier. |
| `scenario_id` | string | Original scenario identifier, `S01`–`S10`. |
| `source_video_id` | string | Identifier of the private original recording used to derive the anonymized clean clip. |
| `split` | enum | Scenario-level split: `dev` or `test`. |
| `clip_type` | enum | `clean` or `corrupted`. |
| `corruption_type` | enum | `clean` or one of the 12 generated corruption categories. |
| `severity_level` | integer | `0` for clean; `1`–`5` for generated corruptions. |
| `severity_name` | string | `s0`–`s5`, corresponding to `severity_level`. |
| `severity_value` | float | Continuous generator severity: 0, 0.08, 0.18, 0.35, 0.55, or 0.75. |
| `is_clean` | boolean integer | `1` for clean and `0` for corrupted. |
| `gshi` | float | Legacy generation placeholder (`1.0` clean, `-1.0` corrupted); do not treat as the final reference label. |
| `weather` | enum | Capture-condition weather label (`cloudy` or `sunny`). |
| `time_of_day` | enum | Capture lighting period (`day` or `night`). |
| `scene_type` | enum | High-level scene: urban, rural, freeway, or parking lot. |
| `traffic_level` | enum | Coarse capture traffic label: none, low, or moderate. |
| `fps` | integer | Nominal frame rate; 30 in this release. |
| `resolution` | string | Released frame dimensions; `1280x720`. |
| `num_frames` | integer | Frames in the clip; 128 in this release. |
| `output_path` | string | Dataset-root-relative path intended to locate the MP4. |
| `extra_json` | JSON object | Corruption-specific generation parameters; `{}` for clean clips. Keys vary by corruption and require per-corruption schemas. |
| `gshi_gt` | float | Deterministic severity-derived reference GSHI, not independently measured reliability. |
| `gshi_pred` | float | PerceptionHealthNet clip prediction, averaged over sampled frames. |
| `pred_top1_issue` | enum | Highest mean predicted-presence class from the model’s 12-class taxonomy vocabulary. |
| `pred_top1_prob` | float | Mean sigmoid probability associated with `pred_top1_issue`. |
| `pred_presence_json` | JSON object | Mapping from all 12 model issue names to mean predicted-presence probabilities. |
| `pred_severity_json` | JSON object | Mapping from all 12 model issue names to mean predicted severities. |
| `pred_present_thresh_json` | JSON array | Descending list of issue/probability objects meeting the inference threshold (default 0.25). |

## Important distinctions

- `gshi` is a legacy placeholder; `gshi_gt` is the generated reference label;
  `gshi_pred` is a model output. They are not interchangeable.
- The model vocabulary includes `vignetting`; the dataset corruption set does
  not. Overexposure and underexposure both map to model/taxonomy class
  `exposure_shift`; fog maps to `haze_fog`; JPEG compression maps to
  `compression`.
- Clean paths use the canonical `clean_clips/S01_clean.mp4` through
  `clean_clips/S10_clean.mp4` convention and resolve in the validated release.

## Nested metadata and legacy field

Allowed structures for every `extra_json` category are documented under
`extra_json_schemas/`. The legacy `gshi` column is retained for schema
compatibility and formally deprecated; use `gshi_gt` for the severity-derived
reference and `gshi_pred` for the model prediction.
