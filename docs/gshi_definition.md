# GSHI provenance freeze (WP1)

This document freezes implementation facts for later manuscript work in WP3.
It does not yet make a scientific-validity claim.

## Implementation

Source: `simulation/gshi_utils.py`  
Configuration: `configs/taxonomy/camera_issues.yaml`

For issue severities `s_i`, base weights `w_i`, group scales `q_i`, and
`beta = 0.85`, the released implementation computes

```text
log_h = sum_i (w_i * q_i) * log(clip(1 - s_i, eps, 1))
h_raw = exp(beta * log_h)
GSHI = clip(h_raw, floor, ceil)
```

with `eps = 1e-6`, `floor = 1e-3`, and `ceil = 0.99`. Clean clips bypass this
function in `add_gshi_gt.py` and are assigned `1.0`; this explains why clean
values may exceed the configured `ceil`.

## Frozen weights

| Dataset corruption | Taxonomy issue | Base weight | Group scale | Effective weight |
|---|---|---:|---:|---:|
| fog | haze_fog | 1.30 | 1.00 | 1.300 |
| rain | rain | 1.10 | 1.00 | 1.100 |
| snow | snow | 1.20 | 1.00 | 1.200 |
| low_light | low_light | 1.30 | 1.00 | 1.300 |
| motion_blur | motion_blur | 1.40 | 1.10 | 1.540 |
| defocus_blur | defocus_blur | 1.40 | 1.10 | 1.540 |
| glare_flare | glare_flare | 1.50 | 1.10 | 1.650 |
| sensor_noise | sensor_noise | 1.10 | 0.95 | 1.045 |
| overexposure | exposure_shift | 1.00 | 0.95 | 0.950 |
| underexposure | exposure_shift | 1.00 | 0.95 | 0.950 |
| jpeg_compression | compression | 0.80 | 0.95 | 0.760 |
| lens_occlusion | lens_occlusion | 1.60 | 1.15 | 1.840 |

`vignetting` is present in the model/taxonomy vocabulary with effective weight
`0.7 * 1.10 = 0.770`, but it is not one of the 12 generated DRIVE-C corruption
categories. Consequently, a model prediction of vignetting is an output-vocabulary
prediction, not a dataset corruption label.

## Interpretation boundary

`gshi_gt` is a deterministic, severity-derived reference value. It is not an
independent measurement of physical sensor condition and does not by itself
establish downstream perception reliability. Figure 1 demonstrates consistency
with the prescribed ordering because the index is constructed from that severity.

## Validation and severity rationale

The configuration SHA-256 is recorded in `RELEASE_MANIFEST.json`. All 610
values reproduce within the six-decimal serialization tolerance (maximum
absolute error `4.95555e-07`). The fixed severities `0.08, 0.18, 0.35, 0.55,
0.75` were selected heuristically to sample mild effects more densely and span
larger moderate-to-severe changes; they are generator control points, not equal
psychophysical intervals.
