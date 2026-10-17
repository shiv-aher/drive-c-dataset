# Changelog

## 1.1.0 — 2026-10-17

- Corrected the pixel head to emit logits for binary-cross-entropy with logits.
- Materialized the pixel head before optimizer construction and added gradient
  and optimizer-coverage tests.
- Corrected the health target to use the configured GSHI beta, floor, and
  ceiling.
- Added the corrected epoch-7 checkpoint selected by held-out KITTI health MAE.
- Added frozen corrected-training provenance, environment information, and
  release checksums.
- Updated documentation to distinguish configured health-index estimation from
  physical sensor-health measurement.

## 1.0.1 — 2026-09-09

- Historical Version 2 release; retained for provenance only.
