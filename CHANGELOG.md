# Changelog

## 1.0.1 — 2026-09-09

- Documented explicit input and output roots for corruption regeneration.
- Added separate smoke-test and full-dataset generation commands to prevent
  skipped outputs from producing incomplete regenerated metadata.

## 1.0.0 — 2026-09-09

- Added the complete PerceptionHealthNet implementation, training program,
  checkpoint, KITTI split provenance, frozen configurations, and inference
  documentation.
- Made scripts portable through repository-relative paths and
  `DRIVE_C_DATASET_ROOT`.
- Added all missing corruption generators and benchmark-figure generation.
- Added the 27-field metadata dictionary and schemas for all 13 `extra_json`
  categories.
- Added release validation, checksums, citation metadata, and environment pins.
- Standardized GSHI language as a severity-derived reference value rather than
  independently measured sensor health or downstream reliability.
- Documented the public regeneration boundary from anonymized clean clips.
- Removed obsolete empty configuration files.
