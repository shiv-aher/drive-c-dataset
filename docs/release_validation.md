# DRIVE-C v1.0.0 release validation

Candidate status: **technical validation passed; public tag/archive pending**

| Area | Status | Result |
|---|---|---|
| Repository layout | PASS | Portable roots; no empty configuration files |
| Required artifacts | PASS | Model, checkpoint, training files, generators, analysis and documentation present |
| Fresh environment | PASS | Python 3.10 installation from pinned requirements |
| Clean-clone test | PASS | Compilation, CLI imports and one-clip exact inference |
| Dataset structure | PASS | 610 clips: 10 clean and 600 corrupted |
| Metadata | PASS | 610 rows, 27 fields, valid JSON, unique/resolving paths |
| Coverage | PASS | 10 scenarios, 12 corruptions, 5 severity levels, 305/305 splits |
| Video integrity | PASS | All 610 clips have 128 frames, 1280x720 resolution and 30 fps |
| Reference GSHI | PASS | All 610 values reproduced; maximum serialized error `4.95555e-07` |
| Baseline inference | PASS | All 610 stored scalar, class and JSON predictions reproduced exactly |
| Statistical outputs | PASS | Existing summary outputs reproduced byte-for-byte |
| Corruption regeneration | PASS/QUALIFIED | Bit-identical from frozen PNG/depth inputs; functional but codec-nonexact from decoded H.264 clean clips |
| Metadata documentation | PASS/REVIEW | Complete 27-field dictionary and 13 category schemas; inferred semantics should be reviewed by the author |
| Dataset identifiers | PASS | Version DOI `10.5281/zenodo.19656444`; concept DOI `10.5281/zenodo.19656443` |
| Public code identity | PENDING | Final `v1.0.0` tag, Git commit and software DOI are filled at publication |

Run the structural/video validator with:

```bash
python scripts/validate_release.py /absolute/path/to/drive-c-core-v1
```

The final publication gate must verify that `v1.0.0`, the Git commit recorded in
`RELEASE_MANIFEST.json`, the GitHub release archive and the software DOI all
refer to the same immutable code tree.
