# Corruption regeneration audit

Date: 2026-09-08  
Scenario: S01  
Corruption: motion blur  
Severity: s3

## Frozen-input path

Using the original frozen clean PNG frames and depth maps, RC1 regenerated both
the clean clip and `S01_motion_blur_s3.mp4` exactly:

- file SHA-256: identical;
- decoded frames: 128/128;
- maximum absolute pixel difference: 0;
- mean absolute pixel difference: 0.

Status: **PASS — bit-exact**.

## Public released-MP4 path

The released anonymized clean MP4 was decoded to PNG and then passed through the
same generator. The workflow completes, but it is not bit-exact because the
released H.264 clean clip has already undergone lossy encoding:

| Output | Max absolute pixel difference | Mean absolute pixel difference |
|---|---:|---:|
| Re-encoded clean | 44 | 2.13082619 |
| Motion blur s3 | 39 | 2.49268657 |

Status: **PASS for functional regeneration; FAIL for bit-exact regeneration**.

The final documentation must distinguish these guarantees. Exact reproduction
requires preserving the frozen clean PNG frames and depth inputs. Regeneration
from the released clean MP4 reproduces the configured corruption procedure but
cannot recreate identical pixels after another lossy encode/decode cycle.
