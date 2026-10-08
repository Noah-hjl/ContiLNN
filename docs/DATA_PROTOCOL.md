# Data and evaluation protocol

<!-- Editor: Jialei.He -->

## Cohorts

| Modality | Training | Validation | Test | Restoration task |
|---|---:|---:|---:|---|
| CT | 8 | 1 | 1 | Denoising |
| MRI | 405 | 58 | 114 | Super-resolution |
| PET | 120 | 10 | 29 | Reduced-count restoration |

The cohort counts are enforced by the modality-specific readers before
training or evaluation begins.

## Ordered-window construction

- Each sample is an ordered sequence of at most seven slices from one case and,
  for patch-based training data, one patch location.
- The primary release uses adjacent observed slices (`slice_index_gap = 1`).
- The final short sequence is retained instead of discarded.
- Every sample includes one normalized slice-index interval for each position.
- Case boundaries are never crossed.

## Evaluation

Validation and test operate on full cases. Seven-slice windows start from
offsets 0 and 3; predictions at positions covered by both traversals are
averaged. The evaluator verifies that every real slice is covered, computes
PSNR, SSIM, and RMSE per slice, averages within each case, and then averages
across cases.

CT, MRI, and PET retain separate normalization ranges and filename parsers.
The release does not infer modality from a filename and does not substitute a
reader when parsing fails.
