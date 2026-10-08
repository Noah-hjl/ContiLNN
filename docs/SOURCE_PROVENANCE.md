# Source provenance

<!-- Editor: Jialei.He -->

## Restore-RWKV integration

The released Restore-RWKV backbone and CUDA kernel derive from
<https://github.com/Yaziwel/Restore-RWKV> at commit
`27dc8dedd139a34f99b9a7b0c7439322525d8a16`. The upstream project is licensed
under Apache-2.0. Modified files retain the Shanghai AI Laboratory copyright
notice.

The ContiLNN–RWKV implementation preserves the Bi-CfC topology and checkpoint
key structure used for the reported primary experiments. Package imports,
runtime path handling, and CUDA-extension loading were made portable for this
distribution.

## DASMamba integration

DASMamba is maintained at <https://github.com/cc111mp/DASMamba-MedIR>. The
released integration is aligned to commit
`8baea1839d800563a1670bb27410509aa7fcbf28`, which is licensed under
Apache-2.0. The upstream backbone is not copied into this repository; users
obtain the pinned revision separately and pass the resulting backbone instance
to `ContiLNNDASMamba`.

The Bi-CfC pathways in `src/contilnn/models/contilnn_dasmamba.py` follow the
transfer configuration used for the DASMamba experiments.

## Data readers

The public CT, MRI, and PET readers preserve the implemented filename grammars,
normalization ranges, short-tail windows, and patient-level partition checks.
Their isolation is tested explicitly; no parser falls back to another
modality's naming convention.

## Evaluation metrics

The Gaussian-window SSIM implementation in
`src/contilnn/evaluation/metrics.py` is adapted from
<https://github.com/Po-Hsun-Su/pytorch-ssim>, distributed under the MIT
License. The local implementation accepts the modality-specific intensity
range used by the evaluation protocol. PSNR and RMSE are implemented directly
in the same module.
