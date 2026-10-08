# Third-party licenses and attribution

<!-- Editor: Jialei.He -->

ContiLNN is distributed under the Apache License 2.0. The following components
retain their upstream terms and attribution.

## Restore-RWKV

The Restore-RWKV backbone and CUDA kernel are derived from
<https://github.com/Yaziwel/Restore-RWKV> at commit
`27dc8dedd139a34f99b9a7b0c7439322525d8a16`. The upstream project is licensed
under Apache-2.0. Modified source files retain the Shanghai AI Laboratory
copyright notice.

## DASMamba

The DASMamba integration targets
<https://github.com/cc111mp/DASMamba-MedIR> at commit
`8baea1839d800563a1670bb27410509aa7fcbf28`. DASMamba is licensed under
Apache-2.0 and is not redistributed in this repository.

## pytorch-ssim

Parts of `src/contilnn/evaluation/metrics.py` are adapted from
<https://github.com/Po-Hsun-Su/pytorch-ssim>, distributed under the MIT
License. The adaptation preserves the Gaussian-window construction and SSIM
formulation while accepting an explicit intensity range.

## Dataset-derived figures

The repository license does not replace or broaden the terms of the datasets
from which qualitative figures were derived. Rendered figures contain no raw
volume files or medical-image headers. Permission to redistribute or reuse a
figure remains subject to the applicable source-dataset terms.
