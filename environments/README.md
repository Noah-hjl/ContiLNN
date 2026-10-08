# Reference environments

<!-- Editor: Jialei.He -->

The repository provides separate top-level environments for the two backbone
integrations because their tested PyTorch and CUDA toolchains differ. Create an
environment from the repository root:

```bash
conda env create -f environments/environment-rwkv.yml
conda activate contilnn-rwkv
```

or:

```bash
conda env create -f environments/environment-dasmamba.yml
conda activate contilnn-dasmamba
```

The Restore-RWKV reference environment uses PyTorch 2.0.1 with CUDA 11.8. It
includes Ninja and the matching CUDA compiler because the WKV extension is
compiled on its first CUDA execution. The DASMamba reference environment uses
PyTorch 2.4.1 with CUDA 12.1 and requires the upstream DASMamba revision listed
in `docs/SOURCE_PROVENANCE.md`.
