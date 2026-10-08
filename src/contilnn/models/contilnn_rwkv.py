#!/usr/bin/env python3
# Editor: Jialei.He
"""Public name for the ContiLNN model built on Restore-RWKV."""

from .slice_operators import RestoreRWKVRefine


class ContiLNNRWKV(RestoreRWKVRefine):
    """The manuscript-selected ContiLNN–RWKV topology.

    Slice dynamics are applied at encoder level 2, decoder level 2, skip
    levels 2 and 3, the deepest bottleneck, and the refinement pathway. The
    bottleneck insertion is serial; the remaining insertions are gated
    residual paths.
    """

    def __init__(
        self,
        dim: int = 48,
        cfc_hidden: int = 256,
    ) -> None:
        super().__init__(dim=dim, cfc_hidden=cfc_hidden)


__all__ = ["ContiLNNRWKV"]
