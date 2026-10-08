#!/usr/bin/env python3
# Editor: Jialei.He
"""CT-only ordered-slice readers for the reported experiments."""

from __future__ import annotations

from .common import (
    OrderedSliceCaseDataset,
    OrderedSliceTrainDataset,
)
from .names import parse_ct_stem


INTENSITY_RANGE = (-1024.0, 3072.0)
EVALUATION_RANGE = (-160.0, 240.0)
EXPECTED_TRAIN_CASES = 8
EXPECTED_CASES = {"validation": 1, "test": 1}


class CtSequenceTrainDataset(OrderedSliceTrainDataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int = 7,
        patch_size: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__(
            data_root=data_root,
            modality="CT",
            parser=parse_ct_stem,
            intensity_range=INTENSITY_RANGE,
            expected_train_cases=EXPECTED_TRAIN_CASES,
            training_auxiliary_required=True,
            seq_len=seq_len,
            patch_size=patch_size,
            seed=seed,
        )


class CtCaseDataset(OrderedSliceCaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            modality="CT",
            parser=parse_ct_stem,
            intensity_range=INTENSITY_RANGE,
            expected_counts=EXPECTED_CASES,
            split=split,
        )
