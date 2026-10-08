#!/usr/bin/env python3
# Editor: Jialei.He
"""MRI-only ordered-slice readers for the reported experiments."""

from __future__ import annotations

from .common import (
    OrderedSliceCaseDataset,
    OrderedSliceTrainDataset,
)
from .names import parse_mri_stem


INTENSITY_RANGE = (0.0, 4095.0)
EVALUATION_RANGE = (0.0, 4095.0)
EXPECTED_TRAIN_CASES = 405
EXPECTED_CASES = {"validation": 58, "test": 114}


class MriSequenceTrainDataset(OrderedSliceTrainDataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int = 7,
        patch_size: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__(
            data_root=data_root,
            modality="MRI",
            parser=parse_mri_stem,
            intensity_range=INTENSITY_RANGE,
            expected_train_cases=EXPECTED_TRAIN_CASES,
            training_auxiliary_required=False,
            seq_len=seq_len,
            patch_size=patch_size,
            seed=seed,
        )


class MriCaseDataset(OrderedSliceCaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            modality="MRI",
            parser=parse_mri_stem,
            intensity_range=INTENSITY_RANGE,
            expected_counts=EXPECTED_CASES,
            split=split,
        )
