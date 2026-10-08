#!/usr/bin/env python3
# Editor: Jialei.He
"""Shared mechanics for modality-specific ordered-slice readers.

Filename grammar, intensity ranges, and expected cohort counts deliberately
remain in the CT, MRI, and PET modules. This file only implements operations
that are invariant across modalities.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .names import ParsedSliceName


@dataclass(frozen=True)
class SlicePair:
    lq_path: str
    hq_path: str
    parsed: ParsedSliceName
    stem: str


def strip_medical_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def medical_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(name)[1]


def load_medical_array(path: str) -> np.ndarray:
    extension = medical_extension(path)
    if extension == ".bin":
        with open(path, "rb") as handle:
            return np.asarray(pickle.load(handle))
    if extension in {".nii", ".nii.gz"}:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    raise ValueError(f"Unsupported medical file extension: {path}")


def normalize_array(array: np.ndarray, intensity_range: Tuple[float, float]) -> np.ndarray:
    low, high = (float(intensity_range[0]), float(intensity_range[1]))
    if not high > low:
        raise ValueError(f"Invalid intensity range: {intensity_range}")
    value = np.asarray(array, dtype=np.float32)
    value = np.clip(value, low, high)
    value = (value - low) / (high - low)
    return np.ascontiguousarray(value, dtype=np.float32)


def denormalize_tensor(
    value: torch.Tensor, intensity_range: Tuple[float, float]
) -> torch.Tensor:
    low, high = (float(intensity_range[0]), float(intensity_range[1]))
    return torch.clamp(value * (high - low) + low, min=low, max=high)


def to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    value = np.squeeze(np.asarray(array, dtype=np.float32))
    if value.ndim != 2:
        raise ValueError(f"Expected one 2D slice after squeeze, got {value.shape}")
    return torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)


def _list_files(directory: Path, extensions: Sequence[str]) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    files = sorted(
        path for path in directory.iterdir() if path.name.endswith(tuple(extensions))
    )
    if not files:
        raise RuntimeError(f"No files with {tuple(extensions)} under {directory}")
    return files


def collect_groups(
    data_root: str,
    modality: str,
    split: str,
    parser: Callable[[str], ParsedSliceName],
    training_auxiliary_required: bool,
) -> Dict[str, List[SlicePair]]:
    """Collect paired slices without guessing another modality's grammar."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split {split!r}")
    training = split == "train"
    modality = str(modality).upper()
    lq_dir = Path(data_root) / modality / split / "LQ"
    hq_dir = Path(data_root) / modality / split / "HQ"
    extensions = (".bin",) if training else (".nii", ".nii.gz")
    files = _list_files(lq_dir, extensions)
    if not hq_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {hq_dir}")

    groups: Dict[str, List[SlicePair]] = {}
    evaluation_auxiliary: Dict[str, Optional[int]] = {}
    for lq_path in files:
        hq_path = hq_dir / lq_path.name
        if not hq_path.is_file():
            raise FileNotFoundError(f"Missing paired HQ file: {hq_path}")
        stem = strip_medical_extension(lq_path.name)
        parsed = parser(stem)
        if training and training_auxiliary_required:
            if parsed.auxiliary_index is None:
                raise ValueError(
                    f"{modality} training file lacks the required patch index: {lq_path.name}"
                )
            key = f"{modality}:{parsed.case_id}:p{parsed.auxiliary_index}"
        else:
            key = f"{modality}:{parsed.case_id}"
            if not training:
                previous = evaluation_auxiliary.setdefault(
                    parsed.case_id, parsed.auxiliary_index
                )
                if previous != parsed.auxiliary_index:
                    raise RuntimeError(
                        f"Mixed auxiliary indices in {modality} case {parsed.case_id}: "
                        f"{previous} versus {parsed.auxiliary_index}"
                    )
        groups.setdefault(key, []).append(
            SlicePair(str(lq_path), str(hq_path), parsed, stem)
        )

    for key, items in groups.items():
        items.sort(key=lambda item: item.parsed.slice_index)
        indices = [item.parsed.slice_index for item in items]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"Duplicate slice index in {key}")
        discontinuities = [
            (left, right)
            for left, right in zip(indices, indices[1:])
            if right - left != 1
        ]
        if discontinuities:
            raise RuntimeError(
                f"Non-contiguous source sequence in {key}: {discontinuities[:8]}"
            )
    return groups


class OrderedSliceTrainDataset(Dataset):
    """Non-overlapping seven-slice windows with retained short tails."""

    def __init__(
        self,
        data_root: str,
        modality: str,
        parser: Callable[[str], ParsedSliceName],
        intensity_range: Tuple[float, float],
        expected_train_cases: int,
        training_auxiliary_required: bool,
        seq_len: int = 7,
        patch_size: int = 128,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.modality = str(modality).upper()
        self.parser = parser
        self.intensity_range = intensity_range
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.seq_len != 7:
            raise ValueError(f"The primary protocol requires seq_len=7, got {self.seq_len}")
        if self.patch_size != 128:
            raise ValueError(
                f"The primary protocol requires patch_size=128, got {self.patch_size}"
            )
        self.groups = collect_groups(
            self.data_root,
            self.modality,
            "train",
            self.parser,
            training_auxiliary_required,
        )
        self.group_keys = sorted(self.groups)
        self.case_ids = sorted(
            {item.parsed.case_id for items in self.groups.values() for item in items}
        )
        if len(self.case_ids) != int(expected_train_cases):
            raise RuntimeError(
                f"Unexpected {self.modality} training cases: got={len(self.case_ids)} "
                f"expected={expected_train_cases}"
            )
        self.windows: List[Tuple[str, int, int]] = []
        self._build_windows()

    def _build_windows(self) -> None:
        windows: List[Tuple[str, int, int]] = []
        for key in self.group_keys:
            length = len(self.groups[key])
            for start in range(0, length, self.seq_len):
                windows.append((key, start, min(length, start + self.seq_len)))
        if not windows:
            raise RuntimeError(f"No {self.modality} training windows were built")
        rng = np.random.RandomState(self.seed + self.epoch * 1_000_003)
        rng.shuffle(windows)
        self.windows = windows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._build_windows()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        key, start, end = self.windows[int(index)]
        items = self.groups[key][start:end]
        inputs = [
            to_chw_tensor(normalize_array(load_medical_array(item.lq_path), self.intensity_range))
            for item in items
        ]
        targets = [
            to_chw_tensor(normalize_array(load_medical_array(item.hq_path), self.intensity_range))
            for item in items
        ]
        inp = torch.stack(inputs, dim=0)
        target = torch.stack(targets, dim=0)
        if inp.shape != target.shape:
            raise RuntimeError(f"LQ/HQ shape mismatch in {key}: {inp.shape} vs {target.shape}")
        if not torch.isfinite(inp).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite normalized data in {key}")

        height, width = inp.shape[-2:]
        pad_height = max(0, self.patch_size - height)
        pad_width = max(0, self.patch_size - width)
        if pad_height or pad_width:
            inp = F.pad(inp, (0, pad_width, 0, pad_height), mode="replicate")
            target = F.pad(target, (0, pad_width, 0, pad_height), mode="replicate")
            height, width = inp.shape[-2:]
        rng = np.random.RandomState(
            self.seed + self.epoch * 1_000_003 + int(index) * 97
        )
        top = int(rng.randint(0, height - self.patch_size + 1))
        left = int(rng.randint(0, width - self.patch_size + 1))
        inp = inp[:, :, top : top + self.patch_size, left : left + self.patch_size]
        target = target[:, :, top : top + self.patch_size, left : left + self.patch_size]
        length = int(inp.shape[0])
        dt = torch.full((length, 1), 1.0 / float(length), dtype=torch.float32)
        return inp.contiguous(), target.contiguous(), dt, key


class OrderedSliceCaseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        modality: str,
        parser: Callable[[str], ParsedSliceName],
        intensity_range: Tuple[float, float],
        expected_counts: Dict[str, int],
        split: str,
    ) -> None:
        if split not in {"validation", "test"}:
            raise ValueError(f"Evaluation split must be validation or test, got {split!r}")
        self.data_root = str(data_root)
        self.modality = str(modality).upper()
        self.split = split
        self.intensity_range = intensity_range
        self.groups = collect_groups(
            self.data_root, self.modality, split, parser, False
        )
        self.case_keys = sorted(self.groups)
        if len(self.case_keys) != int(expected_counts[split]):
            raise RuntimeError(
                f"Unexpected {self.modality} {split} cases: got={len(self.case_keys)} "
                f"expected={expected_counts[split]}"
            )

    def __len__(self) -> int:
        return len(self.case_keys)

    def __getitem__(self, index: int):
        key = self.case_keys[int(index)]
        items = self.groups[key]
        inp = torch.stack(
            [
                to_chw_tensor(
                    normalize_array(load_medical_array(item.lq_path), self.intensity_range)
                )
                for item in items
            ],
            dim=0,
        )
        target = torch.stack(
            [
                to_chw_tensor(
                    normalize_array(load_medical_array(item.hq_path), self.intensity_range)
                )
                for item in items
            ],
            dim=0,
        )
        if inp.shape != target.shape:
            raise RuntimeError(f"LQ/HQ shape mismatch in evaluation case {key}")
        if not torch.isfinite(inp).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite normalized evaluation data in {key}")
        return (
            inp.contiguous(),
            target.contiguous(),
            self.modality,
            key,
            [item.stem for item in items],
        )
