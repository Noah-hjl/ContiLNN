#!/usr/bin/env python3
# Editor: Jialei.He
"""PET-only ordered-slice readers for the primary experiments."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .names import ParsedPetName, parse_pet_stem

PET_RANGE = (0.0, 20.0)
EXPECTED_TRAIN_PATIENTS = 120
EXPECTED_CASES = {"validation": 10, "test": 29}


@dataclass(frozen=True)
class PetSlice:
    lq: str
    hq: str
    stem: str
    parsed: ParsedPetName


def strip_medical_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def load_array(path: str) -> np.ndarray:
    if path.endswith(".bin"):
        with open(path, "rb") as handle:
            value = pickle.load(handle)
        return np.asarray(value)
    if path.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    raise ValueError(f"Unsupported PET file extension: {path}")


def normalize_pet(value: np.ndarray) -> np.ndarray:
    low, high = PET_RANGE
    array = np.asarray(value, dtype=np.float32)
    array = np.clip(array, low, high)
    array = (array - low) / (high - low)
    return np.ascontiguousarray(array, dtype=np.float32)


def to_chw(value: np.ndarray) -> torch.Tensor:
    array = np.squeeze(np.asarray(value, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected one 2D PET slice after squeeze, got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _list_files(directory: Path, suffixes: Sequence[str]) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing PET directory: {directory}")
    files = sorted(path for path in directory.iterdir() if path.name.endswith(tuple(suffixes)))
    if not files:
        raise RuntimeError(f"No PET files with {tuple(suffixes)} under {directory}")
    return files


def collect_pet_groups(data_root: str, split: str) -> Dict[str, List[PetSlice]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported PET split {split!r}")
    training = split == "train"
    lq_dir = Path(data_root) / "PET" / split / "LQ"
    hq_dir = Path(data_root) / "PET" / split / "HQ"
    suffixes = (".bin",) if training else (".nii", ".nii.gz")
    files = _list_files(lq_dir, suffixes)
    if not hq_dir.is_dir():
        raise FileNotFoundError(f"Missing PET directory: {hq_dir}")

    groups: Dict[str, List[PetSlice]] = {}
    eval_patch_by_patient: Dict[str, Optional[int]] = {}
    for lq in files:
        hq = hq_dir / lq.name
        if not hq.is_file():
            raise FileNotFoundError(f"Missing paired PET HQ file: {hq}")
        stem = strip_medical_extension(lq.name)
        parsed = parse_pet_stem(stem)
        if training:
            if parsed.patch is None:
                raise ValueError(f"PET train file has no patch index: {lq.name}")
            key = f"PET:{parsed.patient}:p{parsed.patch}"
        else:
            previous = eval_patch_by_patient.setdefault(parsed.patient, parsed.patch)
            if previous != parsed.patch:
                raise RuntimeError(
                    "Mixed PET auxiliary indices for "
                    f"{parsed.patient}: {previous} vs {parsed.patch}"
                )
            key = f"PET:{parsed.patient}"
        groups.setdefault(key, []).append(PetSlice(str(lq), str(hq), stem, parsed))

    for key, items in groups.items():
        items.sort(key=lambda item: item.parsed.z)
        indices = [item.parsed.z for item in items]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"Duplicate PET z indices in {key}")
        bad = [(a, b) for a, b in zip(indices, indices[1:]) if b - a != 1]
        if bad:
            raise RuntimeError(f"Non-contiguous PET sequence in {key}: {bad[:8]}")
    return groups


class PetSequenceTrainDataset(Dataset):
    """Gap-1 PET training windows used by the primary RWKV experiments.

    Windows are non-overlapping within each patch sequence and the final short
    window is retained. The case and patch identifiers never cross window
    boundaries.
    """

    def __init__(
        self,
        data_root: str,
        seq_len: int = 7,
        patch_size: int = 128,
        seed: int = 0,
    ):
        self.data_root = str(data_root)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.seq_len != 7:
            raise ValueError(f"The primary PET protocol requires seq_len=7, got {self.seq_len}")
        if self.patch_size != 128:
            raise ValueError(
                f"The primary PET protocol requires patch_size=128, got {self.patch_size}"
            )
        self.groups = collect_pet_groups(self.data_root, "train")
        self.group_keys = sorted(self.groups)
        self.patient_ids = sorted(
            {item.parsed.patient for items in self.groups.values() for item in items}
        )
        if len(self.patient_ids) != EXPECTED_TRAIN_PATIENTS:
            raise RuntimeError(
                f"Unexpected PET train patients: got={len(self.patient_ids)} "
                f"expected={EXPECTED_TRAIN_PATIENTS}"
            )
        self.windows: List[tuple] = []
        self._build_windows()

    def _build_windows(self) -> None:
        windows = []
        for key in self.group_keys:
            length = len(self.groups[key])
            for start in range(0, length, self.seq_len):
                windows.append((key, start, min(length, start + self.seq_len)))
        if not windows:
            raise RuntimeError("No PET Gap-1 training windows were built")
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
        inputs = [to_chw(normalize_pet(load_array(item.lq))) for item in items]
        targets = [to_chw(normalize_pet(load_array(item.hq))) for item in items]
        inp = torch.stack(inputs, dim=0)
        target = torch.stack(targets, dim=0)
        if inp.shape != target.shape:
            raise RuntimeError(f"PET LQ/HQ shape mismatch in {key}: {inp.shape} vs {target.shape}")
        if not torch.isfinite(inp).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite PET training tensor in {key}")

        height, width = inp.shape[-2:]
        pad_h = max(0, self.patch_size - height)
        pad_w = max(0, self.patch_size - width)
        if pad_h or pad_w:
            inp = F.pad(inp, (0, pad_w, 0, pad_h), mode="replicate")
            target = F.pad(target, (0, pad_w, 0, pad_h), mode="replicate")
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


class PetCaseDataset(Dataset):
    def __init__(self, data_root: str, split: str):
        if split not in EXPECTED_CASES:
            raise ValueError(f"PET evaluation split must be validation or test, got {split!r}")
        self.data_root = str(data_root)
        self.split = split
        self.groups = collect_pet_groups(self.data_root, split)
        self.case_keys = sorted(self.groups)
        if len(self.case_keys) != EXPECTED_CASES[split]:
            raise RuntimeError(
                f"Unexpected PET {split} cases: got={len(self.case_keys)} "
                f"expected={EXPECTED_CASES[split]} keys={self.case_keys[:8]}"
            )

    def __len__(self) -> int:
        return len(self.case_keys)

    def __getitem__(self, index: int):
        key = self.case_keys[int(index)]
        items = self.groups[key]
        inp = torch.stack([to_chw(normalize_pet(load_array(item.lq))) for item in items], dim=0)
        target = torch.stack([to_chw(normalize_pet(load_array(item.hq))) for item in items], dim=0)
        if inp.shape != target.shape:
            raise RuntimeError(f"PET LQ/HQ shape mismatch in evaluation case {key}")
        if not torch.isfinite(inp).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite PET evaluation tensor in {key}")
        names = [item.stem for item in items]
        return inp.contiguous(), target.contiguous(), "PET", key, names
