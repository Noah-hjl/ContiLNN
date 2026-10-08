#!/usr/bin/env python3
# Editor: Jialei.He
"""Case-complete overlapping-window inference for CT, MRI, and PET."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..data.common import denormalize_tensor
from ..data.ct import EVALUATION_RANGE as CT_EVALUATION_RANGE
from ..data.ct import INTENSITY_RANGE as CT_INTENSITY_RANGE
from ..data.mri import EVALUATION_RANGE as MRI_EVALUATION_RANGE
from ..data.mri import INTENSITY_RANGE as MRI_INTENSITY_RANGE
from ..data.pet import PET_RANGE
from .metrics import compute_measure


INTENSITY_RANGES = {
    "CT": CT_INTENSITY_RANGE,
    "MRI": MRI_INTENSITY_RANGE,
    "PET": PET_RANGE,
}
EVALUATION_RANGES = {
    "CT": CT_EVALUATION_RANGE,
    "MRI": MRI_EVALUATION_RANGE,
    "PET": PET_RANGE,
}


def _restore_scale(value: torch.Tensor, modality: str) -> torch.Tensor:
    modality = str(modality).upper()
    if modality not in INTENSITY_RANGES:
        raise ValueError(f"Unsupported modality {modality!r}")
    restored = denormalize_tensor(value, INTENSITY_RANGES[modality])
    low, high = EVALUATION_RANGES[modality]
    return torch.clamp(restored, min=float(low), max=float(high))


def infer_case(
    model: torch.nn.Module,
    case_input: torch.Tensor,
    device: torch.device,
    sequence_length: int = 7,
    offsets: Tuple[int, int] = (0, 3),
) -> torch.Tensor:
    """Restore every observed slice and average overlapping predictions."""

    if case_input.ndim != 4:
        raise ValueError(f"Case input must be [D,C,H,W], got {tuple(case_input.shape)}")
    depth = int(case_input.shape[0])
    if depth <= 0:
        raise ValueError("Cannot evaluate an empty case")
    source = case_input.detach().float().cpu()
    prediction_sum = torch.zeros_like(source)
    prediction_count = torch.zeros((depth, 1, 1, 1), dtype=source.dtype)
    model.eval()
    with torch.no_grad():
        for offset in offsets:
            for start in range(int(offset), depth, int(sequence_length)):
                end = min(depth, start + int(sequence_length))
                if end <= start:
                    continue
                window = source[start:end].unsqueeze(0).to(device=device)
                length = int(window.shape[1])
                dt = torch.full(
                    (1, length, 1),
                    1.0 / float(length),
                    device=device,
                    dtype=window.dtype,
                )
                prediction = model(window, dt=dt)[0].detach().cpu()
                if prediction.shape != source[start:end].shape:
                    raise RuntimeError(
                        f"Prediction shape mismatch at {start}:{end}: "
                        f"{tuple(prediction.shape)} vs {tuple(source[start:end].shape)}"
                    )
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite prediction at {start}:{end}")
                prediction_sum[start:end] += prediction
                prediction_count[start:end] += 1.0
    missing = torch.nonzero(prediction_count[:, 0, 0, 0] <= 0).flatten().tolist()
    if missing:
        raise RuntimeError(f"Inference did not cover slices: {missing[:16]}")
    return prediction_sum / prediction_count


def evaluate_case_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    sequence_length: int = 7,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]]]:
    per_case: List[Dict[str, float]] = []
    per_slice: List[Dict[str, float]] = []
    for index in range(len(dataset)):
        inp, target, modality, case_key, names = dataset[index]
        if not str(case_key).startswith(f"{modality}:"):
            raise RuntimeError(f"Case key and modality disagree: {modality} {case_key}")
        prediction = _restore_scale(infer_case(model, inp, device, sequence_length), modality)
        reference = _restore_scale(target.float(), modality)
        if prediction.shape != reference.shape or len(names) != int(reference.shape[0]):
            raise RuntimeError(f"Evaluation alignment changed for {case_key}")
        rows = []
        for slice_index, name in enumerate(names):
            pred_slice = prediction[slice_index : slice_index + 1]
            ref_slice = reference[slice_index : slice_index + 1]
            data_range = float((ref_slice.max() - ref_slice.min()).item())
            if not data_range > 0.0:
                raise RuntimeError(f"Zero target range for {case_key} slice {name}")
            psnr, ssim, rmse = compute_measure(pred_slice, ref_slice, data_range)
            row = {
                "case": str(case_key),
                "slice": str(name),
                "psnr": float(psnr),
                "ssim": float(ssim),
                "rmse": float(rmse),
            }
            if not all(np.isfinite(row[key]) for key in ("psnr", "ssim", "rmse")):
                raise FloatingPointError(f"Non-finite metric for {case_key} slice {name}")
            rows.append(row)
            per_slice.append(row)
        per_case.append(
            {
                "case": str(case_key),
                "slices": int(len(rows)),
                "psnr": float(np.mean([row["psnr"] for row in rows])),
                "ssim": float(np.mean([row["ssim"] for row in rows])),
                "rmse": float(np.mean([row["rmse"] for row in rows])),
            }
        )
    summary = {
        "cases": int(len(per_case)),
        "psnr": float(np.mean([row["psnr"] for row in per_case])),
        "ssim": float(np.mean([row["ssim"] for row in per_case])),
        "rmse": float(np.mean([row["rmse"] for row in per_case])),
    }
    return summary, per_case, per_slice
