#!/usr/bin/env python3
# Editor: Jialei.He
"""Loss decomposition used by the two reported backbone integrations."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


_SUPPORTED_BACKBONES = {"restore-rwkv", "dasmamba"}


def _non_negative_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _validate_volume_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Prediction and target must be tensors")
    if prediction.ndim != 5 or target.shape != prediction.shape:
        raise ValueError(
            "Prediction and target must have matching [B,T,C,H,W] shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if any(int(size) <= 0 for size in prediction.shape):
        raise ValueError(
            f"Prediction shape must be non-empty, got {tuple(prediction.shape)}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise FloatingPointError("Prediction or target contains non-finite values")


def _normalized_backbone(backbone: str) -> str:
    value = str(backbone).strip().lower()
    if value not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone must be one of {sorted(_SUPPORTED_BACKBONES)}, "
            f"got {backbone!r}"
        )
    return value


def _consistency_terms_validated(
    prediction: torch.Tensor,
    target: torch.Tensor,
    second_order_multiplier: float,
) -> Dict[str, torch.Tensor]:
    zero = prediction.new_zeros(())
    if prediction.shape[1] < 2:
        first = zero
    else:
        first = F.l1_loss(
            prediction[:, 1:] - prediction[:, :-1],
            target[:, 1:] - target[:, :-1],
        )
    if prediction.shape[1] < 3:
        second = zero
    else:
        second = F.l1_loss(
            prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2],
            target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2],
        )
    combined = first + second_order_multiplier * second
    return {"consistency": combined, "first_order": first, "second_order": second}


def _reconstruction_terms_validated(
    prediction: torch.Tensor,
    target: torch.Tensor,
    backbone: str,
    fourier_weight: float,
) -> Dict[str, torch.Tensor]:
    spatial = F.l1_loss(prediction.float(), target.float())
    if backbone == "dasmamba":
        spectral = torch.mean(
            torch.abs(
                torch.fft.fft2(prediction.float(), dim=(-2, -1))
                - torch.fft.fft2(target.float(), dim=(-2, -1))
            )
        )
        reconstruction = spatial + fourier_weight * spectral
    else:
        spectral = prediction.new_zeros(())
        reconstruction = spatial
    return {
        "reconstruction": reconstruction,
        "spatial_reconstruction": spatial,
        "fourier_reconstruction": spectral,
    }


def consistency_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    second_order_multiplier: float = 2.0,
) -> Dict[str, torch.Tensor]:
    _validate_volume_pair(prediction, target)
    second_order_multiplier = _non_negative_finite(
        "second_order_multiplier", second_order_multiplier
    )
    return _consistency_terms_validated(
        prediction, target, second_order_multiplier
    )


def reconstruction_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    backbone: str,
    fourier_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    _validate_volume_pair(prediction, target)
    normalized_backbone = _normalized_backbone(backbone)
    fourier_weight = _non_negative_finite("fourier_weight", fourier_weight)
    return _reconstruction_terms_validated(
        prediction, target, normalized_backbone, fourier_weight
    )


def contilnn_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher_prediction: torch.Tensor,
    backbone: str,
    consistency_weight: float = 0.30,
    second_order_multiplier: float = 2.0,
    distillation_weight: float = 0.05,
    fourier_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return the complete objective and its reported components."""

    _validate_volume_pair(prediction, target)
    if not isinstance(teacher_prediction, torch.Tensor):
        raise TypeError("A tensor-valued Stage-A teacher prediction is required")
    if teacher_prediction.shape != prediction.shape:
        raise ValueError(
            f"Teacher shape {tuple(teacher_prediction.shape)} does not match "
            f"student shape {tuple(prediction.shape)}"
        )
    if not teacher_prediction.is_floating_point():
        raise TypeError("Teacher prediction must be a floating-point tensor")
    if teacher_prediction.device != prediction.device:
        raise ValueError("Teacher and student predictions must be on the same device")
    if not bool(torch.isfinite(teacher_prediction).all()):
        raise FloatingPointError("Teacher prediction contains non-finite values")
    consistency_weight = _non_negative_finite(
        "consistency_weight", consistency_weight
    )
    distillation_weight = _non_negative_finite(
        "distillation_weight", distillation_weight
    )
    second_order_multiplier = _non_negative_finite(
        "second_order_multiplier", second_order_multiplier
    )
    fourier_weight = _non_negative_finite("fourier_weight", fourier_weight)
    normalized_backbone = _normalized_backbone(backbone)

    reconstruction = _reconstruction_terms_validated(
        prediction,
        target,
        backbone=normalized_backbone,
        fourier_weight=fourier_weight,
    )
    consistency = _consistency_terms_validated(
        prediction,
        target,
        second_order_multiplier=second_order_multiplier,
    )
    distillation = F.l1_loss(
        prediction.float(), teacher_prediction.detach().float()
    )

    total = (
        reconstruction["reconstruction"]
        + consistency_weight * consistency["consistency"]
        + distillation_weight * distillation
    )
    terms = {
        **reconstruction,
        **consistency,
        "distillation": distillation,
        "total": total,
    }
    nonfinite = [name for name, value in terms.items() if not bool(torch.isfinite(value).all())]
    if nonfinite:
        raise FloatingPointError(f"Non-finite objective terms: {nonfinite}")
    return terms
