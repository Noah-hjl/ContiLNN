#!/usr/bin/env python3
# Editor: Jialei.He
"""Load and validate the ContiLNN experiment protocols."""

from __future__ import annotations

import json
import math
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union


PathLike = Union[str, Path]

PROTOCOL_ALIASES = {
    "rwkv": Path("01_rwkv_main") / "protocol.json",
    "contilnn-rwkv": Path("01_rwkv_main") / "protocol.json",
    "dasmamba": Path("02_dasmamba_transfer") / "protocol.json",
    "contilnn-dasmamba": Path("02_dasmamba_transfer") / "protocol.json",
}
EXPECTED_MODALITIES = ("CT", "MRI", "PET")
EXPECTED_EXPERIMENTS = {
    "contilnn_rwkv_main": (1, "Restore-RWKV"),
    "contilnn_dasmamba_transfer": (2, "DASMamba"),
}


class ProtocolError(ValueError):
    """Raised when a protocol is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ProtocolConfig:
    """A validated protocol and its source path."""

    path: Path
    payload: Mapping[str, Any]

    @property
    def experiment(self) -> str:
        return str(self.payload["experiment"])

    @property
    def backbone(self) -> str:
        return str(self.payload["backbone"])

    @property
    def sequence_length(self) -> int:
        return int(self.payload["data"]["sequence_length"])

    def modality(self, name: str) -> Mapping[str, Any]:
        value = str(name).strip().upper()
        if value not in EXPECTED_MODALITIES:
            raise ProtocolError(
                f"Unsupported modality {name!r}; choose one of {EXPECTED_MODALITIES}"
            )
        return self.payload["modalities"][value]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible copy of the protocol."""

        return json.loads(json.dumps(self.payload))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{label} must be a positive integer")
    return int(value)


def _require_finite_number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < float(minimum):
        raise ProtocolError(f"{label} must be finite and at least {minimum}")
    return number


def _require_keys(
    value: Mapping[str, Any],
    label: str,
    required,
    optional=(),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    unexpected = set(value) - allowed
    if missing or unexpected:
        raise ProtocolError(
            f"{label} fields changed: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ProtocolError(f"{label} must be {expected!r}, got {value!r}")


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    experiment = payload.get("experiment")
    if experiment not in EXPECTED_EXPERIMENTS:
        raise ProtocolError(
            f"Unsupported experiment {experiment!r}; expected one of "
            f"{tuple(EXPECTED_EXPERIMENTS)}"
        )
    expected_order, expected_backbone = EXPECTED_EXPERIMENTS[str(experiment)]
    experiment_fields = {
        "contilnn_rwkv_main": (
            "paper_order",
            "experiment",
            "backbone",
            "operator",
            "num_seeds",
            "model",
            "modalities",
            "data",
            "stage_a",
            "stage_b",
            "objective",
        ),
        "contilnn_dasmamba_transfer": (
            "paper_order",
            "experiment",
            "backbone",
            "operator",
            "num_seeds",
            "model",
            "modalities",
            "data",
            "training",
        ),
    }
    _require_keys(
        payload,
        "protocol",
        experiment_fields[str(experiment)],
        optional=("editor",),
    )
    if payload.get("paper_order") != expected_order:
        raise ProtocolError(
            f"paper_order for {experiment} must be {expected_order}"
        )
    if payload.get("backbone") != expected_backbone:
        raise ProtocolError(
            f"backbone for {experiment} must be {expected_backbone!r}"
        )
    if payload.get("operator") != "bidirectional_cfc":
        raise ProtocolError("operator must be bidirectional_cfc")
    if _require_positive_int(payload.get("num_seeds"), "num_seeds") != 5:
        raise ProtocolError("num_seeds must be 5 for the reported protocols")

    modalities = _require_mapping(payload.get("modalities"), "modalities")
    if set(modalities) != set(EXPECTED_MODALITIES):
        raise ProtocolError(
            f"modalities must contain exactly {EXPECTED_MODALITIES}; got {tuple(modalities)}"
        )
    for modality in EXPECTED_MODALITIES:
        record = _require_mapping(modalities[modality], f"modalities.{modality}")
        _require_keys(
            record,
            f"modalities.{modality}",
            ("train_cases", "validation_cases", "test_cases", "epoch_cap"),
        )
        for field in ("train_cases", "validation_cases", "test_cases"):
            _require_positive_int(record.get(field), f"modalities.{modality}.{field}")
        _require_positive_int(record.get("epoch_cap"), f"modalities.{modality}.epoch_cap")

    data = _require_mapping(payload.get("data"), "data")
    _require_keys(
        data,
        "data",
        (
            "sequence_length",
            "patch_size",
            "slice_index_gap",
            "retain_short_tail",
            "evaluation_offsets",
            "overlap_fusion",
        ),
    )
    if _require_positive_int(data.get("sequence_length"), "data.sequence_length") != 7:
        raise ProtocolError("data.sequence_length must be 7")
    if _require_positive_int(data.get("patch_size"), "data.patch_size") != 128:
        raise ProtocolError("data.patch_size must be 128")
    if _require_positive_int(data.get("slice_index_gap"), "data.slice_index_gap") != 1:
        raise ProtocolError("data.slice_index_gap must be 1 for the primary protocols")
    if data.get("retain_short_tail") is not True:
        raise ProtocolError("data.retain_short_tail must be true")
    offsets = data.get("evaluation_offsets")
    if offsets != [0, 3]:
        raise ProtocolError("data.evaluation_offsets must be [0, 3]")
    if data.get("overlap_fusion") != "mean":
        raise ProtocolError("data.overlap_fusion must be mean")

    model = _require_mapping(payload.get("model"), "model")
    _require_keys(model, "model", ("dim", "cfc_hidden"))
    _require_exact(_require_positive_int(model.get("dim"), "model.dim"), 48, "model.dim")
    _require_exact(
        _require_positive_int(model.get("cfc_hidden"), "model.cfc_hidden"),
        256,
        "model.cfc_hidden",
    )

    if experiment == "contilnn_rwkv_main":
        stage_a = _require_mapping(payload.get("stage_a"), "stage_a")
        _require_keys(
            stage_a,
            "stage_a",
            (
                "updates",
                "batch_size_slices",
                "optimizer",
                "objective",
                "learning_rate_start",
                "learning_rate_end",
                "checkpoint_selection",
            ),
        )
        _require_positive_int(stage_a.get("updates"), "stage_a.updates")
        _require_positive_int(
            stage_a.get("batch_size_slices"), "stage_a.batch_size_slices"
        )
        _require_exact(stage_a.get("optimizer"), "Adam", "stage_a.optimizer")
        _require_exact(stage_a.get("objective"), "L1", "stage_a.objective")
        _require_finite_number(
            stage_a.get("learning_rate_start"), "stage_a.learning_rate_start", 1e-300
        )
        _require_finite_number(
            stage_a.get("learning_rate_end"), "stage_a.learning_rate_end", 1e-300
        )
        _require_exact(
            stage_a.get("checkpoint_selection"),
            "highest_validation_psnr",
            "stage_a.checkpoint_selection",
        )
        training = _require_mapping(payload.get("stage_b"), "stage_b")
        objective = _require_mapping(payload.get("objective"), "objective")
        _require_keys(
            training,
            "stage_b",
            (
                "optimizer",
                "scheduler",
                "scheduler_interval",
                "precision",
                "batch_size",
                "weight_decay",
                "learning_rate_backbone",
                "learning_rate_operator",
                "learning_rate_gate",
                "learning_rate_end",
                "gradient_clip_norm",
                "early_stopping_start_epoch",
                "early_stopping_patience",
                "early_stopping_min_delta_db",
                "checkpoint_selection",
            ),
        )
        _require_keys(
            objective,
            "objective",
            (
                "reconstruction",
                "consistency",
                "second_order_multiplier",
                "teacher_distillation",
            ),
        )
    else:
        training = _require_mapping(payload.get("training"), "training")
        objective = training
        _require_keys(
            training,
            "training",
            (
                "optimizer",
                "scheduler",
                "scheduler_interval",
                "precision",
                "batch_size",
                "initialization",
                "teacher",
                "retain_native_spatial_loss",
                "retain_native_fourier_loss",
                "native_fourier_weight",
                "consistency",
                "second_order_multiplier",
                "teacher_distillation",
                "checkpoint_selection",
            ),
            (
                "learning_rate_backbone",
                "learning_rate_operator",
                "learning_rate_gate",
                "learning_rate_end",
                "weight_decay",
                "gradient_clip_norm",
                "early_stopping_start_epoch",
                "early_stopping_patience",
                "early_stopping_min_delta_db",
            ),
        )
    if training.get("optimizer") != "AdamW":
        raise ProtocolError("The training optimizer must be AdamW")
    if training.get("scheduler") != "CosineAnnealingLR":
        raise ProtocolError("The public training scheduler must be CosineAnnealingLR")
    expected_interval = "epoch" if experiment == "contilnn_rwkv_main" else "optimizer_step"
    _require_exact(
        training.get("scheduler_interval"),
        expected_interval,
        "training.scheduler_interval",
    )
    expected_precision = "amp_fp16" if experiment == "contilnn_rwkv_main" else "fp32"
    _require_exact(
        training.get("precision"), expected_precision, "training.precision"
    )
    if training.get("batch_size") != 1:
        raise ProtocolError("Training batch_size must be 1 to retain variable short tails")
    if "weight_decay" in training:
        _require_finite_number(
            training["weight_decay"], "training.weight_decay"
        )
    for field in (
        "learning_rate_backbone",
        "learning_rate_operator",
        "learning_rate_gate",
        "learning_rate_end",
        "gradient_clip_norm",
    ):
        if field in training:
            _require_finite_number(training[field], f"training.{field}", 1e-300)
    early_start = training.get("early_stopping_start_epoch")
    early_patience = training.get("early_stopping_patience")
    if (early_start is None) != (early_patience is None):
        raise ProtocolError("Early-stopping start and patience must be provided together")
    if early_start is not None:
        _require_positive_int(early_start, "training.early_stopping_start_epoch")
        _require_positive_int(early_patience, "training.early_stopping_patience")
        _require_finite_number(
            training.get("early_stopping_min_delta_db", 0.0),
            "training.early_stopping_min_delta_db",
        )
    for field in ("consistency", "second_order_multiplier", "teacher_distillation"):
        _require_finite_number(objective.get(field), f"objective.{field}")
    _require_exact(float(objective["consistency"]), 0.30, "objective.consistency")
    _require_exact(
        float(objective["second_order_multiplier"]),
        2.0,
        "objective.second_order_multiplier",
    )
    _require_exact(
        float(objective["teacher_distillation"]),
        0.05,
        "objective.teacher_distillation",
    )
    if experiment == "contilnn_rwkv_main":
        _require_exact(float(objective["reconstruction"]), 1.0, "objective.reconstruction")
        _require_exact(
            training.get("checkpoint_selection"),
            "highest_validation_psnr",
            "stage_b.checkpoint_selection",
        )
    else:
        _require_exact(
            training.get("initialization"),
            "highest_validation_psnr_backbone_checkpoint",
            "training.initialization",
        )
        _require_exact(
            training.get("teacher"),
            "frozen_matching_backbone",
            "training.teacher",
        )
        _require_exact(
            training.get("retain_native_spatial_loss"),
            True,
            "training.retain_native_spatial_loss",
        )
        _require_exact(
            training.get("retain_native_fourier_loss"),
            True,
            "training.retain_native_fourier_loss",
        )
        _require_exact(
            _require_finite_number(
                training.get("native_fourier_weight"),
                "training.native_fourier_weight",
            ),
            1.0,
            "training.native_fourier_weight",
        )
        _require_exact(
            training.get("checkpoint_selection"),
            "highest_validation_psnr",
            "training.checkpoint_selection",
        )


def _built_in_candidates(relative: Path):
    # Editable/source checkout.
    yield Path(__file__).resolve().parents[2] / "configs" / relative
    # Wheel installation via setuptools data-files.
    yield Path(sysconfig.get_path("data")) / "share" / "contilnn" / "configs" / relative


def resolve_protocol_path(value: PathLike) -> Path:
    """Resolve an explicit path or one of the bundled protocol aliases."""

    raw = str(value).strip()
    alias = raw.lower()
    if alias in PROTOCOL_ALIASES:
        candidates = list(_built_in_candidates(PROTOCOL_ALIASES[alias]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Bundled protocol {raw!r} was not found; checked "
            + ", ".join(str(path) for path in candidates)
        )

    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")
    return path.resolve()


def load_protocol(value: PathLike) -> ProtocolConfig:
    """Resolve, parse, and strictly validate a protocol JSON file."""

    path = resolve_protocol_path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON in {path}: {exc}") from exc
    mapping = _require_mapping(payload, "protocol")
    _validate_protocol(mapping)
    return ProtocolConfig(path=path, payload=mapping)


__all__ = [
    "EXPECTED_MODALITIES",
    "PROTOCOL_ALIASES",
    "ProtocolConfig",
    "ProtocolError",
    "load_protocol",
    "resolve_protocol_path",
]
