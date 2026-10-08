#!/usr/bin/env python3
# Editor: Jialei.He
"""Command-line entry points for protocol inspection, training, and evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .config import EXPECTED_MODALITIES, ProtocolConfig, ProtocolError, load_protocol


def _reader_classes(modality: str):
    """Select one modality-specific reader without a cross-modality fallback."""

    value = str(modality).strip().upper()
    if value == "CT":
        from .data.ct import CtCaseDataset, CtSequenceTrainDataset

        return CtSequenceTrainDataset, CtCaseDataset
    if value == "MRI":
        from .data.mri import MriCaseDataset, MriSequenceTrainDataset

        return MriSequenceTrainDataset, MriCaseDataset
    if value == "PET":
        from .data.pet import PetCaseDataset, PetSequenceTrainDataset

        return PetSequenceTrainDataset, PetCaseDataset
    raise ProtocolError(
        f"Unsupported modality {modality!r}; choose one of {EXPECTED_MODALITIES}"
    )


def _load_factory(specification: str):
    if ":" not in specification:
        raise ValueError("A factory must use the form package.module:callable")
    module_name, attribute_name = specification.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("A factory must use the form package.module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"Factory is not callable: {specification}")
    return factory


def _build_model(protocol: ProtocolConfig, backbone_factory: Optional[str]):
    model_settings = protocol.payload["model"]
    dim = int(model_settings["dim"])
    cfc_hidden = int(model_settings["cfc_hidden"])
    if protocol.experiment == "contilnn_rwkv_main":
        if backbone_factory is not None:
            raise ValueError("--backbone-factory is only valid for the DASMamba protocol")
        from .models import ContiLNNRWKV

        return ContiLNNRWKV(dim=dim, cfc_hidden=cfc_hidden)
    if protocol.experiment == "contilnn_dasmamba_transfer":
        if backbone_factory is None:
            raise ValueError(
                "The DASMamba protocol requires --backbone-factory package.module:callable "
                "for the pinned upstream backbone"
            )
        from .models import ContiLNNDASMamba

        backbone = _load_factory(backbone_factory)()
        return ContiLNNDASMamba(backbone=backbone, dim=dim, cfc_hidden=cfc_hidden)
    raise ProtocolError(f"Unsupported experiment: {protocol.experiment}")


def _checkpoint_state(payload: Any) -> Mapping[str, Any]:
    import torch

    if not isinstance(payload, dict) or payload.get("format") != "contilnn-training-v2":
        raise TypeError("Evaluation requires a ContiLNN training-v2 checkpoint")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint model state is empty or invalid")
    if any(str(key).startswith("module.") for key in state):
        raise RuntimeError("Checkpoint model keys must not use a module. prefix")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint parameter {name!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise FloatingPointError(f"Checkpoint parameter {name!r} is non-finite")
    return state


def _load_checkpoint(
    model, checkpoint: Path, protocol: ProtocolConfig, modality: str
) -> None:
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu")
    state = _checkpoint_state(payload)
    expected = {
        "experiment": protocol.experiment,
        "backbone": protocol.backbone,
        "modality": str(modality).strip().upper(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"Checkpoint {name} must be {value!r}, got {payload.get(name)!r}"
            )
    model.load_state_dict(state, strict=True)


def _resolve_device(specification: str):
    import torch

    value = str(specification).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    return device


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(
    protocol: ProtocolConfig,
    data_root: Path,
    modality: str,
    split: str,
    checkpoint: Path,
    output_dir: Path,
    device_specification: str,
    backbone_factory: Optional[str],
) -> Dict[str, Any]:
    _, case_class = _reader_classes(modality)
    protocol.modality(modality)
    dataset = case_class(data_root=str(data_root), split=split)
    model = _build_model(protocol, backbone_factory)
    _load_checkpoint(model, checkpoint, protocol, modality)
    device = _resolve_device(device_specification)
    if protocol.experiment == "contilnn_rwkv_main" and device.type != "cuda":
        raise RuntimeError("ContiLNN-RWKV evaluation requires CUDA for its WKV operator")
    model = model.to(device)

    from .evaluation import evaluate_case_dataset

    summary, per_case, per_slice = evaluate_case_dataset(
        model=model,
        dataset=dataset,
        device=device,
        sequence_length=protocol.sequence_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "per_case.csv",
        per_case,
        ("case", "slices", "psnr", "ssim", "rmse"),
    )
    _write_csv(
        output_dir / "per_slice.csv",
        per_slice,
        ("case", "slice", "psnr", "ssim", "rmse"),
    )
    record = {
        "status": "PASS",
        "experiment": protocol.experiment,
        "backbone": protocol.backbone,
        "modality": str(modality).upper(),
        "split": split,
        "checkpoint": checkpoint.name,
        "protocol": protocol.experiment,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _training_mapping(protocol: ProtocolConfig) -> Mapping[str, Any]:
    key = "stage_b" if protocol.experiment == "contilnn_rwkv_main" else "training"
    value = protocol.payload.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"{key} must be a JSON object")
    return value


def _objective_mapping(protocol: ProtocolConfig) -> Mapping[str, Any]:
    key = "objective" if protocol.experiment == "contilnn_rwkv_main" else "training"
    value = protocol.payload.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"{key} must be a JSON object")
    return value


def _resolved_float(
    configured: Mapping[str, Any],
    key: str,
    override: Optional[float],
    option: str,
) -> float:
    if key in configured:
        if override is not None:
            raise ValueError(
                f"{option} cannot override the value fixed by the selected protocol"
            )
        return float(configured[key])
    if override is None:
        raise ValueError(
            f"{option} is required because {key!r} is not fixed by this protocol"
        )
    return float(override)


def _training_settings(protocol: ProtocolConfig, args: argparse.Namespace):
    from .training.engine import TrainingSettings

    training = _training_mapping(protocol)
    objective = _objective_mapping(protocol)
    modality = protocol.modality(args.modality)
    early_start = training.get("early_stopping_start_epoch")
    early_patience = training.get("early_stopping_patience")
    return TrainingSettings(
        experiment=protocol.experiment,
        backbone=protocol.backbone,
        modality=str(args.modality).upper(),
        sequence_length=protocol.sequence_length,
        max_epochs=int(modality["epoch_cap"]),
        scheduler_interval=str(training["scheduler_interval"]),
        precision=str(training["precision"]),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        learning_rate_base=_resolved_float(
            training,
            "learning_rate_backbone",
            args.learning_rate_base,
            "--learning-rate-base",
        ),
        learning_rate_operator=_resolved_float(
            training,
            "learning_rate_operator",
            args.learning_rate_operator,
            "--learning-rate-operator",
        ),
        learning_rate_gate=_resolved_float(
            training,
            "learning_rate_gate",
            args.learning_rate_gate,
            "--learning-rate-gate",
        ),
        learning_rate_end=_resolved_float(
            training,
            "learning_rate_end",
            args.learning_rate_end,
            "--learning-rate-end",
        ),
        weight_decay=_resolved_float(
            training, "weight_decay", args.weight_decay, "--weight-decay"
        ),
        gradient_clip_norm=_resolved_float(
            training,
            "gradient_clip_norm",
            args.gradient_clip_norm,
            "--gradient-clip-norm",
        ),
        consistency_weight=float(objective["consistency"]),
        second_order_multiplier=float(objective["second_order_multiplier"]),
        distillation_weight=float(objective["teacher_distillation"]),
        fourier_weight=float(objective.get("native_fourier_weight", 0.0)),
        early_stopping_start_epoch=(int(early_start) if early_start is not None else None),
        early_stopping_patience=(
            int(early_patience) if early_patience is not None else None
        ),
        early_stopping_min_delta=float(
            training.get("early_stopping_min_delta_db", 0.0)
        ),
    )


def _load_stage_a_and_teacher(
    model, protocol: ProtocolConfig, checkpoint: Path
):
    if protocol.experiment == "contilnn_rwkv_main":
        loader = getattr(model, "load_stageA_checkpoint", None)
    else:
        loader = getattr(model, "load_backbone_checkpoint", None)
    if not callable(loader):
        raise TypeError(
            f"{protocol.experiment} does not expose the required backbone checkpoint loader"
        )
    loader(str(checkpoint))
    return _frozen_teacher(model)


def _frozen_teacher(model):
    backbone = getattr(model, "base", None)
    if backbone is None:
        raise TypeError("ContiLNN wrapper does not expose its in-plane backbone as .base")
    teacher = copy.deepcopy(backbone)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _train(protocol: ProtocolConfig, args: argparse.Namespace):
    from .training.engine import run_training, seed_model_initialization

    train_class, case_class = _reader_classes(args.modality)
    expected = protocol.modality(args.modality)
    settings = _training_settings(protocol, args)
    data_root = args.data_root.expanduser().resolve()
    train_dataset = train_class(
        data_root=str(data_root),
        seq_len=protocol.sequence_length,
        patch_size=int(protocol.payload["data"]["patch_size"]),
        seed=settings.seed,
    )
    validation_dataset = case_class(data_root=str(data_root), split="validation")
    test_dataset = case_class(data_root=str(data_root), split="test")
    train_cases = getattr(
        train_dataset, "case_ids", getattr(train_dataset, "patient_ids", ())
    )
    observed = {
        "train_cases": len(train_cases),
        "validation_cases": len(validation_dataset),
        "test_cases": len(test_dataset),
    }
    required = {
        "train_cases": int(expected["train_cases"]),
        "validation_cases": int(expected["validation_cases"]),
        "test_cases": int(expected["test_cases"]),
    }
    if observed != required:
        raise RuntimeError(
            f"Protocol/reader cohort mismatch for {args.modality}: "
            f"observed={observed} expected={required}"
        )

    seed_model_initialization(settings.seed)
    model = _build_model(protocol, args.backbone_factory)
    stage_a = args.stage_a_checkpoint.expanduser().resolve()
    teacher = _load_stage_a_and_teacher(model, protocol, stage_a)
    return run_training(
        model=model,
        teacher=teacher,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        settings=settings,
        output_dir=args.output_dir.expanduser().resolve(),
        device_specification=args.device,
        requires_cuda=protocol.experiment == "contilnn_rwkv_main",
    )


def _add_protocol_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol",
        required=True,
        help="Protocol path or alias: rwkv, contilnn-rwkv, dasmamba, contilnn-dasmamba",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contilnn", description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0.dev0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-protocol", help="validate a protocol and print its settings"
    )
    _add_protocol_argument(inspect_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="run case-complete evaluation from a validation-selected checkpoint"
    )
    _add_protocol_argument(evaluate_parser)
    evaluate_parser.add_argument("--data-root", required=True, type=Path)
    evaluate_parser.add_argument("--modality", required=True, choices=EXPECTED_MODALITIES)
    evaluate_parser.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--output-dir", required=True, type=Path)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument(
        "--backbone-factory",
        help="Required only for DASMamba; package.module:callable returning the pinned backbone",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="train Stage B from an explicit validation-selected backbone checkpoint",
    )
    _add_protocol_argument(train_parser)
    train_parser.add_argument("--data-root", required=True, type=Path)
    train_parser.add_argument("--modality", required=True, choices=EXPECTED_MODALITIES)
    train_parser.add_argument(
        "--stage-a-checkpoint",
        required=True,
        type=Path,
        help="Validation-selected Stage-A backbone checkpoint",
    )
    train_parser.add_argument("--output-dir", required=True, type=Path)
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--learning-rate-base", type=float)
    train_parser.add_argument("--learning-rate-operator", type=float)
    train_parser.add_argument("--learning-rate-gate", type=float)
    train_parser.add_argument("--learning-rate-end", type=float)
    train_parser.add_argument("--weight-decay", type=float)
    train_parser.add_argument("--gradient-clip-norm", type=float)
    train_parser.add_argument(
        "--backbone-factory",
        help="Required only for DASMamba; package.module:callable returning the pinned backbone",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if args.command == "inspect-protocol":
            print(json.dumps(protocol.as_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            result = _evaluate(
                protocol=protocol,
                data_root=args.data_root.expanduser().resolve(),
                modality=args.modality,
                split=args.split,
                checkpoint=checkpoint,
                output_dir=args.output_dir.expanduser().resolve(),
                device_specification=args.device,
                backbone_factory=args.backbone_factory,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "train":
            stage_a = args.stage_a_checkpoint.expanduser().resolve()
            if not stage_a.is_file():
                raise FileNotFoundError(
                    f"Stage-A/backbone checkpoint not found: {stage_a}"
                )
            result = _train(protocol, args)
            if result is not None:
                print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        raise RuntimeError(f"Unhandled command: {args.command}")
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        ProtocolError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"contilnn: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
