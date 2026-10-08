#!/usr/bin/env python3
# Editor: Jialei.He
"""Single-GPU and torchrun-DDP training for the released protocols."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .losses import contilnn_objective


HISTORY_FIELDS = (
    "epoch",
    "train_total",
    "train_reconstruction",
    "train_spatial_reconstruction",
    "train_fourier_reconstruction",
    "train_consistency",
    "train_first_order",
    "train_second_order",
    "train_distillation",
    "validation_cases",
    "validation_psnr",
    "validation_ssim",
    "validation_rmse",
    "learning_rate_base",
    "learning_rate_operator",
    "learning_rate_gate",
)


@dataclass(frozen=True)
class TrainingSettings:
    """Validated settings resolved from a protocol and explicit CLI values."""

    experiment: str
    backbone: str
    modality: str
    sequence_length: int
    max_epochs: int
    scheduler_interval: str
    precision: str
    num_workers: int
    seed: int
    learning_rate_base: float
    learning_rate_operator: float
    learning_rate_gate: float
    learning_rate_end: float
    weight_decay: float
    gradient_clip_norm: float
    consistency_weight: float
    second_order_multiplier: float
    distillation_weight: float
    fourier_weight: float
    early_stopping_start_epoch: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        positive_integers = {
            "sequence_length": self.sequence_length,
            "max_epochs": self.max_epochs,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.num_workers, bool) or int(self.num_workers) < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not 0 <= int(self.seed) < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.scheduler_interval not in {"epoch", "optimizer_step"}:
            raise ValueError(
                "scheduler_interval must be 'epoch' or 'optimizer_step'"
            )
        if self.precision not in {"fp32", "amp_fp16"}:
            raise ValueError("precision must be 'fp32' or 'amp_fp16'")
        for name in (
            "learning_rate_base",
            "learning_rate_operator",
            "learning_rate_gate",
            "learning_rate_end",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_decay",
            "consistency_weight",
            "second_order_multiplier",
            "distillation_weight",
            "fourier_weight",
            "early_stopping_min_delta",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        enabled = self.early_stopping_start_epoch is not None
        if enabled != (self.early_stopping_patience is not None):
            raise ValueError("Early-stopping start and patience must be provided together")
        if enabled:
            if int(self.early_stopping_start_epoch) <= 0:
                raise ValueError("early_stopping_start_epoch must be positive")
            if int(self.early_stopping_patience) <= 0:
                raise ValueError("early_stopping_patience must be positive")


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative, got {value}")
    return value


def initialize_distributed(
    device_specification: str,
    requires_cuda: bool,
) -> DistributedContext:
    """Initialize from torchrun environment variables, or select one local device."""

    world_size = _environment_integer("WORLD_SIZE", 1)
    rank = _environment_integer("RANK", 0)
    local_rank = _environment_integer("LOCAL_RANK", 0)
    if world_size == 0 or rank >= world_size:
        raise RuntimeError(f"Invalid distributed ranks: rank={rank}, world_size={world_size}")
    value = str(device_specification).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if world_size > 1 and value.startswith("cuda:"):
        raise RuntimeError(
            "Do not pass an indexed CUDA device under torchrun; LOCAL_RANK selects the device"
        )
    if world_size > 1 and value == "cuda":
        value = f"cuda:{local_rank}"
    device = torch.device(value)
    if requires_cuda and device.type != "cuda":
        raise RuntimeError("This backbone requires CUDA training")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    owns_process_group = False
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=6),
        )
        owns_process_group = True
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("WORLD_SIZE > 1 but torch.distributed is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(rank, local_rank, world_size, device, owns_process_group)


def _seed_all(seed: int, rank: int) -> None:
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def seed_model_initialization(seed: int) -> None:
    """Seed wrapper/operator construction before any model object is created."""

    if isinstance(seed, bool) or not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    _seed_all(int(seed), rank=0)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_optimizer_groups(
    model: torch.nn.Module, settings: TrainingSettings
) -> List[Dict[str, Any]]:
    """Build the required base/operator/gate parameter groups."""

    builder = getattr(model, "trainable_param_groups", None)
    if not callable(builder):
        raise TypeError("Model must implement trainable_param_groups")
    groups = builder(
        settings.learning_rate_base,
        settings.learning_rate_operator,
        settings.learning_rate_gate,
    )
    if not isinstance(groups, list):
        raise TypeError("trainable_param_groups must return a list")

    names = [str(group.get("name")) for group in groups]
    if names != ["base", "operator", "gate"]:
        raise RuntimeError(
            "Optimizer groups must be ordered exactly as base/operator/gate; "
            f"got {names}"
        )
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    seen = []
    for group in groups:
        parameters = list(group["params"])
        if not parameters:
            raise RuntimeError(f"Optimizer group {group['name']} is empty")
        group["params"] = parameters
        seen.extend(id(parameter) for parameter in parameters)
    if len(seen) != len(set(seen)):
        raise RuntimeError("A trainable parameter appears in multiple optimizer groups")
    if set(seen) != expected:
        raise RuntimeError(
            "Optimizer groups do not cover the complete trainable model: "
            f"grouped={len(set(seen))} expected={len(expected)}"
        )
    return groups


def teacher_prediction(
    teacher: torch.nn.Module, inputs: torch.Tensor
) -> torch.Tensor:
    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("A frozen Stage-A teacher module is required")
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("Teacher input must be a tensor")
    if inputs.ndim != 5:
        raise ValueError(f"Teacher input must be [B,T,C,H,W], got {tuple(inputs.shape)}")
    if not inputs.is_floating_point():
        raise TypeError("Teacher input must be a floating-point tensor")
    if not bool(torch.isfinite(inputs).all()):
        raise FloatingPointError("Teacher input contains non-finite values")
    batch, length, channels, height, width = inputs.shape
    with torch.no_grad():
        output = teacher(inputs.reshape(batch * length, channels, height, width))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Teacher must return a tensor, got {type(output)!r}")
    expected = (batch * length, channels, height, width)
    if tuple(output.shape) != expected:
        raise RuntimeError(
            f"Teacher output shape changed: got={tuple(output.shape)} expected={expected}"
        )
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("Teacher output contains non-finite values")
    return output.reshape(batch, length, channels, height, width)


def train_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    settings: TrainingSettings,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    model.train()
    teacher.eval()
    totals: Dict[str, float] = {}
    batches = 0
    amp_enabled = settings.precision == "amp_fp16"
    if amp_enabled and device.type != "cuda":
        raise RuntimeError("amp_fp16 training requires CUDA")
    for inputs, target, interval, _case_key in loader:
        if not all(
            isinstance(value, torch.Tensor) for value in (inputs, target, interval)
        ):
            raise TypeError("Training batches must contain tensor inputs, targets, and intervals")
        inputs = inputs.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        interval = interval.to(device=device, non_blocking=True)
        if inputs.ndim != 5 or target.shape != inputs.shape:
            raise ValueError(
                "Training tensors must have matching [B,T,C,H,W] shapes; "
                f"got inputs={tuple(inputs.shape)} target={tuple(target.shape)}"
            )
        batch, length, channels, height, width = inputs.shape
        if batch != 1 or not 1 <= length <= settings.sequence_length:
            raise ValueError(
                "Training batches must contain one window of at most the configured "
                f"sequence length; got B={batch}, T={length}"
            )
        if channels != 1 or height != 128 or width != 128:
            raise ValueError(
                "Training tensors must have one channel and 128x128 spatial patches; "
                f"got C={channels}, H={height}, W={width}"
            )
        if not all(value.is_floating_point() for value in (inputs, target, interval)):
            raise TypeError("Training inputs, targets, and intervals must be floating point")
        if interval.shape != (inputs.shape[0], inputs.shape[1], 1):
            raise ValueError(
                "Slice-index intervals must have shape [B,T,1]; "
                f"got {tuple(interval.shape)}"
            )
        if not bool(torch.isfinite(inputs).all()) or not bool(
            torch.isfinite(target).all()
        ):
            raise FloatingPointError("Training input or target contains non-finite values")
        if not bool(torch.isfinite(interval).all()) or not bool((interval > 0).all()):
            raise ValueError("Slice-index intervals must be finite and positive")
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(inputs, dt=interval)
            teacher_output = teacher_prediction(teacher, inputs)
            terms = contilnn_objective(
                prediction=prediction,
                target=target,
                teacher_prediction=teacher_output,
                backbone=settings.backbone,
                consistency_weight=settings.consistency_weight,
                second_order_multiplier=settings.second_order_multiplier,
                distillation_weight=settings.distillation_weight,
                fourier_weight=settings.fourier_weight,
            )
        scaler.scale(terms["total"]).backward()
        scaler.unscale_(optimizer)
        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(
                "Trainable parameters did not receive gradients: "
                + ", ".join(missing_gradients[:8])
            )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            settings.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        if settings.scheduler_interval == "optimizer_step":
            scheduler.step()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item())
        batches += 1
    if batches == 0:
        raise RuntimeError("Training dataset produced no batches")
    return {name: value / float(batches) for name, value in totals.items()}


def _distributed_mean(
    values: Mapping[str, float], context: DistributedContext
) -> Dict[str, float]:
    if context.world_size == 1:
        return {name: float(value) for name, value in values.items()}
    names = sorted(values)
    tensor = torch.tensor(
        [float(values[name]) for name in names],
        device=context.device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(context.world_size)
    return {name: float(value) for name, value in zip(names, tensor.cpu().tolist())}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _checkpoint_training_config(settings: TrainingSettings) -> Dict[str, Any]:
    """Return the scientific settings recorded with a model checkpoint."""

    return {
        "experiment": settings.experiment,
        "backbone": settings.backbone,
        "modality": settings.modality,
        "sequence_length": settings.sequence_length,
        "batch_size": 1,
        "max_epochs": settings.max_epochs,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_interval": settings.scheduler_interval,
        "precision": settings.precision,
        "learning_rate_base": settings.learning_rate_base,
        "learning_rate_operator": settings.learning_rate_operator,
        "learning_rate_gate": settings.learning_rate_gate,
        "learning_rate_end": settings.learning_rate_end,
        "weight_decay": settings.weight_decay,
        "gradient_clip_norm": settings.gradient_clip_norm,
        "consistency_weight": settings.consistency_weight,
        "second_order_multiplier": settings.second_order_multiplier,
        "distillation_weight": settings.distillation_weight,
        "fourier_weight": settings.fourier_weight,
        "early_stopping_start_epoch": settings.early_stopping_start_epoch,
        "early_stopping_patience": settings.early_stopping_patience,
        "early_stopping_min_delta": settings.early_stopping_min_delta,
}


def run_training(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    train_dataset,
    validation_dataset,
    settings: TrainingSettings,
    output_dir: Path,
    device_specification: str = "auto",
    requires_cuda: bool = False,
) -> Optional[Dict[str, Any]]:
    """Train one protocol run; return a summary on rank zero only."""

    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("A frozen Stage-A teacher module is required")
    context = initialize_distributed(device_specification, requires_cuda=requires_cuda)
    try:
        _seed_all(settings.seed, context.rank)
        output_dir = Path(output_dir).resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Training refuses a non-empty output directory: {output_dir}"
            )
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        context.barrier()

        model = model.to(context.device)
        teacher = teacher.to(context.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        groups = build_optimizer_groups(model, settings)
        optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
        sampler = None
        if context.world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=settings.seed,
                drop_last=False,
            )
        generator = torch.Generator()
        generator.manual_seed(settings.seed + context.rank)
        loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=settings.num_workers,
            pin_memory=context.device.type == "cuda",
            drop_last=False,
            worker_init_fn=_seed_worker,
            generator=generator,
            # set_epoch() rebuilds window order and crop coordinates. Recreating
            # workers each epoch ensures they receive the updated dataset copy.
            persistent_workers=False,
        )
        scheduler_horizon = settings.max_epochs
        if settings.scheduler_interval == "optimizer_step":
            scheduler_horizon *= len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_horizon,
            eta_min=settings.learning_rate_end,
        )
        if settings.precision == "amp_fp16" and context.device.type != "cuda":
            raise RuntimeError("amp_fp16 training requires CUDA")
        scaler = torch.cuda.amp.GradScaler(
            enabled=settings.precision == "amp_fp16"
        )

        start_epoch = 1
        best_psnr = float("-inf")
        early_reference = float("-inf")
        bad_epochs = 0
        history: List[Dict[str, Any]] = []

        training_model: torch.nn.Module = model
        if context.world_size > 1:
            kwargs: Dict[str, Any] = {}
            if context.device.type == "cuda":
                kwargs.update(
                    device_ids=[context.device.index], output_device=context.device.index
                )
            training_model = DistributedDataParallel(model, **kwargs)

        from ..evaluation import evaluate_case_dataset

        set_epoch = getattr(train_dataset, "set_epoch", None)
        if not callable(set_epoch):
            raise TypeError("The training dataset must implement set_epoch(epoch)")

        stopped_early = False
        for epoch in range(start_epoch, settings.max_epochs + 1):
            set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            learning_rates = {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            }
            train_terms = train_one_epoch(
                training_model,
                teacher,
                loader,
                optimizer,
                context.device,
                settings,
                scheduler=scheduler,
                scaler=scaler,
            )
            train_terms = _distributed_mean(train_terms, context)

            stop = False
            raw_improvement = False
            if context.is_main:
                summary, _per_case, _per_slice = evaluate_case_dataset(
                    model=_unwrap(training_model),
                    dataset=validation_dataset,
                    device=context.device,
                    sequence_length=settings.sequence_length,
                )
                validation_psnr = float(summary["psnr"])
                row = {
                    "epoch": epoch,
                    **{
                        f"train_{name}": float(train_terms[name])
                        for name in (
                            "total",
                            "reconstruction",
                            "spatial_reconstruction",
                            "fourier_reconstruction",
                            "consistency",
                            "first_order",
                            "second_order",
                            "distillation",
                        )
                    },
                    "validation_cases": int(summary["cases"]),
                    "validation_psnr": validation_psnr,
                    "validation_ssim": float(summary["ssim"]),
                    "validation_rmse": float(summary["rmse"]),
                    "learning_rate_base": learning_rates["base"],
                    "learning_rate_operator": learning_rates["operator"],
                    "learning_rate_gate": learning_rates["gate"],
                }
                history.append(row)
                raw_improvement = validation_psnr > best_psnr
                if raw_improvement:
                    best_psnr = validation_psnr

                early_improvement = validation_psnr > (
                    early_reference + settings.early_stopping_min_delta
                )
                if early_improvement:
                    early_reference = validation_psnr
                    bad_epochs = 0
                elif (
                    settings.early_stopping_start_epoch is not None
                    and epoch >= settings.early_stopping_start_epoch
                ):
                    bad_epochs += 1
                if (
                    settings.early_stopping_start_epoch is not None
                    and epoch >= settings.early_stopping_start_epoch
                ):
                    stop = bad_epochs >= int(settings.early_stopping_patience)

            if settings.scheduler_interval == "epoch":
                scheduler.step()

            if context.world_size > 1:
                signal = torch.tensor(
                    [1 if stop else 0], device=context.device, dtype=torch.int32
                )
                dist.broadcast(signal, src=0)
                stop = bool(signal.item())

            if context.is_main:
                checkpoint = {
                    "format": "contilnn-training-v2",
                    "experiment": settings.experiment,
                    "backbone": settings.backbone,
                    "modality": settings.modality,
                    "epoch": epoch,
                    "best_validation_psnr": best_psnr,
                    "early_reference": early_reference,
                    "bad_epochs": bad_epochs,
                    "model": _unwrap(training_model).state_dict(),
                    "history": history,
                    "training": _checkpoint_training_config(settings),
                }
                if raw_improvement:
                    _save_checkpoint(output_dir / "best.pth", checkpoint)
                _write_history(output_dir / "history.csv", history)
            context.barrier()
            if stop:
                stopped_early = True
                break

        if not context.is_main:
            return None
        return {
            "status": "PASS",
            "experiment": settings.experiment,
            "modality": settings.modality,
            "epochs_completed": int(history[-1]["epoch"]) if history else start_epoch - 1,
            "best_validation_psnr": best_psnr,
            "stopped_early": stopped_early,
            "best_checkpoint": "best.pth",
            "history": "history.csv",
        }
    finally:
        context.close()


__all__ = [
    "DistributedContext",
    "TrainingSettings",
    "build_optimizer_groups",
    "initialize_distributed",
    "run_training",
    "seed_model_initialization",
    "teacher_prediction",
    "train_one_epoch",
]
