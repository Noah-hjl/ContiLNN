#!/usr/bin/env python3
# Editor: Jialei.He
"""Position-aligned bidirectional CfC operators for ContiLNN-RWKV."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .restore_rwkv import Restore_RWKV

try:
    from ncps.torch.cfc_cell import CfCCell as NCPS_CfCCell
except Exception as exc:
    raise ImportError(
        "Failed to import ncps CfCCell. Install the dependencies from the "
        "selected ContiLNN environment file."
    ) from exc


def _strip_uniform_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize the standard prefix written by DistributedDataParallel."""

    keys = [str(key) for key in state]
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("Checkpoint mixes prefixed and unprefixed parameter keys")
    if all(prefixed):
        return {str(key)[7:]: value for key, value in state.items()}
    return {str(key): value for key, value in state.items()}


def _validate_intervals(
    dt: torch.Tensor,
    batch: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dt is None:
        raise ValueError("Slice-index intervals are required")
    if dt.ndim != 3 or tuple(dt.shape) != (batch, length, 1):
        raise ValueError(
            f"Slice-index intervals must be [B,T,1], got {tuple(dt.shape)}"
        )
    value = dt.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError("Slice-index intervals must be finite and positive")
    return value


class CfC1D_NCPS(nn.Module):
    """Apply one CfC cell along an explicitly aligned slice sequence."""

    def __init__(self, dim: int, backbone_units: int = 256) -> None:
        super().__init__()
        self.dim = int(dim)
        self.cell = NCPS_CfCCell(
            input_size=self.dim,
            hidden_size=self.dim,
            mode="default",
            backbone_activation="lecun_tanh",
            backbone_units=int(backbone_units),
            backbone_layers=1,
            backbone_dropout=0.0,
            sparsity_mask=None,
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"CfC input must be [N,T,C], got {tuple(x.shape)}")
        batch, length, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"CfC channel mismatch: got={channels} expected={self.dim}")
        interval = _validate_intervals(dt, batch, length, x.device, x.dtype)

        state = torch.zeros(batch, channels, device=x.device, dtype=x.dtype)
        outputs = []
        for position in range(length):
            state, _ = self.cell(x[:, position], state, interval[:, position])
            outputs.append(state.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class LatentLNN2p5D(nn.Module):
    """Bidirectional CfC over the slice axis at every spatial location."""

    def __init__(self, dim: int, cfc_hidden: int = 256) -> None:
        super().__init__()
        self.dim = int(dim)
        self.z_proj = nn.Linear(1, self.dim)
        self.core_fwd = CfC1D_NCPS(self.dim, backbone_units=int(cfc_hidden))
        self.core_bwd = CfC1D_NCPS(self.dim, backbone_units=int(cfc_hidden))
        self.bi_proj = nn.Linear(2 * self.dim, self.dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Bi-CfC input must be [B,T,C,H,W], got {tuple(x.shape)}")
        batch, length, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Bi-CfC channel mismatch: got={channels} expected={self.dim}"
            )
        interval = _validate_intervals(dt, batch, length, x.device, x.dtype)

        sequence = (
            x.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(batch * height * width, length, channels)
        )
        position = torch.linspace(
            0.0, 1.0, steps=length, device=x.device, dtype=x.dtype
        ).view(1, length, 1)
        position = position.expand(batch, length, 1)
        position = (
            position.unsqueeze(1)
            .expand(batch, height * width, length, 1)
            .reshape(batch * height * width, length, 1)
        )
        sequence = sequence + self.z_proj(position)

        interval = (
            interval.unsqueeze(1)
            .expand(batch, height * width, length, 1)
            .reshape(batch * height * width, length, 1)
        )
        forward = self.core_fwd(sequence, interval)
        reverse = torch.flip(sequence, dims=[1])
        reverse_interval = torch.flip(interval, dims=[1])
        backward = torch.flip(self.core_bwd(reverse, reverse_interval), dims=[1])
        fused = self.bi_proj(torch.cat([forward, backward], dim=-1))
        return (
            fused.view(batch, height, width, length, channels)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )


class RestoreRWKVRefine(nn.Module):
    """The manuscript-selected ContiLNN-RWKV topology.

    Bi-CfC modules are inserted at encoder level 2, decoder level 2, skip
    levels 2 and 3, the deepest bottleneck, and the refinement pathway. The
    bottleneck is serial; all remaining insertions are gated residual paths.
    """

    def __init__(self, dim: int = 48, cfc_hidden: int = 256) -> None:
        super().__init__()
        num_blocks = (4, 6, 6, 8)
        self.base = Restore_RWKV(
            inp_channels=1,
            out_channels=1,
            dim=int(dim),
            num_blocks=list(num_blocks),
            num_refinement_blocks=4,
        )
        self._freeze_base_parameters()

        def make_module(channels: int) -> LatentLNN2p5D:
            return LatentLNN2p5D(channels, cfc_hidden=int(cfc_hidden))

        self.encoder_l2 = nn.ModuleList(
            [make_module(int(dim) * 2) for _ in range(num_blocks[1])]
        )
        self.encoder_gate_l2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.encoder_l2]
        )
        self.decoder_l2 = nn.ModuleList(
            [make_module(int(dim) * 2) for _ in range(num_blocks[1])]
        )
        self.decoder_gate_l2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.decoder_l2]
        )

        self.skip_l2_lnn = make_module(int(dim) * 2)
        self.skip_l3_lnn = make_module(int(dim) * 4)
        self.skip_gate_l2 = nn.Parameter(torch.tensor(0.0))
        self.skip_gate_l3 = nn.Parameter(torch.tensor(0.0))

        self.lnn = make_module(int(dim) * 8)
        self.refine_lnn = make_module(int(dim) * 2)
        self.gate_refine_lnn = nn.Parameter(torch.tensor(-4.0))

    def _freeze_base_parameters(self) -> None:
        for module in self.base.modules():
            convolution = getattr(module, "conv5x5_reparam", None)
            if isinstance(convolution, nn.Module):
                for parameter in convolution.parameters():
                    parameter.requires_grad_(False)
            for name in ("spatial_decay", "spatial_first"):
                parameter = getattr(module, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)

    @staticmethod
    def _to_5d(value: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        flattened, channels, height, width = value.shape
        if flattened != batch * length:
            raise ValueError(
                f"Feature alignment changed: BK={flattened} B*T={batch * length}"
            )
        return value.view(batch, length, channels, height, width)

    @staticmethod
    def _to_4d(value: torch.Tensor) -> torch.Tensor:
        batch, length, channels, height, width = value.shape
        return value.view(batch * length, channels, height, width)

    def _apply_gated(
        self,
        value: torch.Tensor,
        batch: int,
        length: int,
        dt: torch.Tensor,
        module: LatentLNN2p5D,
        gate: nn.Parameter,
    ) -> torch.Tensor:
        source = self._to_5d(value, batch, length)
        transformed = module(source, dt)
        coefficient = torch.sigmoid(gate).to(device=value.device, dtype=value.dtype)
        return self._to_4d(source + coefficient * (transformed - source))

    def _apply_serial(
        self,
        value: torch.Tensor,
        batch: int,
        length: int,
        dt: torch.Tensor,
        module: LatentLNN2p5D,
    ) -> torch.Tensor:
        return self._to_4d(module(self._to_5d(value, batch, length), dt))

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Input must be [B,T,1,H,W], got {tuple(x.shape)}")
        batch, length, channels, height, width = x.shape
        if channels != 1:
            raise ValueError(f"ContiLNN-RWKV expects one input channel, got {channels}")
        if not 1 <= length <= 7:
            raise ValueError(f"ContiLNN-RWKV expects 1--7 ordered slices, got {length}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"Restore-RWKV input height and width must be divisible by 8, got {height}x{width}"
            )
        if (height // 8) * (width // 8) < 64:
            raise ValueError(
                "Restore-RWKV requires at least 64 spatial tokens at the deepest level"
            )
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("ContiLNN-RWKV input must be finite")
        interval = _validate_intervals(dt, batch, length, x.device, x.dtype)
        base = self.base
        flattened = x.view(batch * length, channels, height, width)

        encoder_l1 = base.encoder_level1(base.patch_embed(flattened))

        feature = base.down1_2(encoder_l1)
        for index, block in enumerate(base.encoder_level2):
            feature = block(feature)
            feature = self._apply_gated(
                feature,
                batch,
                length,
                interval,
                self.encoder_l2[index],
                self.encoder_gate_l2[index],
            )
        encoder_l2 = feature

        encoder_l3 = base.encoder_level3(base.down2_3(encoder_l2))
        latent = base.latent(base.down3_4(encoder_l3))
        latent = self._apply_serial(latent, batch, length, interval, self.lnn)

        skip_l3 = self._apply_gated(
            encoder_l3,
            batch,
            length,
            interval,
            self.skip_l3_lnn,
            self.skip_gate_l3,
        )
        feature = base.up4_3(latent)
        feature = base.reduce_chan_level3(torch.cat([feature, skip_l3], dim=1))
        decoder_l3 = base.decoder_level3(feature)

        skip_l2 = self._apply_gated(
            encoder_l2,
            batch,
            length,
            interval,
            self.skip_l2_lnn,
            self.skip_gate_l2,
        )
        feature = base.up3_2(decoder_l3)
        feature = base.reduce_chan_level2(torch.cat([feature, skip_l2], dim=1))
        for index, block in enumerate(base.decoder_level2):
            feature = block(feature)
            feature = self._apply_gated(
                feature,
                batch,
                length,
                interval,
                self.decoder_l2[index],
                self.decoder_gate_l2[index],
            )
        decoder_l2 = feature

        feature = base.up2_1(decoder_l2)
        feature = base.decoder_level1(torch.cat([feature, encoder_l1], dim=1))
        feature = base.refinement(feature)
        feature = self._apply_gated(
            feature,
            batch,
            length,
            interval,
            self.refine_lnn,
            self.gate_refine_lnn,
        )
        output = base.output(feature) + flattened
        return output.view(batch, length, channels, height, width)

    def load_stageA_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            raise RuntimeError("A Stage-B checkpoint cannot initialize the Stage-A backbone")
        else:
            state = payload
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"Invalid Restore-RWKV Stage-A checkpoint: {checkpoint_path}")
        state = _strip_uniform_module_prefix(state)
        if any(str(key).startswith("base.") for key in state):
            raise RuntimeError("A Stage-B checkpoint cannot initialize the Stage-A backbone")
        self.base.load_state_dict(state, strict=True)

    def base_parameters(self) -> Iterable[nn.Parameter]:
        return self.base.parameters()

    def operator_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        for modules in (self.encoder_l2, self.decoder_l2):
            for module in modules:
                parameters.extend(module.parameters())
        for module in (
            self.skip_l2_lnn,
            self.skip_l3_lnn,
            self.lnn,
            self.refine_lnn,
        ):
            parameters.extend(module.parameters())
        return parameters

    def gate_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        parameters.extend(self.encoder_gate_l2)
        parameters.extend(self.decoder_gate_l2)
        parameters.extend(
            [self.skip_gate_l2, self.skip_gate_l3, self.gate_refine_lnn]
        )
        return parameters

    def trainable_param_groups(
        self,
        learning_rate_base: float,
        learning_rate_operator: float,
        learning_rate_gate: float,
    ) -> List[Dict[str, object]]:
        return _complete_parameter_groups(
            self,
            self.base_parameters(),
            self.operator_parameters(),
            self.gate_parameters(),
            learning_rate_base,
            learning_rate_operator,
            learning_rate_gate,
        )


def _complete_parameter_groups(
    model: nn.Module,
    base: Iterable[nn.Parameter],
    operator: Iterable[nn.Parameter],
    gate: Iterable[nn.Parameter],
    learning_rate_base: float,
    learning_rate_operator: float,
    learning_rate_gate: float,
) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = [
        {
            "name": "base",
            "params": [parameter for parameter in base if parameter.requires_grad],
            "lr": float(learning_rate_base),
        },
        {
            "name": "operator",
            "params": [parameter for parameter in operator if parameter.requires_grad],
            "lr": float(learning_rate_operator),
        },
        {
            "name": "gate",
            "params": [parameter for parameter in gate if parameter.requires_grad],
            "lr": float(learning_rate_gate),
        },
    ]
    if any(not group["params"] for group in groups):
        empty = [str(group["name"]) for group in groups if not group["params"]]
        raise RuntimeError(f"Empty optimizer parameter groups: {empty}")

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    identifiers = [
        id(parameter) for group in groups for parameter in group["params"]
    ]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
        raise RuntimeError("Optimizer groups must cover every trainable parameter exactly once")
    return groups


__all__ = ["CfC1D_NCPS", "LatentLNN2p5D", "RestoreRWKVRefine"]
