#!/usr/bin/env python3
# Editor: Jialei.He
"""ContiLNN integration for the official DASMamba in-plane backbone."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .slice_operators import (
    LatentLNN2p5D,
    _complete_parameter_groups,
    _strip_uniform_module_prefix,
    _validate_intervals,
)


class ContiLNNDASMamba(nn.Module):
    """Augment an official DASMamba instance with bidirectional Bi-CfC modules.

    The backbone is injected rather than imported so the original DASMamba
    source can remain a pinned third-party dependency. The expected backbone
    interface is validated when this module is constructed.
    """

    @staticmethod
    def _stage_blocks(stage: nn.Module, stage_name: str) -> nn.ModuleList:
        blocks = getattr(stage, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise TypeError(
                f"{stage_name} must expose a non-empty ModuleList named blocks; "
                f"got stage={type(stage)!r}, blocks={type(blocks)!r}"
            )
        return blocks

    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 48,
        cfc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.base = backbone
        self.dim = int(dim)
        self.cfc_hidden = int(cfc_hidden)

        required = (
            "patch_embed",
            "encoder_level1",
            "encoder_level2",
            "encoder_level3",
            "down1_2",
            "down2_3",
            "down3_4",
            "latent",
            "up4_3",
            "reduce_chan_level3",
            "decoder_level3",
            "up3_2",
            "reduce_chan_level2",
            "decoder_level2",
            "up2_1",
            "decoder_level1",
            "refinement",
            "output",
        )
        missing = [name for name in required if not hasattr(self.base, name)]
        if missing:
            raise TypeError(f"DASMamba backbone is missing required modules: {missing}")

        def make_module(channels: int) -> LatentLNN2p5D:
            return LatentLNN2p5D(dim=channels, cfc_hidden=self.cfc_hidden)

        encoder_l2_blocks = self._stage_blocks(self.base.encoder_level2, "encoder_level2")
        decoder_l2_blocks = self._stage_blocks(self.base.decoder_level2, "decoder_level2")

        self.enc_l2 = nn.ModuleList(
            [make_module(self.dim * 2) for _ in encoder_l2_blocks]
        )
        self.dec_l2 = nn.ModuleList(
            [make_module(self.dim * 2) for _ in decoder_l2_blocks]
        )
        self.gate_enc_l2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.enc_l2]
        )
        self.gate_dec_l2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.dec_l2]
        )

        self.skip_l2 = make_module(self.dim * 2)
        self.skip_l3 = make_module(self.dim * 4)
        self.gate_skip_l2 = nn.Parameter(torch.tensor(0.0))
        self.gate_skip_l3 = nn.Parameter(torch.tensor(0.0))

        # Serial bottleneck processing matches the selected RWKV topology.
        self.latent_lnn = make_module(self.dim * 8)

        self.refine_lnn = make_module(self.dim * 2)
        self.gate_refine = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def _to_5d(value: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        bk, channels, height, width = value.shape
        if bk != batch * length:
            raise ValueError(f"Shape mismatch: BK={bk}, B*K={batch * length}")
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
        value_5d = self._to_5d(value, batch, length)
        transformed = module(value_5d, dt)
        coefficient = torch.sigmoid(gate).to(device=value.device, dtype=value.dtype)
        return self._to_4d(value_5d + coefficient * (transformed - value_5d))

    def _apply_serial(
        self,
        value: torch.Tensor,
        batch: int,
        length: int,
        dt: torch.Tensor,
        module: LatentLNN2p5D,
    ) -> torch.Tensor:
        return self._to_4d(module(self._to_5d(value, batch, length), dt))

    def load_backbone_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected DASMamba checkpoint payload: {type(payload)!r}")
        state = payload.get("model_state_dict", payload)
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"Invalid DASMamba backbone checkpoint: {path}")
        state = _strip_uniform_module_prefix(state)
        self.base.load_state_dict(state, strict=True)

    def forward(self, x_seq: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x_seq.shape)}")
        batch, length, channels, height, width = x_seq.shape
        if channels != 1:
            raise ValueError(f"Expected one input channel, got {channels}")
        dt = _validate_intervals(dt, batch, length, x_seq.device, x_seq.dtype)

        x = x_seq.view(batch * length, channels, height, width)
        out_enc_level1 = self.base.encoder_level1(self.base.patch_embed(x))

        feature = self.base.down1_2(out_enc_level1)
        for index, block in enumerate(
            self._stage_blocks(self.base.encoder_level2, "encoder_level2")
        ):
            feature = block(feature)
            feature = self._apply_gated(
                feature,
                batch,
                length,
                dt,
                self.enc_l2[index],
                self.gate_enc_l2[index],
            )
        out_enc_level2 = feature

        out_enc_level3 = self.base.encoder_level3(
            self.base.down2_3(out_enc_level2)
        )
        latent = self.base.latent(self.base.down3_4(out_enc_level3))
        latent = self._apply_serial(latent, batch, length, dt, self.latent_lnn)

        skip_l3 = self._apply_gated(
            out_enc_level3,
            batch,
            length,
            dt,
            self.skip_l3,
            self.gate_skip_l3,
        )
        feature = self.base.up4_3(latent)
        feature = self.base.reduce_chan_level3(torch.cat([feature, skip_l3], dim=1))
        out_dec_level3 = self.base.decoder_level3(feature)

        skip_l2 = self._apply_gated(
            out_enc_level2,
            batch,
            length,
            dt,
            self.skip_l2,
            self.gate_skip_l2,
        )
        feature = self.base.up3_2(out_dec_level3)
        feature = self.base.reduce_chan_level2(torch.cat([feature, skip_l2], dim=1))
        for index, block in enumerate(
            self._stage_blocks(self.base.decoder_level2, "decoder_level2")
        ):
            feature = block(feature)
            feature = self._apply_gated(
                feature,
                batch,
                length,
                dt,
                self.dec_l2[index],
                self.gate_dec_l2[index],
            )
        out_dec_level2 = feature

        feature = self.base.up2_1(out_dec_level2)
        feature = torch.cat([feature, out_enc_level1], dim=1)
        feature = self.base.decoder_level1(feature)
        feature = self.base.refinement(feature)
        feature = self._apply_gated(
            feature,
            batch,
            length,
            dt,
            self.refine_lnn,
            self.gate_refine,
        )
        output = self.base.output(feature) + x
        return output.view(batch, length, channels, height, width)

    def base_parameters(self) -> Iterable[nn.Parameter]:
        return self.base.parameters()

    def operator_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        for modules in (self.enc_l2, self.dec_l2):
            for module in modules:
                parameters.extend(module.parameters())
        for module in (self.skip_l2, self.skip_l3, self.latent_lnn, self.refine_lnn):
            parameters.extend(module.parameters())
        return parameters

    def gate_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        parameters.extend(self.gate_enc_l2)
        parameters.extend(self.gate_dec_l2)
        parameters.extend(
            [self.gate_skip_l2, self.gate_skip_l3, self.gate_refine]
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


__all__ = ["ContiLNNDASMamba"]
