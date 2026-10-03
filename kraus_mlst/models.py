# Combines a backbone and a head into one filtering model.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .backbones import build_backbone
from .heads import HeadOutput, build_head
from .quantum import density_to_features_torch


@dataclass
class ModelOutput:
    states: torch.Tensor
    aux: dict[str, torch.Tensor | float]
    kraus: torch.Tensor | None = None


class QuantumFilterModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        variant: str,
        measurement_dim: int,
        d: int,
        rank: int,
        backbone_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        seed: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.variant = variant
        self.measurement_dim = measurement_dim
        self.d = d
        self.rank = rank
        self.use_rho0_input = variant != "direct_legacy"
        input_dim = measurement_dim + (2 * d * d if self.use_rho0_input else 0)
        self.backbone = build_backbone(backbone_name, input_dim, backbone_cfg, seed=seed)
        self.head = build_head(variant, self.backbone.output_dim, d, rank, head_cfg)
        self.stabilize_hermiticity = True
        self.stabilize_trace = True

    def augmented_input(self, y: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        if not self.use_rho0_input:
            return y
        init_features = density_to_features_torch(rho0).unsqueeze(1).expand(-1, y.shape[1], -1)
        return torch.cat([y, init_features], dim=-1)

    def forward(
        self,
        y: torch.Tensor,
        rho0: torch.Tensor,
        return_kraus: bool = False,
        profile_qr: bool = False,
    ) -> ModelOutput:
        hidden = self.backbone(self.augmented_input(y, rho0))
        out: HeadOutput = self.head(
            hidden,
            rho0,
            stabilize_hermiticity=self.stabilize_hermiticity,
            stabilize_trace=self.stabilize_trace,
            return_kraus=return_kraus,
            profile_qr=profile_qr,
        )
        aux = dict(out.aux)
        aux["hidden_rms"] = float(getattr(self.backbone, "last_hidden_rms", float("nan")))
        return ModelOutput(states=out.states, aux=aux, kraus=out.kraus)


def build_model(run_cfg: dict[str, Any], data_cfg: dict[str, Any], seed: int) -> QuantumFilterModel:
    n_qubits = int(data_cfg["simulation"].get("n_qubits", 1))
    d = 2 ** n_qubits
    rank = int(run_cfg.get("rank", min(d, 4)))
    return QuantumFilterModel(
        backbone_name=run_cfg["backbone"],
        variant=run_cfg["variant"],
        measurement_dim=n_qubits,
        d=d,
        rank=rank,
        backbone_cfg=run_cfg.get("backbone_cfg", {}),
        head_cfg=run_cfg.get("head_cfg", {}),
        seed=seed,
    )
