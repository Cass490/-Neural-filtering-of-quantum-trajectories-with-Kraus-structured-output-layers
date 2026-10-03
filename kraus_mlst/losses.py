# Training loss: squared Frobenius error plus soft-penalty and Choi trace-preservation terms.
from __future__ import annotations

from typing import Any

import torch

from .models import ModelOutput
from .quantum import eigvalsh_chunked, hermitian_part_torch, trace_torch


def frobenius_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().square().sum(dim=(-2, -1)).mean()


def physicality_penalties(states: torch.Tensor) -> dict[str, torch.Tensor]:
    herm = hermitian_part_torch(states)
    trace_penalty = (trace_torch(states).real - 1.0).square().mean()
    hermiticity_penalty = (states - states.conj().transpose(-1, -2)).abs().square().sum(dim=(-2, -1)).mean()
    eigvals = eigvalsh_chunked(herm)
    psd_penalty = torch.relu(-eigvals).square().sum(dim=-1).mean()
    return {
        "trace_penalty": trace_penalty,
        "hermiticity_penalty": hermiticity_penalty,
        "psd_penalty": psd_penalty,
    }


def compute_training_loss(
    output: ModelOutput,
    target: torch.Tensor,
    variant: str,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = frobenius_mse(output.states, target)
    total = prediction
    components: dict[str, torch.Tensor] = {"prediction_mse": prediction}

    if variant == "soft":
        penalties = physicality_penalties(output.states)
        lambdas = cfg.get("soft_lambdas", {})
        total = total + float(lambdas.get("trace", 1.0)) * penalties["trace_penalty"]
        total = total + float(lambdas.get("psd", 1.0)) * penalties["psd_penalty"]
        total = total + float(lambdas.get("hermiticity", 1.0)) * penalties["hermiticity_penalty"]
        components.update(penalties)

    if variant == "choi":
        penalty = output.aux.get("choi_tp_penalty")
        if not torch.is_tensor(penalty):
            raise RuntimeError("Choi head did not provide choi_tp_penalty")
        weight = float(cfg.get("choi_tp_lambda", 1.0))
        total = total + weight * penalty
        components["choi_tp_penalty"] = penalty

    components["total_loss"] = total
    logged = {k: float(v.detach().cpu()) for k, v in components.items()}
    return total, logged
