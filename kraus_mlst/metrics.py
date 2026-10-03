# State metrics: projected fidelity, Bures and trace distance, physicality, switch recovery, Choi diagnostics.
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .quantum import (
    eigh_chunked,
    eigvalsh_chunked,
    choi_partial_trace_output_torch,
    effective_rank_torch,
    hermitian_part_torch,
    kraus_to_choi_torch,
    project_density_torch,
    trace_torch,
)


def state_fidelity(rho: torch.Tensor, sigma: torch.Tensor, project_inputs: bool = True) -> torch.Tensor:
    if project_inputs:
        rho = project_density_torch(rho)
        sigma = project_density_torch(sigma)
    sqrt_vals, vecs = eigh_chunked(hermitian_part_torch(rho))
    sqrt_vals = sqrt_vals.real.clamp_min(0.0).sqrt().to(vecs.dtype)
    sqrt_rho = (vecs * sqrt_vals.unsqueeze(-2)) @ vecs.conj().transpose(-1, -2)
    middle = hermitian_part_torch(sqrt_rho @ sigma @ sqrt_rho)
    vals = eigvalsh_chunked(middle).clamp_min(0.0)
    return vals.sqrt().sum(dim=-1).square().clamp(0.0, 1.0)


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor, project_inputs: bool = True) -> torch.Tensor:
    if project_inputs:
        rho = project_density_torch(rho)
        sigma = project_density_torch(sigma)
    diff = hermitian_part_torch(rho - sigma)
    vals = eigvalsh_chunked(diff)
    return 0.5 * vals.abs().sum(dim=-1)


def raw_state_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    herm = hermitian_part_torch(pred)
    eigvals = eigvalsh_chunked(herm)
    traces = trace_torch(pred)
    herm_error = (pred - pred.conj().transpose(-1, -2)).abs().square().sum(dim=(-2, -1)).sqrt()
    frob = (pred - target).abs().square().sum(dim=(-2, -1)).sqrt()
    fidelity = state_fidelity(pred, target, project_inputs=True)
    tdist = trace_distance(pred, target, project_inputs=True)
    return {
        "fidelity_projected": fidelity,
        "bures_distance_projected": torch.sqrt(torch.clamp(2.0 * (1.0 - torch.sqrt(fidelity)), min=0.0)),
        "trace_distance_projected": tdist,
        "frobenius_error_raw": frob,
        "trace_error_raw": (traces.real - 1.0).abs(),
        "trace_imag_raw": traces.imag.abs(),
        "hermiticity_error_raw": herm_error,
        "min_eigenvalue_hermitian_part": eigvals.min(dim=-1).values,
        "psd_violation_raw": torch.relu(-eigvals).sum(dim=-1),
        "physical_state_indicator": (
            ((traces.real - 1.0).abs() < 1e-6)
            & (traces.imag.abs() < 1e-6)
            & (herm_error < 1e-6)
            & (eigvals.min(dim=-1).values >= -1e-6)
        ).float(),
    }


def aggregate_state_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, tensor in metrics.items():
        out[f"{key}_mean"] = float(tensor.mean().detach().cpu())
        out[f"{key}_std"] = float(tensor.std(unbiased=False).detach().cpu())
        out[f"{key}_max"] = float(tensor.max().detach().cpu())
    return out


def per_trajectory_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, tensor in metrics.items():
        if tensor.ndim < 2:
            result[key] = tensor.detach().cpu().numpy()
        else:
            result[key] = tensor.mean(dim=1).detach().cpu().numpy()
    return result


def switch_recovery_metrics(
    fidelity: torch.Tensor,
    switch_mask: torch.Tensor,
    recovery_fraction: float = 0.9,
    consecutive: int = 10,
    max_horizon: int = 500,
) -> dict[str, np.ndarray]:
    fid = fidelity.detach().cpu().numpy()
    masks = switch_mask.detach().cpu().numpy()
    n, t_steps = fid.shape
    recovery = np.full(n, np.nan, dtype=np.float64)
    pre = np.full(n, np.nan, dtype=np.float64)
    post = np.full(n, np.nan, dtype=np.float64)
    worst50 = np.full(n, np.nan, dtype=np.float64)
    auc100 = np.full(n, np.nan, dtype=np.float64)
    switch_index = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        indices = np.flatnonzero(masks[i] > 0)
        if len(indices) == 0:
            continue
        s = int(indices[0])
        switch_index[i] = s
        pre_start = max(0, s - 50)
        pre[i] = float(fid[i, pre_start:s].mean()) if s > pre_start else float(fid[i, s])
        post_end = min(t_steps, s + 50)
        post[i] = float(fid[i, s:post_end].mean())
        worst50[i] = float(fid[i, s:post_end].min())
        auc_end = min(t_steps, s + 100)
        auc100[i] = float(fid[i, s:auc_end].mean())
        threshold = recovery_fraction * pre[i]
        end = min(t_steps, s + max_horizon)
        for j in range(s, max(s, end - consecutive + 1)):
            if np.all(fid[i, j : j + consecutive] >= threshold):
                recovery[i] = float(j - s)
                break
        if np.isnan(recovery[i]):
            recovery[i] = float(max_horizon)

    return {
        "switch_index": switch_index,
        "pre_switch_fidelity": pre,
        "post_switch_fidelity_50": post,
        "worst_fidelity_50": worst50,
        "post_switch_auc_100": auc100,
        "recovery_steps": recovery,
    }


def choi_diagnostics(kraus: torch.Tensor, d: int) -> dict[str, torch.Tensor]:
    choi = kraus_to_choi_torch(kraus)
    eigvals = eigvalsh_chunked(hermitian_part_torch(choi))
    ptr = choi_partial_trace_output_torch(choi, d)
    eye = torch.eye(d, dtype=ptr.dtype, device=ptr.device)
    tp_error = (ptr - eye).abs().square().sum(dim=(-2, -1)).sqrt()
    delta = torch.zeros_like(tp_error)
    if choi.shape[-3] > 1:
        delta[..., 1:] = (choi[..., 1:, :, :] - choi[..., :-1, :, :]).abs().square().sum(dim=(-2, -1)).sqrt()
    return {
        "choi_min_eigenvalue": eigvals.min(dim=-1).values,
        "choi_tp_error": tp_error,
        "choi_effective_rank": effective_rank_torch(choi),
        "choi_temporal_delta": delta,
        "choi_trace": trace_torch(choi).real,
    }


def normal_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    z = 1.959963984540054 if confidence == 0.95 else 1.96
    half = z * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half
