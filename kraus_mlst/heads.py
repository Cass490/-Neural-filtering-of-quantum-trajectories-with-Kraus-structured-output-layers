# Output heads: direct, recursive, soft, Cholesky, Choi and Kraus (QR/Stiefel CPTP).
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .quantum import (
    apply_kraus_torch,
    density_to_features_torch,
    kraus_completeness_torch,
    trace_torch,
)


@dataclass
class HeadOutput:
    states: torch.Tensor
    aux: dict[str, torch.Tensor | float]
    kraus: torch.Tensor | None = None


class DirectStateHead(nn.Module):
    def __init__(self, hidden_dim: int, d: int):
        super().__init__()
        self.d = d
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * d * d))

    def forward(self, hidden: torch.Tensor, rho0: torch.Tensor, **_: Any) -> HeadOutput:
        raw = self.proj(hidden)
        n = self.d * self.d
        real = raw[..., :n].reshape(*raw.shape[:-1], self.d, self.d)
        imag = raw[..., n:].reshape(*raw.shape[:-1], self.d, self.d)
        return HeadOutput(torch.complex(real, imag), aux={})


class RecursiveDeltaHead(nn.Module):
    def __init__(self, hidden_dim: int, d: int, residual_scale: float = 0.1):
        super().__init__()
        self.d = d
        self.residual_scale = float(residual_scale)
        self.eval_stabilize = False
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 2 * d * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d * d),
        )

    def forward(self, hidden: torch.Tensor, rho0: torch.Tensor, **_: Any) -> HeadOutput:
        rho = rho0
        states = []
        for t in range(hidden.shape[1]):
            features = torch.cat([hidden[:, t], density_to_features_torch(rho)], dim=-1)
            raw = self.proj(features)
            n = self.d * self.d
            delta = torch.complex(
                raw[..., :n].reshape(-1, self.d, self.d),
                raw[..., n:].reshape(-1, self.d, self.d),
            )
            rho = rho + self.residual_scale * torch.tanh(delta.real) + 1j * self.residual_scale * torch.tanh(delta.imag)
            if self.eval_stabilize:
                rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
                rho = rho / trace_torch(rho).real.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
            states.append(rho)
        return HeadOutput(torch.stack(states, dim=1), aux={})


class CholeskyStateHead(nn.Module):

    def __init__(self, hidden_dim: int, d: int):
        super().__init__()
        self.d = d
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 2 * d * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d * d),
        )

    def _matrix(self, raw: torch.Tensor) -> torch.Tensor:
        n = self.d * self.d
        mat = torch.complex(raw[..., :n].reshape(-1, self.d, self.d), raw[..., n:].reshape(-1, self.d, self.d))
        lower = torch.tril(mat, diagonal=-1)
        diag_raw = torch.diagonal(mat.real, dim1=-2, dim2=-1)
        diag = F.softplus(diag_raw) + 1e-5
        lower = lower + torch.diag_embed(diag).to(lower.dtype)
        rho = lower @ lower.conj().transpose(-1, -2)
        return rho / trace_torch(rho).real.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)

    def forward(self, hidden: torch.Tensor, rho0: torch.Tensor, **_: Any) -> HeadOutput:
        rho = rho0
        states = []
        for t in range(hidden.shape[1]):
            features = torch.cat([hidden[:, t], density_to_features_torch(rho)], dim=-1)
            rho = self._matrix(self.proj(features))
            states.append(rho)
        return HeadOutput(torch.stack(states, dim=1), aux={})


class ChoiSoftHead(nn.Module):

    def __init__(self, hidden_dim: int, d: int, rank: int):
        super().__init__()
        self.d = d
        self.rank = rank
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * rank * d * d))

    def forward(self, hidden: torch.Tensor, rho0: torch.Tensor, return_kraus: bool = False, **_: Any) -> HeadOutput:
        raw = self.proj(hidden)
        n = self.rank * self.d * self.d
        kraus = torch.complex(
            raw[..., :n].reshape(*raw.shape[:-1], self.rank, self.d, self.d),
            raw[..., n:].reshape(*raw.shape[:-1], self.rank, self.d, self.d),
        )
        rho = rho0
        states = []
        pre_trace_errors = []
        for t in range(hidden.shape[1]):
            rho_next = apply_kraus_torch(rho, kraus[:, t])
            tr = trace_torch(rho_next).real.clamp_min(1e-12)
            pre_trace_errors.append((tr - 1.0).abs())
            rho = rho_next / tr.unsqueeze(-1).unsqueeze(-1)
            states.append(rho)
        complete = kraus_completeness_torch(kraus)
        eye = torch.eye(self.d, device=hidden.device, dtype=complete.dtype)
        tp_residual = (complete - eye).abs().square().sum(dim=(-2, -1)).mean()
        return HeadOutput(
            torch.stack(states, dim=1),
            aux={"choi_tp_penalty": tp_residual, "choi_pre_normalization_trace_error": torch.stack(pre_trace_errors, dim=1).mean()},
            kraus=kraus if return_kraus else None,
        )


class KrausStiefelHead(nn.Module):

    def __init__(self, hidden_dim: int, d: int, rank: int):
        super().__init__()
        self.d = d
        self.rank = rank
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * rank * d * d))
        self.last_qr_seconds = 0.0
        self.last_qr_min_diag = float("nan")

    def make_kraus(self, hidden: torch.Tensor, profile_qr: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        raw = self.proj(hidden)
        n = self.rank * self.d * self.d
        v = torch.complex(
            raw[..., :n].reshape(*raw.shape[:-1], self.rank * self.d, self.d),
            raw[..., n:].reshape(*raw.shape[:-1], self.rank * self.d, self.d),
        )
        if profile_qr and v.is_cuda:
            torch.cuda.synchronize(v.device)
        start = perf_counter()
        q, r = torch.linalg.qr(v, mode="reduced")
        if profile_qr and v.is_cuda:
            torch.cuda.synchronize(v.device)
        self.last_qr_seconds = perf_counter() - start
        with torch.no_grad():
            diag = torch.diagonal(r, dim1=-2, dim2=-1).abs()
            self.last_qr_min_diag = float(diag.min().detach().cpu())
        kraus = q.reshape(*q.shape[:-2], self.rank, self.d, self.d)
        completeness = kraus_completeness_torch(kraus)
        eye = torch.eye(self.d, device=hidden.device, dtype=completeness.dtype)
        completeness_error = (completeness - eye).abs().square().sum(dim=(-2, -1)).mean()
        return kraus, {
            "kraus_completeness_error": completeness_error,
            "qr_seconds": self.last_qr_seconds,
            "qr_min_diag": self.last_qr_min_diag,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        rho0: torch.Tensor,
        stabilize_hermiticity: bool = True,
        stabilize_trace: bool = True,
        return_kraus: bool = False,
        profile_qr: bool = False,
        **_: Any,
    ) -> HeadOutput:
        kraus, aux = self.make_kraus(hidden, profile_qr=profile_qr)
        rho = rho0
        states = []
        pre_trace_errors = []
        pre_herm_errors = []
        for t in range(hidden.shape[1]):
            rho_next = apply_kraus_torch(rho, kraus[:, t])
            pre_trace_errors.append((trace_torch(rho_next).real - 1.0).abs())
            pre_herm_errors.append((rho_next - rho_next.conj().transpose(-1, -2)).abs().square().sum(dim=(-2, -1)).sqrt())
            if stabilize_hermiticity:
                rho_next = 0.5 * (rho_next + rho_next.conj().transpose(-1, -2))
            if stabilize_trace:
                tr = trace_torch(rho_next).real.clamp_min(1e-12)
                rho_next = rho_next / tr.unsqueeze(-1).unsqueeze(-1)
            rho = rho_next
            states.append(rho)
        aux.update(
            {
                "pre_correction_trace_error": torch.stack(pre_trace_errors, dim=1).mean(),
                "pre_correction_hermiticity_error": torch.stack(pre_herm_errors, dim=1).mean(),
            }
        )
        return HeadOutput(
            torch.stack(states, dim=1),
            aux=aux,
            kraus=kraus if return_kraus else None,
        )


def build_head(name: str, hidden_dim: int, d: int, rank: int, cfg: dict[str, Any]) -> nn.Module:
    name = name.lower()
    if name in {"direct", "direct_legacy", "direct_rho0"}:
        return DirectStateHead(hidden_dim, d)
    if name in {"recursive", "soft"}:
        return RecursiveDeltaHead(hidden_dim, d, residual_scale=float(cfg.get("residual_scale", 0.1)))
    if name == "cholesky":
        return CholeskyStateHead(hidden_dim, d)
    if name == "choi":
        return ChoiSoftHead(hidden_dim, d, rank)
    if name == "kraus":
        return KrausStiefelHead(hidden_dim, d, rank)
    raise ValueError(f"Unknown head/variant: {name}")
