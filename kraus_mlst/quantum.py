# Pauli operators, density-matrix utilities, Kraus application and chunked eigensolvers.
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch


@lru_cache(maxsize=None)
def pauli_np(name: str) -> np.ndarray:
    name = name.lower()
    if name == "i":
        return np.eye(2, dtype=np.complex128)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if name == "z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if name == "minus":
        return np.array([[0, 1], [0, 0]], dtype=np.complex128)
    raise ValueError(f"Unknown operator: {name}")


def kron_all_np(ops: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def local_operator_np(name: str, qubit: int, n_qubits: int) -> np.ndarray:
    ops = [pauli_np("i") for _ in range(n_qubits)]
    ops[qubit] = pauli_np(name)
    return kron_all_np(ops)


def random_pure_state_np(d: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi /= np.linalg.norm(psi) + 1e-15
    return np.outer(psi, psi.conj())


def basis_state_np(d: int, index: int = 0) -> np.ndarray:
    psi = np.zeros(d, dtype=np.complex128)
    psi[index] = 1.0
    return np.outer(psi, psi.conj())


def project_density_np(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals.real, 0.0, None)
    total = vals.sum()
    if total <= eps:
        return np.eye(rho.shape[0], dtype=np.complex128) / rho.shape[0]
    vals /= total
    return (vecs * vals[None, :]) @ vecs.conj().T


def dissipator_np(c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    cdagc = c.conj().T @ c
    return c @ rho @ c.conj().T - 0.5 * (cdagc @ rho + rho @ cdagc)


def density_to_features_torch(rho: torch.Tensor) -> torch.Tensor:
    return torch.cat([rho.real.flatten(start_dim=-2), rho.imag.flatten(start_dim=-2)], dim=-1)


def features_to_complex_matrix_torch(features: torch.Tensor, d: int) -> torch.Tensor:
    n = d * d
    if features.shape[-1] != 2 * n:
        raise ValueError(f"Expected last dimension {2*n}, got {features.shape[-1]}")
    real = features[..., :n].reshape(*features.shape[:-1], d, d)
    imag = features[..., n:].reshape(*features.shape[:-1], d, d)
    return torch.complex(real, imag)


def hermitian_part_torch(rho: torch.Tensor) -> torch.Tensor:
    return 0.5 * (rho + rho.conj().transpose(-1, -2))


def trace_torch(rho: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)


def project_density_torch(rho: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    herm = hermitian_part_torch(rho)
    vals, vecs = eigh_chunked(herm)
    vals = vals.real.clamp_min(0.0)
    denom = vals.sum(dim=-1, keepdim=True).clamp_min(eps)
    vals = vals / denom
    return (vecs * vals.to(vecs.dtype).unsqueeze(-2)) @ vecs.conj().transpose(-1, -2)


def apply_kraus_torch(rho: torch.Tensor, kraus: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...rij,...jk,...rlk->...il", kraus, rho, kraus.conj())


def kraus_completeness_torch(kraus: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...rji,...rjk->...ik", kraus.conj(), kraus)


def kraus_to_choi_torch(kraus: torch.Tensor) -> torch.Tensor:
    vec = kraus.reshape(*kraus.shape[:-2], -1)
    return torch.einsum("...ra,...rb->...ab", vec, vec.conj())


def choi_partial_trace_output_torch(choi: torch.Tensor, d: int) -> torch.Tensor:
    shaped = choi.reshape(*choi.shape[:-2], d, d, d, d)
    return torch.einsum("...aiaj->...ij", shaped)


def effective_rank_torch(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vals = eigvalsh_chunked(hermitian_part_torch(matrix)).clamp_min(0.0)
    probs = vals / vals.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    return torch.exp(entropy)


EIG_CHUNK = 512


def eigvalsh_chunked(matrix: torch.Tensor, chunk: int = EIG_CHUNK) -> torch.Tensor:
    flat = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
    if flat.shape[0] <= chunk:
        vals = torch.linalg.eigvalsh(flat).real
    else:
        vals = torch.cat([torch.linalg.eigvalsh(flat[i : i + chunk]).real for i in range(0, flat.shape[0], chunk)])
    return vals.reshape(*matrix.shape[:-2], matrix.shape[-1])


def eigh_chunked(matrix: torch.Tensor, chunk: int = EIG_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    flat = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
    if flat.shape[0] <= chunk:
        vals, vecs = torch.linalg.eigh(flat)
    else:
        parts = [torch.linalg.eigh(flat[i : i + chunk]) for i in range(0, flat.shape[0], chunk)]
        vals = torch.cat([p[0] for p in parts])
        vecs = torch.cat([p[1] for p in parts])
    return vals.reshape(*matrix.shape[:-1]), vecs.reshape(matrix.shape)
