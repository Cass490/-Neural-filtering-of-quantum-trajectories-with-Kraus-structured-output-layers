# Tests that the Kraus head is CPTP and keeps states physical.
import torch

from kraus_mlst.heads import KrausStiefelHead
from kraus_mlst.quantum import kraus_completeness_torch, trace_torch


def _random_density(batch: int, d: int) -> torch.Tensor:
    a = torch.randn(batch, d, d, dtype=torch.complex64)
    rho = a @ a.conj().transpose(-1, -2)
    return rho / trace_torch(rho).real.unsqueeze(-1).unsqueeze(-1)


def test_kraus_completeness_and_state_physicality_d2():
    torch.manual_seed(3)
    head = KrausStiefelHead(hidden_dim=12, d=2, rank=2)
    hidden = torch.randn(4, 7, 12)
    rho0 = _random_density(4, 2)
    output = head(hidden, rho0, stabilize_hermiticity=False, stabilize_trace=False, return_kraus=True)
    complete = kraus_completeness_torch(output.kraus)
    eye = torch.eye(2, dtype=complete.dtype)
    assert torch.max(torch.abs(complete - eye)) < 2e-5
    traces = trace_torch(output.states)
    assert torch.max(torch.abs(traces.real - 1.0)) < 2e-5
    eigvals = torch.linalg.eigvalsh(0.5 * (output.states + output.states.conj().transpose(-1, -2)))
    assert eigvals.min() > -2e-5


def test_kraus_completeness_d4_rank8():
    torch.manual_seed(7)
    head = KrausStiefelHead(hidden_dim=10, d=4, rank=8)
    hidden = torch.randn(2, 3, 10)
    rho0 = _random_density(2, 4)
    output = head(hidden, rho0, stabilize_hermiticity=False, stabilize_trace=False, return_kraus=True)
    complete = kraus_completeness_torch(output.kraus)
    eye = torch.eye(4, dtype=complete.dtype)
    assert torch.max(torch.abs(complete - eye)) < 3e-5
