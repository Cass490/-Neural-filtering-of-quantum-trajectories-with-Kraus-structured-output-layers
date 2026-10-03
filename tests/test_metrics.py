# Tests the state metrics.
import torch

from kraus_mlst.metrics import raw_state_metrics, state_fidelity


def test_fidelity_identical_states():
    rho = torch.zeros(2, 5, 2, 2, dtype=torch.complex64)
    rho[..., 0, 0] = 1
    fidelity = state_fidelity(rho, rho)
    assert torch.allclose(fidelity, torch.ones_like(fidelity), atol=1e-5)


def test_unphysical_raw_metrics_are_reported_but_fidelity_is_projected():
    pred = torch.zeros(1, 1, 2, 2, dtype=torch.complex64)
    pred[0, 0, 0, 0] = 1.2
    pred[0, 0, 1, 1] = -0.2
    target = torch.zeros_like(pred)
    target[0, 0, 0, 0] = 1.0
    metrics = raw_state_metrics(pred, target)
    assert metrics["psd_violation_raw"].item() > 0
    assert 0 <= metrics["fidelity_projected"].item() <= 1
