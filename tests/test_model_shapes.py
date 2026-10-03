# Tests model output shapes for every backbone and head.
import pytest
import torch

from kraus_mlst.models import QuantumFilterModel


@pytest.mark.parametrize("variant", ["direct_legacy", "direct_rho0", "recursive", "soft", "cholesky", "choi", "kraus"])
def test_gru_variant_shapes(variant: str):
    model = QuantumFilterModel(
        backbone_name="gru",
        variant=variant,
        measurement_dim=1,
        d=2,
        rank=2,
        backbone_cfg={"hidden_dim": 16, "layers": 1},
        head_cfg={"residual_scale": 0.1},
        seed=1,
    )
    y = torch.randn(3, 11, 1)
    rho0 = torch.zeros(3, 2, 2, dtype=torch.complex64)
    rho0[:, 0, 0] = 1
    output = model(y, rho0, return_kraus=True)
    assert output.states.shape == (3, 11, 2, 2)
    if variant in {"choi", "kraus"}:
        assert output.kraus is not None
        assert output.kraus.shape == (3, 11, 2, 2, 2)
