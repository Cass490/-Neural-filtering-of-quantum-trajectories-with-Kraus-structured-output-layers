# Tests that measurement y[t] maps rho[t] to rho[t+1].
import numpy as np

from kraus_mlst.simulator import simulate_trajectory


def test_simulator_shapes_and_alignment():
    cfg = {
        "n_qubits": 1,
        "steps": 20,
        "dt": 0.01,
        "initial_state": "random_pure",
        "dynamics": {
            "omega": [0.5, 1.0],
            "kappa": [0.2, 0.4],
            "eta": [0.8, 1.0],
            "coupling": 0.0,
            "switching": {"n_switches": 1, "low": 5, "high": 15, "axes": ["x", "y"]},
        },
        "readout_noise": {"white_std": 0.0, "ou_std": 0.0},
    }
    result = simulate_trajectory(cfg, seed=9)
    assert result.y_raw.shape == (20, 1)
    assert result.rho.shape == (21, 2, 2)
    traces = np.trace(result.rho, axis1=-2, axis2=-1)
    assert np.max(np.abs(traces.real - 1.0)) < 1e-5
    assert result.switch_mask.sum() == 1
