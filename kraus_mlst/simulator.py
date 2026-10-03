# Single-qubit continuous-measurement simulator (Rouchon-form Kraus update) with a mid-trajectory Hamiltonian switch.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .quantum import (
    basis_state_np,
    dissipator_np,
    local_operator_np,
    project_density_np,
    random_pure_state_np,
)


@dataclass
class TrajectoryResult:
    y_raw: np.ndarray
    rho: np.ndarray
    omega: np.ndarray
    kappa: np.ndarray
    eta: np.ndarray
    axis_code: np.ndarray
    coupling: np.ndarray
    switch_mask: np.ndarray
    seed: int


AXIS_TO_CODE = {"x": 0, "y": 1, "z": 2}
CODE_TO_AXIS = {v: k for k, v in AXIS_TO_CODE.items()}


def _sample_range(rng: np.random.Generator, value: Any, size: int | tuple[int, ...] = 1) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(size, float(value), dtype=np.float64)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return rng.uniform(float(value[0]), float(value[1]), size=size)
    raise ValueError(f"Expected scalar or [low, high], got {value!r}")


def _ou_process(
    rng: np.random.Generator,
    length: int,
    x0: float,
    mean: float,
    theta: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    x = np.empty(length, dtype=np.float64)
    x[0] = x0
    for t in range(1, length):
        x[t] = x[t - 1] + theta * (mean - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


def _axes_schedule(n_qubits: int, steps: int, switch_indices: list[int], axes: list[str]) -> np.ndarray:
    if not axes:
        axes = ["x", "y"]
    codes = np.zeros((steps, n_qubits), dtype=np.int8)
    boundaries = [0] + sorted(int(s) for s in switch_indices if 0 < int(s) < steps) + [steps]
    for phase, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        axis = axes[phase % len(axes)].lower()
        codes[a:b, :] = AXIS_TO_CODE[axis]
    return codes


def _build_hamiltonian(
    n_qubits: int,
    omega_t: np.ndarray,
    axis_codes_t: np.ndarray,
    coupling: float,
) -> np.ndarray:
    if n_qubits != 1:
        raise ValueError("This public release supports the single-qubit study only")
    axis = CODE_TO_AXIS[int(axis_codes_t[0])]
    return float(omega_t[0]) * local_operator_np(axis, 0, 1)


def simulate_trajectory(config: dict[str, Any], seed: int) -> TrajectoryResult:
    rng = np.random.default_rng(seed)
    n_qubits = int(config.get("n_qubits", 1))
    if n_qubits != 1:
        raise ValueError("This public release supports the single-qubit study only")
    d = 2
    steps = int(config.get("steps", 2000))
    dt = float(config.get("dt", 0.005))

    init = str(config.get("initial_state", "random_pure"))
    if init == "ground":
        rho0 = basis_state_np(d, 0)
    elif init == "random_pure":
        rho0 = random_pure_state_np(d, rng)
    else:
        raise ValueError(f"Unsupported initial_state={init}")

    dynamics = config.get("dynamics", {})
    omega_base = _sample_range(rng, dynamics.get("omega", [0.5, 4.0]), size=n_qubits)
    kappa_base = _sample_range(rng, dynamics.get("kappa", [0.3, 0.8]), size=n_qubits)
    eta_base = _sample_range(rng, dynamics.get("eta", 1.0), size=n_qubits)
    coupling_base = float(_sample_range(rng, dynamics.get("coupling", 0.0), size=1)[0])

    switch_cfg = dynamics.get("switching", {})
    n_switches = int(switch_cfg.get("n_switches", 1))
    switch_low = int(switch_cfg.get("low", max(1, int(0.2 * steps))))
    switch_high = int(switch_cfg.get("high", min(steps - 1, int(0.8 * steps))))
    if n_switches > 0:
        candidates = np.arange(switch_low, switch_high, dtype=int)
        if len(candidates) < n_switches:
            raise ValueError("Switch interval is too small for requested number of switches")
        switch_indices = sorted(rng.choice(candidates, size=n_switches, replace=False).tolist())
    else:
        switch_indices = []
    axis_codes = _axes_schedule(
        n_qubits=n_qubits,
        steps=steps,
        switch_indices=switch_indices,
        axes=list(switch_cfg.get("axes", ["x", "y"])),
    )
    switch_mask = np.zeros(steps, dtype=np.int8)
    for idx in switch_indices:
        switch_mask[idx] = 1

    boundaries = [0] + switch_indices + [steps]
    omega = np.empty((steps, n_qubits), dtype=np.float64)
    kappa = np.empty((steps, n_qubits), dtype=np.float64)
    eta = np.empty((steps, n_qubits), dtype=np.float64)
    coupling = np.empty(steps, dtype=np.float64)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        omega[a:b] = _sample_range(rng, dynamics.get("omega", [0.5, 4.0]), size=n_qubits)
        if switch_cfg.get("resample_kappa", True):
            kappa[a:b] = _sample_range(rng, dynamics.get("kappa", [0.3, 0.8]), size=n_qubits)
        else:
            kappa[a:b] = kappa_base
        eta[a:b] = eta_base
        coupling[a:b] = float(_sample_range(rng, dynamics.get("coupling", 0.0), size=1)[0])

    drift = dynamics.get("drift", {})
    if drift.get("enabled", False):
        theta = float(drift.get("theta", 0.5))
        sigma_omega = float(drift.get("sigma_omega", 0.15))
        sigma_kappa = float(drift.get("sigma_kappa", 0.03))
        for q in range(n_qubits):
            omega[:, q] += _ou_process(rng, steps, 0.0, 0.0, theta, sigma_omega, dt)
            kappa[:, q] += _ou_process(rng, steps, 0.0, 0.0, theta, sigma_kappa, dt)
        omega = np.clip(omega, 0.01, None)
        kappa = np.clip(kappa, 1e-4, None)

    colored_h = dynamics.get("colored_hamiltonian_noise", {})
    if colored_h.get("enabled", False):
        for q in range(n_qubits):
            omega[:, q] += _ou_process(
                rng,
                steps,
                0.0,
                0.0,
                float(colored_h.get("theta", 0.2)),
                float(colored_h.get("sigma", 0.3)),
                dt,
            )

    measurement_ops_unit = [local_operator_np("z", q, n_qubits) for q in range(n_qubits)]
    amp_rate = float(dynamics.get("amplitude_damping", 0.0))
    dephase_rate = float(dynamics.get("extra_dephasing", 0.0))
    amp_ops = [np.sqrt(max(amp_rate, 0.0)) * local_operator_np("minus", q, n_qubits) for q in range(n_qubits)]
    dephase_ops = [np.sqrt(max(dephase_rate, 0.0)) * local_operator_np("z", q, n_qubits) for q in range(n_qubits)]

    readout = config.get("readout_noise", {})
    white_std = float(readout.get("white_std", 0.0))
    ou_std = float(readout.get("ou_std", 0.0))
    ou_theta = float(readout.get("ou_theta", 0.5))
    readout_ou = np.zeros((steps, n_qubits), dtype=np.float64)
    if ou_std > 0:
        for q in range(n_qubits):
            readout_ou[:, q] = _ou_process(rng, steps, 0.0, 0.0, ou_theta, ou_std, dt)

    rho = np.empty((steps + 1, d, d), dtype=np.complex128)
    y_raw = np.empty((steps, n_qubits), dtype=np.float64)
    rho[0] = rho0
    ident = np.eye(d, dtype=np.complex128)

    for t in range(steps):
        state = rho[t]
        h = _build_hamiltonian(n_qubits, omega[t], axis_codes[t], coupling[t])
        ls = [np.sqrt(kappa[t, q]) * measurement_ops_unit[q] for q in range(n_qubits)]

        dy_physical = np.empty(n_qubits, dtype=np.float64)
        for q, l_op in enumerate(ls):
            expectation = np.trace((l_op + l_op.conj().T) @ state).real
            d_w = np.sqrt(dt) * rng.normal()
            dy_physical[q] = np.sqrt(eta[t, q]) * expectation * dt + d_w

        y_raw[t] = dy_physical + white_std * np.sqrt(dt) * rng.normal(size=n_qubits) + readout_ou[t]

        cdagc_sum = np.zeros((d, d), dtype=np.complex128)
        measurement_increment = np.zeros((d, d), dtype=np.complex128)
        for q, l_op in enumerate(ls):
            cdagc_sum += l_op.conj().T @ l_op
            measurement_increment += np.sqrt(eta[t, q]) * l_op * dy_physical[q]

        m = ident - (1j * h + 0.5 * cdagc_sum) * dt + measurement_increment
        next_state = m @ state @ m.conj().T
        for q, l_op in enumerate(ls):
            next_state += (1.0 - eta[t, q]) * (l_op @ state @ l_op.conj().T) * dt
        for c_op in amp_ops:
            if amp_rate > 0:
                next_state += dissipator_np(c_op, state) * dt
        for c_op in dephase_ops:
            if dephase_rate > 0:
                next_state += dissipator_np(c_op, state) * dt

        tr = np.trace(next_state).real
        if not np.isfinite(tr) or tr <= 1e-14:
            next_state = state.copy()
        else:
            next_state /= tr
        rho[t + 1] = project_density_np(next_state)

    return TrajectoryResult(
        y_raw=y_raw.astype(np.float32),
        rho=rho.astype(np.complex64),
        omega=omega.astype(np.float32),
        kappa=kappa.astype(np.float32),
        eta=eta.astype(np.float32),
        axis_code=axis_codes,
        coupling=coupling.astype(np.float32),
        switch_mask=switch_mask,
        seed=seed,
    )
