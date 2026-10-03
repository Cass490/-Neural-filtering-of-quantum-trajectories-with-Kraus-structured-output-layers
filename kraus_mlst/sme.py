# Model-based SME filter baselines: oracle, mismatched and adaptive bank.
from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import load_manifest
from .metrics import normal_ci
from .quantum import dissipator_np, local_operator_np, project_density_np
from .simulator import _build_hamiltonian
from .utils import ensure_dir, save_json


def rouchon_filter_step(
    rho: np.ndarray,
    dy: np.ndarray,
    hamiltonian: np.ndarray,
    measurement_ops: list[np.ndarray],
    eta: np.ndarray,
    dt: float,
    unmonitored_ops: list[np.ndarray] | None = None,
) -> np.ndarray:
    d = rho.shape[0]
    ident = np.eye(d, dtype=np.complex128)
    cdagc = np.zeros_like(rho, dtype=np.complex128)
    increment = np.zeros_like(rho, dtype=np.complex128)
    for q, l_op in enumerate(measurement_ops):
        cdagc += l_op.conj().T @ l_op
        increment += np.sqrt(float(eta[q])) * l_op * float(dy[q])
    m = ident - (1j * hamiltonian + 0.5 * cdagc) * dt + increment
    next_state = m @ rho @ m.conj().T
    for q, l_op in enumerate(measurement_ops):
        next_state += (1.0 - float(eta[q])) * l_op @ rho @ l_op.conj().T * dt
    for c_op in unmonitored_ops or []:
        next_state += dissipator_np(c_op, rho) * dt
    tr = np.trace(next_state).real
    if not np.isfinite(tr) or tr <= 1e-14:
        return rho.copy()
    return project_density_np(next_state / tr)


def _fidelity_np(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho = project_density_np(rho)
    sigma = project_density_np(sigma)
    vals, vecs = np.linalg.eigh(rho)
    sqrt_rho = (vecs * np.sqrt(np.clip(vals.real, 0.0, None))[None, :]) @ vecs.conj().T
    vals2 = np.linalg.eigvalsh(sqrt_rho @ sigma @ sqrt_rho)
    return float(np.square(np.sqrt(np.clip(vals2.real, 0.0, None)).sum()).clip(0.0, 1.0))


def oracle_filter_trajectory(arrays: dict[str, np.ndarray], i: int, dt: float, scenario_cfg: dict[str, Any] | None = None) -> np.ndarray:
    y = arrays["y_raw"][i]
    true_rho = arrays["rho"][i]
    omega = arrays["omega"][i]
    kappa = arrays["kappa"][i]
    eta = arrays["eta"][i]
    axis_code = arrays["axis_code"][i]
    coupling = arrays["coupling"][i]
    n_qubits = y.shape[-1]
    if n_qubits != 1:
        raise ValueError("This public release supports the single-qubit study only")
    unit_z = [local_operator_np("z", 0, 1)]
    dynamics = (scenario_cfg or {}).get("dynamics", {})
    amp_rate = float(dynamics.get("amplitude_damping", 0.0))
    dephase_rate = float(dynamics.get("extra_dephasing", 0.0))
    unmonitored = []
    if amp_rate > 0:
        unmonitored.extend([np.sqrt(amp_rate) * local_operator_np("minus", q, n_qubits) for q in range(n_qubits)])
    if dephase_rate > 0:
        unmonitored.extend([np.sqrt(dephase_rate) * local_operator_np("z", q, n_qubits) for q in range(n_qubits)])
    rho = true_rho[0].astype(np.complex128)
    pred = []
    for t in range(y.shape[0]):
        h = _build_hamiltonian(n_qubits, omega[t], axis_code[t], float(coupling[t]))
        ls = [np.sqrt(float(kappa[t, q])) * unit_z[q] for q in range(n_qubits)]
        rho = rouchon_filter_step(rho, y[t], h, ls, eta[t], dt, unmonitored_ops=unmonitored)
        pred.append(rho)
    return np.asarray(pred, dtype=np.complex64)


def mismatched_filter_trajectory(
    arrays: dict[str, np.ndarray],
    i: int,
    dt: float,
    omega_assumed: float = 1.5,
    kappa_assumed: float = 0.5,
    eta_assumed: float = 1.0,
    axis_assumed: str = "x",
) -> np.ndarray:
    y = arrays["y_raw"][i]
    true_rho = arrays["rho"][i]
    n_qubits = y.shape[-1]
    if n_qubits != 1:
        raise ValueError("This public release supports the single-qubit study only")
    unit_z = [local_operator_np("z", 0, 1)]
    axis_code = np.full(n_qubits, {"x": 0, "y": 1, "z": 2}[axis_assumed], dtype=np.int8)
    omega = np.full(n_qubits, omega_assumed, dtype=np.float64)
    eta = np.full(n_qubits, eta_assumed, dtype=np.float64)
    h = _build_hamiltonian(n_qubits, omega, axis_code, 0.0)
    ls = [np.sqrt(kappa_assumed) * unit_z[q] for q in range(n_qubits)]
    rho = true_rho[0].astype(np.complex128)
    pred = []
    for t in range(y.shape[0]):
        rho = rouchon_filter_step(rho, y[t], h, ls, eta, dt)
        pred.append(rho)
    return np.asarray(pred, dtype=np.complex64)


def adaptive_bank_filter_trajectory(
    arrays: dict[str, np.ndarray],
    i: int,
    dt: float,
    omega_grid: list[float],
    kappa_grid: list[float],
    eta_assumed: float,
    axes: list[str],
    transition_mix: float = 0.005,
    observation_variance_extra: float = 0.0,
) -> np.ndarray:
    y = arrays["y_raw"][i]
    true_rho = arrays["rho"][i]
    if y.shape[-1] != 1:
        raise ValueError("Adaptive filter bank currently supports one measurement channel")
    hypotheses = list(itertools.product(omega_grid, kappa_grid, axes))
    n_h = len(hypotheses)
    states = np.repeat(true_rho[0][None, :, :].astype(np.complex128), n_h, axis=0)
    weights = np.full(n_h, 1.0 / n_h, dtype=np.float64)
    z = local_operator_np("z", 0, 1)
    eta = np.array([eta_assumed], dtype=np.float64)
    outputs = []
    variance = dt + float(observation_variance_extra)

    for t in range(y.shape[0]):
        log_likelihood = np.empty(n_h, dtype=np.float64)
        new_states = np.empty_like(states)
        for j, (omega, kappa, axis) in enumerate(hypotheses):
            axis_code = np.array([{"x": 0, "y": 1, "z": 2}[axis]], dtype=np.int8)
            h = _build_hamiltonian(1, np.array([omega]), axis_code, 0.0)
            l_op = np.sqrt(kappa) * z
            expected = np.trace((l_op + l_op.conj().T) @ states[j]).real
            mean_dy = np.sqrt(eta_assumed) * expected * dt
            innovation = float(y[t, 0]) - mean_dy
            log_likelihood[j] = -0.5 * (innovation * innovation / max(variance, 1e-12) + np.log(2 * np.pi * max(variance, 1e-12)))
            new_states[j] = rouchon_filter_step(states[j], y[t], h, [l_op], eta, dt)
        states = new_states
        log_weights = np.log(weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum() + 1e-300
        weights = (1.0 - transition_mix) * weights + transition_mix / n_h
        mixture = np.einsum("h,hij->ij", weights, states)
        outputs.append(project_density_np(mixture))
    return np.asarray(outputs, dtype=np.complex64)


def evaluate_sme_baselines(project_cfg: dict[str, Any], output_dir: str | Path) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    data_root = Path(project_cfg["data"]["root"])
    manifest = load_manifest(data_root)
    dt = float(project_cfg["data"]["simulation"].get("dt", 0.005))
    cfg = project_cfg.get("sme_baselines", {})
    scenarios = cfg.get("scenarios", [name for name in manifest["scenario_files"] if name != "val"])
    max_trajectories = int(cfg.get("max_trajectories", 50))
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        with np.load(data_root / manifest["scenario_files"][scenario], allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        n = min(max_trajectories, arrays["y_raw"].shape[0])
        methods = ["oracle", "mismatched"]
        if arrays["y_raw"].shape[-1] == 1 and cfg.get("adaptive", {}).get("enabled", True):
            methods.append("adaptive_bank")
        for method in methods:
            fidelities = []
            latencies = []
            for i in range(n):
                start = time.perf_counter()
                if method == "oracle":
                    pred = oracle_filter_trajectory(arrays, i, dt, manifest.get("scenario_configs", {}).get(scenario, {}))
                elif method == "mismatched":
                    mismatch = cfg.get("mismatched", {})
                    pred = mismatched_filter_trajectory(
                        arrays,
                        i,
                        dt,
                        omega_assumed=float(mismatch.get("omega", 1.5)),
                        kappa_assumed=float(mismatch.get("kappa", 0.5)),
                        eta_assumed=float(mismatch.get("eta", 1.0)),
                        axis_assumed=str(mismatch.get("axis", "x")),
                    )
                else:
                    adapt = cfg.get("adaptive", {})
                    pred = adaptive_bank_filter_trajectory(
                        arrays,
                        i,
                        dt,
                        omega_grid=[float(x) for x in adapt.get("omega_grid", [0.5, 1.5, 3.0, 4.5])],
                        kappa_grid=[float(x) for x in adapt.get("kappa_grid", [0.3, 0.55, 0.8])],
                        eta_assumed=float(adapt.get("eta", 0.8)),
                        axes=list(adapt.get("axes", ["x", "y"])),
                        transition_mix=float(adapt.get("transition_mix", 0.005)),
                        observation_variance_extra=float(adapt.get("observation_variance_extra", 0.0)),
                    )
                latencies.append(time.perf_counter() - start)
                true = arrays["rho"][i, 1:]
                fidelities.append(np.mean([_fidelity_np(pred[t], true[t]) for t in range(pred.shape[0])]))
            mean, low, high = normal_ci(np.asarray(fidelities))
            row = {
                "scenario": scenario,
                "method": method,
                "n_trajectories": n,
                "fidelity_mean": mean,
                "fidelity_ci95_low": low,
                "fidelity_ci95_high": high,
                "rollout_seconds_mean": float(np.mean(latencies)),
                "microseconds_per_timestep": float(1e6 * np.mean(latencies) / arrays["y_raw"].shape[1]),
            }
            rows.append(row)
            save_json(output_dir / f"{scenario}_{method}.json", row)

    fields = sorted({k for row in rows for k in row})
    with (output_dir / "sme_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
