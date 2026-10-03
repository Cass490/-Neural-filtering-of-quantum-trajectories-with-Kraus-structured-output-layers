# Generates the train/val/test and held-out scenario datasets and serves training windows.
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .simulator import TrajectoryResult, simulate_trajectory
from .utils import ensure_dir, save_json


def _stack_results(results: list[TrajectoryResult]) -> dict[str, np.ndarray]:
    return {
        "y_raw": np.stack([r.y_raw for r in results]),
        "rho": np.stack([r.rho for r in results]),
        "omega": np.stack([r.omega for r in results]),
        "kappa": np.stack([r.kappa for r in results]),
        "eta": np.stack([r.eta for r in results]),
        "axis_code": np.stack([r.axis_code for r in results]),
        "coupling": np.stack([r.coupling for r in results]),
        "switch_mask": np.stack([r.switch_mask for r in results]),
        "seeds": np.asarray([r.seed for r in results], dtype=np.int64),
    }


def _simulate_job(args: tuple[dict[str, Any], int]) -> TrajectoryResult:
    sim_cfg, seed = args
    return simulate_trajectory(sim_cfg, seed=seed)


def generate_split(
    sim_cfg: dict[str, Any],
    n_trajectories: int,
    base_seed: int,
    desc: str,
    workers: int = 0,
) -> dict[str, np.ndarray]:
    jobs = [(sim_cfg, base_seed + i) for i in range(n_trajectories)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(_simulate_job, jobs, chunksize=1), total=n_trajectories, desc=desc))
    else:
        results = [_simulate_job(job) for job in tqdm(jobs, desc=desc)]
    return _stack_results(results)


def standardize_measurements(
    train: dict[str, np.ndarray],
    others: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    mean = train["y_raw"].mean(axis=(0, 1), keepdims=True).astype(np.float64)
    std = train["y_raw"].std(axis=(0, 1), keepdims=True).astype(np.float64)
    std = np.maximum(std, 1e-8)
    train["y"] = ((train["y_raw"] - mean) / std).astype(np.float32)
    for data in others.values():
        data["y"] = ((data["y_raw"] - mean) / std).astype(np.float32)
    return {"mean_per_channel": mean.reshape(-1).tolist(), "std_per_channel": std.reshape(-1).tolist()}


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def prepare_dataset_suite(cfg: dict[str, Any], force: bool = False) -> Path:
    data_cfg = cfg["data"]
    root = ensure_dir(data_cfg["root"])
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        return manifest_path

    base_seed = int(data_cfg.get("seed", 12345))
    sim_base = data_cfg["simulation"]
    counts = data_cfg["counts"]
    workers = int(data_cfg.get("workers", 0))

    train = generate_split(sim_base, int(counts["train"]), base_seed, "Generate train", workers=workers)
    val = generate_split(sim_base, int(counts["val"]), base_seed + 1_000_000, "Generate val", workers=workers)
    test_id = generate_split(sim_base, int(counts["test"]), base_seed + 2_000_000, "Generate ID test", workers=workers)

    scenarios: dict[str, dict[str, np.ndarray]] = {"val": val, "id": test_id}
    scenario_cfgs: dict[str, dict[str, Any]] = {"val": sim_base, "id": sim_base}
    for offset, (name, overrides) in enumerate(data_cfg.get("ood_scenarios", {}).items(), start=3):
        from .config import deep_update

        scenario_sim = deep_update(sim_base, overrides)
        scenarios[name] = generate_split(
            scenario_sim,
            int(overrides.get("count", counts["test"])),
            base_seed + int(overrides.get("seed_offset", offset)) * 1_000_000,
            f"Generate {name}",
            workers=workers,
        )
        scenario_cfgs[name] = scenario_sim

    norm = standardize_measurements(train, scenarios)
    save_npz(root / "train.npz", train)
    for name, arrays in scenarios.items():
        save_npz(root / f"{name}.npz", arrays)

    manifest = {
        "format_version": 1,
        "alignment": "y[t] maps rho[t] to rho[t+1]",
        "train_file": "train.npz",
        "scenario_files": {name: f"{name}.npz" for name in scenarios},
        "normalization": norm,
        "counts": {
            "train": int(train["y"].shape[0]),
            **{name: int(arrays["y"].shape[0]) for name, arrays in scenarios.items()},
        },
        "simulation": sim_base,
        "scenario_configs": scenario_cfgs,
        "seed_schedule": {
            "train": base_seed,
            "val": base_seed + 1_000_000,
            "id": base_seed + 2_000_000,
        },
    }
    save_json(manifest_path, manifest)
    return manifest_path


class QuantumTrajectoryDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        window: int | None = None,
        windows_per_trajectory: int = 1,
        random_windows: bool = False,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.arrays = {k: data[k] for k in data.files}
        self.window = window
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.random_windows = bool(random_windows)
        self.seed = int(seed)
        self.epoch = 0
        self.n = int(self.arrays["y"].shape[0])
        self.steps = int(self.arrays["y"].shape[1])
        if self.window is not None and not (1 <= self.window <= self.steps):
            raise ValueError(f"window must be in [1,{self.steps}], got {self.window}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n * self.windows_per_trajectory

    def _window_start(self, traj_idx: int, replica: int) -> int:
        if self.window is None or self.window == self.steps:
            return 0
        max_start = self.steps - self.window
        if self.random_windows:
            local_seed = self.seed + 1_000_003 * self.epoch + 1009 * traj_idx + replica
            rng = np.random.default_rng(local_seed)
            return int(rng.integers(0, max_start + 1))
        if self.windows_per_trajectory == 1:
            return max_start // 2
        return int(round(replica * max_start / max(1, self.windows_per_trajectory - 1)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_idx = index // self.windows_per_trajectory
        replica = index % self.windows_per_trajectory
        start = self._window_start(traj_idx, replica)
        length = self.steps if self.window is None else self.window
        end = start + length

        y = torch.from_numpy(self.arrays["y"][traj_idx, start:end]).float()
        y_raw = torch.from_numpy(self.arrays["y_raw"][traj_idx, start:end]).float()
        rho_full = torch.from_numpy(self.arrays["rho"][traj_idx, start : end + 1]).to(torch.complex64)
        switch_mask = torch.from_numpy(self.arrays["switch_mask"][traj_idx, start:end]).to(torch.int64)

        return {
            "y": y,
            "y_raw": y_raw,
            "rho0": rho_full[0],
            "target": rho_full[1:],
            "rho_full": rho_full,
            "switch_mask": switch_mask,
            "trajectory_index": torch.tensor(traj_idx, dtype=torch.int64),
            "window_start": torch.tensor(start, dtype=torch.int64),
        }


def load_manifest(root: str | Path) -> dict[str, Any]:
    with (Path(root) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)
