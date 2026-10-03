# Evaluates a checkpoint on the identity split and every held-out scenario.
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import QuantumTrajectoryDataset, load_manifest
from .metrics import (
    aggregate_state_metrics,
    choi_diagnostics,
    per_trajectory_metrics,
    raw_state_metrics,
    switch_recovery_metrics,
)
from .models import build_model
from .utils import ensure_dir, save_json


def _device(training_cfg: dict[str, Any]) -> torch.device:
    requested = str(training_cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    project_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    scenarios: list[str] | None = None,
    save_prediction_trajectories: int = 3,
    stabilize_recursive: bool = False,
) -> list[dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = ensure_dir(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_cfg = checkpoint["run_cfg"]
    seed = int(checkpoint["seed"])
    model = build_model(run_cfg, project_cfg["data"], seed)
    model.load_state_dict(checkpoint["model_state"])
    if stabilize_recursive:
        model.head.eval_stabilize = True
    device = _device(project_cfg["training"])
    model.to(device).eval()

    data_root = Path(project_cfg["data"]["root"])
    manifest = load_manifest(data_root)
    available = list(manifest["scenario_files"].keys())
    scenarios = scenarios or [name for name in available if name != "val"]
    eval_batch_size = int(project_cfg.get("evaluation", {}).get("batch_size", 8))

    summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario not in manifest["scenario_files"]:
            raise KeyError(f"Unknown scenario {scenario}; available={available}")
        dataset = QuantumTrajectoryDataset(data_root / manifest["scenario_files"][scenario], window=None)
        loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

        metric_chunks: dict[str, list[np.ndarray]] = {}
        switch_chunks: dict[str, list[np.ndarray]] = {}
        choi_chunks: dict[str, list[np.ndarray]] = {}
        aux_rows: list[dict[str, float]] = []
        saved_predictions: list[np.ndarray] = []
        saved_targets: list[np.ndarray] = []
        saved_inputs: list[np.ndarray] = []
        temporal_sum: dict[str, np.ndarray] = {}
        temporal_sumsq: dict[str, np.ndarray] = {}
        temporal_count = 0
        choi_aligned: dict[str, list[np.ndarray]] = {}
        choi_window = int(project_cfg.get("evaluation", {}).get("choi_switch_window", 100))

        for batch in tqdm(loader, desc=f"Evaluate {scenario}"):
            y = batch["y"].to(device)
            rho0 = batch["rho0"].to(device)
            target = batch["target"].to(device)
            switch_mask = batch["switch_mask"].to(device)
            needs_kraus = run_cfg["variant"] in {"kraus", "choi"}
            with torch.no_grad():
                output = model(y, rho0, return_kraus=needs_kraus)
                state_metrics = raw_state_metrics(output.states, target)
                per_traj = per_trajectory_metrics(state_metrics)
                switch = switch_recovery_metrics(state_metrics["fidelity_projected"], switch_mask)
                for key, tensor in state_metrics.items():
                    arr = tensor.detach().cpu().numpy()
                    temporal_sum[key] = temporal_sum.get(key, np.zeros(arr.shape[1], dtype=np.float64)) + arr.sum(axis=0)
                    temporal_sumsq[key] = temporal_sumsq.get(key, np.zeros(arr.shape[1], dtype=np.float64)) + np.square(arr).sum(axis=0)
                temporal_count += int(y.shape[0])
                for key, values in per_traj.items():
                    metric_chunks.setdefault(key, []).append(values)
                for key, values in switch.items():
                    switch_chunks.setdefault(key, []).append(values)
                if output.kraus is not None:
                    diag = choi_diagnostics(output.kraus, model.d)
                    diag_per = per_trajectory_metrics(diag)
                    for key, values in diag_per.items():
                        choi_chunks.setdefault(key, []).append(values)
                    masks_np = switch_mask.detach().cpu().numpy()
                    diag_np = {key: value.detach().cpu().numpy() for key, value in diag.items()}
                    for i_local in range(masks_np.shape[0]):
                        switches = np.flatnonzero(masks_np[i_local] > 0)
                        if len(switches) == 0:
                            continue
                        center = int(switches[0])
                        for key, arr in diag_np.items():
                            start = max(0, center - choi_window)
                            end = min(arr.shape[1], center + choi_window + 1)
                            segment = arr[i_local, start:end]
                            padded = np.full(2 * choi_window + 1, np.nan, dtype=np.float64)
                            offset = choi_window - (center - start)
                            padded[offset : offset + len(segment)] = segment
                            choi_aligned.setdefault(key, []).append(padded)

            aux_row: dict[str, float] = {}
            for key, value in output.aux.items():
                if torch.is_tensor(value):
                    aux_row[key] = float(value.detach().mean().cpu())
                elif isinstance(value, (int, float)):
                    aux_row[key] = float(value)
            aux_rows.append(aux_row)

            current_saved = sum(x.shape[0] for x in saved_predictions)
            if current_saved < save_prediction_trajectories:
                take = min(save_prediction_trajectories - current_saved, output.states.shape[0])
                saved_predictions.append(output.states[:take].detach().cpu().numpy())
                saved_targets.append(target[:take].detach().cpu().numpy())
                saved_inputs.append(y[:take].detach().cpu().numpy())

        merged_metrics = {k: np.concatenate(v, axis=0) for k, v in metric_chunks.items()}
        merged_switch = {k: np.concatenate(v, axis=0) for k, v in switch_chunks.items()}
        merged_choi = {k: np.concatenate(v, axis=0) for k, v in choi_chunks.items()}

        per_rows: list[dict[str, Any]] = []
        n = len(next(iter(merged_metrics.values())))
        for i in range(n):
            row: dict[str, Any] = {"trajectory": i, "scenario": scenario}
            for key, values in merged_metrics.items():
                row[key] = float(values[i])
            for key, values in merged_switch.items():
                row[key] = float(values[i]) if np.issubdtype(values.dtype, np.floating) else int(values[i])
            for key, values in merged_choi.items():
                row[key] = float(values[i])
            per_rows.append(row)

        scenario_dir = ensure_dir(output_dir / scenario)
        _write_rows(scenario_dir / "per_trajectory.csv", per_rows)
        temporal_rows = []
        if temporal_count > 0:
            t_len = len(next(iter(temporal_sum.values())))
            for t in range(t_len):
                row = {"timestep": t}
                for key in temporal_sum:
                    mean = temporal_sum[key][t] / temporal_count
                    variance = max(0.0, temporal_sumsq[key][t] / temporal_count - mean * mean)
                    row[f"{key}_mean"] = mean
                    row[f"{key}_std"] = float(np.sqrt(variance))
                temporal_rows.append(row)
            _write_rows(scenario_dir / "temporal_metrics.csv", temporal_rows)
        if choi_aligned:
            aligned_rows = []
            relative = np.arange(-choi_window, choi_window + 1)
            for idx, rel in enumerate(relative):
                row = {"relative_step": int(rel)}
                for key, values in choi_aligned.items():
                    matrix = np.asarray(values, dtype=np.float64)
                    column = matrix[:, idx]
                    finite = column[np.isfinite(column)]
                    row[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("nan")
                    row[f"{key}_std"] = float(finite.std()) if len(finite) else float("nan")
                    row[f"{key}_count"] = int(len(finite))
                aligned_rows.append(row)
            _write_rows(scenario_dir / "choi_switch_aligned.csv", aligned_rows)
        if saved_predictions:
            np.savez_compressed(
                scenario_dir / "prediction_examples.npz",
                prediction=np.concatenate(saved_predictions, axis=0),
                target=np.concatenate(saved_targets, axis=0),
                y=np.concatenate(saved_inputs, axis=0),
            )

        summary: dict[str, Any] = {
            "scenario": scenario,
            "backbone": run_cfg["backbone"],
            "variant": run_cfg["variant"],
            "run_tag": run_cfg.get("run_tag", ""),
            "rank": int(run_cfg.get("rank", model.rank)),
            "seed": seed,
            "n_trajectories": n,
            "hilbert_dim": model.d,
            "n_qubits": int(round(np.log2(model.d))),
        }
        for key, values in merged_metrics.items():
            finite = values[np.isfinite(values)]
            summary[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            summary[f"{key}_std"] = float(finite.std()) if len(finite) else float("nan")
        for key, values in merged_switch.items():
            finite = values[np.isfinite(values)]
            summary[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            summary[f"{key}_median"] = float(np.median(finite)) if len(finite) else float("nan")
        for key, values in merged_choi.items():
            finite = values[np.isfinite(values)]
            summary[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            summary[f"{key}_max"] = float(finite.max()) if len(finite) else float("nan")
        if aux_rows:
            for key in sorted({k for row in aux_rows for k in row}):
                vals = [row[key] for row in aux_rows if key in row]
                summary[f"aux_{key}_mean"] = float(np.mean(vals))
        save_json(scenario_dir / "summary.json", summary)
        summary_rows.append(summary)

    _write_rows(output_dir / "all_scenarios_summary.csv", summary_rows)
    save_json(output_dir / "evaluation_manifest.json", {"checkpoint": str(checkpoint_path), "scenarios": scenarios})
    return summary_rows
