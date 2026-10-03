# Numerical audit of the Kraus stabilizers in float32 and float64.
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import QuantumTrajectoryDataset
from .metrics import raw_state_metrics
from .models import build_model
from .utils import ensure_dir, save_json


def numerical_audit_checkpoint(
    project_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    max_trajectories: int = 8,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_cfg = checkpoint["run_cfg"]
    if run_cfg["variant"] != "kraus":
        return []
    seed = int(checkpoint["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data_cfg = checkpoint.get("project_data_cfg", project_cfg["data"])
    dataset = QuantumTrajectoryDataset(Path(checkpoint_data_cfg["root"]) / "id.npz", window=None)
    n = min(max_trajectories, len(dataset))
    y = torch.stack([dataset[i]["y"] for i in range(n)]).to(device)
    rho0 = torch.stack([dataset[i]["rho0"] for i in range(n)]).to(device)
    target = torch.stack([dataset[i]["target"] for i in range(n)]).to(device)

    rows: list[dict[str, Any]] = []
    modes = [
        ("both_stabilizers", True, True, torch.float32),
        ("no_trace_stabilizer", True, False, torch.float32),
        ("no_stabilizers", False, False, torch.float32),
    ]
    modes.append(("float64_no_stabilizers", False, False, torch.float64))

    for name, herm, trace, real_dtype in modes:
        model = build_model(run_cfg, checkpoint_data_cfg, seed)
        model.load_state_dict(checkpoint["model_state"])
        model.stabilize_hermiticity = herm
        model.stabilize_trace = trace
        if real_dtype == torch.float64:
            model = model.double()
            y_mode = y.double()
            rho0_mode = rho0.to(torch.complex128)
            target_mode = target.to(torch.complex128)
        else:
            y_mode, rho0_mode, target_mode = y, rho0, target
        model.to(device).eval()
        with torch.no_grad():
            output = model(y_mode, rho0_mode, return_kraus=True)
            metrics = raw_state_metrics(output.states, target_mode)
        row: dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "backbone": run_cfg["backbone"],
            "seed": seed,
            "mode": name,
            "stabilize_hermiticity": herm,
            "stabilize_trace": trace,
            "dtype": str(real_dtype),
        }
        for key, value in metrics.items():
            row[f"{key}_mean"] = float(value.mean().detach().cpu())
            row[f"{key}_max"] = float(value.max().detach().cpu())
        for key, value in output.aux.items():
            if torch.is_tensor(value):
                row[f"aux_{key}"] = float(value.detach().mean().cpu())
            elif isinstance(value, (int, float)):
                row[f"aux_{key}"] = float(value)
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def numerical_audit_all(project_cfg: dict[str, Any], runs_root: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    cfg = project_cfg.get("numerical_audit", {})
    include_seeds = {int(x) for x in cfg.get("seeds", [])}
    include_stages = set(cfg.get("stages", []))
    for checkpoint in sorted(Path(runs_root).glob("**/checkpoint_best.pt")):
        rel = checkpoint.relative_to(Path(runs_root))
        stage = rel.parts[0] if rel.parts else ""
        parts = checkpoint.parent.name.split("__")
        try:
            seed = int(parts[-1].lstrip("s"))
        except Exception:
            seed = -1
        if include_seeds and seed not in include_seeds:
            continue
        if include_stages and stage not in include_stages:
            continue
        rows.extend(
            numerical_audit_checkpoint(
                project_cfg,
                checkpoint,
                max_trajectories=int(project_cfg.get("numerical_audit", {}).get("max_trajectories", 8)),
            )
        )
    if rows:
        fields = sorted({k for row in rows for k in row})
        with (output_dir / "numerical_stabilizer_audit.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    save_json(output_dir / "numerical_stabilizer_audit.json", rows)
    return rows
