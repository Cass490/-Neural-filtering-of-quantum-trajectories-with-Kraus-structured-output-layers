# Measures inference latency, QR cost and memory of trained checkpoints.
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import QuantumTrajectoryDataset
from .models import build_model
from .utils import count_parameters, ensure_dir, load_json, save_json


def benchmark_checkpoint(
    project_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    batch_sizes: list[int],
    warmup: int = 5,
    repeats: int = 20,
) -> list[dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_cfg = checkpoint["run_cfg"]
    seed = int(checkpoint["seed"])
    checkpoint_data_cfg = checkpoint.get("project_data_cfg", project_cfg["data"])
    model = build_model(run_cfg, checkpoint_data_cfg, seed)
    model.load_state_dict(checkpoint["model_state"])
    requested = str(project_cfg["training"].get("device", "auto"))
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")
    model.to(device).eval()
    dataset = QuantumTrajectoryDataset(Path(checkpoint_data_cfg["root"]) / "id.npz", window=None)
    rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        actual = min(batch_size, len(dataset))
        samples = [dataset[i] for i in range(actual)]
        y = torch.stack([s["y"] for s in samples]).to(device)
        rho0 = torch.stack([s["rho0"] for s in samples]).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                model(y, rho0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            qr_timings = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = model(y, rho0, profile_qr=(run_cfg["variant"] == "kraus"))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                if "qr_seconds" in output.aux:
                    qr_timings.append(float(output.aux["qr_seconds"]))
            mean_time = float(np.mean(timings))
            steps = y.shape[1]
            row = {
                "checkpoint": str(checkpoint_path),
                "backbone": run_cfg["backbone"],
                "variant": run_cfg["variant"],
                "rank": int(run_cfg.get("rank", model.rank)),
                "seed": seed,
                "batch_size": actual,
                "sequence_length": steps,
                "rollout_seconds_mean": mean_time,
                "rollout_seconds_std": float(np.std(timings)),
                "microseconds_per_timestep_per_trajectory": 1e6 * mean_time / (actual * steps),
                "trajectories_per_second": actual / mean_time,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "trainable_parameters": count_parameters(model),
                "qr_seconds_mean": float(np.mean(qr_timings)) if qr_timings else 0.0,
                "qr_time_fraction": float(np.mean(qr_timings) / mean_time) if qr_timings else 0.0,
                "device": str(device),
            }
            rows.append(row)
    return rows


def benchmark_all(project_cfg: dict[str, Any], runs_root: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    runs_root = Path(runs_root)
    output_dir = ensure_dir(output_dir)
    cfg = project_cfg.get("benchmark", {})
    checkpoints = sorted(runs_root.glob("**/checkpoint_best.pt"))
    include_stages = set(cfg.get("stages", []))
    include_seeds = {int(x) for x in cfg.get("seeds", [])}
    include_variants = set(cfg.get("variants", []))
    include_backbones = set(cfg.get("backbones", []))
    filtered = []
    for checkpoint in checkpoints:
        rel = checkpoint.relative_to(runs_root)
        stage = rel.parts[0] if len(rel.parts) > 0 else ""
        run_name = checkpoint.parent.name
        parts = run_name.split("__")
        backbone = parts[0] if parts else ""
        variant = parts[1] if len(parts) > 1 else ""
        try:
            seed = int(parts[-1].lstrip("s"))
        except Exception:
            seed = -1
        if include_stages and stage not in include_stages:
            continue
        if include_seeds and seed not in include_seeds:
            continue
        if include_variants and variant not in include_variants:
            continue
        if include_backbones and backbone not in include_backbones:
            continue
        filtered.append(checkpoint)
    rows: list[dict[str, Any]] = []
    for checkpoint in filtered:
        rows.extend(
            benchmark_checkpoint(
                project_cfg,
                checkpoint,
                batch_sizes=[int(x) for x in cfg.get("batch_sizes", [1, 16, 64])],
                warmup=int(cfg.get("warmup", 5)),
                repeats=int(cfg.get("repeats", 20)),
            )
        )
    if rows:
        fields = sorted({k for row in rows for k in row})
        with (output_dir / "runtime_memory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    save_json(output_dir / "runtime_memory.json", rows)
    return rows
