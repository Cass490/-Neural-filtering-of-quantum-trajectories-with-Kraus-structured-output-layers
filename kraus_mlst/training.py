# Windowed training followed by full-rollout fine-tuning; writes the best checkpoint.
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import deep_update
from .datasets import QuantumTrajectoryDataset
from .losses import compute_training_loss
from .models import build_model
from .utils import count_parameters, ensure_dir, environment_snapshot, save_json, set_global_seed


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def _make_loaders(
    data_root: Path,
    training_cfg: dict[str, Any],
    seed: int,
) -> tuple[QuantumTrajectoryDataset, DataLoader, DataLoader]:
    window = int(training_cfg.get("window", 256))
    train_ds = QuantumTrajectoryDataset(
        data_root / "train.npz",
        window=window,
        windows_per_trajectory=int(training_cfg.get("windows_per_trajectory", 1)),
        random_windows=True,
        seed=seed,
    )
    val_ds = QuantumTrajectoryDataset(
        data_root / "val.npz",
        window=window,
        windows_per_trajectory=int(training_cfg.get("val_windows_per_trajectory", 1)),
        random_windows=False,
        seed=seed,
    )
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_ds, train_loader, val_loader


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / max(1, len(rows)) for key in keys}


def _validation(
    model: torch.nn.Module,
    loader: DataLoader,
    variant: str,
    loss_cfg: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            y = batch["y"].to(device, non_blocking=True)
            rho0 = batch["rho0"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            output = model(y, rho0)
            _, components = compute_training_loss(output, target, variant, loss_cfg)
            rows.append(components)
    return _mean_dict(rows)


def train_run(
    project_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    seed: int,
    run_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not overwrite:
        import json

        return json.loads(done_path.read_text(encoding="utf-8"))

    local_cfg = dict(project_cfg)
    local_cfg["training"] = deep_update(project_cfg["training"], run_cfg.get("training_overrides", {}))
    local_cfg["loss"] = deep_update(project_cfg.get("loss", {}), run_cfg.get("loss_overrides", {}))
    set_global_seed(seed, deterministic=bool(local_cfg["training"].get("deterministic", True)))
    device = _device_from_cfg(local_cfg["training"])
    data_root = Path(local_cfg["data"]["root"])
    train_ds, train_loader, val_loader = _make_loaders(data_root, local_cfg["training"], seed)
    model = build_model(run_cfg, local_cfg["data"], seed).to(device)

    lr = float(local_cfg["training"].get("learning_rate", 5e-4))
    weight_decay = float(local_cfg["training"].get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(local_cfg["training"].get("lr_factor", 0.5)),
        patience=int(local_cfg["training"].get("lr_patience", 5)),
    )

    epochs = int(local_cfg["training"].get("epochs", 100))
    grad_clip = float(local_cfg["training"].get("grad_clip", 1.0))
    accumulation = int(local_cfg["training"].get("gradient_accumulation", 1))
    max_train_batches = local_cfg["training"].get("max_train_batches")
    max_val_batches = local_cfg["training"].get("max_val_batches")
    loss_cfg = local_cfg.get("loss", {})

    save_json(run_dir / "run_config.json", {"project": local_cfg, "run": run_cfg, "seed": seed})
    save_json(run_dir / "environment.json", environment_snapshot())
    save_json(
        run_dir / "model_info.json",
        {
            "trainable_parameters": count_parameters(model),
            "all_parameters": count_parameters(model, trainable_only=False),
            "device": str(device),
        },
    )

    best_val = math.inf
    best_epoch = -1
    epoch_rows: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_rows: list[dict[str, float]] = []
        start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"{run_cfg['backbone']}/{run_cfg['variant']} seed={seed} epoch={epoch}")
        for batch_idx, batch in enumerate(progress):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            y = batch["y"].to(device, non_blocking=True)
            rho0 = batch["rho0"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            output = model(y, rho0)
            loss, components = compute_training_loss(output, target, run_cfg["variant"], loss_cfg)
            (loss / accumulation).backward()

            grad_norm = 0.0
            if (batch_idx + 1) % accumulation == 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm = float(grad_norm_tensor.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            components["grad_norm"] = grad_norm
            components["hidden_rms"] = float(output.aux.get("hidden_rms", float("nan")))
            if "qr_min_diag" in output.aux:
                components["qr_min_diag"] = float(output.aux["qr_min_diag"])
            batch_rows.append(components)
            progress.set_postfix(loss=f"{components['total_loss']:.4g}")

        if len(batch_rows) % accumulation != 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if batch_rows:
                batch_rows[-1]["grad_norm"] = float(grad_norm_tensor.detach().cpu())

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - start
        train_summary = _mean_dict(batch_rows)
        val_summary = _validation(
            model,
            val_loader,
            run_cfg["variant"],
            loss_cfg,
            device,
            max_batches=int(max_val_batches) if max_val_batches is not None else None,
        )
        val_total = val_summary.get("total_loss", math.inf)
        scheduler.step(val_total)

        row: dict[str, float | int] = {
            "phase": "window_train",
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_summary.items()},
            **{f"val_{k}": v for k, v in val_summary.items()},
        }
        if device.type == "cuda":
            row["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        epoch_rows.append(row)
        tqdm.write(
            f"epoch {epoch + 1}/{epochs} train={train_summary.get('total_loss', float('nan')):.4f} "
            f"val_mse={val_summary.get('prediction_mse', float('nan')):.4f} lr={row['learning_rate']:.1e} "
            f"{epoch_seconds:.0f}s"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "run_cfg": run_cfg,
            "seed": seed,
            "epoch": epoch,
            "val_total_loss": val_total,
            "project_data_cfg": local_cfg["data"],
        }
        torch.save(checkpoint, run_dir / "checkpoint_last.pt")
        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "checkpoint_best.pt")

        with (run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for r in epoch_rows for key in r}))
            writer.writeheader()
            writer.writerows(epoch_rows)

    full_epochs = int(local_cfg["training"].get("full_rollout_finetune_epochs", 0))
    if full_epochs > 0:
        best_checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
        torch.save(best_checkpoint, run_dir / "checkpoint_best_window.pt")
        model.load_state_dict(best_checkpoint["model_state"])
        full_best = math.inf
        full_train_base = QuantumTrajectoryDataset(data_root / "train.npz", window=None)
        full_val_base = QuantumTrajectoryDataset(data_root / "val.npz", window=None)
        n_train_full = min(int(local_cfg["training"].get("full_rollout_train_trajectories", 256)), len(full_train_base))
        n_val_full = min(int(local_cfg["training"].get("full_rollout_val_trajectories", 64)), len(full_val_base))
        full_train_loader = DataLoader(
            Subset(full_train_base, range(n_train_full)),
            batch_size=int(local_cfg["training"].get("full_rollout_batch_size", 4)),
            shuffle=True,
            num_workers=0,
        )
        full_val_loader = DataLoader(
            Subset(full_val_base, range(n_val_full)),
            batch_size=int(local_cfg["training"].get("full_rollout_batch_size", 4)),
            shuffle=False,
            num_workers=0,
        )
        full_accum = int(local_cfg["training"].get("full_rollout_gradient_accumulation", 8))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(local_cfg["training"].get("full_rollout_learning_rate", lr * 0.2)),
            weight_decay=weight_decay,
        )
        for full_epoch in range(full_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_rows = []
            start = time.perf_counter()
            progress = tqdm(full_train_loader, desc=f"full-rollout {run_cfg['backbone']}/{run_cfg['variant']} seed={seed} epoch={full_epoch}")
            for batch_idx, batch in enumerate(progress):
                y = batch["y"].to(device, non_blocking=True)
                rho0 = batch["rho0"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                output = model(y, rho0)
                loss, components = compute_training_loss(output, target, run_cfg["variant"], loss_cfg)
                (loss / full_accum).backward()
                grad_norm = 0.0
                if (batch_idx + 1) % full_accum == 0:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm = float(grad_norm_tensor.detach().cpu())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                components["grad_norm"] = grad_norm
                components["hidden_rms"] = float(output.aux.get("hidden_rms", float("nan")))
                if "qr_min_diag" in output.aux:
                    components["qr_min_diag"] = float(output.aux["qr_min_diag"])
                batch_rows.append(components)
            if len(batch_rows) % full_accum != 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if batch_rows:
                    batch_rows[-1]["grad_norm"] = float(grad_norm_tensor.detach().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_summary = _mean_dict(batch_rows)
            val_summary = _validation(model, full_val_loader, run_cfg["variant"], loss_cfg, device)
            val_total = val_summary.get("total_loss", math.inf)
            row = {
                "phase": "full_rollout_finetune",
                "epoch": epochs + full_epoch,
                "epoch_seconds": time.perf_counter() - start,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_summary.items()},
                **{f"val_{k}": v for k, v in val_summary.items()},
            }
            if device.type == "cuda":
                row["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            epoch_rows.append(row)
            tqdm.write(
                f"full-rollout finetune {full_epoch + 1}/{full_epochs} "
                f"val_mse={val_summary.get('prediction_mse', float('nan')):.4f} {row['epoch_seconds']:.0f}s"
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "run_cfg": run_cfg,
                "seed": seed,
                "epoch": epochs + full_epoch,
                "val_total_loss": val_total,
                "project_data_cfg": local_cfg["data"],
            }
            torch.save(checkpoint, run_dir / "checkpoint_last.pt")
            if val_total < full_best:
                full_best = val_total
                best_val = val_total
                best_epoch = epochs + full_epoch
                torch.save(checkpoint, run_dir / "checkpoint_best.pt")
            with (run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted({key for r in epoch_rows for key in r}))
                writer.writeheader()
                writer.writerows(epoch_rows)

    result = {
        "status": "complete",
        "backbone": run_cfg["backbone"],
        "variant": run_cfg["variant"],
        "rank": int(run_cfg.get("rank", 2)),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "trainable_parameters": count_parameters(model),
        "effective_batch_size": int(local_cfg["training"].get("batch_size", 32)) * accumulation,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    save_json(done_path, result)
    return result
