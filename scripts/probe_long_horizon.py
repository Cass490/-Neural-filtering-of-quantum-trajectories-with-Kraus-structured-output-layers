#!/usr/bin/env python
# Evaluates trained checkpoints on 10,000-step rollouts (configs/long_horizon.yaml); recursive and soft heads are also run with the Kraus stabilizers ("+stab").
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd
import torch

from kraus_mlst.config import load_config
from kraus_mlst.datasets import load_manifest, prepare_dataset_suite
from kraus_mlst.evaluation import evaluate_checkpoint

TRAIN_Y_MEAN = -8.419694495387375e-06
TRAIN_Y_STD = 0.07088328152894974
SCENARIO = "long_horizon_10k"
TIME_BINS = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]


def probe(cfg: dict, ckpt: Path, stabilize: bool) -> dict:
    out_dir = ckpt.parent / ("evaluation_long_stab" if stabilize else "evaluation_long")
    if not (out_dir / SCENARIO / "summary.json").exists():
        evaluate_checkpoint(cfg, ckpt, out_dir, scenarios=[SCENARIO], stabilize_recursive=stabilize)
    s = json.loads((out_dir / SCENARIO / "summary.json").read_text())
    t = pd.read_csv(out_dir / SCENARIO / "temporal_metrics.csv")
    row = {
        "fidelity": s["fidelity_projected_mean"],
        "trace_err": s["trace_error_raw_mean"],
        "psd_viol": s["psd_violation_raw_mean"],
        "phys": s["physical_state_indicator_mean"],
    }
    for a, b in TIME_BINS:
        w = t[(t.timestep >= a) & (t.timestep < b)]
        tag = f"{a}-{b}"
        row[f"fid_{tag}"] = float(w.fidelity_projected_mean.mean())
        row[f"trace_err_{tag}"] = float(w.trace_error_raw_mean.mean())
        row[f"psd_viol_{tag}"] = float(w.psd_violation_raw_mean.mean())
        row[f"phys_{tag}"] = float(w.physical_state_indicator_mean.mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon (10k-step) evaluation of trained checkpoints.")
    parser.add_argument("--config", default="configs/long_horizon.yaml")
    parser.add_argument("--checkpoints", nargs="+", default=["outputs/mlst/core/*/checkpoint_best.pt", "outputs/mlst/soft_sweep/*/checkpoint_best.pt"])
    parser.add_argument("--backbones", default="tcn,gru,lstm")
    parser.add_argument("--variants", default="direct_rho0,recursive,soft,cholesky,choi,kraus")
    parser.add_argument("--out", default="outputs/mlst/aggregate/long_horizon_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_dataset_suite(cfg, force=False)
    norm = load_manifest(Path(cfg["data"]["root"]))["normalization"]
    if not (math.isclose(norm["mean_per_channel"][0], TRAIN_Y_MEAN, rel_tol=1e-6)
            and math.isclose(norm["std_per_channel"][0], TRAIN_Y_STD, rel_tol=1e-6)):
        raise SystemExit("Long-horizon data normalization differs from the training data.")
    want_b = {x for x in args.backbones.split(",") if x}
    want_v = {x for x in args.variants.split(",") if x}

    rows = []
    for path in sorted({p for pattern in args.checkpoints for p in glob.glob(pattern)}):
        ckpt = torch.load(path, map_location="cpu")
        run_cfg = ckpt["run_cfg"]
        if run_cfg["backbone"] not in want_b or run_cfg["variant"] not in want_v:
            continue
        tag = run_cfg.get("run_tag") or ""
        for stabilize in ([False, True] if run_cfg["variant"] in {"recursive", "soft"} else [False]):
            label = run_cfg["variant"] + (f"[{tag}]" if tag else "") + ("+stab" if stabilize else "")
            print(f"{Path(path).parent.name} {label}", flush=True)
            rows.append({"backbone": run_cfg["backbone"], "variant": label, "rank": run_cfg.get("rank", 2),
                         "seed": int(ckpt["seed"]), "scenario": SCENARIO, **probe(cfg, Path(path), stabilize)})
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
