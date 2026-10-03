#!/usr/bin/env python
# Trains and evaluates a single run of a stage by its index (see run_suite.py --dry-run); suited to job arrays.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kraus_mlst.config import load_config
from kraus_mlst.datasets import prepare_dataset_suite
from kraus_mlst.evaluation import evaluate_checkpoint
from kraus_mlst.training import train_run
from kraus_mlst.utils import ensure_dir
from run_suite import RUN_KEYS, build_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one run of a stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    plan = build_plan(cfg, args.stage)
    if not 0 <= args.index < len(plan):
        raise IndexError(f"index={args.index}; valid range is [0,{len(plan) - 1}]")
    item = plan[args.index]
    run_id = item["run_id"]
    run_dir = ensure_dir(Path(cfg["output_root"]) / args.stage / run_id)
    prepare_dataset_suite(cfg, force=False)
    start = time.time()
    print(f"\n===== index {args.index}: {run_id} =====", flush=True)
    if (run_dir / "DONE.json").exists() and not args.overwrite:
        print("training already done, skipping to evaluation", flush=True)
    train_run(cfg, {k: item[k] for k in RUN_KEYS}, int(item["seed"]), run_dir, overwrite=args.overwrite)
    if args.skip_evaluation:
        return
    if (run_dir / "evaluation" / "all_scenarios_summary.csv").exists() and not args.overwrite:
        print("evaluation already done", flush=True)
    else:
        evaluate_checkpoint(cfg, run_dir / "checkpoint_best.pt", run_dir / "evaluation")
    s = json.loads((run_dir / "evaluation" / "id" / "summary.json").read_text())
    print(
        f"RESULT {run_id}: fid={s['fidelity_projected_mean']:.4f} trace_err={s['trace_error_raw_mean']:.2e} "
        f"psd={s['psd_violation_raw_mean']:.2e} phys={s['physical_state_indicator_mean']:.2f} "
        f"({(time.time() - start) / 60:.0f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
