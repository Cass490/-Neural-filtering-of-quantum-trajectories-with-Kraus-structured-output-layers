#!/usr/bin/env python
# Trains and evaluates every run of one experiment stage in configs/paper.yaml (core, soft_sweep, supplementary).
from __future__ import annotations

import argparse
import traceback
from typing import Any

from kraus_mlst.config import deep_update, load_config
from kraus_mlst.datasets import prepare_dataset_suite
from kraus_mlst.evaluation import evaluate_checkpoint
from kraus_mlst.training import train_run
from kraus_mlst.utils import ensure_dir, save_json

RUN_KEYS = ["backbone", "variant", "run_tag", "rank", "backbone_cfg", "head_cfg", "training_overrides", "loss_overrides"]


def build_plan(cfg: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    stage_cfg = cfg["experiments"][stage]
    base_backbone_cfg = cfg.get("model", {}).get("backbone_cfg", {})
    base_head_cfg = cfg.get("model", {}).get("head_cfg", {})
    ranks = stage_cfg.get("ranks", [cfg.get("model", {}).get("rank", 2)])
    stage_seeds = stage_cfg.get("seeds", cfg.get("seeds", [11, 23, 37, 51, 73]))
    overrides = stage_cfg.get("training_overrides", {})
    plan = []
    for backbone in stage_cfg["backbones"]:
        variants = stage_cfg.get("variants_by_backbone", {}).get(backbone, stage_cfg["variants"])
        seeds = [int(s) for s in stage_cfg.get("seeds_by_backbone", {}).get(backbone, stage_seeds)]
        for variant in variants:
            variant_ranks = ranks if variant in {"kraus", "choi"} else [ranks[0]]
            profiles = stage_cfg.get("soft_profiles", {"default": {}}) if variant == "soft" else {"default": {}}
            for profile_name, profile_loss in profiles.items():
                tag = profile_name if profile_name != "default" else ""
                for rank in variant_ranks:
                    for seed in seeds:
                        suffix = f"__{tag}" if tag else ""
                        plan.append(
                            {
                                "run_id": f"{backbone}__{variant}{suffix}__r{int(rank)}__s{seed}",
                                "seed": seed,
                                "backbone": backbone,
                                "variant": variant,
                                "run_tag": tag,
                                "rank": int(rank),
                                "backbone_cfg": deep_update(base_backbone_cfg, stage_cfg.get("backbone_overrides", {}).get(backbone, {})),
                                "head_cfg": deep_update(base_head_cfg, stage_cfg.get("head_overrides", {}).get(variant, {})),
                                "training_overrides": deep_update(overrides.get("default", {}), overrides.get(backbone, {})),
                                "loss_overrides": profile_loss,
                            }
                        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one experiment stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage not in cfg["experiments"]:
        raise SystemExit(f"unknown stage {args.stage}; choose from {list(cfg['experiments'])}")
    stage_root = ensure_dir(ensure_dir(cfg["output_root"]) / args.stage)
    plan = build_plan(cfg, args.stage)
    save_json(stage_root / "launch_plan.json", plan)
    if args.dry_run:
        for i, item in enumerate(plan):
            print(f"{i:4d}  {item['run_id']}")
        print(f"{len(plan)} runs")
        return

    prepare_dataset_suite(cfg, force=False)
    allow_skip = bool(cfg["experiments"][args.stage].get("allow_optional_skip", False))
    failures = []
    for item in plan:
        run_dir = stage_root / item["run_id"]
        try:
            print(train_run(cfg, {k: item[k] for k in RUN_KEYS}, int(item["seed"]), run_dir, overwrite=args.overwrite))
            if not args.skip_evaluation and (args.overwrite or not (run_dir / "evaluation" / "all_scenarios_summary.csv").exists()):
                evaluate_checkpoint(cfg, run_dir / "checkpoint_best.pt", run_dir / "evaluation")
        except Exception as exc:
            failure = {"run_id": item["run_id"], "error": repr(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            save_json(run_dir / "FAILED.json", failure)
            if not allow_skip:
                raise
            print(f"FAILED {item['run_id']}: {exc}")
    save_json(stage_root / "failures.json", failures)
    print(f"Stage {args.stage} complete. failures={len(failures)}")


if __name__ == "__main__":
    main()
