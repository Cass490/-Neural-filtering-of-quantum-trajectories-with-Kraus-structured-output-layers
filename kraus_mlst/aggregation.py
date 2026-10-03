# Collects per-run evaluations into seed-aggregated result tables.
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_dir, save_json


def _ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1)) * float(values.std(ddof=1)) / np.sqrt(len(values))


def collect_evaluation_summaries(runs_root: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(runs_root).glob("**/evaluation/all_scenarios_summary.csv"):
        frame = pd.read_csv(path)
        frame["source"] = str(path)
        try:
            frame["stage"] = path.relative_to(Path(runs_root)).parts[0]
        except Exception:
            frame["stage"] = "unknown"
        if "run_tag" not in frame.columns:
            frame["run_tag"] = ""
        frame["run_tag"] = frame["run_tag"].fillna("")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_results(project_cfg: dict[str, Any], runs_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    df = collect_evaluation_summaries(runs_root)
    outputs: dict[str, str] = {}
    if df.empty:
        save_json(output_dir / "aggregation_status.json", {"status": "no_evaluations_found"})
        return outputs

    raw_path = output_dir / "all_run_summaries.csv"
    df.to_csv(raw_path, index=False)
    outputs["all_run_summaries"] = str(raw_path)

    seed_filter = project_cfg.get("aggregation", {}).get("seeds")
    if seed_filter:
        df = df[df["seed"].isin([int(s) for s in seed_filter])].copy()

    group_cols = ["stage", "run_tag", "n_qubits", "hilbert_dim", "scenario", "backbone", "variant", "rank"]
    metrics = [
        c
        for c in df.columns
        if c.endswith("_mean") and c not in group_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean())
            row[f"{metric}_seed_std"] = float(values.std(ddof=1)) if values.count() > 1 else 0.0
            row[f"{metric}_ci95"] = _ci95(values)
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate_path = output_dir / "seed_aggregated_results.csv"
    aggregate.to_csv(aggregate_path, index=False)
    outputs["seed_aggregated_results"] = str(aggregate_path)

    core_scenario = str(project_cfg.get("aggregation", {}).get("core_scenario", "id"))
    core = aggregate[(aggregate["stage"] == "core") & (aggregate["scenario"] == core_scenario)].copy()
    core_metrics = [
        "fidelity_projected_mean",
        "bures_distance_projected_mean",
        "trace_error_raw_mean",
        "psd_violation_raw_mean",
        "physical_state_indicator_mean",
        "recovery_steps_mean",
        "choi_tp_error_mean",
        "choi_effective_rank_mean",
    ]
    present = [c for c in core_metrics if c in core.columns]
    core_table = core[["backbone", "variant", "run_tag", "rank", "n_seeds", *present]].sort_values(["backbone", "variant", "run_tag"])
    core_path = output_dir / "table1_matched_baselines.csv"
    core_table.to_csv(core_path, index=False)
    outputs["table1_matched_baselines"] = str(core_path)

    decomp_rows = []
    for backbone, group in core.groupby("backbone"):
        indexed = group.set_index("variant")
        required = {"direct_legacy", "direct_rho0", "recursive", "kraus"}
        if not required.issubset(set(indexed.index)):
            continue
        metric = "fidelity_projected_mean"
        direct_legacy = float(indexed.loc["direct_legacy", metric])
        direct_rho0 = float(indexed.loc["direct_rho0", metric])
        recursive = float(indexed.loc["recursive", metric])
        kraus = float(indexed.loc["kraus", metric])
        decomp_rows.append(
            {
                "backbone": backbone,
                "direct_legacy": direct_legacy,
                "direct_rho0": direct_rho0,
                "recursive_unconstrained": recursive,
                "kraus": kraus,
                "delta_rho0_information": direct_rho0 - direct_legacy,
                "delta_recursive_structure": recursive - direct_rho0,
                "delta_hard_kraus_over_matched_recursive": kraus - recursive,
            }
        )
    decomp = pd.DataFrame(decomp_rows)
    decomp_path = output_dir / "attribution_decomposition.csv"
    decomp.to_csv(decomp_path, index=False)
    outputs["attribution_decomposition"] = str(decomp_path)

    main_variants = set(project_cfg.get("aggregation", {}).get("main_variants", ["recursive", "soft", "cholesky", "choi", "kraus"]))
    robustness = aggregate[(aggregate["stage"] == "core") & aggregate["variant"].isin(main_variants)].copy()
    robust_cols = ["scenario", "backbone", "variant", "run_tag", "rank", "n_seeds", *[c for c in present if c in robustness.columns]]
    robustness_path = output_dir / "table2_robustness.csv"
    robustness[robust_cols].sort_values(["scenario", "backbone", "variant"]).to_csv(robustness_path, index=False)
    outputs["table2_robustness"] = str(robustness_path)

    soft_cols = [
        "backbone", "run_tag", "n_seeds",
        "fidelity_projected_mean", "trace_error_raw_mean",
        "psd_violation_raw_mean", "hermiticity_error_raw_mean",
        "physical_state_indicator_mean",
    ]
    sweep = aggregate[(aggregate["stage"] == "soft_sweep") & (aggregate["scenario"] == core_scenario)].copy()
    if not sweep.empty:
        medium = aggregate[
            (aggregate["stage"] == "core") & (aggregate["variant"] == "soft") & (aggregate["scenario"] == core_scenario)
            & aggregate["backbone"].isin(set(sweep["backbone"]))
        ].copy()
        medium["run_tag"] = "medium"
        sweep = pd.concat([sweep, medium], ignore_index=True, sort=False)
        sweep_path = output_dir / "soft_penalty_sweep.csv"
        sweep[[c for c in soft_cols if c in sweep.columns]].sort_values(["backbone", "run_tag"]).to_csv(sweep_path, index=False)
        outputs["soft_penalty_sweep"] = str(sweep_path)

    def _aggregate_curve(filename: str, index_col: str, output_name: str) -> None:
        curve_frames = []
        for curve_path in Path(runs_root).glob(f"core/**/evaluation/{core_scenario}/{filename}"):
            run_id = curve_path.parents[2].name
            parts = run_id.split("__")
            if len(parts) < 2:
                continue
            frame = pd.read_csv(curve_path)
            frame["backbone"] = parts[0]
            frame["variant"] = parts[1]
            curve_frames.append(frame)
        if not curve_frames:
            return
        curves = pd.concat(curve_frames, ignore_index=True, sort=False)
        numeric = [c for c in curves.columns if c not in {index_col, "backbone", "variant"} and pd.api.types.is_numeric_dtype(curves[c])]
        grouped = curves.groupby(["backbone", "variant", index_col], as_index=False)[numeric].mean()
        path = output_dir / output_name
        grouped.to_csv(path, index=False)
        outputs[output_name.removesuffix(".csv")] = str(path)

    _aggregate_curve("choi_switch_aligned.csv", "relative_step", "choi_switch_aligned_seed_aggregate.csv")
    _aggregate_curve("temporal_metrics.csv", "timestep", "temporal_metrics_seed_aggregate.csv")

    save_json(output_dir / "aggregation_outputs.json", outputs)
    return outputs
