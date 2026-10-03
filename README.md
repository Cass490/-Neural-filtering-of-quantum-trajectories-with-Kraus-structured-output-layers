# Kraus-constrained sequence learning for quantum trajectories

Code and per-run results for the single-qubit experiments of the paper by Priyanshi Singh and Krishna Bhatia.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Reproduce

```bash
bash scripts/reproduce.sh                          # full paper suite (GPU recommended)
CONFIG=configs/smoke.yaml bash scripts/reproduce.sh  # few-minute CPU check
python -m pytest
```

Step by step:

| Step | Command |
|---|---|
| Generate data (2000/300/300 trajectories, 12 test scenarios) | `python scripts/prepare_data.py --config configs/paper.yaml` |
| List the runs of a stage | `python scripts/run_suite.py --config configs/paper.yaml --stage core --dry-run` |
| Train + evaluate a stage | `python scripts/run_suite.py --config configs/paper.yaml --stage core` |
| Train + evaluate one run (job arrays) | `python scripts/run_one.py --config configs/paper.yaml --stage core --index 0` |
| Aggregate results | `python scripts/aggregate_results.py --config configs/paper.yaml` |
| SME filter baselines | `python scripts/run_sme_baselines.py --config configs/paper.yaml` |
| Latency benchmark | `python scripts/benchmark_runtime.py --config configs/paper.yaml` |
| Stabilizer audit | `python scripts/numerical_audit.py --config configs/paper.yaml` |
| 10,000-step rollouts | `python scripts/probe_long_horizon.py` |

Stages in `configs/paper.yaml`:

- `core`: TCN, GRU, LSTM, RNN, ESN and SSM × direct+ρ0, recursive, soft, Cholesky, Choi and Kraus heads, plus the legacy direct head for GRU, LSTM and SSM; seeds 11, 23, 37, 51, 73.
- `soft_sweep`: GRU, LSTM and SSM soft heads at weak, strong and very strong penalties; seeds 11, 23, 37 (medium is the core soft head).
- `supplementary`: Neural ODE (direct+ρ0, five seeds) and Transformer (direct+ρ0 and Kraus, three seeds).

Outputs go to `outputs/mlst/<stage>/<run_id>/`, aggregates to `outputs/mlst/aggregate/`.

## Results

`results/` holds the results reported in the paper:

- `paper_runs.csv`: every run × scenario
- `long_horizon_results.csv`: 10,000-step rollouts
- `sme_summary.csv`, `runtime_memory.csv`, `numerical_stabilizer_audit.csv`, `temporal_metrics_seed_aggregate.csv`

## License

MIT. See `LICENSE` and `CITATION.cff`.
