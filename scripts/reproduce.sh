#!/usr/bin/env bash
# Runs every experiment of the paper; set CONFIG=configs/smoke.yaml for a quick end-to-end check.
set -euo pipefail
CONFIG=${CONFIG:-configs/paper.yaml}
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

python scripts/prepare_data.py --config "$CONFIG"
for stage in $(python -c "import yaml; print(' '.join(yaml.safe_load(open('$CONFIG'))['experiments']))"); do
  python scripts/run_suite.py --config "$CONFIG" --stage "$stage"
done
python scripts/aggregate_results.py --config "$CONFIG"
python scripts/run_sme_baselines.py --config "$CONFIG"
python scripts/benchmark_runtime.py --config "$CONFIG"
python scripts/numerical_audit.py --config "$CONFIG"
if [ "$CONFIG" = "configs/paper.yaml" ]; then
  python scripts/probe_long_horizon.py --config configs/long_horizon.yaml 
fi
