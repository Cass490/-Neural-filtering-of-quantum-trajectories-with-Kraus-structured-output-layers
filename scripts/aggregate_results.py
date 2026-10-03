#!/usr/bin/env python
# Aggregates all evaluated runs into outputs/mlst/aggregate.
from __future__ import annotations

import argparse

from kraus_mlst.aggregation import aggregate_results
from kraus_mlst.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    outputs = aggregate_results(cfg, cfg["output_root"], f"{cfg['output_root']}/aggregate")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
