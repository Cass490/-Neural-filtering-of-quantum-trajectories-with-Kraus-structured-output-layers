#!/usr/bin/env python
# Benchmarks inference latency of trained checkpoints.
from __future__ import annotations

import argparse

from kraus_mlst.benchmarking import benchmark_all
from kraus_mlst.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = benchmark_all(cfg, cfg["output_root"], f"{cfg['output_root']}/aggregate")
    print(f"Benchmarked {len(rows)} rows")


if __name__ == "__main__":
    main()
