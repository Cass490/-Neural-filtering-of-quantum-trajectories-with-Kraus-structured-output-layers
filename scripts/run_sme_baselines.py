#!/usr/bin/env python
# Runs the model-based SME filter baselines.
from __future__ import annotations

import argparse

from kraus_mlst.config import load_config
from kraus_mlst.sme import evaluate_sme_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = evaluate_sme_baselines(cfg, f"{cfg['output_root']}/aggregate")
    print(f"Wrote {len(rows)} SME baseline rows")


if __name__ == "__main__":
    main()
