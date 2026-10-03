#!/usr/bin/env python
# Generates the datasets defined in a config.
from __future__ import annotations

import argparse

from kraus_mlst.config import load_config
from kraus_mlst.datasets import prepare_dataset_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-free ID/OOD quantum trajectory datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = prepare_dataset_suite(cfg, force=args.force)
    print(f"Dataset manifest: {manifest}")


if __name__ == "__main__":
    main()
