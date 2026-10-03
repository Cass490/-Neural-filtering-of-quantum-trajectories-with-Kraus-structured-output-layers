#!/usr/bin/env python
# Runs the Kraus stabilizer numerical audit on trained checkpoints.
from __future__ import annotations

import argparse

from kraus_mlst.audit import numerical_audit_all
from kraus_mlst.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = numerical_audit_all(cfg, cfg["output_root"], f"{cfg['output_root']}/aggregate")
    print(f"Audited {len(rows)} model/mode rows")


if __name__ == "__main__":
    main()
