#!/usr/bin/env python
"""Run local RDKit ETKDGv3 + MMFF on the synthetic smoke fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.rdkit_etkdg import run_rdkit_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/tiny_smoke.yaml",
        help="Smoke config providing dataset, split, seed, and output root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override. Defaults to config experiment.output_root.",
    )
    parser.add_argument(
        "--run-name",
        default="smoke_random_rdkit_etkdg_mmff_seed0",
        help="Name used for manifest/result/log files.",
    )
    args = parser.parse_args()

    paths = run_rdkit_smoke(
        config_path=args.config,
        output_root_override=args.output_root,
        run_name=args.run_name,
    )
    print(f"manifest: {paths['manifest']}")
    print(f"results: {paths['results']}")
    print(f"log: {paths['log']}")


if __name__ == "__main__":
    main()
