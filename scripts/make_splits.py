#!/usr/bin/env python
"""Create explicit molecule-level split files for the smoke dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.records import load_records, select_records
from src.data.splits import load_or_create_split, split_file_path
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a smoke-test YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_records(config["data"]["processed_root"])
    records = select_records(records, config["data"].get("subset", {}).get("max_molecules"))
    split = load_or_create_split(
        records,
        split_root=config["data"]["split_root"],
        split_type=config["data"]["split_type"],
        dataset_name=config["data"]["dataset_name"],
        seed=int(config["experiment"]["seed"]),
    )
    path = split_file_path(
        config["data"]["split_root"],
        config["data"]["split_type"],
        config["data"]["dataset_name"],
        int(config["experiment"]["seed"]),
    )
    print(json.dumps({"split_path": str(path), "counts": {k: len(split[k]) for k in ["train", "val", "test"]}}, sort_keys=True))


if __name__ == "__main__":
    main()
