"""Bounded training-path scale diagnostic, kept separate from learned-field errors."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .precision import control_target
from .preflight import fingerprint, provenance
from .research_paths import BASE_METHODS
from .smoke import external_directory
from .targets import path_module, precision_target


def summarize_paths(records, method, rows=128, seed=42):
    if not isinstance(rows, int) or not 1 <= rows <= 4096 or method not in BASE_METHODS:
        raise ValueError("Use an original path and 1..4096 diagnostic rows")
    times = np.unique(
        np.r_[0.0, np.geomspace(1e-5, 0.1, 25), np.linspace(0.1, 1.0, 46)]
    )
    sums = np.zeros((len(times), 4))
    module = path_module(method)
    target = (
        precision_target
        if method == "precision"
        else (control_target if module is None else module.sample_target)
    )
    rng = np.random.default_rng([seed, 72])
    selected = rng.choice(
        int(records.offsets[-1]), min(rows, int(records.offsets[-1])), replace=False
    )
    row_ids = []
    for row in selected:
        i = int(np.searchsorted(records.offsets[1:], row, side="right"))
        _, item = records.sample(i, int(row - records.offsets[i]))
        row_ids.append(int(item["row_id"]))
        z = np.random.default_rng([seed, int(item["row_id"]), 73]).standard_normal(
            item["source"].S.shape[0]
        )
        for index, t in enumerate(times):
            x, u = target(item["x1"], t, z, item["source"], item["label"])
            sums[index] += [
                np.square(x).sum(),
                np.square(u).sum(),
                np.square(x).mean(),
                np.square(u).mean(),
            ]
    sums /= len(selected)
    return {
        "method": method,
        "seed": seed,
        "row_ids": row_ids,
        "times": times.tolist(),
        "mean_per_molecule_position_energy": sums[:, 0].tolist(),
        "mean_per_molecule_target_energy": sums[:, 1].tolist(),
        "coordinate_rms": np.sqrt(sums[:, 2]).tolist(),
        "velocity_rms": np.sqrt(sums[:, 3]).tolist(),
        "caveat": "One fixed Gaussian draw per training row, shared across t/method; descriptive conditional-path scales, not marginal-field error or validation performance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=BASE_METHODS, required=True)
    parser.add_argument("--packed-cache-root", type=Path)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Packed path statistics belong in an SHHPC CPU allocation")
    if args.output.exists():
        parser.error("Existing report is protected")
    import yaml

    from .training import load_records

    cfg = yaml.safe_load(args.config.read_text())
    records = load_records(
        args.train_labels,
        "train",
        "all",
        args.method,
        cfg["source"],
        args.packed_cache_root,
    )
    if not hasattr(records, "offsets"):
        raise ValueError("Requires version-3 packed training labels")
    try:
        report = summarize_paths(records, args.method, args.rows)
    finally:
        records.close()
    report.update(
        train_manifest=fingerprint(args.train_labels),
        config=fingerprint(args.config),
        provenance=provenance(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
