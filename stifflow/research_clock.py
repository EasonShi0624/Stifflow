"""Train-only global clock: piecewise-linear q(t), exact quadratic CDF inverse."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .geometry import centered
from .precision import schedules


class PathClock:
    def __init__(self, specification):
        if (
            specification.get("schema_version") != 1
            or specification.get("split") != "train"
        ):
            raise ValueError("Clock must be a version-1 train-only artifact")
        self.t = np.asarray(specification["t"], dtype=np.float64)
        self.q = np.asarray(specification["q"], dtype=np.float64)
        if (
            self.t.ndim != 1
            or len(self.t) < 2
            or self.q.shape != self.t.shape
            or not np.isfinite(self.t).all()
            or not np.isfinite(self.q).all()
        ):
            raise ValueError("Invalid clock arrays")
        if (
            self.t[0] != 0
            or self.t[-1] != 1
            or np.any(np.diff(self.t) <= 0)
            or np.any(self.q <= 0)
        ):
            raise ValueError("Clock requires [0,1], increasing knots and positive q")
        areas = np.diff(self.t) * (self.q[:-1] + self.q[1:]) / 2
        self.cdf = np.r_[0.0, np.cumsum(areas)]
        if not np.isclose(self.cdf[-1], 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Clock density must integrate to one")
        self.cdf[-1] = 1.0
        if np.any(np.diff(self.cdf) <= 0):
            raise ValueError("Clock CDF must remain strictly increasing in float64")

    def inverse(self, tau):
        if not np.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("Clock time must lie in [0,1]")
        i = min(int(np.searchsorted(self.cdf, tau, side="right")) - 1, len(self.t) - 2)
        slope = (self.q[i + 1] - self.q[i]) / (self.t[i + 1] - self.t[i])
        area = tau - self.cdf[i]
        # Rationalized positive root also works when the slope is zero.
        delta = (
            2
            * area
            / (self.q[i] + np.sqrt(max(0.0, self.q[i] ** 2 + 2 * slope * area)))
        )
        t = float(np.clip(self.t[i] + delta, self.t[i], self.t[i + 1]))
        return t, float(self.q[i] + slope * (t - self.t[i]))


def precision_energy(x1, source, label, times):
    """E_z ||U_t||², including the mean term, without dense factors per knot."""
    b = np.asarray(label["values"])
    weights = np.square(source.S @ label["vectors"]).sum(axis=0)
    mean_norm = np.square(centered(x1)).sum()
    energies = []
    for t in times:
        alpha, complement, alphadot, s = schedules(t)
        denominator = complement + alpha * b
        derivative = -(1 + 0.5 * s * alphadot * (b - 1) / denominator) / np.sqrt(
            denominator
        )
        energies.append(alphadot**2 * mean_norm + weights @ derivative**2)
    return np.asarray(energies)


def fit_clock(records, rows=512, seed=42):
    """Small fixed row-uniform training sample; preprocessing belongs on SHHPC."""
    if not isinstance(rows, int) or not 1 <= rows <= 4096:
        raise ValueError("Clock fit is bounded to 1..4096 training rows")
    # Resolve the early precision layer without endpoint truncation.
    times = np.unique(
        np.r_[0.0, np.geomspace(1e-6, 0.1, 257), np.linspace(0.1, 1.0, 257)]
    )
    rng = np.random.default_rng([seed, 71])
    selected = rng.choice(
        int(records.offsets[-1]),
        size=min(rows, int(records.offsets[-1])),
        replace=False,
    )
    energies, row_ids = np.zeros(len(times)), []
    for row in selected:
        i = int(np.searchsorted(records.offsets[1:], row, side="right"))
        _, item = records.sample(i, int(row - records.offsets[i]))
        energies += precision_energy(item["x1"], item["source"], item["label"], times)
        row_ids.append(int(item["row_id"]))
    energies /= len(selected)
    q = np.sqrt(energies)
    q /= np.trapz(q, times)
    specification = {
        "schema_version": 1,
        "split": "train",
        "t": times.tolist(),
        "q": q.tolist(),
        "mean_target_energy": energies.tolist(),
        "row_ids": row_ids,
        "seed": seed,
    }
    PathClock(specification)
    return specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packed-cache-root", type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Clock fitting belongs in an SHHPC allocation")
    if args.output.exists():
        parser.error("Clock output already exists")
    import yaml

    from .preflight import fingerprint, provenance
    from .smoke import external_directory
    from .training import load_records

    external_directory(args.output)
    cfg = yaml.safe_load(args.config.read_text())
    records = load_records(
        args.train_labels,
        "train",
        "all",
        "precision",
        cfg["source"],
        args.packed_cache_root,
    )
    if not hasattr(records, "offsets"):
        raise ValueError("Clock fitting requires version-3 packed training labels")
    try:
        result = fit_clock(records, args.rows, args.seed)
    finally:
        records.close()
    result.update(
        train_manifest=fingerprint(args.train_labels),
        source_config=cfg["source"],
        provenance=provenance(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
