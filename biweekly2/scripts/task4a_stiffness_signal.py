"""Task 4a: quantify how much stiffness signal Alternative A actually carries.

Uses the same 128 training rows, the same fixed z per row, and the same time
grid as the existing P0 path statistics, so results are directly comparable.
For each row and t:

- relative Frobenius difference between A_t (Alternative A) and (1-t) S
  (the control factor);
- per-draw relative velocity difference ||u_A - u_ctrl|| / ||u_ctrl||;
- relative position difference ||x_A - x_ctrl|| / ||x_ctrl||;
- ||L_K - S||_F / ||S||_F and the spectrum of S^{-1} L_K (how anisotropic
  the terminal covariance C_K is relative to the graph prior C_0).

Also aggregates the per-molecule label audit receipts over the validation
split: reference-geometry offset (alignment RMSD before Hessian), floored
modes, curvature calibration ratios, relaxed energies.  These contextualize
the strength and reliability of the stiffness signal.

Outputs runs/biweekly2-20261019/results/task4a-stiffness-signal.json
"""

import json
from pathlib import Path

import numpy as np
import yaml

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
OUT = WS / "runs/biweekly2-20261019/results/task4a-stiffness-signal.json"
TRAIN_MANIFEST = WS / "cache/qm9-v3-packed-strict/train/manifest.json"
CONFIG = WS / "stifflow-v1/configs/qm9-formal-4090d.yaml"
ROWS = 128
SEED = 42
TIMES = np.unique(
    np.r_[0.0, np.geomspace(1e-5, 0.1, 25), np.linspace(0.1, 1.0, 46)]
)


def path_differences(records):
    from stifflow.precision import control_target, schedules
    from stifflow_alt_a.path import factor, sample_target

    rng = np.random.default_rng([SEED, 72])
    total = int(records.offsets[-1])
    selected = rng.choice(total, min(ROWS, total), replace=False)
    factor_diff = {float(t): [] for t in TIMES}
    velocity_diff = {float(t): [] for t in TIMES}
    position_diff = {float(t): [] for t in TIMES}
    lk_norms, lk_spectrum = [], []
    lk_spectrum_stats = {"min": [], "median": [], "max": [], "log_ratio": []}
    row_ids = []
    for row in selected:
        i = int(np.searchsorted(records.offsets[1:], row, side="right"))
        _, item = records.sample(i, int(row - records.offsets[i]))
        row_ids.append(int(item["row_id"]))
        source, label, x1 = item["source"], item["label"], item["x1"]
        z = np.random.default_rng([SEED, int(item["row_id"]), 73]).standard_normal(
            source.S.shape[0]
        )
        S, LK = source.S, np.asarray(label["L_K"], dtype=np.float64)
        lk_norms.append(float(np.linalg.norm(LK - S) / np.linalg.norm(S)))
        spectrum = np.linalg.eigvalsh(np.linalg.solve(S, LK))
        lk_spectrum.append(spectrum)
        lk_spectrum_stats["min"].append(float(spectrum.min()))
        lk_spectrum_stats["median"].append(float(np.median(spectrum)))
        lk_spectrum_stats["max"].append(float(spectrum.max()))
        lk_spectrum_stats["log_ratio"].append(
            float(np.log(spectrum.max() / max(spectrum.min(), 1e-300)))
        )
        for t in TIMES:
            A, _ = factor(float(t), source, label)
            control_A = (1 - float(t)) * S
            denom = np.linalg.norm(control_A)
            if denom > 0:
                factor_diff[float(t)].append(
                    float(np.linalg.norm(A - control_A) / denom)
                )
            xa, ua = sample_target(x1, float(t), z, source, label)
            xc, uc = control_target(x1, float(t), z, source, label)
            un = np.linalg.norm(uc)
            if un > 0:
                velocity_diff[float(t)].append(float(np.linalg.norm(ua - uc) / un))
            xn = np.linalg.norm(xc)
            if xn > 0:
                position_diff[float(t)].append(float(np.linalg.norm(xa - xc) / xn))
    return {
        "rows": len(row_ids),
        "row_ids": row_ids,
        "times": [float(t) for t in TIMES],
        "factor_relative_difference_median": [
            float(np.median(factor_diff[float(t)])) if factor_diff[float(t)] else None
            for t in TIMES
        ],
        "factor_relative_difference_p90": [
            float(np.percentile(factor_diff[float(t)], 90))
            if factor_diff[float(t)]
            else None
            for t in TIMES
        ],
        "velocity_relative_difference_median": [
            float(np.median(velocity_diff[float(t)])) if velocity_diff[float(t)] else None
            for t in TIMES
        ],
        "velocity_relative_difference_p90": [
            float(np.percentile(velocity_diff[float(t)], 90))
            if velocity_diff[float(t)]
            else None
            for t in TIMES
        ],
        "position_relative_difference_median": [
            float(np.median(position_diff[float(t)])) if position_diff[float(t)] else None
            for t in TIMES
        ],
        "L_K_minus_S_relative_norm_median": float(np.median(lk_norms)),
        "L_K_minus_S_relative_norm_p90": float(np.percentile(lk_norms, 90)),
        "S^-1_L_K_spectrum": {
            "min_median": float(np.median(lk_spectrum_stats["min"])),
            "median_median": float(np.median(lk_spectrum_stats["median"])),
            "max_median": float(np.median(lk_spectrum_stats["max"])),
            "log_ratio_median": float(np.median(lk_spectrum_stats["log_ratio"])),
            "log_ratio_p90": float(np.percentile(lk_spectrum_stats["log_ratio"], 90)),
        },
    }


def receipt_audit():
    """Aggregate validation label audit receipts (reference offset, floors)."""
    manifest = json.loads(
        (WS / "cache/qm9-v3-packed-strict/val/manifest.json").read_text()
    )
    alignment, floored, relaxed, calibration, rotations = [], [], [], [], []
    missing = 0
    for record in manifest["records"]:
        receipt_path = record["audit_receipt"]["path"]
        try:
            receipt = json.loads(Path(receipt_path).read_text())
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue
        labels = receipt["record"]["labels"][0]
        alignment.append(labels["alignment_rmsd_angstrom"])
        floored.append(labels["floored_modes"])
        relaxed.append(labels["relaxed_energy_kcal_mol"])
        rotations.append(labels["rotation_rank"])
        calibration.append(
            [c["ratio"] for c in labels.get("calibration", [])]
        )
    alignment = np.array(alignment)
    floored = np.array(floored)
    relaxed = np.array(relaxed)
    rotations = np.array(rotations)
    calibration = np.array(calibration)
    return {
        "molecules": len(alignment),
        "missing_receipts": missing,
        "alignment_rmsd_angstrom": {
            "mean": float(alignment.mean()),
            "median": float(np.median(alignment)),
            "p90": float(np.percentile(alignment, 90)),
            "max": float(alignment.max()),
        },
        "floored_modes": {
            "mean": float(floored.mean()),
            "median": float(np.median(floored)),
            "fraction_molecules_with_floor": float((floored > 0).mean()),
        },
        "relaxed_energy_kcal_mol": {
            "mean": float(relaxed.mean()),
            "median": float(np.median(relaxed)),
        },
        "rotation_rank_all_3": bool((rotations == 3).all()) if len(rotations) else None,
        "curvature_calibration_ratio": {
            "amplitudes": [0.01, 0.005, 0.0025],
            "median_ratio_per_amplitude": [
                float(np.median(calibration[:, k])) for k in range(calibration.shape[1])
            ]
            if calibration.size
            else None,
        },
    }


def main():
    from rdkit import RDLogger

    from stifflow.training import load_records

    RDLogger.DisableLog("rdApp.*")
    cfg = yaml.safe_load(CONFIG.read_text())
    records = load_records(
        TRAIN_MANIFEST, "train", "all", "alternative_a", cfg["source"]
    )
    diffs = path_differences(records)
    audit = receipt_audit()
    result = {
        "schema_version": 1,
        "path_differences": diffs,
        "label_audit": audit,
        "caveat": "Same rows/z/time grid as P0 path statistics; conditional-path "
        "supervision differences, not network errors or generation quality.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "L_K_minus_S_median": diffs["L_K_minus_S_relative_norm_median"],
                "velocity_diff_median_t0.5": diffs[
                    "velocity_relative_difference_median"
                ][diffs["times"].index(0.5)],
                "factor_diff_median_t0.5": diffs["factor_relative_difference_median"][
                    diffs["times"].index(0.5)
                ],
                "alignment_rmsd_median": audit["alignment_rmsd_angstrom"]["median"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
