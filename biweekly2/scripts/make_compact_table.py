"""Assemble the compact results table for Biweekly Report 2.

Produces results/compact-table.json and a markdown rendering.  Covers:
generation metrics (test primary, val secondary) with both uncertainty
types, physical quality, numerical verification, and mechanism quantities.
"""

import json
from pathlib import Path

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
BW = WS / "runs/biweekly2-20261019"
OUT = BW / "results/compact-table.json"


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    data = json.loads((BW / "results/figure-data.json").read_text())
    caliber = data["task1_caliber"]
    phys = data["task3_physical_quality"]
    gauss = data["task2a_gaussian"]
    equiv = data["task2b_equivariance"]
    signal = data["task4a_signal"]

    table = {"test": [], "val": []}
    label = {
        "alternative_a": "StifFlow",
        "control": "ET-Flow-backbone control",
    }
    for split in ("test", "val"):
        metrics = data[f"{split}_metrics"]
        for method in ("alternative_a", "control"):
            row = {
                "method": label[method],
                "mat_r": fmt(metrics[method]["mean"]["mat_r"]),
                "mat_r_seed_std": fmt(metrics[method]["seed_std"]["mat_r"]),
                "cov_r": fmt(metrics[method]["mean"]["cov_r"]),
                "cov_r_seed_std": fmt(metrics[method]["seed_std"]["cov_r"]),
                "chirality_macro": fmt(
                    metrics[method]["per_seed"]["42"]["chirality_macro"], 3
                ),
            }
            table[split].append(row)
    for split in ("test", "val"):
        for pair, pair_label in (("alternative_a-control", "StifFlow − control"),):
            p = caliber["pooled_paired_comparisons"][split][pair]["mat_r"]
            table[split].append(
                {
                    "method": f"paired Δ ({pair_label}), pooled 3 seeds",
                    "mat_r": f"{p['mean']:+.4f}",
                    "mat_r_seed_std": f"CI [{p['ci95_molecule_bootstrap'][0]:+.4f},"
                    f"{p['ci95_molecule_bootstrap'][1]:+.4f}]",
                    "cov_r": f"better for {p['fraction_molecules_better']*100:.1f}% mols",
                    "cov_r_seed_std": "",
                    "chirality_macro": "",
                }
            )

    physical = {}
    for method in ("alternative_a", "control"):
        s = phys["methods"][method]["summary"]
        physical[label[method]] = {
            "strain_median_kcal_mol": fmt(s["relaxable_strain_kcal_mol"]["median"], 2),
            "strain_mean_kcal_mol": fmt(s["relaxable_strain_kcal_mol"]["mean"], 2),
            "bond_dev_A": fmt(s["bond_deviation_angstrom"]["median"], 4),
        }
    physical["references (GEOM)"] = {
        "strain_median_kcal_mol": fmt(
            phys["references"]["summary"]["relaxable_strain_kcal_mol"]["median"], 2
        ),
        "strain_mean_kcal_mol": fmt(
            phys["references"]["summary"]["relaxable_strain_kcal_mol"]["mean"], 2
        ),
        "bond_dev_A": "—",
    }

    t_half = gauss["molecules_data"][min(gauss["molecules_data"])]["times"]["0.5"][
        "M4096"
    ]
    verification = {
        "sigma_identity_residual_max": fmt(
            max(
                gauss["molecules_data"][i]["times"][t]["sigma_identity_residual"]
                for i in gauss["molecules_data"]
                for t in gauss["molecules_data"][i]["times"]
            ),
            16,
        ),
        "cov_rel_error_M4096_median_t0.5": fmt(t_half["covariance_relative_error"], 3),
        "whitened_within_MP_all_molecules_times": all(
            gauss["molecules_data"][i]["times"][t][f"M{m}"]["whitened_eigs_within_mp"]
            for i in gauss["molecules_data"]
            for t in gauss["molecules_data"][i]["times"]
            for m in (256, 1024, 4096)
        ),
        "rotation_equivariance_max": fmt(
            max(
                e
                for m in ("alternative_a", "control")
                for row in equiv["methods"][m]["equivariance"]
                for e in row["rotation_equivariance_relative_error"]
            ),
            7,
        ),
        "permutation_equivariance_max": fmt(
            max(
                e
                for m in ("alternative_a", "control")
                for row in equiv["methods"][m]["equivariance"]
                for e in row["permutation_equivariance_relative_error"]
            ),
            7,
        ),
        "sampling_consistency_rmsd_max_A": fmt(
            max(
                max(
                    row["rotation_endpoint_rmsd_angstrom"],
                    row["permutation_endpoint_rmsd_angstrom"],
                )
                for m in ("alternative_a", "control")
                for row in equiv["methods"][m]["sampling_consistency"]
            ),
            7,
        ),
    }

    mech = signal["path_differences"]
    times = mech["times"]
    import numpy as np

    def nearest(target):
        return int(np.argmin([abs(t - target) for t in times]))

    mechanism = {
        "velocity_diff_median_t0.24": fmt(
            mech["velocity_relative_difference_median"][nearest(0.25)], 3
        ),
        "velocity_diff_median_t0.5": fmt(
            mech["velocity_relative_difference_median"][nearest(0.5)], 3
        ),
        "velocity_diff_p90_t0.5": fmt(
            mech["velocity_relative_difference_p90"][nearest(0.5)], 3
        ),
        "L_K_minus_S_relative_norm_median": fmt(
            mech["L_K_minus_S_relative_norm_median"], 3
        ),
        "alignment_rmsd_median_A": fmt(
            signal["label_audit"]["alignment_rmsd_angstrom"]["median"], 3
        ),
        "floored_mode_fraction_molecules": fmt(
            signal["label_audit"]["floored_modes"]["fraction_molecules_with_floor"], 3
        ),
        "calibration_ratio_median_0.005A": fmt(
            signal["label_audit"]["curvature_calibration_ratio"][
                "median_ratio_per_amplitude"
            ][1],
            6,
        ),
    }

    result = {
        "schema_version": 1,
        "generation": table,
        "physical_quality": physical,
        "numerical_verification": verification,
        "mechanism": mechanism,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
