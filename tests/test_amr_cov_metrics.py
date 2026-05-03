from __future__ import annotations

import math
import unittest

import numpy as np

from src.eval.amr_cov import (
    compute_amr_cov_from_pairwise,
    heavy_atom_aligned_rmsd,
    parse_args,
    _resolve_output_paths,
    _resolve_run_name,
)


class AmrCovMetricTests(unittest.TestCase):
    def test_compute_amr_cov_from_pairwise(self) -> None:
        pairwise = np.array(
            [
                [0.20, 1.20],
                [0.90, 0.80],
            ]
        )

        metrics = compute_amr_cov_from_pairwise(pairwise, delta=0.75)

        self.assertAlmostEqual(metrics["cov_r"], 0.5)
        self.assertAlmostEqual(metrics["amr_r"], 0.50)
        self.assertAlmostEqual(metrics["cov_p"], 0.5)
        self.assertAlmostEqual(metrics["amr_p"], 0.50)

    def test_compute_amr_cov_requires_finite_nonempty_matrix(self) -> None:
        with self.assertRaises(ValueError):
            compute_amr_cov_from_pairwise(np.empty((0, 1)))
        with self.assertRaises(ValueError):
            compute_amr_cov_from_pairwise(np.array([[math.nan]]))
        with self.assertRaises(ValueError):
            compute_amr_cov_from_pairwise(np.array([[0.1]]), delta=0.0)

    def test_heavy_atom_aligned_rmsd_is_rigid_motion_invariant(self) -> None:
        coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        translated_rotated = [
            [4.0, -3.0, 2.0],
            [4.0, -2.0, 2.0],
            [3.0, -3.0, 2.0],
            [4.0, -3.0, 3.0],
        ]

        rmsd = heavy_atom_aligned_rmsd(coords, translated_rotated, heavy_atom_indices=[0, 1, 2, 3])

        self.assertLess(rmsd, 1.0e-12)

    def test_heavy_atom_aligned_rmsd_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            heavy_atom_aligned_rmsd([[0.0, 0.0, 0.0]], [], heavy_atom_indices=[0])
        with self.assertRaises(ValueError):
            heavy_atom_aligned_rmsd([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], heavy_atom_indices=[])

    def test_full_test_output_paths_are_labeled_separately(self) -> None:
        args = parse_args(["--max-molecules-per-split", "0", "--allow-full-test-split"])

        run_name = _resolve_run_name(args, full_test_requested=True)
        paths = _resolve_output_paths(args, run_name, ("scaffold", "random"), full_test_requested=True)

        self.assertEqual(run_name, "geom_full_database_amr_cov_full_test_abde_seed0")
        self.assertEqual(
            str(paths["summary"]),
            "experiments/results/geom_full_database_amr_cov_full_test_abde_seed0.json",
        )
        self.assertEqual(
            str(paths["combined_report"]),
            "reports/tables/geom_full_database_amr_cov_full_test_abde_seed0_combined.md",
        )
        self.assertEqual(
            str(paths["random_figure"]),
            "reports/figures/geom_full_database_amr_cov_full_test_abde_seed0_random.png",
        )


if __name__ == "__main__":
    unittest.main()
