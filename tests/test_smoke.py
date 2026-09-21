"""CPU checks runnable even while the installed torch-cluster is broken."""

import unittest

import numpy as np
from stifflow.preflight import WORKSPACE, fingerprint
from stifflow.smoke import external_directory, validate_positions


class SmokeChecks(unittest.TestCase):
    def test_roundtrip_guards(self):
        positions = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.assertEqual(validate_positions(positions, 1, 2), 0.0)
        for bad in (positions + 1.0, positions * np.nan, positions[:, :1]):
            with self.assertRaises(ValueError):
                validate_positions(bad, 1, 2)
        for directory in ("dataset/new", "ETFlow-reference/new", "stifflow-v1/new"):
            with self.assertRaises(ValueError):
                external_directory(WORKSPACE / directory)
        self.assertEqual(
            external_directory(WORKSPACE / "cache/new"), WORKSPACE / "cache/new"
        )
        self.assertEqual(len(fingerprint(__file__)["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
