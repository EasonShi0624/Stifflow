"""File-only checks: no numerical libraries or model initialization."""

import tempfile
import unittest
from pathlib import Path

from stifflow.preflight import optional_fingerprint


class OptionalArtifactChecks(unittest.TestCase):
    def test_absent_baseline_and_present_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.ckpt"
            self.assertEqual(
                optional_fingerprint(path),
                {"path": str(path.resolve()), "exists": False},
            )
            path.write_bytes(b"abc")
            self.assertEqual(
                optional_fingerprint(path)["sha256"],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(optional_fingerprint(path)["bytes"], 3)


if __name__ == "__main__":
    unittest.main()
