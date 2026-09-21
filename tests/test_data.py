import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stifflow.data import build_manifest


class ManifestChecks(unittest.TestCase):
    def test_split_overlap_and_frozen_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for name in ("C", "N", "O"):
                (raw / f"{name}.pickle").touch()
            files = list(raw.glob("*"))
            np.save(
                root / "split.npy",
                np.array([[0], [1], [2]], dtype=object),
                allow_pickle=True,
            )
            (root / "test.csv").write_text("smiles\n" + files[2].stem + "\n")
            manifest = build_manifest(raw, root / "split.npy", root / "test.csv")
            self.assertEqual(manifest["counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(
                [r["file"] for r in manifest["entries"]], [p.name for p in files]
            )
            json.dumps(manifest)
            np.save(
                root / "split.npy",
                np.array([[0], [0], [2]], dtype=object),
                allow_pickle=True,
            )
            with self.assertRaisesRegex(ValueError, "overlapping"):
                build_manifest(raw, root / "split.npy")


if __name__ == "__main__":
    unittest.main()
