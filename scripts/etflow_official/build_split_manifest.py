"""Build the canonical official-split manifest for the clean-sde memmap scheme.

Reads only official downloaded artifacts (Zenodo v2 zips / v3 tar extracts) and
records hashes, shapes and row-level consistency checks. No data is modified.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

WS = Path(os.environ.get("STIFFLOW_WORKSPACE", "/gpfsnyu/scratch/ys6132/stifflow"))
ZEN = WS / "cache/etflow-official/zenodo"
OUT = WS / "cache/etflow-official/splits/manifest.json"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path.relative_to(WS)), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def array_info(path):
    arr = np.load(path)
    return {
        **file_hash(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "unique": int(len(np.unique(arr))),
    }


def expand(count_indices):
    spans = [
        set(range(int(start), int(start) + int(count)))
        for start, count in zip(count_indices[0], count_indices[1])
    ]
    return set().union(*spans)


def main():
    manifest = {
        "scheme": "ET-Flow clean-sde memmap (branch clean-sde@2f20cad)",
        "sources": {
            "v2_zip": {
                "record": "https://zenodo.org/api/records/14226681",
                "files": {
                    "QM9.zip": "1bed2e29e15806306b9877b32fe27d1c",
                    "DRUGS.zip": "5e9f7dc7e0bf8ccef7fa7ad6e7c75d12",
                    "XL.zip": "4759ae27d64dbe03266795325b15cc2e",
                },
            },
            "v3_tar": {
                "record": "https://zenodo.org/api/records/16551316",
                "files": {"scaffold_data.tar.gz": "142b4651bbe5c9239fd3b4fad65204e2"},
            },
        },
        "semantics": {
            "random": "conformer-level indices into the global memmap; ET-Flow clean-sde RandomDataModule style",
            "scaffold_qm9_indices": "conformer-level indices for the QM9 scaffold split (v2 zip)",
            "scaffold_smiles": "molecule-level SMILES arrays defining the v3 scaffold split; map to memmap rows after preprocessing",
            "test": "count_indices rows are (start, count) per test molecule; expand to rows",
        },
        "files": {},
        "checks": {},
    }

    qm9 = ZEN / "v2/QM9"
    drugs = ZEN / "v2/DRUGS"
    xl = ZEN / "v2/XL"
    scaf = ZEN / "v3/data/scaffold"

    for label, path in {
        "random_qm9_train": qm9 / "train_indices.npy",
        "random_qm9_val": qm9 / "val_indices.npy",
        "random_qm9_count": qm9 / "count_indices.npy",
        "random_drugs_train": drugs / "train_indices.npy",
        "random_drugs_val": drugs / "val_indices.npy",
        "random_drugs_count": drugs / "count_indices.npy",
        "scaffold_qm9_train": qm9 / "scaffold_train_indices.npy",
        "scaffold_qm9_val": qm9 / "scaffold_val_indices.npy",
        "scaffold_qm9_count": qm9 / "scaffold_qm9_count_indices.npy",
    }.items():
        manifest["files"][label] = array_info(path)

    for label, path in {
        "xl_test_mols": xl / "test_mols.pkl",
        "xl_test_smiles": xl / "test_smiles.csv",
    }.items():
        manifest["files"][label] = file_hash(path)

    for label, path in {
        "scaffold_qm9_train_smiles": scaf / "qm9/scaffold_train_smiles.pkl",
        "scaffold_qm9_val_smiles": scaf / "qm9/scaffold_val_smiles.pkl",
        "scaffold_qm9_test_smiles": scaf / "qm9/scaffold_test_smiles.pkl",
        "scaffold_drugs_train_smiles": scaf / "drugs/scaffold_train_smiles.pkl",
        "scaffold_drugs_val_smiles": scaf / "drugs/scaffold_val_smiles.pkl",
        "scaffold_drugs_test_smiles": scaf / "drugs/scaffold_test_smiles.pkl",
    }.items():
        print(f"hashing {label} ...", flush=True)
        info = file_hash(path)
        with path.open("rb") as stream:
            obj = pickle.load(stream)
        info["molecules"] = int(len(obj))
        manifest["files"][label] = info

    def rows(name):
        return set(np.load(ZEN / f"v2/{name}").tolist())

    rq_tr, rq_va = rows("QM9/train_indices.npy"), rows("QM9/val_indices.npy")
    sq_tr, sq_va = rows("QM9/scaffold_train_indices.npy"), rows("QM9/scaffold_val_indices.npy")
    rq_te = expand(np.load(qm9 / "count_indices.npy"))
    sq_te = expand(np.load(qm9 / "scaffold_qm9_count_indices.npy"))
    rd_tr, rd_va = rows("DRUGS/train_indices.npy"), rows("DRUGS/val_indices.npy")
    rd_te = expand(np.load(drugs / "count_indices.npy"))
    manifest["checks"] = {
        "random_qm9_train_val_overlap": len(rq_tr & rq_va),
        "random_qm9_test_in_train": len(rq_te & rq_tr),
        "random_qm9_test_in_val": len(rq_te & rq_va),
        "random_drugs_train_val_overlap": len(rd_tr & rd_va),
        "random_drugs_test_in_train": len(rd_te & rd_tr),
        "random_drugs_test_in_val": len(rd_te & rd_va),
        "scaffold_qm9_train_val_overlap": len(sq_tr & sq_va),
        "scaffold_qm9_test_in_train": len(sq_te & sq_tr),
        "scaffold_qm9_test_in_val": len(sq_te & sq_va),
        "scaffold_qm9_counts": {
            "train_rows": len(sq_tr),
            "val_rows": len(sq_va),
            "test_molecules": int(np.load(qm9 / "scaffold_qm9_count_indices.npy").shape[1]),
            "test_rows": len(sq_te),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["checks"], indent=2))
    print(f"manifest={OUT}")


if __name__ == "__main__":
    main()
