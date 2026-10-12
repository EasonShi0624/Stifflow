"""Phase 1: exact raw-inventory count in official ET-Flow clean-sde order.

Runs the official `read_mol` (imported from ETFlow-clean-sde/scripts/preprocess.py)
in a process pool, counts molecules/conformers/atoms and compares them with the
constants hardcoded in that script. Writes a JSON report and a compact NPZ.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ.get("STIFFLOW_WORKSPACE", "/gpfsnyu/scratch/ys6132/stifflow"))
RAW = WS / "dataset/rdkit_folder"
OUT_JSON = WS / "runs/etflow-official/phase1-inventory.json"
OUT_NPZ = WS / "runs/etflow-official/phase1-inventory.npz"
OFFICIAL = {
    "TOTAL_NUM_MOLS": 437580,
    "TOTAL_NUM_CONFORMERS": 33078483,
    "TOTAL_NUM_ATOMS": 1656320500,
    "TOTAL_ATOMS_UNIQUE_MOLS": 15904498,
}

_READ_MOL = None
_RAW = str(RAW)


def init_worker():
    global _READ_MOL
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    source = (WS / "ETFlow-clean-sde/scripts/preprocess.py").read_text()
    namespace = {}
    exec(compile(source, "preprocess_clean_sde.py", "exec"), namespace)
    _READ_MOL = namespace["read_mol"]


def work(item):
    partition, key, mol_dict = item
    try:
        result = _READ_MOL(key, mol_dict, _RAW, partition, 0)
    except Exception as exc:  # replicate read_mol's skip semantics
        return (key, partition, 0, 0, f"{type(exc).__name__}: {exc}")
    if result is None:
        return (key, partition, 0, 0, "skipped")
    return (key, partition, int(result["num_atoms"]), int(result["num_confs"]), None)


def items_for(partition, summary):
    for key, mol_dict in summary.items():
        yield (partition, key, mol_dict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()

    started = time.perf_counter()
    per_partition = {}
    keys, partitions, num_atoms, num_confs = [], [], [], []
    failures = []
    totals = {"mols": 0, "confs": 0, "atoms": 0, "atoms_unique": 0}
    with mp.Pool(args.workers, initializer=init_worker) as pool:
        for partition in ("qm9", "drugs"):
            with (RAW / f"summary_{partition}.json").open() as stream:
                summary = json.load(stream)
            part = {"mols": 0, "confs": 0, "atoms": 0, "atoms_unique": 0, "skipped": 0}
            done = 0
            for key, part_name, atoms, confs, error in pool.imap(
                work, items_for(partition, summary), chunksize=16
            ):
                done += 1
                if error is not None:
                    part["skipped"] += 1
                    if len(failures) < 100:
                        failures.append({"id": key, "partition": part_name, "reason": error})
                    continue
                keys.append(key)
                partitions.append(part_name)
                num_atoms.append(atoms)
                num_confs.append(confs)
                part["mols"] += 1
                part["confs"] += confs
                part["atoms"] += atoms * confs
                part["atoms_unique"] += atoms
                totals["mols"] += 1
                totals["confs"] += confs
                totals["atoms"] += atoms * confs
                totals["atoms_unique"] += atoms
                if done % 20000 == 0:
                    print(f"{partition}: {done}/{len(summary)}", flush=True)
            per_partition[partition] = part
            print(f"{partition} done: {part}", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        keys=np.asarray(keys),
        partitions=np.asarray(partitions),
        num_atoms=np.asarray(num_atoms, dtype=np.int32),
        num_confs=np.asarray(num_confs, dtype=np.int32),
    )
    mapping = {
        "TOTAL_NUM_MOLS": totals["mols"],
        "TOTAL_NUM_CONFORMERS": totals["confs"],
        "TOTAL_NUM_ATOMS": totals["atoms"],
        "TOTAL_ATOMS_UNIQUE_MOLS": totals["atoms_unique"],
    }
    report = {
        "official_constants": OFFICIAL,
        "measured": mapping,
        "delta": {name: mapping[name] - value for name, value in OFFICIAL.items()},
        "matches_official": {name: mapping[name] == value for name, value in OFFICIAL.items()},
        "per_partition": per_partition,
        "failures": failures,
        "workers": args.workers,
        "seconds": time.perf_counter() - started,
        "counts_npz": str(OUT_NPZ.relative_to(WS)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps({"measured": mapping, "delta": report["delta"], "matches": report["matches_official"]}, indent=2))
    print(f"report={OUT_JSON}")


if __name__ == "__main__":
    main()
