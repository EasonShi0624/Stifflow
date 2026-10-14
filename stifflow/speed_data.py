"""Small real training-only caches for speed tests, never formal label caches."""

import csv
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from .cache_io import atomic_json, read_json
from .features import graph_from_mol
from .formal_cache import _commit_part
from .identity import molecule_identity
from .preflight import fingerprint
from .source import prepare_source
from .stiffness import build_label_arrays


def pack_speed_records(records, output, source_config):
    """Use the production mmap reader and immutable part writer on a tiny cohort."""
    output = Path(output)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(len(record["graph"]["atomic_numbers"]), []).append(index)
    manifest = {
        "schema_version": 3,
        "dataset_kind": "torch_speed_subset",
        "speed_only": True,
        "audit_partial": True,
        "source_config": source_config,
        "records": [
            {
                "id": r["id"],
                "split": "train",
                "row_ids": [i],
                "retained_conformers": 1,
            }
            for i, r in enumerate(records)
        ],
        "graph_shards": [],
        "factor_shards": [],
    }
    index_array = np.empty((len(records), 3), dtype=np.int64)
    for part, (atoms, indices) in enumerate(sorted(groups.items())):
        dimension, count = 3 * atoms - 3, len(indices)
        graph_specs = {
            "numbers": (np.int64, (count, atoms)),
            "nodes": (np.float32, (count, atoms, 10)),
            "bonds": (np.float32, (count, atoms, atoms, 3)),
            "S": (np.float64, (count, dimension, dimension)),
            "C0": (np.float64, (count, dimension, dimension)),
        }
        factor_specs = {
            "positions": (np.float64, (count, atoms, 3)),
            "row_ids": (np.int64, (count,)),
            "precision_values": (np.float64, (count, dimension)),
            "precision_vectors": (np.float64, (count, dimension, dimension)),
            "alt_L_K": (np.float64, (count, dimension, dimension)),
        }

        def fill_graph(arrays, indices=indices):
            arrays["bonds"][:] = 0
            for local, i in enumerate(indices):
                record, graph = records[i], records[i]["graph"]
                arrays["numbers"][local] = graph["atomic_numbers"]
                arrays["nodes"][local] = graph["node_attr"]
                bonds = graph["bonds"]
                arrays["bonds"][local, bonds[:, 0], bonds[:, 1]] = graph["bond_attr"]
                arrays["S"][local] = record["source"].S
                arrays["C0"][local] = record["source"].C0

        def fill_factors(arrays, indices=indices, part=part):
            for local, i in enumerate(indices):
                for key in (
                    "positions",
                    "precision_values",
                    "precision_vectors",
                    "alt_L_K",
                ):
                    arrays[key][local] = records[i]["arrays"][key][0]
                arrays["row_ids"][local] = i
                manifest["records"][i].update(graph_shard=part, graph_row=local)
                index_array[i] = part, local, i

        manifest["graph_shards"].append(
            _commit_part(output, "graph", atoms, part, graph_specs, fill_graph)
        )
        manifest["factor_shards"].append(
            _commit_part(output, "factor", atoms, part, factor_specs, fill_factors)
        )
    index_path = output / "row_index.npy"
    with index_path.open("xb") as stream:
        np.save(stream, index_array, allow_pickle=False)
    manifest["row_index"] = fingerprint(index_path)
    return manifest


def prepare_speed_cache(split_manifest, output, source_config, count=16, attempts=64):
    """One original conformer per real train molecule; all physical kernels reused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    split = read_json(split_manifest)
    if split.get("official_test", {}).get("isolated") is not True:
        raise ValueError(
            "Speed data requires the locally frozen training/test isolation receipt"
        )
    test_file = split["official_test"]
    if fingerprint(test_file["path"])["sha256"] != test_file["sha256"]:
        raise ValueError("Official test-list hash mismatch")
    with Path(test_file["path"]).open(newline="") as stream:
        test_ids = {molecule_identity(row["smiles"]) for row in csv.DictReader(stream)}
    raw_dir = Path(split["raw_dir"]).resolve()
    candidates = sorted(
        (e for e in split["entries"] if e["split"] == "train"),
        key=lambda e: hashlib.sha256(f"42:{e['molecule_id']}".encode()).hexdigest(),
    )[:256]
    sizes, selection_filters = [], []
    for entry in candidates:
        mol = Chem.MolFromSmiles(entry["molecule_id"])
        if mol is not None:
            size = Chem.AddHs(mol).GetNumAtoms()
            if 8 <= size <= 32:
                sizes.append((size, entry))
            else:
                selection_filters.append(
                    {
                        "id": entry["molecule_id"],
                        "reason": "outside 8..32 atoms",
                        "atoms": size,
                    }
                )
        else:
            selection_filters.append(
                {"id": entry["molecule_id"], "reason": "SMILES could not be sanitized"}
            )
    sizes.sort(key=lambda pair: pair[0])
    if len(sizes) < count:
        raise ValueError("Not enough bounded training molecules for the speed cohort")
    preferred = [sizes[i][1] for i in np.linspace(0, len(sizes) - 1, 8, dtype=int)]
    preferred += [entry for _, entry in sizes[-8:]]
    ordered, seen = [], set()
    for entry in preferred + [entry for _, entry in sizes]:
        if entry["file"] not in seen:
            ordered.append(entry)
            seen.add(entry["file"])
    records, ledger, identities = [], [], set()
    for entry in ordered[:attempts]:
        item = {"id": entry["molecule_id"], "status": "started"}
        ledger.append(item)
        try:
            identity = molecule_identity(entry["molecule_id"])
            if identity in test_ids or identity in identities:
                raise ValueError("Canonical test overlap or duplicate speed molecule")
            path = (raw_dir / entry["file"]).resolve()
            if not path.is_relative_to(raw_dir) or path.stat().st_size > 16 * 1024**2:
                raise ValueError(
                    "Raw path escapes dataset or exceeds the small-read limit"
                )
            with path.open("rb") as stream:
                raw = pickle.load(stream)  # Trusted local GEOM artifact.
            mol = Chem.Mol(raw["conformers"][0]["rd_mol"])
            positions = (
                mol.GetConformer().GetPositions().astype(np.float32).astype(np.float64)
            )
            for atom, xyz in enumerate(positions):
                mol.GetConformer().SetAtomPosition(atom, xyz)
            graph = graph_from_mol(mol)
            source = prepare_source(graph, source_config)
            label_started = time.perf_counter()
            try:
                arrays, audits, residuals = build_label_arrays(
                    [
                        {
                            "rd_mol": mol,
                            "positions": positions,
                            "conformer_id": f"{identity}:raw:0",
                        }
                    ],
                    source,
                )
            finally:
                item["stiffness_seconds"] = time.perf_counter() - label_started
            name = hashlib.sha256(identity.encode()).hexdigest()
            archive = output / f"{name}.npz"
            with archive.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            item.update(
                status="accepted",
                atoms=mol.GetNumAtoms(),
                raw_index=0,
                raw=fingerprint(path),
                archive=fingerprint(archive),
                audit=audits,
                covariance_residuals=residuals,
            )
            records.append(
                {
                    "id": "qm9:" + identity,
                    "graph": graph,
                    "source": source,
                    "arrays": arrays,
                }
            )
            identities.add(identity)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            item.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "attempts.json", ledger)
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"Need {count} valid speed molecules; accepted {len(records)}")
    manifest = pack_speed_records(records, output, source_config)
    manifest.update(
        input_manifest=fingerprint(split_manifest),
        cohort="local legacy train split; not SHHPC strict v3 full-data throughput",
        coordinate_rule="original GEOM conformer 0, float32 memmap rounding; target not relaxed",
        preparation_seconds=time.perf_counter() - started,
        stiffness_preprocessing_seconds=sum(
            r.get("stiffness_seconds", 0) for r in ledger
        ),
        atoms=[len(r["graph"]["atomic_numbers"]) for r in records],
        candidate_pool=len(candidates),
        selection_filters=selection_filters,
        attempts=fingerprint(output / "attempts.json"),
    )
    atomic_json(output / "manifest.json", manifest)
    return manifest
