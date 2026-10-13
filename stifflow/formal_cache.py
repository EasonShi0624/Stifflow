"""Resumable all-candidate labels and bounded, atom-count-grouped mmap packing."""

import argparse
import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rdkit import Chem

from .cache_io import atomic_json, read_json
from .freeze_data import require_frozen
from .official_data import KIND, SAMPLING, OfficialDataset, verified_path
from .official_labels import label_record
from .preflight import fingerprint, provenance
from .smoke import external_directory
from .stiffness import DEFAULTS

MOLECULES_PER_SHARD = 1024
ROWS_PER_SHARD = 4096


@contextmanager
def owned_directory(output, resume):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output


def signature(data_manifest, split, source_config, seed):
    from stifflow_alt_a import path as alternative

    specification = read_json(data_manifest)
    require_frozen(specification)
    directory = Path(__file__).parent
    implementation = {
        name: fingerprint(directory / name)["sha256"]
        for name in (
            "stiffness.py",
            "source.py",
            "precision.py",
            "geometry.py",
            "features.py",
            "official_labels.py",
            "identity.py",
        )
    }
    implementation["alternative_a"] = fingerprint(alternative.__file__)["sha256"]
    return {
        "data_manifest": fingerprint(data_manifest),
        "dataset_kind": KIND,
        "sampling": SAMPLING,
        "split": split,
        "selection_seed": seed,
        "source_config": source_config,
        "stiffness_config": DEFAULTS,
        "pipeline_version": 1,
        "label_implementation": implementation,
    }


def receipt_name(identity):
    return hashlib.sha256(identity.encode()).hexdigest() + ".json"


def reuse_index(paths, split, source_config):
    result = {}
    for path in paths:
        old = read_json(path)
        if (
            old.get("schema_version") != 2
            or old.get("status") != "completed"
            or old.get("audit_partial")
            or old.get("source_config") != source_config
            or old.get("stiffness_config") != DEFAULTS
            or old.get("split") != split
        ):
            raise ValueError(f"Incompatible reuse manifest: {path}")
        for record in old["records"]:
            if record["audit_partial"]:
                raise ValueError("Partial reuse molecule")
            if record["id"] in result:
                raise ValueError("Duplicate reused molecule")
            result[record["id"]] = (record, Path(path))
    return result


def reused_record(entry, dataset, split, source_config, reuse):
    """Reuse v2 only after checking the frozen rows, raw target and factor math."""
    from stifflow_alt_a.path import prepare_label as prepare_alternative

    from .precision import prepare_label
    from .source import prepare_source

    if entry["id"] in reuse:
        record, path = reuse[entry["id"]]
        if record["row_ids"] != entry["row_ids"]:
            raise ValueError("Reuse whole-molecule rows differ")
        archive = Path(path).parent / record["npz"]
        if fingerprint(archive)["sha256"] != record["cache"]["sha256"]:
            raise ValueError("Reuse archive hash mismatch")
        molecule = dataset.load_molecule(entry, split)
        if molecule["source"]["sha256"] != record["source"]["sha256"]:
            raise ValueError("Reuse raw molecule hash mismatch")
        source = prepare_source(molecule["graph"], source_config)
        with np.load(archive, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(arrays["row_ids"], entry["row_ids"])
        np.testing.assert_array_equal(
            arrays["positions"],
            np.stack([c["positions"] for c in molecule["conformers"]]),
        )
        for i, K in enumerate(arrays["K"]):
            p = prepare_label(source, K, float(arrays["epsilon_E"]))
            actual = (
                arrays["precision_vectors"][i] * arrays["precision_values"][i]
            ) @ arrays["precision_vectors"][i].T
            expected = (p["vectors"] * p["values"]) @ p["vectors"].T
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
            a = prepare_alternative(source, K, float(arrays["epsilon_E"]))
            np.testing.assert_allclose(
                arrays["alt_L_K"][i], a["L_K"], rtol=1e-9, atol=1e-10
            )
        record = dict(record)
        record["reused_from"] = fingerprint(path)
        return record, arrays, source
    return None


def build_shard(
    dataset,
    data_manifest,
    output,
    split,
    source_config,
    seed=42,
    shard_index=0,
    shard_size=MOLECULES_PER_SHARD,
    resume=False,
    reuse=(),
):
    common = signature(data_manifest, split, source_config, seed)
    candidates = dataset.candidates(split, seed)
    count = (len(candidates) + shard_size - 1) // shard_size
    if not 0 <= shard_index < count:
        raise ValueError(f"shard-index must be in [0,{count})")
    selected = candidates[shard_index * shard_size : (shard_index + 1) * shard_size]
    reuse = reuse_index(reuse, split, source_config)
    expected = dict(
        common,
        shard_index=shard_index,
        shard_count=count,
        shard_size=shard_size,
        candidate_ids=[e["id"] for e in selected],
    )
    output = Path(output)
    with owned_directory(output, resume):
        owner = output / "owner.json"
        if owner.exists():
            if read_json(owner) != expected:
                raise ValueError("Resume label data/configuration/shard differs")
        else:
            atomic_json(owner, expected)
        (output / "receipts").mkdir(exist_ok=True)
        (output / "archives").mkdir(exist_ok=True)
        for entry in selected:
            receipt = output / "receipts" / receipt_name(entry["id"])
            if receipt.exists():
                finished = read_json(receipt)
                if finished["attempt"]["id"] != entry["id"]:
                    raise ValueError("Receipt identity differs")
                if finished["attempt"]["status"] == "success":
                    verified_path(finished["record"]["cache"])
                elif finished["attempt"]["status"] != "rejected":
                    raise ValueError("Nonterminal receipt")
                continue
            started = time.perf_counter()
            attempt = {
                "id": entry["id"],
                "split": split,
                "official_conformers": len(entry["row_ids"]),
            }
            reused = reused_record(entry, dataset, split, source_config, reuse)
            try:
                record, arrays, source = reused or label_record(
                    dataset, entry, split, source_config
                )
                from .features import graph_from_mol

                mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
                if mol is None:
                    raise ValueError("Invalid graph molblock")
                features = graph_from_mol(mol)
                np.testing.assert_array_equal(
                    features["atomic_numbers"], record["graph"]["atomic_numbers"]
                )
                arrays.update(
                    source_S=source.S,
                    source_C0=source.C0,
                    graph_numbers=features["atomic_numbers"],
                    graph_nodes=features["node_attr"],
                    graph_bonds=features["bonds"],
                    graph_attrs=features["bond_attr"],
                )
                archive = output / "archives" / (uuid.uuid4().hex + ".npz")
                with archive.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                record.update(npz=str(archive.resolve()), cache=fingerprint(archive))
                attempt.update(
                    status="success",
                    labeled_rows=len(entry["row_ids"]),
                    reused=bool(reused),
                )
                value = {"attempt": attempt, "record": record}
            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ) as exc:
                attempt.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
                value = {"attempt": attempt}
            attempt["seconds"] = time.perf_counter() - started
            atomic_json(receipt, value)
        receipts = [
            fingerprint(output / "receipts" / receipt_name(e["id"])) for e in selected
        ]
        report = dict(
            expected, status="completed", receipts=receipts, provenance=provenance()
        )
        atomic_json(output / "shard.json", report)
    return report


def _commit_part(output, kind, atoms, part, specs, fill):
    """Only receipts commit a part; interrupted attempts remain untouched."""
    receipt = output / "parts" / f"{kind}-{atoms}-{part}.json"
    if receipt.exists():
        result = read_json(receipt)
        for item in result["files"].values():
            verified_path(item)
        return result
    directory = output / "arrays" / f"{kind}-{atoms}-{part}-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    arrays = {
        key: np.lib.format.open_memmap(
            directory / (key + ".npy"), mode="w+", dtype=dtype, shape=shape
        )
        for key, (dtype, shape) in specs.items()
    }
    try:
        fill(arrays)
        for array in arrays.values():
            array.flush()
    finally:
        for array in arrays.values():
            array._mmap.close()
    files = {
        key: {
            **fingerprint(directory / (key + ".npy")),
            "mtime_ns": (directory / (key + ".npy")).stat().st_mtime_ns,
            "shape": list(shape),
            "dtype": np.dtype(dtype).str,
        }
        for key, (dtype, shape) in specs.items()
    }
    result = {"atoms": atoms, "files": files}
    atomic_json(receipt, result)
    return result


def merge_shards(
    dataset,
    data_manifest,
    shard_paths,
    output,
    split,
    source_config,
    seed=42,
    resume=False,
    rows_per_shard=ROWS_PER_SHARD,
):
    common = signature(data_manifest, split, source_config, seed)
    candidates = dataset.candidates(split, seed)
    shards = sorted(
        [read_json(path) for path in shard_paths], key=lambda s: s["shard_index"]
    )
    if not shards or [s["shard_index"] for s in shards] != list(
        range(shards[0]["shard_count"])
    ):
        raise ValueError("Missing or duplicate label shards")
    records, rejections, seconds = [], [], 0.0
    seen = []
    for shard in shards:
        if shard["status"] != "completed" or any(
            shard[k] != v for k, v in common.items()
        ):
            raise ValueError("Label shard is incomplete or has a different signature")
        if len(shard["receipts"]) != len(shard["candidate_ids"]):
            raise ValueError("Incomplete terminal receipts")
        for identity, receipt in zip(shard["candidate_ids"], shard["receipts"]):
            value = read_json(verified_path(receipt))
            attempt = value["attempt"]
            if attempt["id"] != identity or attempt["split"] != split:
                raise ValueError("Receipt order/split differs")
            seen.append(identity)
            seconds += attempt["seconds"]
            if attempt["status"] == "success":
                record = value["record"]
                verified_path(record["cache"])
                records.append(dict(record, audit_receipt=receipt))
            elif attempt["status"] == "rejected":
                rejections.append(attempt)
            else:
                raise ValueError("Unprocessed candidate")
    if seen != [e["id"] for e in candidates] or not records:
        raise ValueError(
            "All frozen candidates must have terminal receipts and at least one accepted molecule"
        )
    expected_rows = {e["id"]: e["row_ids"] for e in candidates}
    for record in records:
        if record["audit_partial"] or record["row_ids"] != expected_rows[record["id"]]:
            raise ValueError("Whole-molecule row pairing differs")
    owner_value = dict(
        common,
        shards=[fingerprint(p) for p in shard_paths],
        rows_per_shard=rows_per_shard,
    )
    output = Path(output)
    with owned_directory(output, resume):
        owner = output / "owner.json"
        if owner.exists() and read_json(owner) != owner_value:
            raise ValueError("Resume packing inputs/configuration differ")
        if not owner.exists():
            atomic_json(owner, owner_value)
        (output / "parts").mkdir(exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(r["row_ids"]) for r in records])]
        index = np.empty((int(offsets[-1]), 3), dtype=np.int64)
        slim = [
            {
                k: r[k]
                for k in (
                    "id",
                    "split",
                    "row_ids",
                    "retained_conformers",
                    "audit_partial",
                    "audit_receipt",
                )
            }
            for r in records
        ]
        by_atoms = {}
        for i, record in enumerate(records):
            by_atoms.setdefault(len(record["graph"]["atomic_numbers"]), []).append(i)
        graph_parts, factor_parts = [], []
        for n, record_ids in sorted(by_atoms.items()):
            d = 3 * n - 3
            for part, start in enumerate(
                range(0, len(record_ids), MOLECULES_PER_SHARD)
            ):
                ids = record_ids[start : start + MOLECULES_PER_SHARD]
                specs = {
                    "numbers": (np.int64, (len(ids), n)),
                    "nodes": (np.float32, (len(ids), n, 10)),
                    "bonds": (np.float32, (len(ids), n, n, 3)),
                    "S": (np.float64, (len(ids), d, d)),
                    "C0": (np.float64, (len(ids), d, d)),
                }

                def fill_graph(arrays, ids=ids):
                    for local, record_id in enumerate(ids):
                        with np.load(
                            records[record_id]["cache"]["path"], allow_pickle=False
                        ) as saved:
                            arrays["numbers"][local] = saved["graph_numbers"]
                            arrays["nodes"][local] = saved["graph_nodes"]
                            arrays["bonds"][local] = 0
                            bonds = saved["graph_bonds"]
                            arrays["bonds"][local, bonds[:, 0], bonds[:, 1]] = saved[
                                "graph_attrs"
                            ]
                            arrays["S"][local] = saved["source_S"]
                            arrays["C0"][local] = saved["source_C0"]
                        slim[record_id].update(
                            graph_shard=len(graph_parts), graph_row=local
                        )

                result = _commit_part(output, "graph", n, part, specs, fill_graph)
                for local, record_id in enumerate(ids):
                    slim[record_id].update(
                        graph_shard=len(graph_parts), graph_row=local
                    )
                graph_parts.append(result)
            lengths = [len(records[i]["row_ids"]) for i in record_ids]
            group_offsets = np.r_[0, np.cumsum(lengths)]
            for part, start in enumerate(
                range(0, int(group_offsets[-1]), rows_per_shard)
            ):
                end = min(start + rows_per_shard, int(group_offsets[-1]))
                specs = {
                    "positions": (np.float64, (end - start, n, 3)),
                    "precision_values": (np.float64, (end - start, d)),
                    "precision_vectors": (np.float64, (end - start, d, d)),
                    "alt_L_K": (np.float64, (end - start, d, d)),
                    "row_ids": (np.int64, (end - start,)),
                }
                spans = []
                first = int(np.searchsorted(group_offsets[1:], start, side="right"))
                for group_id in range(first, len(record_ids)):
                    lo, hi = max(start, int(group_offsets[group_id])), min(
                        end, int(group_offsets[group_id + 1])
                    )
                    if lo >= hi:
                        break
                    record_id = record_ids[group_id]
                    a, b = lo - int(group_offsets[group_id]), hi - int(
                        group_offsets[group_id]
                    )
                    local = lo - start
                    index[offsets[record_id] + a : offsets[record_id] + b] = (
                        np.column_stack(
                            [
                                np.full(b - a, len(factor_parts)),
                                np.arange(local, local + b - a),
                                records[record_id]["row_ids"][a:b],
                            ]
                        )
                    )
                    spans.append((record_id, a, b, local))

                def fill_factors(arrays, spans=spans, specs=specs):
                    for record_id, a, b, local in spans:
                        with np.load(
                            records[record_id]["cache"]["path"], allow_pickle=False
                        ) as saved:
                            np.testing.assert_array_equal(
                                saved["row_ids"], records[record_id]["row_ids"]
                            )
                            for key in specs:
                                values = saved[key][a:b]
                                if (
                                    values.dtype != arrays[key].dtype
                                    or not np.isfinite(values).all()
                                ):
                                    raise ValueError(
                                        f"Invalid label dtype/finiteness: {key}"
                                    )
                                if key == "precision_values" and np.any(values <= 0):
                                    raise ValueError("Nonpositive precision values")
                                arrays[key][local : local + b - a] = values

                factor_parts.append(
                    _commit_part(output, "factor", n, part, specs, fill_factors)
                )
        index_path = output / "row_index.npy"
        if index_path.exists():
            np.testing.assert_array_equal(
                np.load(index_path, allow_pickle=False), index
            )
        else:
            with index_path.open("xb") as stream:
                np.save(stream, index)
        rejections_path = output / "rejections.jsonl"
        if not rejections_path.exists():
            with rejections_path.open("x") as stream:
                for rejected in rejections:
                    stream.write(json.dumps(rejected) + "\n")
        result = dict(
            common,
            schema_version=3,
            status="completed",
            audit_partial=False,
            requested_attempts_completed=True,
            all_candidates_processed=True,
            candidate_molecules=len(candidates),
            attempted_molecules=len(seen),
            requested_molecules=len(slim),
            rejected_molecules=len(rejections),
            records=slim,
            graph_shards=graph_parts,
            factor_shards=factor_parts,
            row_index=fingerprint(index_path),
            rejections=fingerprint(rejections_path),
            labeled_rows=len(index),
            preprocessing_seconds=seconds,
            archive_bytes=sum(r["cache"]["bytes"] for r in records),
            packed_bytes=sum(
                f["bytes"]
                for p in graph_parts + factor_parts
                for f in p["files"].values()
            ),
            provenance=provenance(),
        )
        atomic_json(output / "manifest.json", result)
    return result


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "merge"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reuse", type=Path, nargs="*", default=[])
    parser.add_argument("--shards", type=Path, nargs="+")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error(
            "Full physical preprocessing/packing belongs in an SHHPC allocation"
        )
    cfg = yaml.safe_load(args.config.read_text())
    dataset = OfficialDataset.from_manifest(args.manifest)
    dataset.validate_splits()
    common = {
        "dataset": dataset,
        "data_manifest": args.manifest,
        "output": args.output,
        "split": args.split,
        "source_config": cfg["source"],
        "seed": cfg["dataset_seed"],
        "resume": args.resume,
    }
    if args.action == "build":
        report = build_shard(**common, shard_index=args.shard_index, reuse=args.reuse)
    else:
        if not args.shards:
            parser.error("merge requires every --shards shard.json")
        report = merge_shards(**common, shard_paths=args.shards)
    print(json.dumps({k: report[k] for k in ("status", "split")}))


if __name__ == "__main__":
    main()
