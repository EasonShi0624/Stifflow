"""Bounded CPU float64 reads; no full dense label array is copied into RAM."""

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import centered_basis
from .preflight import fingerprint
from .source import Source


@lru_cache(maxsize=64)
def basis(atoms):
    return centered_basis(atoms)


class PackedRecords:
    def __init__(self, manifest, method, max_open_parts=16, graph_cache_size=64):
        if (
            method not in ("precision", "alternative_a", "control")
            or min(max_open_parts, graph_cache_size) < 1
        ):
            raise ValueError("Invalid method or cache bounds")
        self.manifest = manifest
        self.method = method
        self.records = manifest["records"]
        self.lengths = np.asarray(
            [len(r["row_ids"]) for r in self.records], dtype=np.int64
        )
        self.offsets = np.r_[0, np.cumsum(self.lengths)]
        item = manifest["row_index"]
        if fingerprint(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Packed row index hash mismatch")
        self.index = np.load(item["path"], allow_pickle=False, mmap_mode="r")
        expected = np.concatenate([r["row_ids"] for r in self.records])
        if (
            self.index.dtype != np.int64
            or self.index.shape != (len(expected), 3)
            or not np.array_equal(self.index[:, 2], expected)
        ):
            raise ValueError("Packed row/label pairing differs")
        self.max_open_parts = max_open_parts
        self.graph_cache_size = graph_cache_size
        self.parts = OrderedDict()
        self.graphs = OrderedDict()
        self.verified = {}

    def __len__(self):
        return len(self.records)

    def _part(self, kind, part_id):
        key = kind, int(part_id)
        if key in self.parts:
            self.parts.move_to_end(key)
            return self.parts[key]
        group = self.manifest[kind + "_shards"]
        if not 0 <= part_id < len(group):
            raise ValueError("Packed shard index out of range")
        metadata = group[part_id]
        n, d = metadata["atoms"], 3 * metadata["atoms"] - 3
        if kind == "graph":
            fields = {
                "numbers": (n,),
                "nodes": (n, 10),
                "bonds": (n, n, 3),
                "S": (d, d),
                "C0": (d, d),
            }
        else:
            fields = {"positions": (n, 3), "row_ids": ()}
            if self.method == "precision":
                fields.update(precision_values=(d,), precision_vectors=(d, d))
            elif self.method == "alternative_a":
                fields["alt_L_K"] = (d, d)
            elif self.method != "control":
                raise ValueError("Unknown method")
        arrays = {}
        try:
            for name, shape in fields.items():
                item = metadata["files"][name]
                stat = Path(item["path"]).stat()
                if stat.st_size != item["bytes"]:
                    raise ValueError("Packed array size changed after freezing")
                stamp = stat.st_size, stat.st_mtime_ns
                if self.verified.get(item["path"]) != stamp:
                    if fingerprint(item["path"])["sha256"] != item["sha256"]:
                        raise ValueError("Packed array hash mismatch")
                    self.verified[item["path"]] = stamp
                array = np.load(item["path"], allow_pickle=False, mmap_mode="r")
                arrays[name] = array
                expected_dtype = (
                    np.int64
                    if name in ("numbers", "row_ids")
                    else (np.float32 if name in ("nodes", "bonds") else np.float64)
                )
                if (
                    array.shape[1:] != shape
                    or array.dtype != expected_dtype
                    or list(array.shape) != item["shape"]
                ):
                    raise ValueError("Packed array shape/dtype differs")
        except BaseException:
            for array in arrays.values():
                array._mmap.close()
            raise
        self.parts[key] = arrays
        while len(self.parts) > self.max_open_parts:
            _, evicted = self.parts.popitem(last=False)
            for array in evicted.values():
                array._mmap.close()
        return arrays

    def _graph(self, record_id):
        from .model import graph_data

        if record_id in self.graphs:
            self.graphs.move_to_end(record_id)
            return self.graphs[record_id]
        record = self.records[record_id]
        arrays = self._part("graph", record["graph_shard"])
        row = record["graph_row"]
        numbers = arrays["numbers"][row].copy()
        dense = arrays["bonds"][row]
        edges = np.column_stack(np.nonzero(dense[:, :, 0]))
        graph = graph_data(
            {
                "atomic_numbers": numbers,
                "node_attr": arrays["nodes"][row].copy(),
                "bonds": edges,
                "bond_attr": dense[edges[:, 0], edges[:, 1]].copy(),
            }
        )
        source = Source(
            basis(len(numbers)), arrays["S"][row].copy(), arrays["C0"][row].copy()
        )
        if not np.isfinite(source.S).all() or not np.isfinite(source.C0).all():
            raise ValueError("Nonfinite graph-only source")
        self.graphs[record_id] = graph, source
        while len(self.graphs) > self.graph_cache_size:
            self.graphs.popitem(last=False)
        return graph, source

    def sample(self, record_id, conformer):
        graph, source = self._graph(record_id)
        part, row, row_id = map(int, self.index[self.offsets[record_id] + conformer])
        arrays = self._part("factor", part)
        if (
            row_id != self.records[record_id]["row_ids"][conformer]
            or int(arrays["row_ids"][row]) != row_id
        ):
            raise ValueError("Unpaired target/factor row")
        x1 = arrays["positions"][row].copy()
        if not np.isfinite(x1).all() or x1.shape != (graph.num_nodes, 3):
            raise ValueError("Invalid target coordinates")
        label = None
        if self.method == "precision":
            label = {
                "values": arrays["precision_values"][row].copy(),
                "vectors": arrays["precision_vectors"][row].copy(),
            }
            if np.any(label["values"] <= 0):
                raise ValueError("Nonpositive precision values")
        elif self.method == "alternative_a":
            label = {"L_K": arrays["alt_L_K"][row].copy()}
        if label and any(not np.isfinite(a).all() for a in label.values()):
            raise ValueError("Nonfinite label")
        return graph, {"x1": x1, "source": source, "label": label, "row_id": row_id}

    def close(self):
        for arrays in self.parts.values():
            for array in arrays.values():
                array._mmap.close()
        self.parts.clear()
        self.graphs.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(parts=OrderedDict(), graphs=OrderedDict(), verified={}, index=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.index = np.load(
            self.manifest["row_index"]["path"], allow_pickle=False, mmap_mode="r"
        )
