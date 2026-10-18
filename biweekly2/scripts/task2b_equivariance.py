"""Task 2b: equivariance and sampling-consistency checks on formal checkpoints.

Loads the formal Alternative A checkpoint (250k updates, seed 42; control as a
contrast) and, on real validation monitor molecules, verifies with the trained
network:

1. SE(3) equivariance of the learned velocity field: for a random proper
   rotation R and translation b, v(Rx + b, t; G) == R v(x, t; G).  The model
   centers positions internally, so translation must be exactly invariant and
   rotation equivariant up to network float32 noise.
2. Atom-permutation equivariance: permuting graph, coordinates and features
   jointly permutes the velocity.
3. Sampling consistency from transformed initial noise: Euler-50 generation
   from R-rotated initial noise reproduces R times the untransformed endpoint
   (no realignment), and permuted noise reproduces the permuted endpoint.

Reports relative errors and endpoint RMSDs.  This is a verification of the
delivered model, not a retraining or performance measurement.

Outputs runs/biweekly2-20261019/results/task2b-equivariance.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
OUT = WS / "runs/biweekly2-20261019/results/task2b-equivariance.json"
MONITOR = WS / "cache/qm9-v3-references-4090d/val/monitor.json"
CHECKPOINTS = {
    "alternative_a": WS
    / "runs/formal-4090d-staged/alternative_a-seed42/checkpoints/step-00250000.ckpt",
    "control": WS
    / "runs/formal-4090d-staged/control-seed42/checkpoints/step-00250000.ckpt",
}
LIMIT = 32
CONSISTENCY_LIMIT = 16
SEED = 42


def select_records(records, limit):
    ordered = sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    return ordered[:limit]


def load_model(checkpoint_path, device):
    import torch

    from stifflow.model import StifFlow

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = StifFlow(**saved["hyper_parameters"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, saved


def velocity(model, graph_data_obj, positions, t_value, device):
    import torch

    n = positions.shape[0]
    data = graph_data_obj
    batch = torch.zeros(n, dtype=torch.long, device=device)
    t = torch.full((n, 1), float(t_value), dtype=torch.float32, device=device)
    pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(
            z=data.atomic_numbers.to(device),
            t=t,
            pos=pos,
            bond_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            node_attr=data.node_attr.to(device),
            batch=batch,
        )
    return out.cpu().numpy()


def generate_from_initial(model, graph, initial, steps, device):
    """Left Euler 0->1 from a given initial centered configuration."""
    import torch

    from stifflow.model import graph_data

    data = graph_data(graph)
    n = initial.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    x = torch.as_tensor(initial, dtype=torch.float32, device=device)
    with torch.no_grad():
        for step in range(steps):
            t = torch.full((n, 1), step / steps, dtype=torch.float32, device=device)
            x = x + model(
                z=data.atomic_numbers.to(device),
                t=t,
                pos=x,
                bond_index=data.edge_index.to(device),
                edge_attr=data.edge_attr.to(device),
                node_attr=data.node_attr.to(device),
                batch=batch,
            ) / steps
    return x.cpu().numpy()


def initial_from_source(graph, rng, samples, source):
    z = rng.standard_normal((samples, source.S.shape[0]))
    initial = (z @ source.S.T @ source.Q.T).reshape(samples, -1, 3)
    return initial


def check_molecule(model, record, rng, device, steps=50):
    import torch
    from rdkit import Chem

    from stifflow.model import graph_data
    from stifflow.source import prepare_source

    mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
    mol.RemoveAllConformers()
    from stifflow.features import graph_from_mol

    graph = graph_from_mol(mol)
    data = graph_data(graph)
    source = prepare_source(graph, model.source_config)
    n = mol.GetNumAtoms()

    # Random configuration near a plausible radius: use the source prior draw.
    rng_np = np.random.default_rng([SEED, hash(record["id"]) % (2**31)])
    x0 = initial_from_source(graph, np.random.default_rng(SEED + 7), 1, source)[0]
    x0 = x0 - x0.mean(axis=0)
    t_value = 0.5

    base = velocity(model, data, x0, t_value, device)

    # --- Rotation equivariance (+ translation invariance) ---
    from scipy.spatial.transform import Rotation

    rotations = []
    for k in range(3):
        R = Rotation.random(random_state=np.random.default_rng(SEED + 100 + k)).as_matrix()
        shift = rng_np.standard_normal(3) * 2.0
        v_rot = velocity(model, data, x0 @ R.T + shift, t_value, device)
        err = float(
            np.linalg.norm(v_rot - base @ R.T) / (np.linalg.norm(base) + 1e-12)
        )
        rotations.append(err)

    # --- Atom permutation equivariance ---
    permutations = []
    for k in range(2):
        perm = np.random.default_rng(SEED + 200 + k).permutation(n)
        inverse = np.argsort(perm)
        # new slot k holds old atom perm[k]; relabel old bonds by inverse.
        permuted_graph = {
            "atomic_numbers": np.asarray(graph["atomic_numbers"])[perm],
            "node_attr": np.asarray(graph["node_attr"])[perm],
            "bonds": inverse[np.asarray(graph["bonds"]).reshape(-1, 2)],
            "bond_attr": np.asarray(graph["bond_attr"]),
        }
        pdata = graph_data(permuted_graph)
        v_perm = velocity(model, pdata, x0[perm], t_value, device)
        err = float(
            np.linalg.norm(v_perm - base[perm])
            / (np.linalg.norm(base) + 1e-12)
        )
        permutations.append(err)

    return {
        "id": record["id"],
        "n_atoms": n,
        "rotation_equivariance_relative_error": rotations,
        "translation_invariance_included": True,
        "permutation_equivariance_relative_error": permutations,
    }


def consistency_check(model, record, device, steps=50):
    """Same initial noise, transformed: endpoint should transform likewise."""
    import torch
    from rdkit import Chem
    from scipy.spatial.transform import Rotation

    from stifflow.features import graph_from_mol
    from stifflow.source import prepare_source

    mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
    mol.RemoveAllConformers()
    graph = graph_from_mol(mol)
    source = prepare_source(graph, model.source_config)
    n = mol.GetNumAtoms()
    rng = np.random.default_rng([SEED, 9001])
    z = rng.standard_normal((1, source.S.shape[0]))
    initial = (z @ source.S.T @ source.Q.T).reshape(1, -1, 3)[0]

    base_endpoint = generate_from_initial(model, graph, initial, steps, device)

    R = Rotation.random(random_state=np.random.default_rng(SEED + 300)).as_matrix()
    rotated_initial = initial @ R.T
    rot_endpoint = generate_from_initial(model, graph, rotated_initial, steps, device)
    rot_err = float(
        np.sqrt(((rot_endpoint - base_endpoint @ R.T) ** 2).sum(axis=1).mean())
    )
    base_scale = float(
        np.sqrt((base_endpoint**2).sum(axis=1).mean())
    )

    perm = np.random.default_rng(SEED + 400).permutation(n)
    inverse = np.argsort(perm)
    perm_graph = {
        "atomic_numbers": np.asarray(graph["atomic_numbers"])[perm],
        "node_attr": np.asarray(graph["node_attr"])[perm],
        "bonds": inverse[np.asarray(graph["bonds"]).reshape(-1, 2)],
        "bond_attr": np.asarray(graph["bond_attr"]),
    }
    perm_initial = initial[perm]
    perm_endpoint = generate_from_initial(model, perm_graph, perm_initial, steps, device)
    perm_err = float(
        np.sqrt(((perm_endpoint - base_endpoint[perm]) ** 2).sum(axis=1).mean())
    )

    return {
        "id": record["id"],
        "n_atoms": n,
        "rotation_endpoint_rmsd_angstrom": rot_err,
        "permutation_endpoint_rmsd_angstrom": perm_err,
        "endpoint_scale_angstrom": base_scale,
        "rotation_relative": rot_err / base_scale,
        "permutation_relative": perm_err / base_scale,
    }


def main():
    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = json.loads(MONITOR.read_text())
    selected = select_records(monitor["records"], LIMIT)

    result = {"schema_version": 1, "steps": 50, "seed": SEED, "methods": {}}
    for method, checkpoint in CHECKPOINTS.items():
        model, saved = load_model(checkpoint, device)
        if saved["global_step"] != 250000:
            raise SystemExit(f"{method}: expected 250k checkpoint")
        rows = []
        for record in selected:
            rng = np.random.default_rng(SEED)
            rows.append(check_molecule(model, record, rng, device))
            print(f"{method} equivariance {record['id']}", flush=True)
        consistent = []
        for record in select_records(monitor["records"], CONSISTENCY_LIMIT):
            consistent.append(consistency_check(model, record, device))
            print(f"{method} consistency {record['id']}", flush=True)
        result["methods"][method] = {
            "checkpoint": str(checkpoint),
            "equivariance": rows,
            "sampling_consistency": consistent,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    summary = {}
    for method in result["methods"]:
        rot = [
            e
            for row in result["methods"][method]["equivariance"]
            for e in row["rotation_equivariance_relative_error"]
        ]
        perm = [
            e
            for row in result["methods"][method]["equivariance"]
            for e in row["permutation_equivariance_relative_error"]
        ]
        cons_r = [
            row["rotation_endpoint_rmsd_angstrom"]
            for row in result["methods"][method]["sampling_consistency"]
        ]
        cons_p = [
            row["permutation_endpoint_rmsd_angstrom"]
            for row in result["methods"][method]["sampling_consistency"]
        ]
        summary[method] = {
            "rotation_equivariance_max": float(max(rot)),
            "permutation_equivariance_max": float(max(perm)),
            "rotation_consistency_rmsd_max": float(max(cons_r)),
            "permutation_consistency_rmsd_max": float(max(cons_p)),
        }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
