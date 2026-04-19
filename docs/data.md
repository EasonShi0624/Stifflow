# Data Contract

## Main dataset

The main benchmark is **GEOM-Drugs**.

QM9 may be used only for debugging or smoke tests.

---

## Data philosophy

This project is designed around a clean separation between:
- raw downloaded data,
- processed molecule/conformer records,
- split definitions,
- experiment-time manifests and outputs.

The repo should avoid ad hoc file conventions that make later reproduction difficult.

---

## Intended directory layout

```text
data/
├── raw/
│   └── geom_drugs/
├── processed/
│   └── geom_drugs_subset/
└── splits/
    ├── random/
    └── scaffold/
```

The exact final filenames may change, but they should follow a consistent and documented convention.

---

## Molecule grouping rule

All conformers belonging to the same molecule must remain together in the same split.

No molecule should have conformers spread across train/validation/test.

---

## Recommended first-pass subset

For an initial medium-scale experiment:
- 20k–50k molecules
- up to 30 conformers per molecule

For a smoke test:
- a tiny subset small enough to run quickly on CPU or a small GPU allocation
- enough molecules to exercise data loading, splitting, sampling, and evaluation paths

---

## Required metadata per molecule

At minimum, processed records should preserve or make derivable:
- molecule identifier
- molecular graph / atom-bond structure
- atom types
- formal charges if relevant
- conformer coordinates
- number of conformers
- ring-related information if needed for corruption heuristics
- rotatable-bond count or enough chemistry information to compute it later

Useful optional metadata:
- heavy atom count
- ring count
- stereocenter presence
- scaffold label

---

## Split files

Split definitions should be explicit and reusable.

Each split artifact should record:
- split type (`random` or `scaffold`)
- random seed if applicable
- molecule IDs in each partition
- subset identifier / data snapshot identifier

This allows later evaluations to reuse exactly the same split without recomputation.

---

## Smoke-test data contract

Before the full benchmark exists, it is acceptable to define a tiny smoke-test format such as:
- a small serialized file containing a handful of molecules and conformers;
- a small JSON or pickle manifest with molecule IDs and coordinates;
- a minimal split file for train/val/test.

The smoke-test dataset should be small, honest, and easy to inspect.

The current smoke fixture is:

```text
data/processed/smoke_fixture/molecules.json
```

It is labeled `dataset_kind: synthetic_smoke` and is hand-authored only to exercise wiring. It is **not** GEOM-Drugs, not QM9, and not benchmark data.

Current record fields include:
- `molecule_id`
- `smiles`
- `scaffold`
- `atom_symbols`
- `bonds`
- `rotatable_bond_count`
- `ring_atom_indices`
- `amide_atom_indices`
- `conformers`

---

## Important warning for coding agents

If the raw GEOM-Drugs files are not yet present in the repo or on the filesystem:
- do **not** pretend the data pipeline already works;
- do **not** invent a fake “processed GEOM” file and label it as real benchmark data;
- do create clean interfaces that can later be connected to the real dataset.

Current GEOM-Drugs readiness check:

```bash
python scripts/prepare_geom.py --dry-run
```

This is a read-only preflight. It inspects `data/raw/geom_drugs`, looks for common GEOM-like layout hints, and writes:
- `experiments/manifests/geom_drugs_preflight.json`
- `reports/tables/geom_drugs_preflight.md`

It does **not** parse molecules, modify raw data, write processed records, create split files, train models, or run benchmark evaluation.

---

## Later expansion

This file should eventually record:
- exact raw file names or source manifests;
- the processed record schema;
- scaffold-splitting implementation details;
- how rotatable-bond bins are computed;
- any filtering rules used during preprocessing.
