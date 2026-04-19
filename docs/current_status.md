# Current Repository Status

_Last updated: synthetic smoke scaffold stage._

## Summary

This repository is currently in an early **synthetic smoke scaffold phase**.

Current status:
- `README.md`, `AGENTS.md`, and project docs exist.
- A tiny synthetic smoke fixture exists at `data/processed/smoke_fixture/molecules.json`.
- Minimal config loading, molecule-level splitting, source corruption, smoke metrics, and manifest/result writing exist.
- Smoke CLI entry points re-exec through `/scratch/ys6132/programs/miniforge3/envs/stifflow/bin/python` instead of relying on ambient shell activation.
- `scripts/train.py --dry-run` validates the smoke path without running neural training.
- `scripts/run_smoke_pipeline.py` runs both tiny smoke configs locally and writes aggregate smoke-only outputs.
- `scripts/run_rdkit_smoke.py` runs a tiny local RDKit ETKDGv3 + MMFF smoke baseline if RDKit is importable, recording Python/RDKit versions and categorized failure modes.
- `scripts/compare_smoke_results.py` writes a smoke-only JSON summary and markdown table across isotropic, stiffness, and RDKit smoke outputs.
- `scripts/prepare_geom.py --dry-run` runs a read-only GEOM-Drugs raw-data preflight plus a tiny metadata-only preprocessing dry-run over bounded `rdkit_folder/drugs/*.pickle` samples.
- `scripts/evaluate.py` currently re-runs the same smoke evaluation path.

This file is the canonical place to record **what is actually present today**, as opposed to the intended target design.

---

## What exists now

The repo contains:
- project framing and scientific scope;
- agent instructions;
- target structure and experimental plan;
- a minimal `src/` package for config, data, splitting, corruption, smoke evaluation, and smoke-run output writing;
- a CPU-only smoke pipeline that runs the isotropic and stiffness-structured synthetic fixture configs together;
- a CPU-only RDKit ETKDGv3 + MMFF smoke wrapper for the synthetic fixture;
- a CPU-only smoke comparison summary/report over current synthetic fixture outputs;
- a read-only GEOM-Drugs preflight that reports raw-data availability and samples bounded raw pickle metadata without writing processed molecules;
- honest placeholders for GEOM preprocessing, model sampling, and ablation orchestration.

Still missing or intentionally incomplete:
- real GEOM-Drugs preprocessing;
- full real GEOM-Drugs raw-data schema/parser beyond bounded metadata inspection;
- real GEOM-scale RDKit ETKDG/MMFF baseline runner;
- neural backbone implementation and model training;
- model sampling;
- real COV/MAT over generated conformer ensembles;
- PoseBusters/MMFF/UFF/xTB integrations;
- full random-vs-scaffold benchmark runs.

If any of the above now exist, update this file.

---

## What the next coding pass should accomplish

The next implementation pass should aim to turn the smoke scaffold into the first real runnable milestone:

1. add a tiny real training loop or explicitly separate `scripts/smoke.py` from future `scripts/train.py`;
2. add scaffold-split tests using fixture scaffold labels;
3. add a real training-loop smoke test or separate smoke scripts from future training scripts;
4. keep all smoke outputs labeled as wiring checks, not scientific results;
5. keep manifest/result outputs stable while expanding internals.

---

## Rules for interpreting this file

- This file should describe the **real repo state**, not the idealized future state.
- Update it whenever major implementation milestones land.
- If `README.md` or `AGENTS.md` say a script is part of the target design, but the script does not actually exist yet, this file should say so plainly.

---

## Known current constraints

- Working environment is expected to be `conda activate stifflow`.
- The repo is intended for use on an HPC system.
- Large training or preprocessing jobs should not be assumed runnable from a login node.

---

## Immediate next milestone

The immediate goal is **not** full benchmark reproduction.

The immediate goal is:

> extend the tiny, honest, config-driven smoke-test pipeline toward the first real baseline without overstating what currently runs.
