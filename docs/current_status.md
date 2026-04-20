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
- `scripts/run_rdkit_tiny_real.py` runs RDKit ETKDGv3 + MMFF on the tiny three-molecule real-data dry-run artifact if RDKit is importable.
- `scripts/run_tiny_real_model_dry_runs.py` runs tiny real-data model-side dry-runs for isotropic and stiffness-structured source configurations on the existing three-molecule processed artifact and tiny split, without neural training.
- `scripts/run_tiny_real_execution_smoke.py` runs one parameter-free CPU forward/loss-style execution smoke pass for isotropic and stiffness-structured tiny real configurations, without optimizers, backward passes, checkpoints, sampling, or training loops.
- `scripts/run_tiny_real_single_step_smoke.py` runs one CPU-only forward/loss/backward/optimizer-step training smoke for isotropic and stiffness-structured tiny real configurations, without a real training loop, checkpoints, sampling, GPUs, or benchmark claims.
- `scripts/run_tiny_real_multi_step_smoke.py` runs a bounded CPU-only three-step training smoke for isotropic and stiffness-structured tiny real configurations, records per-step loss metadata, and remains non-benchmark with no checkpoints or larger training run.
- `scripts/run_tiny_real_checkpoint_sampling_smoke.py` runs a bounded CPU-only train/save/reload/sample smoke for isotropic and stiffness-structured tiny real configurations, writing disposable smoke checkpoints only.
- `scripts/summarize_tiny_real_lifecycle.py` writes an end-to-end tiny real-data smoke lifecycle JSON summary and markdown table across all current tiny real result stages.
- `scripts/run_tiny_real_pilot.py` runs a bounded CPU-only tiny real-data pilot training loop for isotropic and stiffness-structured source configurations, using only the existing three-molecule artifact and writing no checkpoints or samples.
- `scripts/compare_tiny_real_pilot.py` writes a bounded tiny real-data pilot comparison JSON summary and markdown table from the existing isotropic and stiffness-structured pilot results.
- `scripts/compare_tiny_real_results.py` writes a tiny real-data dry-run JSON summary and markdown table across the RDKit, isotropic, and stiffness-structured tiny real dry-run outputs.
- `scripts/compare_smoke_results.py` writes a smoke-only JSON summary and markdown table across isotropic, stiffness, and RDKit smoke outputs.
- `scripts/prepare_geom.py --dry-run` runs a read-only GEOM-Drugs raw-data preflight plus a tiny metadata-only preprocessing dry-run over bounded `rdkit_folder/drugs/*.pickle` samples.
- `scripts/prepare_geom.py --dry-run --write-tiny-processed-output` writes a deliberately tiny real-data processed-output dry-run artifact, currently capped at three molecules, for extraction-schema validation only.
- `scripts/make_geom_tiny_split.py` writes a deliberately tiny real-data split dry-run artifact for split-interface validation only.
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
- a CPU-only RDKit ETKDGv3 + MMFF tiny real-data dry-run wrapper over the bounded GEOM artifact;
- CPU-only tiny real-data model-side dry-runs that validate isotropic and stiffness-structured source construction against the bounded GEOM artifact and tiny split;
- CPU-only tiny real-data model execution smoke tests that run one parameter-free forward/loss-style pass for isotropic and stiffness-structured configurations;
- CPU-only tiny real-data single-step training smoke tests that run exactly one optimizer step for isotropic and stiffness-structured configurations;
- CPU-only tiny real-data bounded multi-step training smoke tests that run a hard-capped tiny optimizer sequence for isotropic and stiffness-structured configurations;
- CPU-only tiny real-data checkpoint-and-sampling smoke tests that write and reload disposable smoke checkpoints before one bounded sampling step;
- an end-to-end tiny real-data smoke lifecycle summary/report that distinguishes baseline, dry-run, execution, training-smoke, and checkpoint/sampling-smoke stages;
- a bounded CPU-only tiny real-data pilot training loop over the existing three-molecule artifact for isotropic and stiffness-structured configurations, labeled as pilot-only and non-benchmark;
- a reporting-only bounded tiny real-data pilot comparison summary/report over the two current pilot outputs;
- a tiny real-data dry-run comparison summary/report over the current RDKit, isotropic, and stiffness-structured tiny real outputs;
- a CPU-only smoke comparison summary/report over current synthetic fixture outputs;
- a read-only GEOM-Drugs preflight that reports raw-data availability and samples bounded raw pickle metadata;
- an opt-in tiny real-data processed-output dry-run that writes a non-benchmark `molecules.json` plus manifest with enough bounded molecules for a non-degenerate split dry-run;
- a tiny real-data split dry-run that writes a non-benchmark `split.json` plus manifest from the tiny processed artifact;
- honest placeholders for GEOM preprocessing, model sampling, and ablation orchestration.

Still missing or intentionally incomplete:
- real GEOM-Drugs preprocessing;
- a real GEOM-Drugs processed training subset;
- real GEOM-Drugs random/scaffold split artifacts;
- full real GEOM-Drugs raw-data schema/parser beyond bounded metadata inspection;
- real GEOM-scale RDKit ETKDG/MMFF baseline runner beyond the three-molecule dry-run;
- real equivariant neural backbone implementation and full model training;
- any real model training beyond the bounded three-molecule pilot;
- trainable model loops beyond the bounded tiny smoke checks and the bounded three-molecule pilot;
- real model sampling beyond bounded tiny smoke checks;
- real COV/MAT over generated conformer ensembles;
- PoseBusters/MMFF/UFF/xTB integrations;
- full random-vs-scaffold benchmark runs.

If any of the above now exist, update this file.

---

## What the next coding pass should accomplish

The next implementation pass should aim to turn the smoke scaffold into the first real runnable milestone:

1. grow the bounded tiny real pilot only after a real subset and stronger backbone path are explicitly approved;
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
