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
- `scripts/prepare_geom.py --dry-run --write-micro-pilot-output` writes a six-molecule, one-conformer-per-molecule, non-benchmark micro-pilot processed-output dry-run artifact.
- `scripts/make_geom_tiny_split.py` writes a deliberately tiny real-data split dry-run artifact for split-interface validation only.
- `scripts/make_geom_tiny_split.py --micro-pilot` writes a bounded six-molecule micro-pilot split dry-run artifact.
- `scripts/run_tiny_real_pilot.py --micro-pilot` runs the existing bounded CPU-only pilot loop on the six-molecule micro-pilot artifact.
- `scripts/compare_tiny_real_pilot.py --micro-pilot` writes a bounded micro-pilot comparison JSON summary and markdown table from the existing isotropic and stiffness-structured micro-pilot results.
- `scripts/run_micro_pilot_multiseed.py` runs the bounded six-molecule micro-pilot for isotropic and stiffness-structured modes over a fixed tiny seed set and writes a multi-seed JSON/markdown summary.
- `scripts/run_micro_pilot_checkpoint_sampling.py` runs a single-seed bounded six-molecule CPU-only micro-pilot checkpoint-and-sampling run for both modes, writing disposable pilot-only checkpoints plus manifest/result/log outputs.
- `scripts/compare_micro_pilot_checkpoint_sampling.py` writes a reporting-only JSON/markdown comparison of the two current bounded micro-pilot checkpoint-and-sampling outputs and flags suspicious runtime inconsistencies.
- `scripts/run_micro_pilot_checkpoint_sampling_multiseed.py` runs the bounded six-molecule CPU-only checkpoint-and-sampling pilot for both modes over the fixed tiny seed set `0,1,2` and writes a deterministic multi-seed JSON/markdown summary.
- `scripts/run_data_scale_check_checkpoint_sampling.py` writes a bounded 30-molecule processed dry-run artifact and split, then runs the same CPU-only checkpoint-and-sampling pilot settings for isotropic and stiffness-structured modes at seed `0`.
- `scripts/run_data_scale_check_checkpoint_sampling_multiseed.py` runs the existing bounded 30-molecule checkpoint-and-sampling setup for both modes over seeds `0,1,2` and writes a deterministic non-benchmark multi-seed JSON/markdown summary.
- `scripts/run_data_scale_check_500_checkpoint_sampling.py` writes a bounded 500-molecule processed dry-run artifact and nondegenerate 450/25/25 split, then runs the same CPU-only checkpoint-and-sampling pilot settings for isotropic and stiffness-structured modes at seed `0`.
- `scripts/run_data_scale_check_500_checkpoint_sampling_multiseed.py` runs the existing bounded 500-molecule checkpoint-and-sampling setup for both modes over seeds `0,1,2` and writes a deterministic non-benchmark multi-seed JSON/markdown summary.
- `scripts/run_data_scale_check_10000_checkpoint_sampling.py` runs the bounded 10000-molecule CPU-only checkpoint-and-sampling pilot for isotropic and stiffness-structured modes at seed `0`.
- `scripts/run_data_scale_check_10000_checkpoint_sampling_multiseed.py` runs the bounded 10000-molecule CPU-only checkpoint-and-sampling multi-seed repeat over seeds `0,1,2` and writes a deterministic non-benchmark multi-seed JSON/markdown summary.
- `scripts/run_data_scale_check_10000_checkpoint_sampling_convergence_multiseed.py` and `scripts/slurm_data_scale_check_10000_checkpoint_sampling_convergence_multiseed.sbatch` cover the bounded 10000-molecule CPU-only checkpoint-and-sampling convergence repeat over the same fixed seed set, with max_steps `200`, patience `20`, and a strict hard cap; any rerun should still be submitted only after explicit approval.
- `scripts/summarize_data_scale_check_10000_convergence_diagnostics.py` reads the existing bounded 10000-molecule steps200 result JSONs and writes a reporting-only convergence diagnostics JSON plus markdown report without running training or sampling.
- `scripts/run_data_scale_check_10000_formal_training_multiseed.py` and `scripts/slurm_data_scale_check_10000_formal_training_multiseed.sbatch` are prepared for the first formal bounded 10000-molecule CPU-only training phase over seeds `0,1,2`, using reusable validation-selected checkpoints and the frozen artifact, split, model, optimizer, checkpoint/reload pattern, and one-step sampling settings; submit only after explicit approval.
- `scripts/run_geom_full_database_formal_training_seed0.py` and `scripts/slurm_geom_full_database_formal_training_seed0.sbatch` are prepared for the first full-database GEOM-Drugs CPU-only formal training path at seed `0`, preserving the formal training design while changing the data scope to the full available one-conformer raw tree and a 90/5/5 random split; submit only after explicit approval.
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
- an opt-in micro-pilot processed-output dry-run capped at six molecules and one conformer per molecule;
- a tiny real-data split dry-run that writes a non-benchmark `split.json` plus manifest from the tiny processed artifact;
- a micro-pilot split dry-run and configurable bounded pilot path over the six-molecule artifact;
- a reporting-only bounded micro-pilot comparison summary/report over the two current six-molecule pilot outputs;
- a bounded CPU-only multi-seed micro-pilot runner and summary over a fixed tiny seed set for both source modes;
- a bounded CPU-only single-seed micro-pilot checkpoint-and-sampling runner that trains only the six-molecule split's train subset, writes disposable pilot-only checkpoints, reloads them, and runs one bounded sampling step;
- a reporting-only bounded micro-pilot checkpoint/sampling comparison that compares reloaded loss, sampled RMSD diagnostics, per-molecule sampled RMSD, and runtime accounting reliability;
- a bounded CPU-only micro-pilot checkpoint/sampling multi-seed runner and deterministic summary/report over seeds `0,1,2`;
- a bounded CPU-only 30-molecule data-scale-check checkpoint/sampling runner with a compact JSON/log comparison of the seed-0 directional signal;
- a bounded CPU-only 30-molecule data-scale-check checkpoint/sampling multi-seed runner and deterministic summary/report over seeds `0,1,2`;
- a bounded CPU-only 500-molecule data-scale-check checkpoint/sampling runner with a compact JSON/log comparison of the seed-0 directional signal;
- a bounded CPU-only 500-molecule data-scale-check checkpoint/sampling multi-seed runner and deterministic summary/report over seeds `0,1,2`;
- a bounded CPU-only 10000-molecule data-scale-check checkpoint/sampling runner with a compact JSON/log comparison of the seed-0 directional signal;
- a bounded CPU-only 10000-molecule data-scale-check checkpoint/sampling multi-seed runner and deterministic summary/report over seeds `0,1,2`;
- bounded CPU-only 10000-molecule data-scale-check checkpoint/sampling convergence multi-seed outputs over seeds `0,1,2` with max_steps `200`;
- a reporting-only bounded 10000-molecule convergence diagnostics summary/report over the existing steps200 loss traces;
- a prepared first formal bounded CPU-only 10000-molecule training path with max_steps `2000`, validation-loss checkpoint selection/stopping, reusable best/latest checkpoints, and a fixed non-benchmark aggregation interface;
- a prepared first full-database CPU-only GEOM-Drugs formal training path that scans the available `rdkit_folder/drugs` raw tree, writes a one-conformer-per-molecule processed artifact and 90/5/5 random split, then runs the frozen formal training design for isotropic and stiffness-structured modes at seed `0` after approved scheduling;
- honest placeholders for GEOM preprocessing, model sampling, and ablation orchestration.

Still missing or intentionally incomplete:
- completed full-database GEOM-Drugs processed/split artifacts;
- real GEOM-Drugs random/scaffold split artifacts;
- full real GEOM-Drugs raw-data schema/parser beyond bounded metadata inspection;
- real GEOM-scale RDKit ETKDG/MMFF baseline runner beyond the three-molecule dry-run;
- real equivariant neural backbone implementation and full model training;
- any benchmark-style model training beyond the prepared bounded 10000-molecule and full-database formal paths;
- trainable model loops beyond the bounded tiny smoke checks, the bounded three-molecule pilot, bounded six-molecule micro-pilot, and bounded 30/500/10000-molecule data-scale checks;
- real model sampling beyond bounded tiny smoke checks, the bounded six-molecule micro-pilot checkpoint/sampling pilot, and bounded 30/500/10000-molecule checkpoint/sampling data-scale checks;
- real COV/MAT over generated conformer ensembles;
- PoseBusters/MMFF/UFF/xTB integrations;
- full random-vs-scaffold benchmark runs.

If any of the above now exist, update this file.

---

## What the next coding pass should accomplish

The next implementation pass should aim to turn the smoke scaffold into the first real runnable milestone:

1. submit the prepared bounded 10000-molecule formal training job only after explicit approval, then review validation-selected checkpoints before making benchmark claims;
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
