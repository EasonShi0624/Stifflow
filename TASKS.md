# TASKS

This file tracks the current prioritized work queue.

## Highest priority

1. **Create the repo skeleton** — done for the smoke scaffold
   - Add `src/`, `scripts/`, `configs/`, `tests/`, and `experiments/` directories if absent.
   - Keep the structure minimal and clean.

2. **Define a tiny smoke-test data interface** — done for `synthetic_smoke`
   - Decide what the smallest acceptable molecule/conformer record format will be.
   - Make sure it supports train/val/test split membership.

3. **Add config loading** — done for the minimal smoke schema
   - Establish a minimal config system.
   - Confirm that configs can specify split type, backbone, corruption mode, and seed.

4. **Implement corruption interface** — done for isotropic and smoke stiffness-structured modes
   - Add a shared interface for source construction.
   - First modes: isotropic and stiffness-structured.

5. **Implement baseline utilities** — partially done
   - Random split and scaffold split helpers exist.
   - Tiny local RDKit ETKDGv3 + MMFF smoke wrapper exists for the synthetic fixture.
   - Tiny local RDKit ETKDGv3 + MMFF dry-run wrapper exists for the bounded three-molecule real GEOM artifact.
   - Real GEOM-scale RDKit baseline runner is still missing.

6. **Implement machine-readable outputs** — done for dry-run smoke outputs
   - Every smoke-test run should emit a manifest JSON and a results JSON.
   - Smoke CLI entry points pin execution to the `stifflow` Python interpreter.

## Next priority

7. **Add GEOM-Drugs raw-data readiness checks** — partially done
   - `scripts/prepare_geom.py --dry-run` inspects `data/raw/geom_drugs` read-only.
   - Preflight writes a manifest/report and runs a tiny bounded metadata-only dry-run over `rdkit_folder/drugs/*.pickle` when present.
   - `scripts/prepare_geom.py --dry-run --write-tiny-processed-output` writes an opt-in, tiny, non-benchmark processed-output dry-run artifact capped at three molecules by default.
   - `scripts/prepare_geom.py --dry-run --write-micro-pilot-output` writes an opt-in, six-molecule, non-benchmark micro-pilot processed-output dry-run artifact.
   - `scripts/make_geom_tiny_split.py` writes an opt-in, tiny, non-benchmark split-interface dry-run artifact from that processed dry-run.
   - `scripts/make_geom_tiny_split.py --micro-pilot` writes a non-benchmark micro-pilot split dry-run from the six-molecule artifact.
   - It still does not write a real processed training subset, real benchmark split files, checkpoints, or training outputs.

8. **Add tiny training / sampling entry points**
   - `scripts/train.py --dry-run` validates config and file wiring.
   - `scripts/run_smoke_pipeline.py` runs both tiny smoke configs as one local CPU-only pipeline.
   - `scripts/run_tiny_real_model_dry_runs.py` validates isotropic and stiffness-structured model-side source construction on the bounded three-molecule real GEOM dry-run artifact without training.
   - `scripts/run_tiny_real_execution_smoke.py` runs one parameter-free CPU forward/loss-style execution smoke pass for isotropic and stiffness-structured tiny real configurations without optimizers, backward passes, checkpoints, sampling, or training loops.
   - `scripts/run_tiny_real_single_step_smoke.py` runs exactly one CPU-only forward/loss/backward/optimizer-step smoke for isotropic and stiffness-structured tiny real configurations without a real training loop, checkpoints, sampling, or benchmark claims.
   - `scripts/run_tiny_real_multi_step_smoke.py` runs a hard-capped CPU-only three-step training smoke for isotropic and stiffness-structured tiny real configurations, records per-step metadata, and does not write checkpoints or start larger training.
   - `scripts/run_tiny_real_checkpoint_sampling_smoke.py` runs a bounded CPU-only train/save/reload/sample smoke for isotropic and stiffness-structured tiny real configurations, writing disposable smoke checkpoints only.
   - `scripts/run_tiny_real_pilot.py` runs a bounded CPU-only tiny real-data pilot training loop for isotropic and stiffness-structured configurations on the existing three-molecule artifact, without checkpoints, sampling, jobs, GPUs, or benchmark claims.
   - `scripts/run_tiny_real_pilot.py --micro-pilot` runs the same bounded CPU-only pilot loop on the six-molecule micro-pilot artifact and split.
   - `scripts/run_micro_pilot_multiseed.py` runs a fixed tiny seed set for both micro-pilot modes on CPU and writes a non-benchmark multi-seed summary.
   - `scripts/run_micro_pilot_checkpoint_sampling.py` runs a single-seed, bounded CPU-only six-molecule micro-pilot checkpoint-and-sampling pilot for both modes, writing disposable pilot-only checkpoints plus manifest/result/log outputs.
   - `scripts/run_micro_pilot_checkpoint_sampling_multiseed.py` runs the bounded CPU-only six-molecule checkpoint-and-sampling pilot across the fixed tiny seed set `0,1,2` for both modes and writes a non-benchmark multi-seed summary.
   - Micro-pilot checkpoint/sampling runtime accounting now excludes one-time PyTorch CPU backend warm-up from comparable per-mode `runtime_seconds`.
   - Larger real training and real sampling are still not implemented.

9. **Add evaluation scaffolding**
   - Start with placeholders for COV/MAT and validity outputs.
   - Ensure the interface is stable even if internals are incomplete.
   - RDKit smoke baseline emits local fixture RMSD, bond-delta, MMFF-energy, and success-rate diagnostics.
   - Tiny real-data RDKit dry-run emits bounded ETKDG/MMFF diagnostics labeled as non-benchmark.
   - Tiny real-data model dry-runs emit bounded source-vs-target wiring diagnostics for isotropic and stiffness-structured configurations labeled as non-benchmark.
   - Tiny real-data execution smoke outputs emit bounded forward/loss diagnostics labeled as non-benchmark and not training.
   - Tiny real-data single-step training smoke outputs emit bounded loss/gradient/optimizer-step diagnostics labeled as non-benchmark and not real training.
   - Tiny real-data bounded multi-step smoke outputs emit per-step loss/gradient diagnostics labeled as non-benchmark and not larger training.
   - Tiny real-data checkpoint-and-sampling smoke outputs emit disposable checkpoint/reload and one-step sampling diagnostics labeled as non-benchmark.
   - Tiny real-data bounded pilot outputs emit train/val/test loss diagnostics for a strictly bounded CPU-only pilot loop labeled as non-benchmark.
   - Bounded micro-pilot checkpoint-and-sampling outputs emit train-split loss traces, disposable checkpoint reload diagnostics, and one-step six-molecule sampling diagnostics labeled as non-benchmark.
   - RDKit smoke outputs record Python/RDKit versions and distinguish import, fixture, ETKDG, MMFF, alignment, and runtime failures.
   - Smoke comparison report aggregates isotropic, stiffness, and RDKit smoke outputs into JSON and markdown.
   - Tiny real-data dry-run comparison report aggregates the current RDKit, isotropic, and stiffness-structured tiny real outputs into JSON and markdown.
   - Tiny real-data lifecycle summary aggregates all current tiny real stages into one non-benchmark JSON and markdown report.
   - Tiny real-data bounded pilot comparison report aggregates the two current pilot outputs into JSON and markdown without running training.
   - Bounded micro-pilot comparison report aggregates the two current six-molecule pilot outputs into JSON and markdown without running training.
   - Bounded micro-pilot multi-seed summary aggregates the fixed tiny seed set for both modes into JSON and markdown.
   - Bounded micro-pilot checkpoint-and-sampling comparison report aggregates the two current checkpoint/sampling outputs into JSON and markdown, including runtime-consistency warnings for legacy outputs.
   - Bounded micro-pilot checkpoint-and-sampling multi-seed summary aggregates the fixed tiny seed set for both modes into JSON and markdown, reporting per-seed winners and aggregate means.

10. **Draft one smoke-test config**
   - Baseline backbone
   - Random split
   - Isotropic corruption
   - Very small dataset size

11. **Draft one main-method config**
    - Same backbone
    - Same objective
    - Stiffness-structured corruption
    - Done for `configs/train/tiny_smoke_stiffness.yaml`

## Defer until core pipeline exists

- MMFF-derived stiffness sensitivity
- xTB/GFN2-xTB subset checks
- force-residual repair
- chirality-aware extensions
- architecture exploration



## Infrastructure and HPC policy tasks
- [ ] Add a canonical SLURM submission template for smoke tests.
- [ ] Add a canonical SLURM submission template for real training/evaluation.
- [ ] Record account choice in manifests for scheduled runs.
- [ ] Keep account selection consistent with `docs/slurm.md`.
