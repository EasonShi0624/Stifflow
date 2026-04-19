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
   - `scripts/make_geom_tiny_split.py` writes an opt-in, tiny, non-benchmark split-interface dry-run artifact from that processed dry-run.
   - It still does not write a real processed training subset, real benchmark split files, checkpoints, or training outputs.

8. **Add tiny training / sampling entry points**
   - `scripts/train.py --dry-run` validates config and file wiring.
   - `scripts/run_smoke_pipeline.py` runs both tiny smoke configs as one local CPU-only pipeline.
   - Real training and sampling are still not implemented.

9. **Add evaluation scaffolding**
   - Start with placeholders for COV/MAT and validity outputs.
   - Ensure the interface is stable even if internals are incomplete.
   - RDKit smoke baseline emits local fixture RMSD, bond-delta, MMFF-energy, and success-rate diagnostics.
   - Tiny real-data RDKit dry-run emits bounded ETKDG/MMFF diagnostics labeled as non-benchmark.
   - RDKit smoke outputs record Python/RDKit versions and distinguish import, fixture, ETKDG, MMFF, alignment, and runtime failures.
   - Smoke comparison report aggregates isotropic, stiffness, and RDKit smoke outputs into JSON and markdown.

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
