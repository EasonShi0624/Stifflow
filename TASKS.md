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
   - Tiny RDKit baseline wrapper is still missing.

6. **Implement machine-readable outputs** — done for dry-run smoke outputs
   - Every smoke-test run should emit a manifest JSON and a results JSON.

## Next priority

7. **Add tiny training / sampling entry points**
   - `scripts/train.py --dry-run` validates config and file wiring.
   - `scripts/run_smoke_pipeline.py` runs both tiny smoke configs as one local CPU-only pipeline.
   - Real training and sampling are still not implemented.

8. **Add evaluation scaffolding**
   - Start with placeholders for COV/MAT and validity outputs.
   - Ensure the interface is stable even if internals are incomplete.

9. **Draft one smoke-test config**
   - Baseline backbone
   - Random split
   - Isotropic corruption
   - Very small dataset size

10. **Draft one main-method config**
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
