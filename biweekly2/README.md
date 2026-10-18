# Biweekly Report 2 — minimal supplementary experiments and core figures

SHHPC, 2026-10-19. Goal: verify the evaluation caliber, answer the numerical
verification request, add physical-quality diagnostics, and explain the
parity between Alternative A (Euclidean A_E) and the stiffness-free
ET-Flow-backbone control — using existing checkpoints, generated conformers
and Hessian labels wherever possible.

## Code versions

| worktree | branch | commit |
|---|---|---|
| `stifflow-v1` | `shhpc-research-20261018` | `dc3e07b650447faceeefe7abac5c7b90d82d0cc4` |
| `stifflow-v1-benchmark` | `shhpc-research-20261018-benchmark` | `01b3865f49b6d01ad91de3f79dee5631c944d9bd` |
| `stifflow-v1-alternative-A` | `alternative-a` | `c56d466cdd4c309e7ce87543475a085413ac57a1` |

No source files in any worktree were modified; all new code lives in
`runs/biweekly2-20261019/scripts/` (committed on the biweekly-2 branch).

## Environment

- `ETFLOW_ENV=/gpfsnyu/home/ys6132/anaconda3/envs/etflow` (Python 3.12.14,
  torch 2.4.1, numpy 1.26.4, rdkit 2026.03.6, scipy 1.17.1, matplotlib 3.11.2)
- Slurm: `parallel` (CPU) and `rooster` 4090D (GPU), account wjg3.

## Data (all pre-existing, unchanged)

- frozen data manifest `5bb7f261…`; monitor `557bcd43…` (512 molecules /
  2,741 references); packed val labels `3022258a…`
- formal checkpoints `runs/formal-4090d-staged/{method}-seed{42,43,44}/`
  (250k) and `snapshots/step-*.ckpt` (25k–250k)
- formal generations/evaluations `runs/formal-eval-20261018/{test,val,monitor}/`

## New experiments and job IDs

| task | what | job(s) | status |
|---|---|---|---|
| 1 caliber | reference-mapping audit + pooled 3-seed paired bootstrap (test/val) | 4349989 | COMPLETED |
| 2a | empirical conditional-Gaussian verification (32 mols × 5 t × M ∈ {256,1024,4096}) | 4349990 | COMPLETED |
| 2b | SE(3)/permutation equivariance + sampling consistency, trained 250k A & control ckpts | 4349993 | COMPLETED |
| 3 | MMFF94 physical quality on monitor (3 methods + references, 19,187 conformers) | 4349992 | COMPLETED |
| 3b | strain × chirality × coverage pairing | login CPU | done |
| 4a | A-vs-control supervision difference (128 rows, shared z) + label-audit aggregation (9,423 receipts) | 4350000 | COMPLETED |
| 4b | monitor learning-curve snapshots: A 25k/50k/100k/175k, control 50k/100k/175k | 4349966_0–6 (gen), 4349967_0–6 (eval) | COMPLETED |

All new generations use the frozen formal protocol: Euler 50 NFE, seed 42,
2× reference count, no postprocessing; monitor only (test untouched).

## Results

- `results/task1-caliber.json` — mapping audit (0 mappings; P0 recheck all
  true) + pooled paired bootstrap CIs.
- `results/task2a-conditional-gaussian.json` — Σ identity residual ≤ 1.3e-15;
  empirical covariance → analytic at 1/√M; whitened spectra within
  Marchenko–Pastur bounds everywhere.
- `results/task2b-equivariance.json` — max equivariance errors ~1e-6;
  sampling-consistency endpoint RMSD ≤ 7.5e-4 Å.
- `results/task3-physical-quality.json`, `results/task3b-strain-chirality.json`
  — A 5.47 vs control 5.52 vs refs 5.28 kcal/mol median strain; paired A−control
  +0.03 [−0.23, +0.35]; stereo changes during relaxation recorded (~1%).
- `results/task4a-stiffness-signal.json` — median A−control supervision
  velocity difference 6–10%; ‖L_K−S‖/‖S‖ median 0.83; alignment-RMSD median
  0.153 Å; floored modes in 14.4% of molecules; calibration ≈ 1.000.
- `results/task4b-learning-curves.json` — A ≈ control at every 25k–250k
  snapshot (MAT-R within 0.01 Å).
- `results/compact-table.json` — report table data.
- `results/FINDINGS.md` — report-ready conclusions and figure captions.
- `figures/figure1-verification-mechanism.{pdf,png}`,
  `figures/figure2-performance-quality.{pdf,png}` — the two core figures.
- `results/figure-data.json` — assembled plotting inputs.

## Reproduce

```bash
WS=/gpfsnyu/scratch/ys6132/stifflow
cd "$WS/runs/biweekly2-20261019"
# CPU jobs (scripts submit sbatch with account=wjg3):
sbatch scripts/task1_caliber.sbatch
sbatch scripts/task2a.sbatch
sbatch scripts/task3.sbatch
sbatch scripts/task4a.sbatch
# GPU jobs:
sbatch scripts/task2b.sbatch
sbatch --array=0-6 scripts/task4b_snapshots.sbatch generate
sbatch --array=0-6 --dependency=afterok:<gen_job> scripts/task4b_snapshots.sbatch evaluate
# Assembly (login node, light):
python scripts/task3b_strain_chirality.py
python scripts/task4b_summary.py
python scripts/collect_figure_data.py
python scripts/make_figures.py
python scripts/make_compact_table.py
```

## Notes and caveats

- First task2b submission (4349963) exposed a harness bug: my permutation
  relabeled bonds with `perm` instead of `argsort(perm)`; fixed and rerun as
  4349993. The upstream unit test (`test_dense_neighbors_equivariance…`)
  uses the correct inverse mapping, so this was a diagnostic-harness bug, not
  a model bug. The stale output was deleted and regenerated.
- Sampling-consistency permutation RMSD (7.5e-4 Å) is larger than rotation
  (~4e-7 Å) because `radius_graph` neighbor counts can differ between permuted
  and original atom orderings at float32 precision; still ~3 orders below
  conformer scales.
- MMFF94 minimization uses forceTol 1e-6; all 19,187 minimizations converged
  (status 0); no missing-parameter failures.
- The four 25k null/linear ablation trainings (research-20261018) remain
  unevaluated by choice — listed as next stage, not run this round.
- Git: results committed on the local branch `shhpc-biweekly2-20261019`
  (not pushed, per user decision). Raw NPZs, checkpoints, caches and the
  per-conformer task3 JSON stay outside Git.
