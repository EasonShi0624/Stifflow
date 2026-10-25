# aligned_linear 250k — full-budget confirmation (SHHPC, 2026-10-25)

This directory records the full-budget (250k updates, 3 seeds) confirmation of
`aligned_linear`, the only candidate that passed the 25k screening gates. It
contains the endpoint geometry comparison, the seed-42 learning curve, and the
250k MMFF94 physical audit. Per-molecule NPZ files, checkpoints and raw logs stay
in the external run directory (`/gpfsnyu/scratch/ys6132/stifflow/runs/`), not in Git.

## Provenance

| item | revision |
|---|---|
| training code | `stifflow-v1@bcdcbfd` (clean tree recorded by the run manifest; the config/scripts were staged in `f49f9d2`) |
| evaluation code | `stifflow-v1@a2ce2c6`, `stifflow-v1-benchmark@01b3865` |
| physics audit | `shhpc-physics-budget-audit@69c38f8` (adds `--budget`) |
| Alternative A | `stifflow-v1-alternative-A@c56d466` |
| ETFlow-reference | `80f82d7` |
| data | canonical train `a0bbf33c…`, val `3022258a…`; monitor reference `557bcd43…` |
| jobs | training `4353269/70/71`, eval `4375135/36/37`, physics `4463573`, learning curve `4463574` |

Data identity: the 25k research runs used relocated (staged) copies of the same
packed cache; `full-budget-comparison.json` verifies that the path-stripped
content hashes of the archived staged manifests equal the canonical manifests
for both control and Alternative A. The aligned_linear runs used the canonical
manifests directly.

## Protocol

250,000 optimizer updates, effective batch 128 (microbatch 32 × accumulation 4),
the original 250k cosine cycle, seeds 42/43/44, dataset seed 42, row RNG domain 2.
Generation: Euler 50 steps, generation seed 42, 2× reference count, no
postprocessing. Splits: test 993, val 9,654, monitor 512 molecules.

## 1. Endpoint comparison (3 seeds, mean ± seed SD)

| split | method | MAT-R (Å) | MAT-P (Å) | COV-R@0.5 | chirality |
|---|---|---|---|---|---|
| test | aligned_linear | **0.1040 ± 0.0017** | **0.1701 ± 0.0072** | **0.9462 ± 0.0048** | **0.8951 ± 0.0112** |
| test | control | 0.1292 ± 0.0040 | 0.2153 ± 0.0038 | 0.9274 ± 0.0052 | 0.8443 ± 0.0075 |
| test | alternative_a | 0.1270 ± 0.0071 | 0.2187 ± 0.0086 | 0.9302 ± 0.0040 | 0.8373 ± 0.0058 |
| val | aligned_linear | **0.1044 ± 0.0005** | **0.1603 ± 0.0026** | **0.9524 ± 0.0013** | **0.8937 ± 0.0073** |
| val | control | 0.1239 ± 0.0009 | 0.2094 ± 0.0012 | 0.9419 ± 0.0006 | 0.8365 ± 0.0017 |
| val | alternative_a | 0.1263 ± 0.0033 | 0.2168 ± 0.0084 | 0.9392 ± 0.0013 | 0.8250 ± 0.0070 |
| monitor | aligned_linear | **0.0972 ± 0.0016** | **0.1687 ± 0.0064** | 0.9505 ± 0.0053 | **0.8870 ± 0.0113** |
| monitor | control | 0.1196 ± 0.0048 | 0.2285 ± 0.0040 | 0.9410 ± 0.0063 | 0.8214 ± 0.0099 |
| monitor | alternative_a | 0.1219 ± 0.0052 | 0.2340 ± 0.0081 | 0.9415 ± 0.0016 | 0.8146 ± 0.0112 |

Paired molecule bootstrap (2,000 draws, shared IDs):

| split | contrast | ΔMAT-R [95% CI] | ΔMAT-P | ΔCOV-R@0.5 | Δchirality |
|---|---|---|---|---|---|
| test | AL − control | −0.0252 [−0.0328, −0.0180] | −0.0452 [−0.0527, −0.0380] | +0.0189 [+0.0100, +0.0279] | +0.0509 [+0.0407, +0.0614] |
| test | AL − A | −0.0231 [−0.0302, −0.0164] | −0.0486 [−0.0564, −0.0411] | +0.0160 [+0.0076, +0.0245] | +0.0578 [+0.0471, +0.0693] |
| val | AL − control | −0.0195 [−0.0216, −0.0175] | −0.0491 [−0.0515, −0.0466] | +0.0106 [+0.0082, +0.0129] | +0.0572 [+0.0538, +0.0606] |
| val | AL − A | −0.0219 [−0.0241, −0.0198] | −0.0565 [−0.0592, −0.0539] | +0.0132 [+0.0107, +0.0157] | +0.0687 [+0.0651, +0.0724] |
| monitor | AL − control | −0.0224 [−0.0331, −0.0127] | −0.0597 [−0.0707, −0.0494] | +0.0094 [−0.0017, +0.0211] | +0.0656 [+0.0511, +0.0806] |
| monitor | AL − A | −0.0247 [−0.0362, −0.0142] | −0.0653 [−0.0769, −0.0544] | +0.0089 [−0.0023, +0.0209] | +0.0725 [+0.0577, +0.0875] |

The test/val directions are identical in all three seeds; for example test
AL − control MAT-R per seed is −0.0237 / −0.0239 / −0.0281.

## 2. Seed-42 monitor learning curve (MAT-R, Å)

| budget | aligned_linear | control | alternative_a | precision |
|---|---|---|---|---|
| 25k | 0.2176 | 0.2791 | 0.2696 | 0.3406 |
| 50k | 0.1611 | 0.2186 | 0.2274 | 0.3127 |
| 75k | 0.1511 | — | — | — |
| 100k | 0.1359 | 0.1726 | 0.1668 | 0.2779 |
| 125k | 0.1223 | — | — | — |
| 150k | 0.1087 | — | — | — |
| 175k | 0.0962 | 0.1280 | 0.1240 | 0.2622 |
| 200k | 0.0974 | — | — | — |
| 225k | 0.0961 | — | — | — |
| 250k | 0.0957 | 0.1188 | 0.1179 | 0.2425 |

AL − control MAT-R gap: −0.0615 (25k), −0.0575 (50k), −0.0367 (100k),
−0.0318 (175k), −0.0230 (250k). AL reaches control's 250k level (~0.119)
already around 130k updates. The 25k screening advantage therefore shrinks with
budget but does not vanish; the curve is consistent with a faster-convergence
effect rather than a constant offset.

## 3. Physics at 250k (frozen 64-molecule monitor cohort, MMFF94)

Raw generation coordinates, atomic RMS force (kcal/mol/Å), conformer median then
equal molecule mean; strain after MMFF94 relaxation.

| method | force (mean ± seed SD) | per-seed (42/43/44) | strain |
|---|---|---|---|
| aligned_linear | 25.356 ± 0.282 | 25.682 / 25.190 / 25.195 | 7.702 |
| control | 24.450 ± 0.210 | 24.208 / 24.575 / 24.567 | 7.561 |
| alternative_a | 24.599 ± 0.102 | 24.569 / 24.712 / 24.514 | 7.697 |
| precision | 30.576 ± 0.997 | 31.717 / 30.141 / 29.869 | 11.174 |

Reference force mean: 24.763. Paired deltas (3-seed per-molecule mean, 2,000 draws):

| contrast | Δforce [95% CI] | fraction lower |
|---|---|---|
| AL − control | +0.906 [+0.624, +1.193] | 0.20 |
| AL − A | +0.757 [+0.402, +1.115] | 0.30 |
| AL − precision | −5.220 [−7.739, −3.363] | 0.83 |
| A − control | +0.149 [−0.216, +0.507] | 0.42 |
| control − precision | −6.125 [−8.569, −4.272] | 0.91 |

At 25k the same audit gave A − control = −6.815 [−9.122, −4.718]; by 250k that
difference is gone (+0.149, CI crosses zero). AL is geometrically/chirally
better than both, but pays a small raw-force penalty (~+0.9 kcal/mol/Å) while
relaxed strain is essentially unchanged (7.70 vs 7.56). In other words, the
global-shape gain does not come with more locally relaxed raw conformers.

## Caveats

- QM9 only; no GEOM-drugs or cross-dataset evidence.
- Physics is MMFF94-only (no GFN2-xTB executable on SHHPC; nothing installed).
- The physics cohort is 64 of the 512 monitor molecules; the metric is not a
  thermodynamic or basin-population statement.
- Endpoint metrics have 3 training seeds; the learning curve is seed 42 only.
- The 25k→250k shrink means the effect size should not be extrapolated beyond
  this budget or dataset.

## Reproduction

```bash
# endpoint comparison (writes analysis/full-budget-comparison.json)
python runs/aligned-linear-250k/scripts/full_budget_compare.py

# seed-42 learning curve (9 GPU jobs; then assemble)
sbatch --array=0-8 runs/aligned-linear-250k/scripts/learning_curve.sbatch
python runs/aligned-linear-250k/scripts/assemble_learning_curve.py

# 250k physics (requires shhpc-physics-budget-audit@69c38f8 for --budget)
python -m stifflow_benchmark.stiffness_diagnostics physics \
  <monitor generated manifest> --evaluation <evaluation.json> \
  --val-labels cache/qm9-v3-packed-strict/val/manifest.json --budget 250000 \
  --output <external/physics-250k/method-seed>
```

## Not run in this round

- `linear` (unaligned coupling) at 250k — the direct alignment ablation.
- GEOM-drugs-scale validation.
- xTB cross-potential check.
- Any new path or schedule variant; those belong to the method-development side.
