# Source-Only Stiffness-Structured Flow Matching for 3D Molecular Conformer Generation

This repository contains a fixed-topology molecular conformer-generation study on GEOM-Drugs. The implementation is now beyond the bootstrap stage: full-database optimized training summaries and complete val/test COV/AMR evaluations have been generated for the current final reporting bundle.

## Current Conclusion

The final interpretation is a precision-recall Pareto frontier, not a single universally best hybrid lambda.

- `hybrid_harmonic_stiffness_v1` with `lambda=5.0` is strongest on optimized validation loss, single-sample RMSD, and AMR-P precision.
- `hybrid_harmonic_stiffness_v1` with `lambda=0.025` is the best tested hybrid for recall/balanced COV/AMR.
- `harmonic_prior` remains the AMR-R recall anchor.
- No tested hybrid lambda dominates `harmonic_prior` on both precision and recall ensemble metrics.
- All final outputs are still labeled non-benchmark reporting; physical benchmark validation has not been run.

The clean claim is:

> Stiffness-modulated harmonic source priors improve precision-oriented metrics and create a tunable precision-recall frontier, but the optimized hybrid does not uniformly dominate harmonic_prior on ensemble COV/AMR metrics.

## Methods In The Final Matrix

The optimized all-method matrix uses:

- `isotropic`
- `stiffness_structured`
- `torsion_only`
- `harmonic_prior`
- `hybrid_harmonic_stiffness_v1 lambda=5.0`

The optimizer selected by the pilot phase is AdamW with learning rate `1e-3`. Final optimized training summaries cover random and scaffold splits for seeds `0,1,2`.

The final hybrid-lambda COV/AMR comparison uses:

- `harmonic_prior` as the lambda-0 reference
- `hybrid_harmonic_stiffness_v1 lambda=0.025`
- `hybrid_harmonic_stiffness_v1 lambda=0.1`
- `hybrid_harmonic_stiffness_v1 lambda=1.0`
- `hybrid_harmonic_stiffness_v1 lambda=3.0`
- `hybrid_harmonic_stiffness_v1 lambda=5.0`

The complete COV/AMR evaluations are val/test only, not train+val+test.

## Evaluation Scope

Reported final ensemble metrics:

- COV-R
- COV-P
- AMR-R
- AMR-P

The full val/test COV/AMR evaluations use:

- random and scaffold splits
- val and test partitions only
- seeds `0,1,2`
- `K=20` generated conformers per molecule
- RMSD threshold `delta=0.75`
- cached processed artifacts, splits, checkpoints, and result JSONs

Train partition COV/AMR is intentionally excluded from the default complete run because it is much larger and was not needed for the final closure question.

## Reproducibility

Use [docs/reproduce.md](docs/reproduce.md) for final-state verification and reporting regeneration commands. Major run artifacts live under:

- `experiments/results/`
- `experiments/manifests/`
- `experiments/checkpoints/`
- `reports/tables/`
- `reports/figures/`

## Future Directions

A natural next step is to move beyond stiffness-aware source design and explore **stiffness-aware flow-matching paths**. In the current project, stiffness information is used to shape the source distribution, while the flow-matching interpolation path remains fixed. Future work could incorporate local chemical stiffness directly into the path construction, so that rigid regions such as rings, amides, and conjugated fragments move more conservatively, while flexible torsional regions are allowed greater displacement. This may help balance local geometric stability with conformational diversity and further clarify how stiffness information should enter molecular flow matching.

Wish to have a chance to develop it~
