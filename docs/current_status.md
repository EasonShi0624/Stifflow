# Current Repository Status

_Last updated: 2026-05-11 project-closure audit._

## Summary

The repository is now in a closure/reporting stage for the current study. The key full-database optimized training summaries, complete val/test COV/AMR evaluations, and final reporting bundle are present.

The final scientific interpretation is:

> Stiffness-modulated harmonic source priors create a tunable precision-recall frontier. `lambda=5.0` is validation/RMSD and AMR-P precision-favorable; `lambda=0.025` is the best tested hybrid recall/balanced setting; harmonic_prior remains the AMR-R recall anchor. No tested hybrid lambda dominates harmonic on both precision and recall.

All final summaries remain explicitly non-benchmark. Physical benchmark validation has not been run.

## Final Outputs

Primary JSON summaries:

- `experiments/results/geom_full_database_final_optimized_all_methods_summary.json`
- `experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json`
- `experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json`
- `experiments/results/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json`
- `experiments/results/geom_full_database_final_claims_and_limitations_summary.json`
- `experiments/results/geom_full_database_project_closure_audit.json`

Primary markdown reports:

- `reports/tables/geom_full_database_final_optimized_all_methods_summary.md`
- `reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md`
- `reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md`
- `reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md`
- `reports/tables/geom_full_database_final_claims_and_limitations_summary.md`
- `reports/tables/geom_full_database_final_results_discussion_narrative.md`
- `reports/tables/geom_full_database_project_closure_audit.md`

Presentation figures:

- `reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png`
- `reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png`
- `reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png`
- `reports/figures/geom_full_database_final_claim_summary_table.png`

Clean presentation bundle:

- `reports/presentation_bundle/final_closure/`

## Completed Work

- Full-database optimized AdamW lr `1e-3` matrix for:
  - `isotropic`
  - `stiffness_structured`
  - `torsion_only`
  - `harmonic_prior`
  - `hybrid_harmonic_stiffness_v1 lambda=5.0`
- Random and scaffold splits, seeds `0,1,2`.
- Final optimized validation/RMSD summary across all five methods.
- Complete val/test all-method COV/AMR evaluation.
- Complete val/test hybrid-lambda COV/AMR evaluation for harmonic and hybrid lambdas `0.025`, `0.1`, `1.0`, `3.0`, and `5.0`.
- Final interpretation summary, claims/limitations summary, concise narrative, and presentation figures.

## Important Claim Boundaries

Supported:

- `hybrid_harmonic_stiffness_v1 lambda=5.0` improves optimized validation loss, single-sample RMSD, and AMR-P precision relative to harmonic_prior.
- `hybrid_harmonic_stiffness_v1 lambda=0.025` is the best tested hybrid for recall/balanced COV/AMR.
- The hybrid lambda sweep forms a precision-recall Pareto frontier.

Not supported:

- A single hybrid lambda is universally best.
- `lambda=5.0` is ensemble-optimal.
- Any tested hybrid lambda dominates harmonic_prior on both precision and recall.
- Physical benchmark validation supports full superiority.

## Historical / Superseded Outputs

Older pilot, lambda-sweep, and lambda500 summaries are still useful as audit trail, but they should not be used as final presentation evidence unless their historical scope is stated. In particular, validation/RMSD-era summaries that call `lambda=5.0` the best source design are superseded for ensemble-quality interpretation by the complete COV/AMR lambda summary.

Older reports with pending or missing state should not be cited as current final status.

## Remaining Work

Must do only if preparing a final deck or paper draft:

- use the clean presentation bundle;
- cite the final summaries above;
- keep the precision-recall limitation explicit.

Optional future work requiring separate approval:

- PoseBusters-style physical validation;
- MMFF/UFF energy or force sweeps;
- xTB/GFN2-xTB subset checks;
- RDKit GEOM-scale baseline expansion;
- any additional training or scheduler submission.

## Safety State

- No raw data modification is required for the closure state.
- No environment modification is required.
- No scheduler jobs should be submitted without new explicit approval.
- The complete COV/AMR runs are val/test only; train partition COV/AMR is intentionally excluded from the default complete evaluation.
