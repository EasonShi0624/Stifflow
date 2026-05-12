# TASKS

This file tracks the current closure-stage work queue. Historical smoke, pilot, lambda-search, and scheduler-preparation tasks remain represented by their scripts/results, but the active project state is now final reporting and claim hygiene.

## Current Project State

Completed:

- Full-database optimized AdamW lr `1e-3` all-method training summaries are complete for random and scaffold splits over seeds `0,1,2`.
- Optimized tracked validation/RMSD ordering is established in `experiments/results/geom_full_database_final_optimized_all_methods_summary.json`.
- Complete val/test all-method COV/AMR evaluation is complete in `experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json`.
- Complete val/test hybrid-lambda COV/AMR evaluation is complete in `experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json`.
- Final reporting bundle is generated:
  - `experiments/results/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json`
  - `reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md`
  - `experiments/results/geom_full_database_final_claims_and_limitations_summary.json`
  - `reports/tables/geom_full_database_final_claims_and_limitations_summary.md`
  - `reports/tables/geom_full_database_final_results_discussion_narrative.md`
  - final figures under `reports/figures/`
- Project closure audit is generated:
  - `experiments/results/geom_full_database_project_closure_audit.json`
  - `reports/tables/geom_full_database_project_closure_audit.md`

Current final interpretation:

- `hybrid_harmonic_stiffness_v1 lambda=5.0` is validation/RMSD and AMR-P precision-favorable.
- `hybrid_harmonic_stiffness_v1 lambda=0.025` is the best tested hybrid recall/balanced COV/AMR setting.
- `harmonic_prior` remains the AMR-R recall anchor.
- No tested hybrid lambda dominates harmonic on both precision and recall.
- The final result is a precision-recall Pareto frontier, not one universally best lambda.

## Must Fix Before Presentation

- Keep `README.md`, `docs/current_status.md`, and `docs/reproduce.md` aligned with the final Pareto-frontier interpretation.
- Use the clean presentation bundle under `reports/presentation_bundle/final_closure/` rather than intermediate historical reports.
- When citing `lambda=5.0`, state that it is RMSD/precision-favorable, not ensemble-optimal.
- When citing `lambda=0.025`, state that it is the best tested hybrid recall/balanced setting, not a dominance result over harmonic.
- Keep all final claims labeled non-benchmark unless separate physical benchmark evaluation is approved and completed.

## Nice To Have

- Add a lightweight report index that separates final reports from historical/intermediate reports.
- Add one page of slide notes using the four final figures.
- Add a compact appendix table mapping every final claim to its source JSON/report path.

## Should Skip For Closure

- More lambda sweeps.
- More training reruns.
- Train-partition COV/AMR.
- Residual hybrid variants.
- PoseBusters/MMFF/UFF/xTB jobs.
- RDKit GEOM-scale baseline expansion.
- Any scheduler submission without a new explicit approval.

## Historical Notes

Older summaries that say `lambda=5.0` is the strongest source design should be read as validation/RMSD-era conclusions only. They are superseded for ensemble interpretation by:

- `reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md`
- `reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md`
- `reports/tables/geom_full_database_final_claims_and_limitations_summary.md`

Older summaries with pending/missing state should not be used for presentation unless explicitly labeled historical.
