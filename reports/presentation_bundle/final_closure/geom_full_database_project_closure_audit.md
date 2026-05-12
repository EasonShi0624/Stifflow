# GEOM Full-Database Project Closure Audit

Generated on 2026-05-11. This is reporting-only and non-benchmark.

## Final Interpretation

Stiffness-modulated harmonic source priors create a tunable precision-recall Pareto frontier rather than one universally best lambda.

- `lambda=5.0`: validation/RMSD and AMR-P precision-favorable; not ensemble-optimal.
- `lambda=0.025`: best tested hybrid recall/balanced COV/AMR setting.
- `harmonic_prior`: AMR-R recall anchor.
- No tested hybrid lambda dominates harmonic_prior on both precision and recall COV/AMR metrics.

## Project Status

| Area | Status | Evidence / Output Path | Remaining Action |
|---|---|---|---|
| Governing docs | refreshed for closure | `README.md`, `TASKS.md`, `docs/current_status.md`, `docs/reproduce.md` | Keep future edits aligned with the Pareto-frontier interpretation. |
| Optimized all-method matrix | complete | `experiments/results/geom_full_database_final_optimized_all_methods_summary.json` | Cite as validation/RMSD evidence only. |
| Complete val/test all-method COV/AMR | complete | `experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json` | Use as ensemble-metric evidence, not physical benchmark evidence. |
| Complete val/test hybrid-lambda COV/AMR | complete | `experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json` | Use as the final lambda-selection interpretation. |
| Final reporting bundle | complete | `reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md`, `reports/tables/geom_full_database_final_claims_and_limitations_summary.md`, `reports/tables/geom_full_database_final_results_discussion_narrative.md` | Present only the qualified Pareto-frontier claim. |
| Presentation figures | complete | `reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png`, `reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png`, `reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png`, `reports/figures/geom_full_database_final_claim_summary_table.png` | Use the clean presentation bundle copies for slides. |
| Physical benchmark validation | not run | `reports/tables/geom_full_database_physical_benchmark_plan.md` | Skip for closure unless separately approved. |

## Key Metrics

| Entry | Lambda | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean | Role |
|---|---:|---:|---:|---:|---:|---|
| harmonic_prior | 0.0 | 0.184841 | 1.000000 | 1.489857 | 0.096041 | AMR-R recall anchor |
| hybrid_lambda0025 | 0.025 | 0.184890 | 1.000000 | 1.490074 | 0.095111 | best tested hybrid recall/balanced |
| hybrid_lambda500 | 5.0 | 0.184111 | 1.000000 | 1.498808 | 0.059748 | precision/RMSD-favorable |

## Must Fix Before Presentation

- Use the clean final-closure presentation bundle.
- Avoid citing historical pending/missing summaries as final evidence.
- Describe `lambda=5.0` as RMSD/precision-favorable, not ensemble-optimal.
- Describe `lambda=0.025` as best tested hybrid recall/balanced, not dominance over harmonic.
- State that harmonic_prior remains the AMR-R recall anchor.

## Nice To Have

- Add a final report index separating final and historical outputs.
- Add slide notes for the four final figures.
- Add a claim-to-artifact appendix table.

## Should Skip

- Additional lambda sweeps.
- Training reruns.
- Train-partition COV/AMR.
- Residual variants.
- PoseBusters/MMFF/UFF/xTB jobs.
- Scheduler submissions without explicit approval.

## Safety

- `benchmark_result`: `False`
- `training_executed_by_summary`: `False`
- `sampling_executed_by_summary`: `False`
- `raw_data_modified`: `False`
- `environment_modified`: `False`
- `scheduler_submitted`: `False`
