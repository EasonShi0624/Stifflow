# Final Hybrid Lambda COV/AMR Interpretation

This report integrates the complete val/test COV/AMR lambda evaluation. It is non-benchmark reporting.

## Key Findings

- Primary conclusion: stiffness-modulated harmonic priors create a tunable precision-recall frontier rather than one universally best lambda.
- lambda=5.0 role: RMSD/validation-loss and AMR-P precision-favorable; not ensemble-optimal.
- lambda=0.025 role: best hybrid recall/balanced COV/AMR lambda by COV-R, AMR-R, and rank aggregation.
- Harmonic role: best AMR-R recall anchor within the harmonic-vs-hybrid lambda comparison.
- Dominance statement: No tested hybrid lambda beats harmonic_prior on both precision and recall COV/AMR metrics.

## Lambda Metrics

| Entry | Lambda | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| harmonic_prior | 0.0 | 0.184841 | 1.000000 | 1.489857 | 0.096041 | lambda=0 harmonic reference |
| hybrid_lambda0025 | 0.025 | 0.184890 | 1.000000 | 1.490074 | 0.095111 | balanced / recall-favorable |
| hybrid_lambda010 | 0.1 | 0.184808 | 1.000000 | 1.490593 | 0.092630 | Pareto frontier point |
| hybrid_lambda100 | 1.0 | 0.184411 | 1.000000 | 1.497081 | 0.065375 | Pareto frontier point |
| hybrid_lambda300 | 3.0 | 0.184131 | 1.000000 | 1.498555 | 0.060516 | Pareto frontier point |
| hybrid_lambda500 | 5.0 | 0.184111 | 1.000000 | 1.498808 | 0.059748 | precision / RMSD-favorable |

## Selection

- Precision-optimal lambda: `{'amr_p': 0.05974827347882525, 'amr_r': 1.4988076131247985, 'cov_p': 1.0, 'cov_r': 0.1841108275887776, 'criterion': 'lowest AMR-P, AMR-R tie-breaker', 'entry_label': 'hybrid_lambda500', 'lambda_value': 5.0}`
- Recall-optimal lambda: `{'amr_p': 0.09511050764880791, 'amr_r': 1.4900735597348704, 'cov_p': 1.0, 'cov_r': 0.18488987622845177, 'criterion': 'highest COV-R, lowest AMR-R tie-breaker', 'entry_label': 'hybrid_lambda0025', 'lambda_value': 0.025}`
- Balanced lambda: `{'criterion': 'rank aggregation over AMR-P, AMR-R, 1-COV-R, and 1-COV-P', 'entry_label': 'hybrid_lambda0025', 'lambda_value': 0.025, 'rank_sum': 8}`
- Pareto frontier: `[{'amr_p': 0.09511050764880791, 'amr_r': 1.4900735597348704, 'cov_p': 1.0, 'cov_r': 0.18488987622845177, 'entry_label': 'hybrid_lambda0025', 'lambda_value': 0.025}, {'amr_p': 0.0926303439584627, 'amr_r': 1.490593208477421, 'cov_p': 1.0, 'cov_r': 0.18480763844865333, 'entry_label': 'hybrid_lambda010', 'lambda_value': 0.1}, {'amr_p': 0.06537465050738851, 'amr_r': 1.4970812040471937, 'cov_p': 1.0, 'cov_r': 0.1844113046004111, 'entry_label': 'hybrid_lambda100', 'lambda_value': 1.0}, {'amr_p': 0.060515990426266465, 'amr_r': 1.4985545681790957, 'cov_p': 1.0, 'cov_r': 0.18413138348653857, 'entry_label': 'hybrid_lambda300', 'lambda_value': 3.0}, {'amr_p': 0.05974827347882525, 'amr_r': 1.4988076131247985, 'cov_p': 1.0, 'cov_r': 0.1841108275887776, 'entry_label': 'hybrid_lambda500', 'lambda_value': 5.0}]`

## Figures

- lambda_tradeoff: `reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png`
- precision_recall_pareto: `reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png`
- method_comparison: `reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png`
- claim_summary_table: `reports/figures/geom_full_database_final_claim_summary_table.png`

## Safety

- `benchmark_result`: `False`
- `training_executed_by_summary`: `False`
- `raw_data_modified`: `False`
- `environment_modified`: `False`
