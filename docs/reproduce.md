# Reproducibility And Final-State Verification

This document records how to verify and regenerate the current final reporting state without launching training, evaluation chunks, scheduler jobs, or environment changes.

## Final Scope

The closure-state results cover:

- full-database random and scaffold splits;
- seeds `0,1,2`;
- optimized AdamW lr `1e-3` training summaries;
- complete val/test COV/AMR, excluding train partition;
- `K=20` generated conformers per molecule;
- RMSD threshold `delta=0.75`;
- non-benchmark reporting only.

Final interpretation:

- `hybrid_harmonic_stiffness_v1 lambda=5.0` is validation/RMSD and AMR-P precision-favorable.
- `hybrid_harmonic_stiffness_v1 lambda=0.025` is the best tested hybrid recall/balanced setting.
- `harmonic_prior` remains the AMR-R recall anchor.
- No tested hybrid lambda dominates harmonic on both precision and recall.

## Key Final Artifacts

JSON summaries:

```text
experiments/results/geom_full_database_final_optimized_all_methods_summary.json
experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json
experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json
experiments/results/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json
experiments/results/geom_full_database_final_claims_and_limitations_summary.json
experiments/results/geom_full_database_project_closure_audit.json
```

Markdown reports:

```text
reports/tables/geom_full_database_final_optimized_all_methods_summary.md
reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md
reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md
reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md
reports/tables/geom_full_database_final_claims_and_limitations_summary.md
reports/tables/geom_full_database_final_results_discussion_narrative.md
reports/tables/geom_full_database_project_closure_audit.md
```

Figures:

```text
reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png
reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png
reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png
reports/figures/geom_full_database_final_claim_summary_table.png
```

Presentation bundle:

```text
reports/presentation_bundle/final_closure/
```

## Verify Final Summary Consistency

```bash
python - <<'PY'
import json

paths = [
    "experiments/results/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json",
    "experiments/results/geom_full_database_final_claims_and_limitations_summary.json",
    "experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json",
    "experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json",
    "experiments/results/geom_full_database_final_optimized_all_methods_summary.json",
    "experiments/results/geom_full_database_project_closure_audit.json",
]

for path in paths:
    data = json.load(open(path))
    print(path)
    print("  benchmark_result:", data.get("benchmark_result"))
    print("  raw_data_modified:", data.get("raw_data_modified"))
    print("  environment_modified:", data.get("environment_modified"))
    coverage = data.get("coverage") or {}
    if coverage:
        print("  coverage:", coverage)
PY
```

Expected high-level checks:

- final COV/AMR all-method coverage is complete;
- hybrid-lambda COV/AMR coverage is complete;
- final reports are marked `benchmark_result: false`;
- no final reporting artifact claims raw-data or environment modification.

## Verify Figures Exist

```bash
for f in \
  reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png \
  reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png \
  reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png \
  reports/figures/geom_full_database_final_claim_summary_table.png
do
  test -s "$f" && echo "ok $f"
done
```

## Regenerate Final Reporting Bundle

This command refreshes reporting-only final summaries and figures from existing results. It does not run training, submit jobs, mutate raw data, or modify the environment.

```bash
python scripts/summarize_geom_full_database_final_reporting_bundle.py
```

Expected refreshed outputs:

```text
experiments/results/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json
reports/tables/geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md
experiments/results/geom_full_database_final_claims_and_limitations_summary.json
reports/tables/geom_full_database_final_claims_and_limitations_summary.md
reports/tables/geom_full_database_final_results_discussion_narrative.md
reports/figures/geom_full_database_hybrid_lambda_cov_amr_tradeoff.png
reports/figures/geom_full_database_hybrid_lambda_precision_recall_pareto.png
reports/figures/geom_full_database_final_method_comparison_rmsd_cov_amr.png
reports/figures/geom_full_database_final_claim_summary_table.png
```

## Stale-Claim Scan

```bash
rg -n -i \
  "universally best|lambda=5.*ensemble-optimal|stiffness beats harmonic|physical benchmark supports|dominates harmonic" \
  README.md TASKS.md docs reports/tables --glob '*.md'
```

Allowed hits should be limited to explicit limitations, not-supported claims, or clearly historical/superseded reports.

## Do Not Run Without New Approval

Do not run:

- training jobs;
- scheduler submissions;
- COV/AMR evaluation chunks;
- PoseBusters jobs;
- MMFF/UFF sweeps;
- xTB/GFN2-xTB checks;
- raw-data preprocessing or mutation;
- environment installation or modification.

Any future physical benchmark or additional evaluation phase requires a new explicit approval.
