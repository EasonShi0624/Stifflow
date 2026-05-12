# Hybrid Lambda COV/AMR Selection Commands

Do not submit without explicit approval. This is one standalone CPU job, not a SLURM array.

```bash
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_lambda_cov_amr_selection_cpu.sbatch
```

The job evaluates the existing scaffold seed0 lambda checkpoints on the bounded scaffold subset and writes:

- `experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_summary.json`
- `reports/tables/geom_full_database_hybrid_lambda_cov_amr_selection_summary.md`
- `experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_per_molecule.jsonl`
