# Hybrid Harmonic-Stiffness Lambda 5.0 Multiseed Commands

These are CPU-only, single-job SLURM wrappers for the fixed
`hybrid_harmonic_stiffness_v1` lambda `5.0` multiseed follow-up. They use the
cached full-database processed artifact and cached random/scaffold splits. They
do not run residual variants, old baselines, additional lambda search,
physical benchmark jobs, xTB, or scaffold seed0 lambda `5.0`.

Use account `torch_pr_872_general`; do not submit without explicit approval.

```bash
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda500_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda500_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda500_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_seed2_cpu.sbatch
```
