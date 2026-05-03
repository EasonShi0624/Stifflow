# Scaffold Hybrid Harmonic-Stiffness Lambda Sweep Commands

These are CPU-only, single-job SLURM wrappers for the scaffold-split seed0
`hybrid_harmonic_stiffness_v1` lambda sweep. They use the cached full-database
processed artifact and cached scaffold split. They do not run residual variants,
old baselines, scaffold multiseed, physical benchmark jobs, or xTB.

Use account `torch_pr_872_general`; do not submit without explicit approval.

```bash
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda025_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda050_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda075_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda100_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda150_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda200_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda300_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_cpu.sbatch
```
