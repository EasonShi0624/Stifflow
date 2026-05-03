# GEOM Full-Database Hybrid Phase 1 CPU Single-Job Commands

These commands submit the six random-split seed0 Phase 1 hybrid pilot jobs as
separate CPU-only SLURM jobs. Do not run them without explicit approval.

```bash
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda010_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda025_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda050_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_harmonic_stiffness_residual_v1_lambda010_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_harmonic_stiffness_residual_v1_lambda025_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_harmonic_stiffness_residual_v1_lambda050_cpu.sbatch
```
