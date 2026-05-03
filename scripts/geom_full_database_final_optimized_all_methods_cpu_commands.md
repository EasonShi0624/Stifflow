# Full-Database Final Optimized All-Methods CPU Commands

Preparation only. Do not submit without explicit approval.

These 18 single-job CPU submissions cover the missing optimized AdamW `lr=1e-3` runs for isotropic, stiffness-structured, and torsion-only on random and scaffold splits over seeds `0,1,2`. Optimized harmonic-prior and hybrid lambda500 results already exist and are reused.

All jobs use cached full-database processed artifacts and cached random/scaffold splits, CPU execution, `torch_pr_872_general`, `AdamW lr=1e-3`, no scheduler, the existing formal model/checkpoint/reload/validation-selection/one-step sampling design, and the existing 2000-step hard cap.

```bash
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_isotropic_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_isotropic_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_isotropic_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_stiffness_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_stiffness_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_stiffness_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_torsion_only_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_torsion_only_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_random_torsion_only_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_isotropic_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_isotropic_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_isotropic_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_stiffness_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_stiffness_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_stiffness_seed2_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_torsion_only_seed0_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_torsion_only_seed1_cpu.sbatch
sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_final_optimized_scaffold_torsion_only_seed2_cpu.sbatch
```

Recommended account: `torch_pr_872_general`, because these are CPU-only single-job follow-ups and previous GPU acceleration was limited. Queue time is usually shorter than the chemistry account. These are real training jobs, not smoke tests, so submit only after explicit approval.
