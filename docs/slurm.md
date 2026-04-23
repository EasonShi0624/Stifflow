# SLURM Guidance

## Purpose

This file records the SLURM submission policy for this repository.

A user-provided example script includes the following pattern:

```bash
#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --account=torch_pr_226_chemistry
```

The important operational takeaway is that the script specifies an explicit SLURM account. Treat that as required repository practice. fileciteturn5file0

---

## Available accounts

### `torch_pr_872_general`
Prefer this for:
- short jobs;
- CPU jobs;
- debugging;
- tiny smoke tests;
- lower-resource GPU jobs;
- work where shorter queue time is more important than peak GPU capability.

Typical GPU expectation:
- standard / more available GPUs such as `l40s`.

### `torch_pr_226_chemistry`
Prefer this for:
- chemistry-heavy workloads;
- real training runs;
- more demanding evaluation or sampling jobs;
- jobs with a real need for stronger GPU resources such as `A100`.

Tradeoff:
- queue time may be longer.

---

## Default account-selection rule

Use:
- `torch_pr_872_general` for short, low-risk, debugging, CPU, and modest-resource jobs;
- `torch_pr_226_chemistry` only when the workload clearly benefits from stronger GPU resources or is a real chemistry-heavy training/evaluation job.

Do not choose the chemistry account by default for trivial work.

---

## Agent behavior rules

Agents may:
- draft SLURM scripts;
- revise requested resources;
- explain why one account is better than another.

Agents must ask before:
- running `sbatch`, `srun`, `qsub`, `scancel`, or similar commands;
- launching any nontrivial GPU job;
- switching to a more expensive account;
- increasing wall time, memory, or GPU count for a real run.

---

## Recommended ask-first wording

For `torch_pr_872_general`:
> This looks like a short or modest-resource job. I recommend `torch_pr_872_general` for faster turnaround. Do you want me to prepare or submit the SLURM script?

For `torch_pr_226_chemistry`:
> This looks like a real chemistry-heavy GPU run. I recommend `torch_pr_226_chemistry` because it may benefit from A100-class resources, but queue time may be longer. Do you want me to prepare or submit the SLURM script?

---

## Minimal template

```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=<n_cpu>
#SBATCH --gres=gpu:<n_gpu>
#SBATCH --mem=<memory>
#SBATCH --time=<hh:mm:ss>
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --account=<torch_pr_872_general|torch_pr_226_chemistry>

conda activate stifflow
python <entrypoint>.py --config <config_path>
```

## Prepared CPU pilot scripts

- `scripts/slurm_data_scale_check_10000_checkpoint_sampling.sbatch` prepares the bounded 10000-molecule GEOM-Drugs data-scale checkpoint/sampling pilot. It is CPU-only, uses `torch_pr_872_general`, and runs `python scripts/run_data_scale_check_10000_checkpoint_sampling.py`.
- `scripts/slurm_data_scale_check_10000_checkpoint_sampling_multiseed.sbatch` prepares the bounded 10000-molecule GEOM-Drugs data-scale checkpoint/sampling multi-seed repeat. It is CPU-only, uses `torch_pr_872_general`, and runs `python scripts/run_data_scale_check_10000_checkpoint_sampling_multiseed.py`.
- `scripts/slurm_data_scale_check_10000_checkpoint_sampling_convergence_multiseed.sbatch` prepares the bounded 10000-molecule GEOM-Drugs data-scale checkpoint/sampling convergence repeat. It is CPU-only, uses `torch_pr_872_general`, and runs `python scripts/run_data_scale_check_10000_checkpoint_sampling_convergence_multiseed.py`.
- `scripts/slurm_data_scale_check_10000_formal_training_multiseed.sbatch` prepares the first formal bounded 10000-molecule GEOM-Drugs training phase. It is CPU-only, uses `torch_pr_872_general`, and runs `python scripts/run_data_scale_check_10000_formal_training_multiseed.py`.
- `scripts/slurm_geom_full_database_formal_training_seed0.sbatch` prepares the first full-database GEOM-Drugs formal training path at seed `0`. It is CPU-only, uses `torch_pr_872_general`, and runs `python scripts/run_geom_full_database_formal_training_seed0.py`.
- `scripts/slurm_geom_full_database_formal_training_isotropic_seed1_gpu.sbatch`, `scripts/slurm_geom_full_database_formal_training_stiffness_seed1_gpu.sbatch`, `scripts/slurm_geom_full_database_formal_training_isotropic_seed2_gpu.sbatch`, and `scripts/slurm_geom_full_database_formal_training_stiffness_seed2_gpu.sbatch` prepare one GPU-enabled full-database formal training job per mode/seed combination. They use `torch_pr_872_general`, request one GPU, regenerate their own processed artifact plus random_seed0 split, and run `python scripts/run_geom_full_database_formal_training_single.py ... --device cuda --require-cuda`.
- `scripts/slurm_geom_full_database_scaffold_split.sbatch` prepares the cached full-database Bemis-Murcko scaffold split at seed `0`. It is CPU-only, uses `torch_pr_872_general`, reads the cached processed artifact, and runs `python scripts/prepare_geom_full_database_scaffold_split.py`.
- `scripts/slurm_geom_full_database_scaffold_formal_training_isotropic_seed0_gpu.sbatch`, `scripts/slurm_geom_full_database_scaffold_formal_training_stiffness_seed0_gpu.sbatch`, `scripts/slurm_geom_full_database_scaffold_formal_training_isotropic_seed1_gpu.sbatch`, `scripts/slurm_geom_full_database_scaffold_formal_training_stiffness_seed1_gpu.sbatch`, `scripts/slurm_geom_full_database_scaffold_formal_training_isotropic_seed2_gpu.sbatch`, and `scripts/slurm_geom_full_database_scaffold_formal_training_stiffness_seed2_gpu.sbatch` prepare one GPU-enabled full-database scaffold-split formal training job per mode/seed combination. They use `torch_pr_872_general`, request one GPU, reuse the cached processed artifact plus cached scaffold split, and run `python scripts/run_geom_full_database_scaffold_formal_training_single.py ... --device cuda --require-cuda`.
- `scripts/slurm_geom_full_database_attribution_formal_training_gpu.sbatch` prepares one cached-input full-database attribution formal training job for a single split/mode/seed. It uses `torch_pr_872_general`, requests one GPU, reuses `data/processed/geom_drugs_full_database/molecules.json` plus the cached random or scaffold split, and runs `python scripts/run_geom_full_database_attribution_formal_training_single.py ... --device cuda --require-cuda`. It requires `STIFFLOW_ATTRIBUTION_SPLIT_TYPE`, `STIFFLOW_ATTRIBUTION_MODE`, and `STIFFLOW_ATTRIBUTION_SEED` to be set at submission.

Do not submit these scripts without explicit approval. The single-seed script reads a bounded subset of real GEOM-Drugs raw pickle files, writes a bounded one-conformer-per-molecule processed dry-run artifact and split, and runs the same non-benchmark checkpoint/sampling pilot settings for isotropic and stiffness-structured modes. The multi-seed script reuses that bounded artifact and split, then repeats the same non-benchmark checkpoint/sampling pilot settings over the fixed seed set `0,1,2`. The convergence script reuses the same bounded artifact, split, seed set, model, optimizer, checkpoint/reload behavior, and one-step sampling settings, but extends only the training horizon to a strict 200-step hard cap with patience-based stopping. The formal 10000 training script keeps the same bounded 10000-molecule artifact, 9000/500/500 split, seed set, model, optimizer, checkpoint/reload pattern, and one-step sampling settings, then extends the horizon to 2000 CPU steps with validation-selected reusable checkpoints. The full-database formal seed0 script preserves that formal design at seed `0`, scans the full available GEOM-Drugs raw tree read-only, writes one-conformer-per-molecule processed/split artifacts, and runs isotropic plus stiffness-structured modes. The GPU full-database scripts preserve the same design for seeds `1` and `2`, split mode/seed combinations into separate jobs, and deliberately avoid a cached shared preprocessing artifact by regenerating run-specific processed/split files. The scaffold split script reuses the cached processed artifact and writes only a cached scaffold split; the scaffold training scripts then reuse both cached inputs for seeds `0`, `1`, and `2`. The attribution wrapper now reuses the cached processed artifact and cached random/scaffold splits for A/B/D/E. All prepared scripts remain explicitly non-benchmark.

For the prepared full-database GPU jobs, prefer `torch_pr_872_general` for faster turnaround unless there is an explicit A100 need. GPU is more worthwhile on cached-input scaffold and attribution paths than it was for per-run preprocessing-heavy random jobs because the processed artifact and split can be reused, so more wall time goes to the model-side tensor work. The current formal model is still minimal and part of the pipeline remains CPU/memory-bound, so `torch_pr_872_general` remains the default recommendation. `torch_pr_226_chemistry` may provide stronger GPUs such as A100, but queue time may be longer and should be reserved for workloads with a clear chemistry-heavy or A100-specific need.

---

## Notes

- Prefer tiny smoke tests before real cluster jobs.
- Record the chosen account in manifests for reproducibility.
- Avoid submitting jobs from an agent without explicit approval.
