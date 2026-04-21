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

Do not submit these scripts without explicit approval. The single-seed script reads a bounded subset of real GEOM-Drugs raw pickle files, writes a bounded one-conformer-per-molecule processed dry-run artifact and split, and runs the same non-benchmark checkpoint/sampling pilot settings for isotropic and stiffness-structured modes. The multi-seed script reuses that bounded artifact and split, then repeats the same non-benchmark checkpoint/sampling pilot settings over the fixed seed set `0,1,2`.

---

## Notes

- Prefer tiny smoke tests before real cluster jobs.
- Record the chosen account in manifests for reproducibility.
- Avoid submitting jobs from an agent without explicit approval.
