# Reproducibility Plan

## Goal

The immediate goal is not yet full-benchmark reproduction.

The immediate reproducibility goal is:

> given a config, a code snapshot, and a data snapshot, reproduce the same smoke-test run and regenerate the same machine-readable outputs.

---

## Minimum reproducibility contract

Every meaningful run should record:
- git commit hash if available
- config path
- config contents or a frozen config snapshot
- random seed
- dataset subset identifier
- split type
- backbone name
- corruption mode
- output directory
- checkpoint path if training occurs
- metrics JSON path
- execution mode: local smoke test vs scheduled cluster job
- SLURM account if a scheduled job is used

---

## Output locations

The intended output structure is:

```text
experiments/
├── manifests/
├── logs/
├── checkpoints/
└── results/
```

A typical run should generate:
- `experiments/manifests/<run_name>.json`
- `experiments/logs/<run_name>.log`
- `experiments/results/<run_name>.json`
- optionally `experiments/checkpoints/<run_name>.pt`

Current smoke command:

```bash
python scripts/train.py --config configs/train/tiny_smoke.yaml --dry-run
```

This validates config/data/split/corruption/evaluation wiring and writes manifest/results/log files. It does **not** run neural model training. Smoke entry points re-exec through `/scratch/ys6132/programs/miniforge3/envs/stifflow/bin/python` when launched from another interpreter.

Current paired smoke pipeline:

```bash
python scripts/run_smoke_pipeline.py
```

This runs `configs/train/tiny_smoke.yaml` and `configs/train/tiny_smoke_stiffness.yaml` locally on the synthetic fixture, then writes aggregate files:
- `experiments/manifests/smoke_pair_synthetic_seed0.json`
- `experiments/results/smoke_pair_synthetic_seed0.json`
- `experiments/logs/smoke_pair_synthetic_seed0.log`

All outputs are labeled `smoke_only_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, and `slurm_account: null`.

Current RDKit smoke baseline:

```bash
python scripts/run_rdkit_smoke.py
```

This runs RDKit ETKDGv3 embedding plus MMFF optimization on the same synthetic fixture, if RDKit is available in the active environment. It writes:
- `experiments/manifests/smoke_random_rdkit_etkdg_mmff_seed0.json`
- `experiments/results/smoke_random_rdkit_etkdg_mmff_seed0.json`
- `experiments/logs/smoke_random_rdkit_etkdg_mmff_seed0.log`

These outputs are also labeled `smoke_only_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, and `slurm_account: null`. The manifest and result JSON record `python_executable`, `python_version`, and `rdkit_version`. If RDKit is not importable, the script writes a clear `execution_status: import_failure` smoke report and does not fabricate RDKit metrics. If a successful RDKit smoke result already exists, an import failure will not overwrite it; a `*.preflight_failure.log` diagnostic is written instead. Per-molecule failures are categorized as `fixture`, `etkdg`, `mmff`, `alignment`, or `runtime`.

---

## Run naming

Use informative names that encode at least:
- split type
- corruption mode
- seed
- dataset size or subset tag

Example style:
- `smoke_random_isotropic_seed0`
- `smoke_random_stiffness_seed0`
- `core_scaffold_torsion_only_seed1`

---

## What counts as reproducible

A run counts as reproducible if another person can:
1. activate the same environment,
2. locate the same data subset,
3. use the same config,
4. run the same entry point,
5. recover the same key outputs and approximately the same metrics.

---

## Early-stage policy

Because this repository is still being bootstrapped, early scripts may be stubs.
That is acceptable.

However, every stub should make the future reproducibility path clearer, not blurrier.

Good stub behavior:
- validate config loading;
- print or write a manifest;
- fail clearly when a required file is missing.

Bad stub behavior:
- silently skip missing data;
- write fake metrics;
- claim to have executed training when nothing ran.

---

## Cluster-job policy for reproducibility

For any scheduled run, reproducibility also requires:
- the SLURM script or launch command;
- the chosen account;
- requested resources (CPU, GPU, memory, wall time);
- output and error log paths.

Two accounts are available:
- `torch_pr_872_general` for shorter, lighter, debugging, CPU, or modest-GPU jobs;
- `torch_pr_226_chemistry` for chemistry-heavy or higher-end GPU jobs, including cases that may need A100-class resources.

Account choice should be recorded in the manifest when a job is submitted.

---

## Prompt behavior before expensive runs

Before launching a nontrivial run, a coding agent should ask for approval and include:
- what will be run;
- whether it is a smoke test or real experiment;
- whether it is CPU or GPU;
- which account it recommends and why;
- whether queue time may be longer because of the account / GPU class.

Example:

> This is a real GPU training run, not a tiny smoke test. I recommend `torch_pr_226_chemistry` because it may benefit from A100-class resources, but queue time may be longer. Do you want me to prepare or submit the SLURM job?

---

## Later additions

This file should later expand to include:
- exact training entry points;
- canonical SLURM launch examples;
- expected runtime classes for smoke tests vs full runs;
- evaluation aggregation conventions;
- seed-averaging policy.
