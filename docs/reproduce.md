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

Current smoke comparison report:

```bash
python scripts/compare_smoke_results.py
```

This reads the isotropic smoke result, stiffness-structured smoke result, and RDKit smoke baseline result, validates that all inputs are labeled smoke-only, and writes:
- `experiments/results/smoke_comparison_summary.json`
- `reports/tables/smoke_comparison_summary.md`

The summary and markdown report are labeled as smoke-only synthetic-fixture diagnostics, not benchmark results.

Current GEOM-Drugs raw-data preflight:

```bash
python scripts/prepare_geom.py --dry-run
```

This read-only command inspects `data/raw/geom_drugs` and writes:
- `experiments/manifests/geom_drugs_preflight.json`
- `reports/tables/geom_drugs_preflight.md`

The output is labeled `preflight_not_benchmark`, `benchmark_result: false`, `raw_data_modified: false`, `processed_data_written: false`, and `preprocessing_executed: false`. When `rdkit_folder/drugs/*.pickle` is present, it additionally reads a tiny bounded subset of raw pickle metadata and records `preprocessing_dry_run_executed: true`; this remains metadata-only and does not write processed records.

Current tiny GEOM-Drugs processed-output dry-run:

```bash
python scripts/prepare_geom.py --dry-run --write-tiny-processed-output
```

This is still not real preprocessing. It writes only:
- `data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json`
- `data/processed/geom_drugs_subset/tiny_real_dry_run/manifest.json`

The artifact is capped by explicit molecule, conformer, pickle-size, and atom-count bounds. By default it writes at most three molecules and one conformer per molecule. It is labeled `processed_output_dry_run_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, `raw_data_modified: false`, and `full_processed_data_written: false`.

Current tiny GEOM-Drugs split dry-run:

```bash
python scripts/make_geom_tiny_split.py
```

This reads the tiny processed-output dry-run artifact and writes only:
- `data/splits/geom_drugs_tiny_real_dry_run/random_seed0/split.json`
- `data/splits/geom_drugs_tiny_real_dry_run/random_seed0/manifest.json`

The artifact is labeled `split_dry_run_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, `raw_data_modified: false`, `processed_data_modified: false`, and `full_split_written: false`. With the default three-molecule processed dry-run it is non-degenerate; if the tiny input contains fewer than three molecules, the split is explicitly marked `degenerate_tiny_split: true`.

Current micro-pilot processed-output and split dry-run:

```bash
python scripts/prepare_geom.py --dry-run --write-micro-pilot-output
python scripts/make_geom_tiny_split.py --micro-pilot
```

This writes only:
- `data/processed/geom_drugs_subset/micro_pilot/molecules.json`
- `data/processed/geom_drugs_subset/micro_pilot/manifest.json`
- `data/splits/geom_drugs_micro_pilot/random_seed0/split.json`
- `data/splits/geom_drugs_micro_pilot/random_seed0/manifest.json`

The micro-pilot processed artifact is still labeled `processed_output_dry_run_not_benchmark` and `benchmark_result: false`, with `subset_scope: micro_pilot_processed_dry_run`, `max_molecules: 6`, `hard_max_molecules: 6`, and one conformer per molecule by default. The split dry-run is labeled `split_dry_run_not_benchmark`; with six molecules it writes four train IDs, one validation ID, and one test ID. This is not a real processed training subset or benchmark split.

Current tiny real-data RDKit baseline dry-run:

```bash
python scripts/run_rdkit_tiny_real.py
```

This reads only the three-molecule tiny processed artifact and tiny split dry-run, then runs RDKit ETKDGv3 + MMFF locally on CPU if RDKit is importable. It writes:
- `experiments/manifests/geom_tiny_real_rdkit_etkdg_mmff_seed0.json`
- `experiments/results/geom_tiny_real_rdkit_etkdg_mmff_seed0.json`
- `experiments/logs/geom_tiny_real_rdkit_etkdg_mmff_seed0.log`

The outputs are labeled `tiny_real_data_dry_run_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, and `slurm_account: null`. This is not the real GEOM-scale RDKit baseline.

Current tiny real-data model-side dry-runs:

```bash
python scripts/run_tiny_real_model_dry_runs.py
```

This reads only:
- `data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json`
- `data/splits/geom_drugs_tiny_real_dry_run/random_seed0/split.json`

It runs the isotropic and stiffness-structured source-corruption paths locally on CPU for the bounded three-molecule artifact, constructs flow-matching sample tensors, and writes:
- `experiments/manifests/geom_tiny_real_model_isotropic_seed0.json`
- `experiments/results/geom_tiny_real_model_isotropic_seed0.json`
- `experiments/logs/geom_tiny_real_model_isotropic_seed0.log`
- `experiments/manifests/geom_tiny_real_model_stiffness_seed0.json`
- `experiments/results/geom_tiny_real_model_stiffness_seed0.json`
- `experiments/logs/geom_tiny_real_model_stiffness_seed0.log`

The outputs are labeled `tiny_real_data_model_dry_run_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, `model_training_started: false`, `sampling_executed: false`, and `slurm_account: null`. They are model-side plumbing checks only, not model training, model sampling, or benchmark results.

Current tiny real-data model execution smoke:

```bash
python scripts/run_tiny_real_execution_smoke.py
```

This reads the same three-molecule tiny processed artifact and tiny split dry-run, then runs one parameter-free CPU forward/loss-style execution pass for both isotropic and stiffness-structured source configurations. It writes:
- `experiments/manifests/geom_tiny_real_execution_isotropic_seed0.json`
- `experiments/results/geom_tiny_real_execution_isotropic_seed0.json`
- `experiments/logs/geom_tiny_real_execution_isotropic_seed0.log`
- `experiments/manifests/geom_tiny_real_execution_stiffness_seed0.json`
- `experiments/results/geom_tiny_real_execution_stiffness_seed0.json`
- `experiments/logs/geom_tiny_real_execution_stiffness_seed0.log`

The outputs are labeled `tiny_real_data_model_execution_smoke_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed: false`, `model_training_started: false`, `training_loop_executed: false`, `optimizer_created: false`, `backward_executed: false`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. This validates forward/loss plumbing only and is not model training, model sampling, or a benchmark result.

Current tiny real-data single-step training smoke:

```bash
python scripts/run_tiny_real_single_step_smoke.py
```

This reads the same three-molecule tiny processed artifact and tiny split dry-run, then runs exactly one CPU-only forward/loss/backward/optimizer step for both isotropic and stiffness-structured source configurations. It writes:
- `experiments/manifests/geom_tiny_real_single_step_isotropic_seed0.json`
- `experiments/results/geom_tiny_real_single_step_isotropic_seed0.json`
- `experiments/logs/geom_tiny_real_single_step_isotropic_seed0.log`
- `experiments/manifests/geom_tiny_real_single_step_stiffness_seed0.json`
- `experiments/results/geom_tiny_real_single_step_stiffness_seed0.json`
- `experiments/logs/geom_tiny_real_single_step_stiffness_seed0.log`

The outputs are labeled `tiny_real_data_single_step_training_smoke_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed: true`, `model_training_started: true`, `training_loop_executed: false`, `single_step_training_smoke_executed: true`, `max_steps: 1`, `optimizer_created: true`, `optimizer_step_count: 1`, `backward_executed: true`, `backward_pass_count: 1`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. This validates one bounded training step only and is not a real training run or benchmark result.

Current tiny real-data bounded multi-step training smoke:

```bash
python scripts/run_tiny_real_multi_step_smoke.py
```

This reads the same three-molecule tiny processed artifact and tiny split dry-run, then runs a very short CPU-only training sequence for both isotropic and stiffness-structured source configurations. The default is three optimizer steps, with a hard cap of five steps. It writes:
- `experiments/manifests/geom_tiny_real_multistep_isotropic_steps3_seed0.json`
- `experiments/results/geom_tiny_real_multistep_isotropic_steps3_seed0.json`
- `experiments/logs/geom_tiny_real_multistep_isotropic_steps3_seed0.log`
- `experiments/manifests/geom_tiny_real_multistep_stiffness_steps3_seed0.json`
- `experiments/results/geom_tiny_real_multistep_stiffness_steps3_seed0.json`
- `experiments/logs/geom_tiny_real_multistep_stiffness_steps3_seed0.log`

The outputs are labeled `tiny_real_data_bounded_multistep_training_smoke_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed: true`, `bounded_training_smoke_executed: true`, `larger_real_training_run_started: false`, `max_steps: 3`, `hard_max_steps: 5`, `optimizer_step_count: 3`, `backward_pass_count: 3`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. They include per-step loss and gradient metadata only; this is not a benchmark result or larger real training run.

Current tiny real-data checkpoint-and-sampling smoke:

```bash
python scripts/run_tiny_real_checkpoint_sampling_smoke.py
```

This reads the same three-molecule tiny processed artifact and tiny split dry-run, runs a bounded CPU-only mini training smoke for both isotropic and stiffness-structured source configurations, writes disposable smoke checkpoints, reloads them, and runs one bounded sampling update per molecule. It writes:
- `experiments/manifests/geom_tiny_real_ckpt_sample_isotropic_steps3_seed0.json`
- `experiments/results/geom_tiny_real_ckpt_sample_isotropic_steps3_seed0.json`
- `experiments/logs/geom_tiny_real_ckpt_sample_isotropic_steps3_seed0.log`
- `experiments/checkpoints/geom_tiny_real_ckpt_sample_isotropic_steps3_seed0.pt`
- `experiments/manifests/geom_tiny_real_ckpt_sample_stiffness_steps3_seed0.json`
- `experiments/results/geom_tiny_real_ckpt_sample_stiffness_steps3_seed0.json`
- `experiments/logs/geom_tiny_real_ckpt_sample_stiffness_steps3_seed0.log`
- `experiments/checkpoints/geom_tiny_real_ckpt_sample_stiffness_steps3_seed0.pt`

The outputs are labeled `tiny_real_data_checkpoint_sampling_smoke_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `bounded_training_smoke_executed: true`, `larger_real_training_run_started: false`, `checkpoint_written: true`, `checkpoint_reloaded: true`, `checkpoint_scope: disposable_smoke_only`, `sampling_executed: true`, `sample_step_count: 1`, and `slurm_account: null`. The checkpoints are disposable smoke artifacts, not reusable research checkpoints, and the sampled coordinates are plumbing diagnostics only, not benchmark samples.

Current bounded tiny real-data pilot:

```bash
python scripts/run_tiny_real_pilot.py
```

This reads the same three-molecule tiny processed artifact and tiny split dry-run, then runs a strictly bounded CPU-only pilot training loop for both isotropic and stiffness-structured source configurations. Each pilot trains only the one train-split molecule, records validation/test losses on the one-molecule validation/test splits as interface diagnostics, and writes no checkpoints or samples. It writes:
- `experiments/manifests/geom_tiny_real_pilot_isotropic_steps6_seed0.json`
- `experiments/results/geom_tiny_real_pilot_isotropic_steps6_seed0.json`
- `experiments/logs/geom_tiny_real_pilot_isotropic_steps6_seed0.log`
- `experiments/manifests/geom_tiny_real_pilot_stiffness_steps6_seed0.json`
- `experiments/results/geom_tiny_real_pilot_stiffness_steps6_seed0.json`
- `experiments/logs/geom_tiny_real_pilot_stiffness_steps6_seed0.log`

The outputs are labeled `tiny_real_data_bounded_pilot_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `training_executed: true`, `training_loop_executed: true`, `bounded_real_training_loop_executed: true`, `larger_real_training_run_started: false`, `max_steps: 6`, `hard_max_steps: 8`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. This is the first bounded pilot training loop on the tiny real artifact, not a benchmark result or evidence of model quality.

Current bounded micro-pilot run:

```bash
python scripts/run_tiny_real_pilot.py --micro-pilot
```

This reads the six-molecule micro-pilot processed artifact and split dry-run, then runs the same bounded CPU-only pilot loop for isotropic and stiffness-structured source configurations. It writes:
- `experiments/manifests/geom_micro_pilot_isotropic_mols6_steps6_seed0.json`
- `experiments/results/geom_micro_pilot_isotropic_mols6_steps6_seed0.json`
- `experiments/logs/geom_micro_pilot_isotropic_mols6_steps6_seed0.log`
- `experiments/manifests/geom_micro_pilot_stiffness_mols6_steps6_seed0.json`
- `experiments/results/geom_micro_pilot_stiffness_mols6_steps6_seed0.json`
- `experiments/logs/geom_micro_pilot_stiffness_mols6_steps6_seed0.log`

The outputs remain labeled `tiny_real_data_bounded_pilot_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `micro_pilot_run: true`, `bounded_pilot: true`, `larger_real_training_run_started: false`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. This is a bounded micro-pilot only, not a larger real training run or benchmark result.

Current bounded micro-pilot comparison report:

```bash
python scripts/compare_tiny_real_pilot.py --micro-pilot
```

This reads the two current bounded six-molecule micro-pilot result JSON files and writes a reporting-only comparison of subset metadata, train/validation/test counts, final losses, per-step loss traces, and safety labels. It writes:
- `experiments/results/geom_micro_pilot_summary.json`
- `reports/tables/geom_micro_pilot_summary.md`

The summary is labeled `tiny_real_data_bounded_micro_pilot_comparison_not_benchmark`, `comparison_scope: micro_pilot`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed_by_summary: false`, `sampling_executed_by_summary: false`, `larger_real_training_run_started: false`, and `slurm_account: null`. It does not read raw data, start training, run sampling, write checkpoints, use GPUs, submit jobs, or modify the environment.

Current bounded multi-seed micro-pilot:

```bash
python scripts/run_micro_pilot_multiseed.py
```

This runs the bounded six-molecule micro-pilot for isotropic and stiffness-structured source configurations over the fixed tiny seed set `0,1`, then writes one multi-seed summary. It writes per-seed pilot outputs such as:
- `experiments/manifests/geom_micro_pilot_isotropic_mols6_steps6_seed0.json`
- `experiments/results/geom_micro_pilot_isotropic_mols6_steps6_seed0.json`
- `experiments/logs/geom_micro_pilot_isotropic_mols6_steps6_seed0.log`
- `experiments/manifests/geom_micro_pilot_isotropic_mols6_steps6_seed1.json`
- `experiments/results/geom_micro_pilot_isotropic_mols6_steps6_seed1.json`
- `experiments/logs/geom_micro_pilot_isotropic_mols6_steps6_seed1.log`
- matching `stiffness` files for seeds `0` and `1`

It also writes:
- `experiments/results/geom_micro_pilot_multiseed_summary.json`
- `reports/tables/geom_micro_pilot_multiseed_summary.md`

The per-seed outputs remain labeled `tiny_real_data_bounded_pilot_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `micro_pilot_run: true`, `bounded_pilot: true`, `larger_real_training_run_started: false`, `checkpoint_written: false`, `sampling_executed: false`, and `slurm_account: null`. The summary is labeled `tiny_real_data_bounded_micro_pilot_multiseed_summary_not_benchmark`, with `seed_set: [0, 1]`. This is a bounded micro-pilot stability diagnostic only, not a benchmark result.

Current bounded micro-pilot checkpoint-and-sampling pilot:

```bash
python scripts/run_micro_pilot_checkpoint_sampling.py
```

This reads the existing six-molecule micro-pilot processed artifact and split, runs a single fixed seed for both isotropic and stiffness-structured source configurations, trains only the four-molecule train split for five CPU optimizer steps, writes a disposable pilot-only checkpoint, reloads it, and runs one bounded sampling step across the six-molecule artifact. It writes:
- `experiments/manifests/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.json`
- `experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.json`
- `experiments/logs/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.log`
- `experiments/checkpoints/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.pt`
- `experiments/manifests/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed0.json`
- `experiments/results/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed0.json`
- `experiments/logs/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed0.log`
- `experiments/checkpoints/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed0.pt`

The outputs are labeled `tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `micro_pilot_run: true`, `larger_real_training_run_started: false`, `checkpoint_written: true`, `checkpoint_reloaded: true`, `checkpoint_scope: disposable_micro_pilot_checkpoint_sampling_only`, `sampling_executed: true`, `sample_step_count: 1`, and `slurm_account: null`. The checkpoints are disposable pilot artifacts, not reusable research checkpoints, and the sampled coordinates are plumbing diagnostics only, not benchmark samples.

Current bounded micro-pilot checkpoint-and-sampling comparison:

```bash
python scripts/compare_micro_pilot_checkpoint_sampling.py
```

This reads the current isotropic and stiffness-structured bounded micro-pilot checkpoint/sampling result JSON files and writes:
- `experiments/results/geom_micro_pilot_ckpt_sample_summary.json`
- `reports/tables/geom_micro_pilot_ckpt_sample_summary.md`

The summary is labeled `tiny_real_data_bounded_micro_pilot_checkpoint_sampling_comparison_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed_by_summary: false`, `sampling_executed_by_summary: false`, `larger_real_training_run_started: false`, and `slurm_account: null`. It compares `reloaded_loss`, overall and split-wise `sampled_rmsd_to_target`, per-molecule sampled RMSD, and runtime fields. Older checkpoint/sampling outputs that measured one-time PyTorch import/runtime warm-up in only the first mode are flagged as runtime-inconsistent until regenerated with the fixed `runtime_scope`.

Current bounded micro-pilot checkpoint-and-sampling multi-seed run:

```bash
python scripts/run_micro_pilot_checkpoint_sampling_multiseed.py
```

This runs the same bounded six-molecule checkpoint/sampling pilot for isotropic and stiffness-structured source configurations over seeds `0,1,2`. It does not change the artifact, split, molecule count, model, optimizer, five-step cap, or one-step sampling settings. It writes per-mode/per-seed manifest, result, log, and disposable checkpoint files named like:
- `experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.json`
- `experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed1.json`
- `experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed2.json`
- matching `stiffness` files for seeds `0`, `1`, and `2`

It also writes:
- `experiments/results/geom_micro_pilot_ckpt_sample_multiseed_summary.json`
- `reports/tables/geom_micro_pilot_ckpt_sample_multiseed_summary.md`

The per-run outputs are labeled `tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `micro_pilot_run: true`, `checkpoint_scope: disposable_micro_pilot_checkpoint_sampling_only`, `sampling_executed: true`, and `slurm_account: null`. The multi-seed summary is labeled `tiny_real_data_bounded_micro_pilot_checkpoint_sampling_multiseed_summary_not_benchmark` and reports per-seed winners, aggregate means, stiffness consistency, and runtime metadata for accounting only.

Current bounded 30-molecule data-scale checkpoint-and-sampling check:

```bash
python scripts/run_data_scale_check_checkpoint_sampling.py
```

This writes a bounded 30-molecule processed dry-run artifact and 28/1/1 random split, then runs the same CPU-only checkpoint/sampling pilot settings for isotropic and stiffness-structured source configurations at seed `0`. It writes:
- `data/processed/geom_drugs_subset/data_scale_check/molecules.json`
- `data/splits/geom_drugs_data_scale_check/random_seed0/split.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols30_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_stiffness_mols30_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_comparison_mols30_steps5_seed0.json`

The per-run outputs are labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `data_scale_check_run: true`, `checkpoint_scope: disposable_data_scale_check_checkpoint_sampling_only`, `sampling_executed: true`, and `slurm_account: null`. The comparison is labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_comparison_not_benchmark` and reports whether the seed-0 directional signal is preserved. This is still a bounded non-benchmark data-scale check, not a larger real training run.

Current bounded 30-molecule data-scale checkpoint-and-sampling multi-seed run:

```bash
python scripts/run_data_scale_check_checkpoint_sampling_multiseed.py
```

This reads the existing bounded 30-molecule processed artifact and fixed 28/1/1 split through the data-scale-check configs, then runs the same CPU-only checkpoint/sampling pilot settings for isotropic and stiffness-structured source configurations over seeds `0,1,2`. It does not rebuild the processed artifact or split, and it does not change the molecule count, split logic, model, optimizer, five-step cap, or one-step sampling settings. It writes per-mode/per-seed manifest, result, log, and disposable checkpoint files named like:
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols30_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols30_steps5_seed1.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols30_steps5_seed2.json`
- matching `stiffness` files for seeds `0`, `1`, and `2`

It also writes:
- `experiments/results/geom_data_scale_check_ckpt_sample_multiseed_summary.json`
- `reports/tables/geom_data_scale_check_ckpt_sample_multiseed_summary.md`

The per-run outputs remain labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `data_scale_check_run: true`, `checkpoint_scope: disposable_data_scale_check_checkpoint_sampling_only`, `sampling_executed: true`, and `slurm_account: null`. The multi-seed summary is labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_multiseed_summary_not_benchmark` and reports per-seed winners, aggregate means, stiffness consistency, and runtime metadata for accounting only. This is still a bounded non-benchmark data-scale check, not a larger real training run.

Current bounded 500-molecule data-scale checkpoint-and-sampling check:

```bash
python scripts/run_data_scale_check_500_checkpoint_sampling.py
```

This writes a bounded 500-molecule processed dry-run artifact and nondegenerate 450/25/25 random split, then runs the same CPU-only checkpoint/sampling pilot settings used by the 30-molecule check for isotropic and stiffness-structured source configurations at seed `0`. It preserves the processed-output schema, one-conformer-per-molecule policy, model, optimizer, five-step cap, checkpoint/reload path, and one-step sampling settings. It writes:
- `data/processed/geom_drugs_subset/data_scale_check_500/molecules.json`
- `data/splits/geom_drugs_data_scale_check_500/random_seed0/split.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols500_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_stiffness_mols500_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_comparison_mols500_steps5_seed0.json`

The per-run outputs are labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `data_scale_check_run: true`, `data_scale_check_500_run: true`, `checkpoint_scope: disposable_data_scale_check_500_checkpoint_sampling_only`, `sampling_executed: true`, and `slurm_account: null`. The comparison is labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_comparison_not_benchmark` and reports whether the seed-0 directional signal is preserved. This is a bounded non-benchmark 500-molecule data-scale check, not a benchmark result or larger real training run.

Current bounded 500-molecule data-scale checkpoint-and-sampling multi-seed run:

```bash
python scripts/run_data_scale_check_500_checkpoint_sampling_multiseed.py
```

This reads the existing bounded 500-molecule processed artifact and fixed 450/25/25 split through the data-scale-check configs, then runs the same CPU-only checkpoint/sampling pilot settings for isotropic and stiffness-structured source configurations over seeds `0,1,2`. It does not rebuild the processed artifact or split, and it does not change the molecule count, split logic, model, optimizer, five-step cap, checkpoint/reload path, or one-step sampling settings. It writes per-mode/per-seed manifest, result, log, and disposable checkpoint files named like:
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols500_steps5_seed0.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols500_steps5_seed1.json`
- `experiments/results/geom_data_scale_check_ckpt_sample_isotropic_mols500_steps5_seed2.json`
- matching `stiffness` files for seeds `0`, `1`, and `2`

It also writes:
- `experiments/results/geom_data_scale_check_ckpt_sample_multiseed_summary_mols500.json`
- `reports/tables/geom_data_scale_check_ckpt_sample_multiseed_summary_mols500.md`

The per-run outputs remain labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `pilot_run: true`, `bounded_pilot: true`, `data_scale_check_run: true`, `data_scale_check_500_run: true`, `checkpoint_scope: disposable_data_scale_check_500_checkpoint_sampling_only`, `sampling_executed: true`, and `slurm_account: null`. The multi-seed summary is labeled `tiny_real_data_bounded_data_scale_check_checkpoint_sampling_multiseed_summary_not_benchmark` and reports per-seed winners, aggregate means, stiffness consistency, and runtime metadata for accounting only. This is a bounded non-benchmark 500-molecule data-scale check, not a larger real training run.

Current bounded tiny real-data pilot comparison report:

```bash
python scripts/compare_tiny_real_pilot.py
```

This reads the two current bounded pilot result JSON files and writes a reporting-only comparison of pilot metadata, train/validation/test counts, final losses, per-step loss traces, and safety labels. It writes:
- `experiments/results/geom_tiny_real_pilot_summary.json`
- `reports/tables/geom_tiny_real_pilot_summary.md`

The summary is labeled `tiny_real_data_bounded_pilot_comparison_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `gpu_used: false`, `training_executed_by_summary: false`, `sampling_executed_by_summary: false`, `larger_real_training_run_started: false`, and `slurm_account: null`. It does not read raw data, start training, run sampling, write checkpoints, use GPUs, submit jobs, or modify the environment.

Current tiny real-data end-to-end smoke lifecycle summary:

```bash
python scripts/summarize_tiny_real_lifecycle.py
```

This reads the current tiny real-data result JSON files across the RDKit baseline, model dry-runs, execution smoke, single-step training smoke, bounded multi-step training smoke, and checkpoint/sampling smoke stages. It writes:
- `experiments/results/geom_tiny_real_lifecycle_summary.json`
- `reports/tables/geom_tiny_real_lifecycle_summary.md`

The summary is labeled `tiny_real_data_lifecycle_smoke_summary_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `larger_real_training_run_started: false`, and `slurm_account: null`. It is a reporting-only aggregation of existing smoke artifacts; it does not read raw data, load checkpoint contents, start training, run sampling, or produce benchmark results.

Current tiny real-data dry-run comparison report:

```bash
python scripts/compare_tiny_real_results.py
```

This reads the current tiny real-data RDKit baseline result plus the isotropic and stiffness-structured model-side dry-run results, then writes:
- `experiments/results/geom_tiny_real_dry_run_summary.json`
- `reports/tables/geom_tiny_real_dry_run_summary.md`

The summary is labeled `tiny_real_data_comparison_dry_run_not_benchmark`, `benchmark_result: false`, `hardware: cpu_only`, `training_executed: false`, `model_training_started: false`, `sampling_executed: false`, and `slurm_account: null`. It is a reporting-only aggregation of dry-run artifacts, not a benchmark comparison.

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
