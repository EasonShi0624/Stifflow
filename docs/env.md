# Environment and HPC Usage Notes

## Canonical environment

Use the existing conda environment:

```bash
conda activate stifflow
```

This project assumes that the environment has already been created and is usable.

Unless explicitly instructed otherwise:
- do **not** create a new conda environment;
- do **not** reinstall everything from scratch;
- do **not** install packages system-wide.

---

## Intended usage model

This repository is expected to be developed on a school HPC system.

Safe default behavior on the cluster:
- inspect files;
- edit code;
- run lightweight Python import checks;
- run unit tests and very small smoke tests;
- avoid heavy preprocessing or full training runs from the login node.

Unsafe default behavior unless explicitly approved:
- long GPU training runs;
- full GEOM-Drugs preprocessing on the login node;
- large RDKit/xTB/MMFF sweeps;
- automatic SLURM submission.

---

## Minimal shell startup

A typical session should begin with:

```bash
cd /path/to/repo
conda activate stifflow
python -V
which python
```

Optional environment check:

```bash
python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

---

## SLURM accounts and resource-selection policy

Two SLURM accounts are available for this project:

### `torch_pr_872_general`
Use this by default for:
- CPU jobs;
- short debugging jobs;
- tiny smoke tests;
- low-resource or modest GPU jobs;
- work where fast turnaround matters more than peak GPU performance.

Operational expectation:
- queue is often shorter;
- standard / more available GPUs such as `l40s` are the natural fit.

### `torch_pr_226_chemistry`
Use this when there is a clear reason for stronger resources, especially:
- chemistry-heavy training or evaluation;
- larger GPU workloads;
- jobs that benefit from higher-end GPUs such as `A100`.

Operational expectation:
- queue can be longer;
- use it when the resource need is real, not just because it is available.

### Default recommendation rule
If the job is:
- **short / debugging / CPU / low-resource** → prefer `torch_pr_872_general`
- **real training / expensive chemistry workflow / A100-worthy GPU job** → prefer `torch_pr_226_chemistry`

Every SLURM script should include an explicit account line.
A user-provided example includes:

```bash
#SBATCH --account=torch_pr_226_chemistry
```

This should be treated as mandatory style, not optional. fileciteturn5file0

---

## Prompt policy for coding agents

Agents may:
- inspect the repository;
- write or modify code;
- draft SLURM scripts;
- run cheap import checks and tiny smoke tests.

Agents must ask before:
- submitting any scheduler job;
- running any GPU training or GPU-heavy evaluation;
- starting a job likely to consume meaningful shared cluster resources;
- choosing between accounts for a nontrivial job;
- installing packages or changing the environment;
- downloading large datasets or checkpoints.

When asking, the prompt should state:
1. what the job does;
2. whether it is CPU or GPU;
3. which account is recommended;
4. why that account is appropriate;
5. whether queue time may be longer because of that choice.

Recommended wording:

> This looks like a short debugging or low-resource job. I recommend `torch_pr_872_general` for faster turnaround. Do you want me to prepare or submit the SLURM script?

> This looks like a real GPU training run and may benefit from A100-class resources. I recommend `torch_pr_226_chemistry`, but queue time may be longer. Do you want me to prepare or submit the SLURM script?

---

## Expected Python-side dependencies

The exact package set may evolve, but the project will likely require most of the following:
- Python 3.x
- PyTorch
- NumPy
- SciPy
- RDKit
- PyYAML or OmegaConf / Hydra-style config tooling
- pandas
- tqdm
- pytest

Potential later additions:
- torch-geometric or equivalent graph tooling
- xTB / GFN2-xTB interface
- PoseBusters or a compatible validity-check wrapper

Because the environment already exists, this file intentionally records **assumptions**, not a fresh installation recipe.

---

## Recommended first checks

Before implementing anything substantial, verify:

```bash
conda activate stifflow
python -V
python -c "import torch; print(torch.__version__)"
python -c "import rdkit; print('rdkit ok')"
```

If one of these imports fails, document the failure instead of guessing silently.

---

## Optional future additions

This file should later be expanded with:
- exact package versions;
- CUDA / PyTorch compatibility notes;
- module-load requirements if needed;
- resource profiles for smoke tests vs full runs.
