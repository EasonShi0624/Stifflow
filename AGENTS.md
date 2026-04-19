# AGENTS.md

This file contains the working rules for coding agents operating in this repository.

The repository is currently in a **bootstrap stage**. Treat it as a docs-first research scaffold, not as a mature implementation. Planned interfaces, file paths, and experiments must be verified against the actual repository contents before use.

---

## 0. Mission

Build a **fixed-topology molecular conformer generation** project centered on a **source-only stiffness-structured flow-matching intervention**.

The scientific question is:

> Under a fixed equivariant flow-matching backbone and a simple Cartesian linear interpolation objective, can stiffness-structured source corruption improve physical plausibility and scaffold-split robustness without collapsing conformer diversity?

This repository is **not** for:
- de novo graph generation;
- a full internal-coordinate probability path;
- broad architecture exploration;
- speculative features that weaken attribution.

---

## 1. Source of truth

If anything is ambiguous, follow this order:

1. `AGENTS.md`
2. `docs/current_status.md`
3. `docs/env.md`
4. `docs/data.md`
5. `docs/slurm.md`
6. `README.md`
7. the current codebase and tests
8. old experiment scripts or notes

If the repo state conflicts with the planned structure described in the docs, trust the **actual repo state** and update the docs if needed.

---

## 2. Current repository reality

Assume the following unless the repository tree proves otherwise:
- the repo may contain only documentation plus minimal scaffolding;
- many scripts and config files may still be missing;
- command examples in `README.md` may describe the **target interface**, not a currently runnable pipeline.

Therefore:
- do not pretend placeholder commands already work;
- do not fabricate successful experiments;
- do not silently fill major gaps with unapproved complexity;
- prefer to build the smallest runnable path first.

---

## 3. Hard scope boundaries

### In scope
- Fixed-topology conformer generation from molecular graph + reference conformer set.
- GEOM-Drugs as the main dataset.
- QM9 only for debugging or smoke tests.
- Random split and Bemis–Murcko scaffold split.
- Baselines:
  - RDKit ETKDGv3 + MMFF
  - isotropic Cartesian flow matching baseline
  - harmonic-prior / ET-Flow-style baseline
  - torsion-focused corruption baseline
- Main method:
  - **same backbone, same FM objective, different source corruption**
- Evaluation:
  - COV / MAT
  - PoseBusters-style validity checks or closest available intramolecular equivalent
  - bond / angle / torsion distribution diagnostics
  - MMFF/UFF energies and force norms
  - optional xTB / GFN2-xTB subset checks

### Out of scope for the first successful milestone
- Full internal-coordinate probability-path transport.
- New neural architecture research.
- Large-scale hyperparameter sweeps.
- Force-residual repair head.
- Chirality-aware representation changes.
- Pocket-conditioned generation.
- Full de novo atom/bond generation.

If unsure whether something is in scope, choose the **simpler** option.

---

## 4. Non-negotiable design decisions

### 4.1 Keep the backbone fixed
Do **not** change the backbone unless the repository is blocked and cannot train at all.

Backbone selection rule:
1. Prefer an existing ET-Flow-style implementation **if it already exists or can be made to run with minimal surgery**.
2. Otherwise use a **minimal EGNN-style fallback**.
3. Once selected for the core study, that backbone must remain fixed across the ablation suite.

Do not switch backbones mid-ablation.

### 4.2 Keep the flow objective fixed
Use a simple Cartesian interpolation objective. Default:
- source sample: `x0`
- target conformer: `x1`
- interpolation: `xt = (1 - t) * x0 + t * x1`

Do **not** implement a full internal-coordinate path as the main method.

### 4.3 Main intervention = source corruption only
The main method changes only how `x0` is constructed.

Default stiffness classes:
- **stiff**: bond lengths, constrained rings, amide-like nearly planar motifs
- **medium**: bond angles
- **soft**: rotatable torsions and selected long-range arrangements

Default first implementation:
- a **3-level heuristic stiffness table**;
- very small perturbations for bonds;
- small/moderate perturbations for angles;
- largest perturbations for rotatable torsions;
- strong protection for rings and amide-like motifs.

Do **not** start with MMFF force constants.

---

## 5. Scientific risks that must be respected

### Risk A: method is not closed
Avoid fragile internal-coordinate-to-Cartesian reconstruction chains.

### Risk B: proxy mismatch
A geometrically cleaner corruption does not automatically improve conformer-ensemble metrics. Always report both geometry/energy checks and ensemble-matching metrics.

### Risk C: unclear attribution
If performance improves, reviewers may say the source is simply easier, the path is shorter, or the corruption is milder. The ablation matrix must isolate:
- stiffness-structured corruption vs simple low-noise corruption
- stiffness-structured corruption vs torsion-only corruption
- stiffness-structured corruption vs generic geometry distortion
- stiffness-structured corruption vs harmonic-prior / ET-Flow-style source design

### Risk D: teacher-aligned evaluation bias
If any MMFF-based prior or teacher is used, MMFF energy cannot be the only meaningful metric.

---

## 6. HPC and environment rules

The intended environment is a preconfigured conda environment:

```bash
conda activate stifflow
```

### 6.1 Basic environment constraints
Unless explicitly instructed otherwise:
- do not create a new environment;
- do not install packages system-wide;
- do not reinstall the whole stack;
- do not modify cluster-wide module state for all users;
- do not launch heavy training jobs from the login node;
- do not submit SLURM jobs automatically without permission.

### 6.2 Safe default behavior
Allowed without asking first:
- inspect files;
- edit code inside this repository;
- run lightweight import checks;
- run small unit tests;
- run tiny CPU or very small local smoke tests that clearly do **not** consume meaningful shared cluster resources;
- write or revise SLURM scripts and manifests.

### 6.3 Ask-first behavior
Codex must explicitly request approval before:
- any `sbatch`, `srun`, `qsub`, `scancel`, or comparable scheduler action;
- any GPU training, GPU evaluation, or GPU-heavy sampling;
- any job expected to run longer than a brief smoke test;
- any preprocessing likely to touch a substantial fraction of GEOM-Drugs;
- any large RDKit / MMFF / xTB / GFN2-xTB sweep;
- any package installation or environment mutation;
- any network activity beyond lightweight documentation / paper / API lookup;
- any deletion or move affecting large directories, datasets, checkpoints, or result trees.

When uncertain, ask first.

### 6.4 Cluster account policy
Two SLURM accounts are available:

- `torch_pr_872_general`
  - usually shorter queue times;
  - preferred for short jobs, CPU jobs, debugging, tiny smoke tests, and modest GPU jobs;
  - expected GPU class is typically standard / more available GPUs such as `l40s`.

- `torch_pr_226_chemistry`
  - preferred when the job genuinely benefits from higher-end GPUs or chemistry-heavy workflows;
  - can access more advanced GPU resources such as `A100`;
  - may queue longer.

Default account-selection policy:
- choose `torch_pr_872_general` for short, low-risk, low-resource, debugging, CPU, or lightweight GPU work;
- choose `torch_pr_226_chemistry` only when the job has a clear reason to need higher-end resources, longer chemistry workloads, or A100-class GPUs.

Never omit `--account=...` in a SLURM script.

### 6.5 Required prompt behavior for job launches
Before proposing or launching a cluster job, Codex should state:
1. what the job will do;
2. whether it is CPU or GPU;
3. which account it recommends and why;
4. the expected scale (tiny smoke test vs real training/eval);
5. whether queue time may be longer because of the chosen account / GPU class.

Example wording:

> This looks like a real GPU training run rather than a smoke test. I recommend `torch_pr_226_chemistry` because it may need A100-class resources, but queue time may be longer. Do you want me to prepare or submit the SLURM job?

> This is a short debugging or light-evaluation job. I recommend `torch_pr_872_general` for faster turnaround unless you specifically want higher-end GPUs. Do you want me to prepare or submit it?

### 6.6 SLURM script requirements
A user-provided example script uses an explicit account flag:
`#SBATCH --account=torch_pr_226_chemistry`

Treat that as a required pattern: every SLURM script in this repo should specify an account explicitly.
See `docs/slurm.md` for detailed guidance.

---

## 7. Required project layout

If the repo already has an equivalent structure, adapt to it. Otherwise create the following directories as needed:

```text
.
├── .codex/
│   └── config.toml
├── AGENTS.md
├── README.md
├── TASKS.md
├── docs/
│   ├── current_status.md
│   ├── data.md
│   ├── env.md
│   ├── reproduce.md
│   └── slurm.md
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── eval/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── models/
│   ├── corruption/
│   ├── train/
│   ├── eval/
│   ├── metrics/
│   └── utils/
├── scripts/
│   ├── activate_env.sh
│   ├── prepare_geom.py
│   ├── make_splits.py
│   ├── train.py
│   ├── sample.py
│   ├── evaluate.py
│   └── run_ablation.py
├── experiments/
│   ├── manifests/
│   ├── logs/
│   ├── checkpoints/
│   └── results/
├── tests/
└── reports/
    ├── figures/
    └── tables/
```

---

## 8. Milestone order

### Milestone 1 — Data and baselines
You must be able to:
1. preprocess a GEOM-Drugs subset,
2. construct random and scaffold splits,
3. run RDKit ETKDGv3 + MMFF,
4. train or at least execute a minimal Cartesian FM baseline,
5. produce COV/MAT and basic validity metrics on a tiny subset.

### Milestone 2 — Core method
Add source-only stiffness-structured corruption and compare it against:
1. isotropic corruption,
2. harmonic-prior / ET-Flow-style source,
3. torsion-only corruption,
4. generic geometry distortion if feasible.

### Milestone 3 — Robust evaluation
Run the main evaluation on:
- random split,
- scaffold split,
- stratified subsets by rotatable bond count.

### Milestone 4 — Optional extensions
Only after the core claim is stable:
- MMFF-derived stiffness sensitivity,
- small xTB/GFN2-xTB subset checks,
- force-residual repair prototype,
- chirality diagnostic.

---

## 9. Definition of done for the core project

The core project is “done” only if all of the following are true:
1. a fixed-backbone FM baseline and the stiffness-structured variant both train and sample successfully;
2. results exist for both random split and scaffold split;
3. at least one flexibility-stratified analysis exists;
4. the ablation suite isolates the effect of stiffness structure from generic low-noise or torsion-only alternatives;
5. the evaluation includes both ensemble metrics and physical validity metrics;
6. all runs are reproducible from config files and saved manifests.

---

## 10. Implementation rules

### 10.1 Config-driven only
New functionality must be toggled by config. Do not hard-code experiment choices.

At minimum, configs must expose:
- dataset subset size
- split type
- backbone type
- corruption mode
- stiffness table
- ring / amide protection flags
- training hyperparameters
- sampling steps
- evaluation subset size

### 10.2 Every run gets a manifest
Each training or evaluation run must write a machine-readable manifest, e.g. JSON or YAML, including:
- git commit hash if available
- config path
- dataset subset
- split type
- model name
- corruption mode
- random seed
- checkpoint path
- metrics output path
- execution mode: local smoke test vs scheduled job
- SLURM account if applicable

### 10.3 Fail loudly on chemistry-breaking corruption
If corruption breaks obvious constraints, do not silently continue. Log counts for:
- ring violations if detectable
- invalid stereochemistry if detectable
- NaNs / exploding coordinates
- invalid RDKit molecules

### 10.4 Prefer small, composable modules
Suggested module boundaries:
- `src/corruption/isotropic.py`
- `src/corruption/harmonic.py`
- `src/corruption/torsion_only.py`
- `src/corruption/stiffness_structured.py`
- `src/eval/cov_mat.py`
- `src/eval/posebusters_wrapper.py`
- `src/eval/energy_checks.py`
- `src/data/scaffold_split.py`

---

## 11. Minimum ablation matrix

These ablations are required unless impossible for a documented reason.

| ID | Backbone | Source / corruption | Purpose |
|---|---|---|---|
| A | fixed | isotropic Cartesian | naive baseline |
| B | fixed | harmonic-prior / ET-Flow-style | strong baseline |
| C | fixed | simple low-noise Cartesian | tests “easier source” explanation |
| D | fixed | torsion-only | tests whether torsion-only is enough |
| E | fixed | stiffness-structured | main method |
| F | fixed | stiffness-structured, no ring protection | tests role of rigid motifs |
| G | fixed | stiffness-structured, weak angle preservation | tests role of angle control |

If time is short, run A, B, D, E first.

---

## 12. Metrics that must be reported

### Primary metrics
- COV-R / COV-P
- MAT-R / MAT-P
- PoseBusters pass rate or closest available intramolecular validity checks
- bond-length / bond-angle / torsion distribution diagnostics
- results on scaffold split

### Secondary metrics
- MMFF/UFF energy
- MMFF/UFF force norm
- xTB / GFN2-xTB subset checks if feasible
- runtime and number of sampling steps

### Required stratification
At minimum, stratify by:
- rotatable bond count
- random vs scaffold split

---

## 13. What not to do

Do not:
- claim the method works because a single energy metric improves;
- compare models with different backbones and call it a corruption ablation;
- add residual heads before the core comparison is stable;
- add chirality features before the core comparison is stable;
- silently change the evaluation pipeline mid-project;
- tune each baseline separately until the comparison becomes unfair;
- describe the method as a full internal-coordinate path if it is not.

---

## 14. Default fallback decisions

If anything becomes blocked, take the simpler fallback:
- if ET-Flow is difficult to reproduce exactly, use a minimal EGNN-style FM and keep the story as **same backbone, different source corruption**;
- if MMFF-derived stiffness is messy, use a 3-level heuristic table only;
- if xTB is too expensive, run it on a small stratified subset and label it secondary;
- if PoseBusters integration is difficult, implement the nearest available intramolecular validity checks now and keep a clean wrapper for later replacement;
- if scaffold split is expensive, do it on a subset rather than skipping it entirely.

---

## 15. Output contract

By default, every major run should create:
- `experiments/manifests/<run_name>.json`
- `experiments/results/<run_name>.json`
- `experiments/logs/<run_name>.log`
- optional plots/tables under `reports/`

At minimum, each result JSON should include:
- dataset and split info
- corruption mode
- sample count
- COV/MAT metrics
- validity metrics
- energy diagnostics if present
- runtime
- execution mode
- SLURM account if applicable

---

## 16. Suggested first tasks for Codex

If starting from an incomplete repo, do the following in order:
1. inspect repo structure and summarize missing pieces;
2. implement dataset preparation and split scripts;
3. add a tiny smoke-test subset;
4. implement a clean corruption interface;
5. add isotropic and stiffness-structured corruption modules;
6. make training config-driven;
7. implement evaluation script with JSON output;
8. run one tiny baseline and one tiny stiffness-structured experiment;
9. verify reproducibility with a second seed;
10. only then scale up.

---

## 17. Commit and PR style

When making commits or PRs, use titles like:
- `data: add geom subset preprocessing and scaffold split`
- `corruption: add stiffness-structured source module`
- `eval: add cov-mat and validity report writer`
- `exp: run isotropic vs stiffness smoke tests`

In the description, always state:
- what changed,
- why it matters scientifically,
- what experiment it enables,
- what remains risky.

---

## 18. Short literature notes for the agent

These are not a full review; they are operational guidance.

- **GEOM (Axelrod & Gómez-Bombarelli, 2022):** standard large-scale conformer dataset; use GEOM-Drugs as the main benchmark.
- **Torsional Diffusion (Jing et al., NeurIPS 2022):** strong evidence that torsional degrees of freedom matter most for conformer variation.
- **ET-Flow (Hassan et al., NeurIPS 2024):** strong flow-matching baseline operating directly on Cartesian coordinates with a harmonic prior; avoid unnecessary internal-coordinate complexity.
- **GEOM-Drugs Revisited (Nikitin et al., 2025):** evaluation quality matters; use chemically careful preprocessing and independent checks.
- **PoseBusters (Buttenschoen et al., Chemical Science 2024):** RMSD alone is not enough; always report physical validity checks.
- **FlowMol3 (Dunn & Koes, 2025):** generic geometry distortion already improves transport models, so our contribution must be isolated from that explanation.
- **SO(3)-Averaged Flow (Cao et al., ICML 2025):** flow-path improvements can also come from training-objective design; do not overclaim that every gain is specifically from stiffness structure.
- **Chirality limitation paper (Dumitrescu et al., ICLR 2025):** chirality is not a trivial add-on for E(3)-equivariant models; keep it out of the core scope.

---

## 19. Final reminder

The clean version of the project is better than the glamorous version.

If forced to choose, prefer:
- a smaller but fairer experiment,
- a simpler but closed method,
- a narrower but more defensible claim.
