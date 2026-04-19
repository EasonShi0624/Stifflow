# Source-Only Stiffness-Structured Flow Matching for 3D Molecular Conformer Generation

This repository is a **research bootstrap** for a fixed-topology molecular conformer generation project on **GEOM-Drugs**.

The central claim is intentionally narrow:

> Keep the backbone and Cartesian flow-matching objective fixed, and test whether **source-only stiffness-structured corruption** improves physical plausibility and scaffold-split robustness over generic source constructions.

This repository is currently in a **docs-first bootstrap stage**:
- the scientific scope is defined;
- the intended experiments are specified;
- the code pipeline is **not yet fully implemented**.

That means this README is partly a project overview and partly an implementation contract. Do **not** assume that every script listed below already exists unless the current repository tree confirms it.

For agent-specific rules, priorities, and constraints, read [`AGENTS.md`](./AGENTS.md) first.

---

## 1. Project thesis

Recent conformer-generation work suggests two complementary lessons:

1. **Torsional Diffusion** motivates the idea that conformer variability is concentrated largely in torsional degrees of freedom rather than arbitrary Cartesian noise.
2. **ET-Flow-style Cartesian flow matching** suggests that one can stay in Cartesian coordinates and still obtain a strong equivariant conformer-generation baseline without a full internal-coordinate transport path.

This project sits between those ideas.

We do **not** attempt a full internal-coordinate probability path. Instead, we keep a simple Cartesian interpolation objective and inject chemical structure into the **source construction step**:
- bonds and rigid motifs should move very little;
- angles should move moderately;
- rotatable torsions should move the most.

The scientific challenge is to show that any gain comes from **stiffness structure itself**, not merely from milder noise, an easier source, or generic train-time geometry distortion.

---

## 2. Core scientific question

The main question is:

> Under a fixed equivariant flow-matching backbone and a simple Cartesian linear interpolation objective, can stiffness-structured source corruption improve physical plausibility and scaffold-split robustness without clearly collapsing conformer diversity?

A convincing answer must separate this from weaker explanations:
- the source is simply easier;
- the path is shorter;
- torsion-only perturbation already explains the gain;
- generic geometry distortion already explains the gain.

---

## 3. Scope

### In scope
- Fixed-topology molecular conformer generation.
- **GEOM-Drugs** as the main dataset.
- Random split and Bemis–Murcko scaffold split.
- A fixed Cartesian equivariant flow-matching backbone.
- Source-only corruption ablations.
- Evaluation using both **ensemble quality** and **physical validity** diagnostics.

### Out of scope for the core milestone
- De novo graph generation.
- Full internal-coordinate probability paths.
- Major architecture redesign.
- Force-residual repair head.
- Chirality-aware representation redesign.
- Pocket-conditioned or docking-conditioned generation.
- Large hyperparameter sweeps before the basic claim is stable.

---

## 4. Method summary

### Baseline formulation
Notation:
- `x1`: target conformer
- `x0`: source geometry
- `xt = (1 - t) * x0 + t * x1`

The core ablation keeps the **same backbone** and **same interpolation objective**, and changes only how `x0` is constructed.

### Main intervention: source-only stiffness-structured corruption
Default heuristic stiffness classes:
- **stiff**: bond lengths, constrained rings, amide-like nearly planar motifs
- **medium**: bond angles
- **soft**: rotatable torsions and selected long-range arrangements

Default first implementation:
- very small perturbations on bonds;
- small-to-moderate perturbations on angles;
- largest perturbations on rotatable torsions;
- strong protection for rings and amide-like motifs.

The first version should use a **3-level heuristic stiffness table**. MMFF-derived stiffness is a later sensitivity study, not the starting point.

---

## 5. Repository status

At the time of writing, this repository is assumed to be in a **bootstrap state**:
- `README.md` and `AGENTS.md` are present;
- the remaining project files are being drafted;
- the training / sampling / evaluation code may not exist yet.

Read [`docs/current_status.md`](./docs/current_status.md) for the up-to-date implementation status.

This distinction matters: the repo contains both **present reality** and **target design**. Coding agents must not confuse planned interfaces with already implemented code.

---

## 6. HPC usage and job-submission policy

This project is expected to run on a school HPC cluster inside the existing environment:

```bash
conda activate stifflow
```

Two SLURM accounts are available and should be chosen deliberately:
- `torch_pr_872_general`: usually better for short jobs, debugging, CPU work, and modest GPU work with shorter queue times.
- `torch_pr_226_chemistry`: better for chemistry-heavy or higher-end GPU work such as A100-class jobs, but queue times may be longer.

Coding agents may edit files and run lightweight checks without asking first, but must **request approval before**:
- submitting or launching scheduler jobs;
- starting GPU training or meaningful evaluation runs;
- using expensive preprocessing or large chemistry sweeps;
- mutating the environment.

See [`docs/env.md`](./docs/env.md), [`docs/reproduce.md`](./docs/reproduce.md), and [`docs/slurm.md`](./docs/slurm.md) for the exact policy and prompt expectations.

---

## 7. Target repository structure

The intended structure is:

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
│   └── train/
│       └── tiny_smoke.yaml
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
│   └── activate_env.sh
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

The actual repository may currently be a strict subset of this.

---

## 8. Dataset and splits

### Main dataset
Use **GEOM-Drugs** as the main benchmark.

Recommended first pass:
- 20k–50k molecules for a medium-scale study;
- up to 30 conformers per molecule;
- QM9 only for smoke tests or debugging.

### Required split policy
Always report both:
- **random split**
- **Bemis–Murcko scaffold split**

All conformers from the same molecule must remain in the same split.

### Minimum stratification
At minimum, stratify by:
- rotatable bond count;
- heavy atom count;
- ring count.

See [`docs/data.md`](./docs/data.md) for the intended data contract.

---

## 9. Baselines and required ablations

At minimum, compare:
1. **RDKit ETKDGv3 + MMFF**
2. **Isotropic Cartesian FM**
3. **Harmonic-prior / ET-Flow-style FM**
4. **Simple low-noise Cartesian corruption**
5. **Torsion-only corruption**
6. **Stiffness-structured corruption**
7. **Stiffness-structured corruption without ring protection**
8. **Stiffness-structured corruption with weak angle preservation**

If time is limited, prioritize:
- harmonic-prior baseline,
- torsion-only baseline,
- stiffness-structured main method.

---

## 10. Evaluation

### Primary metrics
These should anchor the core paper-style comparison:
- **COV-R / COV-P**
- **MAT-R / MAT-P**
- **PoseBusters pass rate** or the closest intramolecular validity equivalent
- bond / angle / torsion distribution diagnostics
- random vs scaffold split comparison
- performance vs rotatable-bond count

### Secondary metrics
These are useful but should not dominate the conclusion:
- MMFF/UFF energies
- MMFF/UFF force norms
- xTB / GFN2-xTB checks on a stratified subset
- runtime and number of sampling steps

### Evaluation principles
- RMSD-based ensemble quality is necessary but not sufficient.
- Physical validity must be checked independently.
- If MMFF-derived information is used in the method, MMFF energy must not be the only key metric.
- Results on scaffold split are more scientifically informative than random split alone.

---

## 11. Suggested workflow

### Step 1 — Data pipeline
- prepare a GEOM-Drugs subset;
- standardize hydrogens and conformer grouping;
- create random and scaffold splits;
- save reusable processed files.

### Step 2 — Baseline reproduction
- run RDKit ETKDGv3 + MMFF;
- run one Cartesian FM baseline;
- verify end-to-end training/sampling/evaluation on a tiny subset.

### Step 3 — Main method
- implement the stiffness-structured corruption interface;
- add the 3-level heuristic table;
- run the minimum ablation matrix.

### Step 4 — Robust evaluation
- compare random vs scaffold split;
- stratify by rotatable bond count;
- report both ensemble metrics and physical-validity metrics.

### Step 5 — Optional extensions
Only after the core result is stable:
- MMFF-derived stiffness sensitivity;
- xTB subset checks;
- force-residual repair;
- chirality diagnostic.

---

## 12. Quickstart template

These commands are placeholders and should be adapted to the actual repo implementation.

### Prepare data
```bash
python scripts/prepare_geom.py \
  --input data/raw/geom_drugs \
  --output data/processed/geom_drugs_subset \
  --max-molecules 20000 \
  --max-conformers-per-mol 30
```

### Make splits
```bash
python scripts/make_splits.py \
  --input data/processed/geom_drugs_subset \
  --output data/splits \
  --split-types random scaffold \
  --seed 2026
```

### Train a baseline
```bash
python scripts/train.py \
  --config configs/train/baseline_harmonic.yaml
```

### Train the main method
```bash
python scripts/train.py \
  --config configs/train/stiffness_structured.yaml
```

### Sample and evaluate
```bash
python scripts/sample.py \
  --config configs/eval/sample_baseline.yaml

python scripts/evaluate.py \
  --config configs/eval/eval_baseline.yaml
```

### Run ablation suite
```bash
python scripts/run_ablation.py \
  --manifest configs/ablation/core_suite.yaml
```

---

## 13. Reproducibility requirements

Every run should save:
- config snapshot;
- random seed;
- dataset subset identifier;
- split type;
- model/backbone name;
- corruption mode;
- checkpoint path;
- output metrics as JSON.

A run is not considered valid if it cannot be reproduced from config + manifest.

---

## 14. What success looks like

The core project succeeds if it can support a claim like this:

> With a fixed equivariant flow-matching backbone and simple Cartesian interpolation, stiffness-structured source corruption reduces geometry/energy failure modes without clearly harming diversity, and its gains are strongest on scaffold-split and high-rotatable-bond molecules.

This is a better outcome than a more ambitious but ambiguous story.
