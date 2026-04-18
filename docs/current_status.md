# Current Repository Status

_Last updated: bootstrap stage._

## Summary

This repository is currently in a **documentation-first setup phase**.

Presently assumed status:
- `README.md` exists.
- `AGENTS.md` exists.
- Core code for data preparation, training, sampling, and evaluation may **not yet be implemented**.
- Config files and scripts are currently being drafted.

This file is the canonical place to record **what is actually present today**, as opposed to the intended target design.

---

## What exists now

At minimum, the repo contains:
- project framing and scientific scope;
- agent instructions;
- target structure and experimental plan.

Potentially missing right now:
- `src/` implementation files;
- `scripts/prepare_geom.py`;
- `scripts/make_splits.py`;
- `scripts/train.py`;
- `scripts/sample.py`;
- `scripts/evaluate.py`;
- data-processing code;
- baseline wrappers;
- tests.

If any of the above now exist, update this file.

---

## What the next coding pass should accomplish

The next implementation pass should aim for the smallest useful milestone:

1. create the basic repo skeleton if absent;
2. define a tiny smoke-test data contract;
3. add a config-driven experiment interface;
4. add a baseline corruption interface;
5. implement at least one tiny end-to-end path with machine-readable outputs.

---

## Rules for interpreting this file

- This file should describe the **real repo state**, not the idealized future state.
- Update it whenever major implementation milestones land.
- If `README.md` or `AGENTS.md` say a script is part of the target design, but the script does not actually exist yet, this file should say so plainly.

---

## Known current constraints

- Working environment is expected to be `conda activate stifflow`.
- The repo is intended for use on an HPC system.
- Large training or preprocessing jobs should not be assumed runnable from a login node.

---

## Immediate next milestone

The immediate goal is **not** full benchmark reproduction.

The immediate goal is:

> establish a tiny, honest, config-driven smoke-test pipeline that can later be scaled into the full conformer-generation study.
