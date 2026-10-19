# StifFlow shared benchmark

CPU evaluator for saved conformers; no model import or label cache is required. The neural generation
entry is prepared but unexecuted under the user's request to defer GPU runs. The current environment
now imports ET-Flow successfully on CPU; this agent did not change installed packages.

Set `PYTHONPATH` to all three worktree roots and use the existing `etflow` Python. From the workspace:

```bash
python -m stifflow_benchmark.evaluate runs/example/manifest.json --output runs/example/evaluation.json
```

Output paths must be new. Evaluation uses the exact `RemoveHs` + RDKit `GetBestRMS` defaults used by
ET-Flow's inner RMSD function: proper alignment and symmetry atom matching, Angstrom units, strict
`RMSD < threshold`, default thresholds 0.05..3.00, and the first `2 * N_ref` generated conformers.
There is no force-field postprocessing. Unlike upstream, any invalid RMSD pair fails the entire
molecule instead of silently dropping NaNs or molecules. Full molblocks preserve the producer's atom
order; atomic numbers alone cannot prove the order of equal-element atoms.

Every attempted molecule is retained with status/reason. Reports contain COV-R/P curves and MAT-R/P
mean/median for successful molecules, as well as attempted coverage with failures counted as zero.
MAT over all attempts is undefined when any molecule fails, and is emitted as null. Tetrahedral CIP
agreement is reported separately for specified centers; no E/Z accuracy metric is claimed. Achiral
molecules are excluded from that chirality denominator. RMSD alone must not be read as stereo validity.
Multi-fragment molecules are rejected by default; pass `--allow-disconnected` to evaluate them with
the same RMSD/chirality protocol. Formal StifFlow evaluation opts in because the frozen val split
contains a few multi-fragment QM9 species, and records the count in `aggregate.multi_fragment_molecules`.

The run manifest is JSON:

```json
{
  "schema_version": 1,
  "method": "precision",
  "baseline_kind": "controlled",
  "seed": 42,
  "split": "val",
  "config": {},
  "training_budget": {"steps": 1000},
  "generation": {},
  "provenance": {},
  "records": [
    {"id": "stable-id", "status": "ok", "molblock": "full RDKit molblock", "npz": "sample.npz"},
    {"id": "failed-id", "status": "failed", "reason": "record original failure"}
  ]
}
```

Each NPZ contains `pos_ref [N_ref,N_atoms,3]`, `pos_gen [N_gen,N_atoms,3]`, and `atomic_numbers [N_atoms]`.
Controlled methods include `precision`, `alternative_a`, `control` and the research variants
`a_rot_null`, `a_trace_null`, `linear`, `aligned_linear`, `precision_clock`. Official pretrained ET-Flow
must use `method=etflow_pretrained`, `baseline_kind=external_pretrained`, and state its different
budget/prior/loss/parity treatment. An external pretrained score is not a budget-matched comparison.

Reports save input hashes, versions, full run metadata, per-molecule failures and evaluator provenance.
Repeating evaluation on the same file produces identical metrics; elapsed times naturally differ.
Synthetic unit tests verify round-trip evaluation, reflected chirality, missing files, wrong atom
numbers and duplicate IDs. These are not GEOM model-quality results.

## Deferred neural generation

```bash
# Run only later in an authorized Slurm allocation after environment repair/training.
python -m stifflow_benchmark.generate runs/references.json \
  --checkpoint runs/precision/final.ckpt --steps 50 --output runs/precision-generated
```

The reference manifest contains `split` and `records` with `id`, `molblock`, `reference_count`, and
`reference_npz` (keys pos_ref/atomic_numbers). Test runs additionally require `selection_frozen=true`
after validation selection. The generator constructs graph features after removing molblock
conformers, calls `stifflow.model.generate` with graph-only inputs, and reads reference coordinates
only after generation. It never reads stiffness caches. Output follows the evaluator schema.
The model checkpoint carries its source configuration; changing the inference source is rejected.
Steps 25/50/100 are available for later validation sensitivity checks. Runtime, NFE and peak GPU
memory are recorded per molecule; stiffness label cost remains in preprocessing manifests.

Long generations atomically rewrite `manifest.json` every `--flush-every` processed molecules
(default 50). Pass `--resume` with the same checkpoint/reference/config to skip records whose NPZ
file and stored fingerprint are intact; failed or interrupted molecules are regenerated. Resume is
rejected if the checkpoint, reference manifest, sampler/model configuration, training signature or
code provenance changed.

No GPU generation, controlled pilot, formal test-set evaluation or DRUGS experiment has been run.

```bash
python -m stifflow_benchmark.summarize runs/precision/evaluation.json \
  runs/alternative-a/evaluation.json runs/control/evaluation.json --output runs/comparison.json
```

The summarizer refuses controlled comparisons with different molecule IDs, split, seed, training
budget, source/model configuration, integration steps or evaluation protocol. When training staged
the byte-identical label cache onto node-local storage and rewrote absolute paths, pass
`--normalize-staged-labels`: the frozen `official_data` hash remains the canonical data identity,
while each run's staged `train`/`val` manifest hashes are recorded as evidence instead of being
compared. External pretrained ET-Flow remains a separately identified row. Comparing different NFE
settings is a separate validation sensitivity study, not one matched comparison table.

Post-formal research diagnostics and the validation-only 25k paired screening workflow
are documented in
[the core research handoff](https://github.com/EasonShi0624/Stifflow/blob/torch/shhpc-training-safety/docs/research-next-stage.md).
Use `--allow-disconnected` explicitly to match the formal multi-fragment protocol.
Research comparisons use `stifflow_benchmark.research_compare`; the original formal
250k summary remains separate.

The next bounded experiment uses `stifflow_benchmark.stiffness_diagnostics`
(`locality` for a frozen train-only gate; `physics` for saved validation ensembles)
and `stifflow_benchmark.stiffness_compare` for the five-run physical/geometry screen.
The executable protocol and SHHPC commands are in the core branch
[`scripts/shhpc/README.md`](https://github.com/EasonShi0624/Stifflow/blob/torch/stiffness-locality-core/scripts/shhpc/README.md#stiffness-locality-screen).
All real diagnostics require a Slurm allocation. Labels and original generated coordinates
remain read-only; failed molecules cannot produce a passing physical subset. New delayed
methods cannot use the old geometry-only `research_compare` advancement gate.
Optional GFN2-xTB gradients use an existing executable and explicit graph charge/closed-shell
spin; parsing/unit checks do not substitute for an SHHPC oracle run.
