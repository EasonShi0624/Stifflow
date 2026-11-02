#!/bin/bash
#SBATCH --job-name=stifflow-torsion
#SBATCH --partition=debug,parallel,chem,rooster
#SBATCH --cpus-per-task=1
# Set any required account, memory and walltime when submitting.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script as a Slurm CPU job}"
: "${STIFFLOW_WORKSPACE:?Set the existing SHHPC workspace}"
: "${ETFLOW_ENV:?Set the existing etflow environment}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
export MPLCONFIGDIR="$TMPDIR/matplotlib"

export TORSION_OUTPUT="$STIFFLOW_WORKSPACE/runs/torsion-screen-$SLURM_JOB_ID"
mkdir "$TORSION_OUTPUT"
TORSION_REFERENCE="$STIFFLOW_WORKSPACE/cache/qm9-v3-references-4090d/val/manifest.json"
TORSION_LABELS="$STIFFLOW_WORKSPACE/cache/qm9-v3-packed-strict/val/manifest.json"
TORSION_VAL="$STIFFLOW_WORKSPACE/runs/formal-eval-20261018/val"

echo "Output: $TORSION_OUTPUT"
echo "Stage 1: freeze the validation molecule panel"
python -m stifflow_benchmark.torsion_screen select \
  --references "$TORSION_REFERENCE" --output "$TORSION_OUTPUT/selection"

python - <<'PY'
import json
import os
from pathlib import Path

directory = Path(os.environ["TORSION_OUTPUT"]) / "selection"
panel = json.loads((directory / "panel.json").read_text())
inventory = json.loads((directory / "inventory.json").read_text())
records = {r["id"]: r for r in inventory["records"]}
print("Panel group counts:", panel["groups"], flush=True)
for identity in panel["ids"]:
    r = records[identity]
    print(r["group"], r["smiles"], identity, "rotors=", r["strict_rotors"], flush=True)
PY

echo "Stage 2: verify saved training-label identities"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["STIFFLOW_WORKSPACE"]) / "runs/formal-eval-20261018/val"
pairs = []
for method in ("alternative_a", "control"):
    for seed in (42, 43, 44):
        path = root / f"{method}-seed{seed}/generated/manifest.json"
        data = json.loads(path.read_text())
        train = data["training_data"]["train"]
        val = data["training_data"]["val"]
        pairs.append((train, val))
        print(method, seed, "train:", train, "val:", val, flush=True)
if len(set(pairs)) != 1:
    raise SystemExit("Label hashes differ: stop and obtain the archived staging manifests; see docs/torsion-screen.md")
print("All six train/val label hashes agree", flush=True)
PY

TORSION_RUNS=()
for TORSION_METHOD in alternative_a control; do
  for TORSION_SEED in 42 43 44; do
    TORSION_RUN="$TORSION_VAL/${TORSION_METHOD}-seed${TORSION_SEED}"
    TORSION_RUNS+=(--run "$TORSION_RUN/generated/manifest.json" "$TORSION_RUN/evaluation.json")
  done
done

echo "Stage 3: two-molecule acceptance"
python -m stifflow_benchmark.torsion_screen analyze \
  --panel "$TORSION_OUTPUT/selection/panel.json" "${TORSION_RUNS[@]}" \
  --physics --val-labels "$TORSION_LABELS" \
  --limit 2 --output "$TORSION_OUTPUT/acceptance2"

echo "Stage 4: full frozen panel"
python -m stifflow_benchmark.torsion_screen analyze \
  --panel "$TORSION_OUTPUT/selection/panel.json" "${TORSION_RUNS[@]}" \
  --physics --val-labels "$TORSION_LABELS" --output "$TORSION_OUTPUT/full"

echo "Completed. Read $TORSION_OUTPUT/acceptance2/report.json and $TORSION_OUTPUT/full/report.json"
