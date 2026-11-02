"""Synthetic Slurm-script flow checks; no cluster or GEOM data is used."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/torsion_screen.sbatch"


@pytest.mark.parametrize(
    "scenario,expected_calls",
    [
        ("success", ["select", "acceptance", "full"]),
        ("different-labels", ["select"]),
        ("failed-acceptance", ["select", "acceptance"]),
    ],
)
def test_job_is_ordered_and_fails_closed(tmp_path, scenario, expected_calls):
    workspace = tmp_path / "workspace"
    (workspace / "runs").mkdir(parents=True)
    helpers = workspace / "stifflow-v1/scripts"
    helpers.mkdir(parents=True)
    (helpers / "shhpc_env.sh").write_text(
        'export TMPDIR="$STIFFLOW_WORKSPACE/tmp"\nmkdir -p "$TMPDIR"\n'
    )
    for method in ("alternative_a", "control"):
        for seed in (42, 43, 44):
            directory = (
                workspace
                / f"runs/formal-eval-20261018/val/{method}-seed{seed}/generated"
            )
            directory.mkdir(parents=True)
            train = (
                "different"
                if scenario == "different-labels" and method == "control" and seed == 44
                else "same"
            )
            (directory / "manifest.json").write_text(
                json.dumps({"training_data": {"train": train, "val": "same"}})
            )

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_python = fakebin / "python"
    fake_python.write_text("""#!/bin/bash
set -eu
if [[ "$1" == "-m" ]]; then
    if [[ "$3" == "select" ]]; then
        echo select >> "$TORSION_TEST_LOG"
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "--output" ]]; then output="$2"; break; fi
            shift
        done
        mkdir "$output"
        printf '%s\\n' '{"groups": {}, "ids": []}' > "$output/panel.json"
        printf '%s\\n' '{"records": []}' > "$output/inventory.json"
    elif [[ " $* " == *" --limit 2 "* ]]; then
        echo acceptance >> "$TORSION_TEST_LOG"
        if [[ "${TORSION_FAIL_ACCEPTANCE:-0}" == 1 ]]; then exit 9; fi
    else
        echo full >> "$TORSION_TEST_LOG"
    fi
    exit 0
fi
exec "$TORSION_REAL_PYTHON" "$@"
""")
    fake_python.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = os.environ.copy()
    env.update(
        STIFFLOW_WORKSPACE=str(workspace),
        ETFLOW_ENV=str(tmp_path / "env"),
        SLURM_JOB_ID="12345",
        PATH=str(fakebin) + os.pathsep + env["PATH"],
        TORSION_TEST_LOG=str(log),
        TORSION_REAL_PYTHON=sys.executable,
        TORSION_FAIL_ACCEPTANCE="1" if scenario == "failed-acceptance" else "0",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert (result.returncode == 0) == (scenario == "success"), result.stderr
    assert log.read_text().splitlines() == expected_calls
    if scenario == "different-labels":
        assert "Label hashes differ" in result.stderr
    if scenario == "success":
        output = workspace / "runs/torsion-screen-12345"
        marker = output / "keep.txt"
        marker.write_text("existing result")
        repeated = subprocess.run(
            ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
        )
        assert repeated.returncode != 0
        assert marker.read_text() == "existing result"
        assert log.read_text().splitlines() == expected_calls
