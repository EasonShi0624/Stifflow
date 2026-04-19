"""Helpers for running smoke scripts under the project interpreter."""

from __future__ import annotations

import os
import sys
from pathlib import Path


STIFFLOW_PYTHON = Path("/scratch/ys6132/programs/miniforge3/envs/stifflow/bin/python")


def ensure_stifflow_python() -> None:
    """Re-exec the current script with the stifflow interpreter when needed."""

    current = Path(sys.executable).resolve()
    target = STIFFLOW_PYTHON.resolve()
    if current == target:
        return
    if os.environ.get("STIFFLOW_REEXEC_ATTEMPTED") == "1":
        raise RuntimeError(
            f"Expected stifflow python at {target}, but re-exec is already in progress "
            f"and current interpreter is {current}."
        )
    if not target.exists():
        raise RuntimeError(f"Required stifflow interpreter does not exist: {target}")

    env = os.environ.copy()
    env["STIFFLOW_REEXEC_ATTEMPTED"] = "1"
    os.execve(str(target), [str(target), *sys.argv], env)
