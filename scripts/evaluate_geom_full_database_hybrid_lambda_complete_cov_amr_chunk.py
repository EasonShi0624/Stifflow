#!/usr/bin/env python
"""Evaluate one hybrid-lambda complete val/test COV/AMR chunk."""

from __future__ import annotations

import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_geom_full_database_final_cov_amr_complete_valtest_chunk as base


base.DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_plan.json")
base.PLAN_RESULT_KIND = "geom_drugs_full_database_hybrid_lambda_complete_cov_amr_plan_not_benchmark"
base.CHUNK_RESULT_KIND = "geom_drugs_full_database_hybrid_lambda_complete_cov_amr_chunk_not_benchmark"


if __name__ == "__main__":
    base.main()
