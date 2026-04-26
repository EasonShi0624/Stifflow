#!/usr/bin/env python
"""Write a bounded xTB/GFN2-xTB subset design without running xTB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.physical_benchmark import (  # noqa: E402
    DEFAULT_XTB_PLAN_JSON,
    DEFAULT_XTB_PLAN_MD,
    write_xtb_subset_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_XTB_PLAN_JSON),
        help="Path for the xTB subset design JSON.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_XTB_PLAN_MD),
        help="Path for the xTB subset design markdown table.",
    )
    parser.add_argument(
        "--samples-per-stratum",
        type=int,
        default=2,
        help="Bounded planned sample count per split/method/seed/rotatable-bin/partition stratum.",
    )
    args = parser.parse_args()

    outputs = write_xtb_subset_plan(
        json_path=args.output_json,
        markdown_path=args.output_md,
        samples_per_stratum=args.samples_per_stratum,
    )
    print(f"xtb_subset_plan_json: {outputs['json']}")
    print(f"xtb_subset_plan_markdown: {outputs['markdown']}")
    print("xtb_execution_status: not_run")
    print("scheduler_status: not_submitted")
    print("benchmark_result: false")
    print("note: this writes a subset design only; ask before running any xTB/GFN2-xTB job.")


if __name__ == "__main__":
    main()
