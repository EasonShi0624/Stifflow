#!/usr/bin/env python
"""Compare full-database GEOM-Drugs formal seed0 result JSONs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_full_database_compare import (
    DEFAULT_REPORT_PATH,
    DEFAULT_RESULT_PATHS,
    DEFAULT_SUMMARY_PATH,
    write_formal_full_database_seed0_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        dest="results",
        help="Input result JSON. Pass twice to override the default isotropic/stiffness full-database result paths.",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="Output JSON summary path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output markdown report path.")
    args = parser.parse_args()

    result_paths = [Path(path) for path in args.results] if args.results else DEFAULT_RESULT_PATHS
    paths = write_formal_full_database_seed0_comparison(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
