#!/usr/bin/env python
"""Plan a bounded final-optimized GEOM AMR/COV subset without running evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import (  # noqa: E402
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SCOPE,
    FULL_DATABASE_SPLIT_PATH,
)
from src.eval.amr_cov import MoleculeSkip, _load_reference_molecule  # noqa: E402
from src.eval.final_optimized_all_methods import (  # noqa: E402
    METHODS,
    METHOD_LABELS,
    SEEDS,
    _candidate_paths,
)


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_optimized_amr_cov_subset_plan.md")
SPLIT_PATHS = {"random": FULL_DATABASE_SPLIT_PATH, "scaffold": FULL_DATABASE_SCAFFOLD_SPLIT_PATH}
PARTITIONS = ("val", "test")
ROTATABLE_BINS = (
    ("0", 0, 0),
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7+", 7, None),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("random", "scaffold"), default=["random", "scaffold"])
    parser.add_argument("--partitions", nargs="+", choices=("val", "test"), default=list(PARTITIONS))
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--molecules-per-stratum", type=int, default=10)
    parser.add_argument("--generated-conformers-per-mol", type=int, default=20)
    parser.add_argument("--min-reference-conformers", type=int, default=2)
    parser.add_argument("--max-reference-conformers-per-mol", type=int, default=40)
    parser.add_argument("--max-pickle-bytes", type=int, default=25_000_000)
    parser.add_argument("--max-scan-per-stratum", type=int, default=10000)
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    plan = build_plan(args)
    plan_path = Path(args.plan_path)
    report_path = Path(args.report_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(plan), encoding="utf-8")
    print(f"plan: {plan_path}")
    print(f"report: {report_path}")
    print("benchmark_result: false")
    print("note: Planning-only bounded AMR/COV subset; no model sampling, training, scheduler submission, raw-data mutation, or environment mutation.")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    raw_root = Path(args.raw_root)
    selections: list[dict[str, Any]] = []
    skips_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stratum_summaries = []
    for split_type in args.splits:
        split = _load_split(SPLIT_PATHS[split_type], split_type)
        for partition in args.partitions:
            candidate_ids = list(split[partition])
            for bin_label, low, high in ROTATABLE_BINS:
                stratum_key = f"{split_type}/{partition}/{bin_label}"
                shuffled = _shuffled_ids(candidate_ids, args.subset_seed, stratum_key)
                selected_for_stratum: list[dict[str, Any]] = []
                scanned = 0
                for molecule_id in shuffled:
                    if scanned >= int(args.max_scan_per_stratum):
                        break
                    scanned += 1
                    try:
                        molecule = _load_reference_molecule(
                            molecule_id=molecule_id,
                            raw_root=raw_root,
                            max_reference_conformers=int(args.max_reference_conformers_per_mol),
                            max_pickle_bytes=int(args.max_pickle_bytes),
                        )
                    except MoleculeSkip as exc:
                        _record_skip(skips_by_stratum, stratum_key, molecule_id, exc.reason, exc.detail)
                        continue
                    rotatable = molecule.record.get("rotatable_bond_count")
                    if rotatable is None or not _in_bin(int(rotatable), low, high):
                        continue
                    if molecule.used_reference_conformer_count < int(args.min_reference_conformers):
                        _record_skip(
                            skips_by_stratum,
                            stratum_key,
                            molecule_id,
                            "insufficient_reference_conformers",
                            f"{molecule.used_reference_conformer_count} < {int(args.min_reference_conformers)}",
                        )
                        continue
                    row = {
                        "selection_index": len(selections),
                        "split_type": split_type,
                        "partition": partition,
                        "rotatable_bin": bin_label,
                        "molecule_id": molecule.molecule_id,
                        "smiles": molecule.smiles,
                        "source_path": molecule.source_path,
                        "atom_count": len(molecule.anchor_coords),
                        "heavy_atom_count": len(molecule.heavy_atom_indices),
                        "rotatable_bond_count": int(rotatable),
                        "reference_conformer_count_available": molecule.available_reference_conformer_count,
                        "reference_conformer_count_used": molecule.used_reference_conformer_count,
                        "reference_conformer_count_skipped": molecule.skipped_reference_conformer_count,
                        "raw_pickle_bytes": molecule.raw_pickle_bytes,
                    }
                    selections.append(row)
                    selected_for_stratum.append(row)
                    if len(selected_for_stratum) >= int(args.molecules_per_stratum):
                        break
                stratum_summaries.append(
                    {
                        "split_type": split_type,
                        "partition": partition,
                        "rotatable_bin": bin_label,
                        "target_molecule_count": int(args.molecules_per_stratum),
                        "selected_molecule_count": len(selected_for_stratum),
                        "scanned_candidate_count": scanned,
                        "skip_count_recorded": len(skips_by_stratum[stratum_key]),
                        "complete": len(selected_for_stratum) >= int(args.molecules_per_stratum),
                    }
                )

    molecule_counts = Counter(row["molecule_id"] for row in selections)
    missing_training = _missing_training_results()
    total_slots = len(selections)
    plan = {
        "summary_name": "geom_full_database_final_optimized_amr_cov_subset_plan",
        "result_kind": "geom_drugs_full_database_final_optimized_amr_cov_subset_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_amr_cov_subset_planning",
        "hardware": "reporting_only",
        "gpu_used_by_plan": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_plan": False,
        "sampling_executed_by_plan": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "raw_root": str(raw_root),
        "split_paths": {split: str(SPLIT_PATHS[split]) for split in args.splits},
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "partitions": list(args.partitions),
        "rotatable_bins": [label for label, _low, _high in ROTATABLE_BINS],
        "subset_design": {
            "subset_seed": int(args.subset_seed),
            "molecules_per_stratum": int(args.molecules_per_stratum),
            "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
            "min_reference_conformers": int(args.min_reference_conformers),
            "max_reference_conformers_per_molecule": int(args.max_reference_conformers_per_mol),
            "max_pickle_bytes": int(args.max_pickle_bytes),
            "max_scan_per_stratum": int(args.max_scan_per_stratum),
        },
        "coverage": {
            "stratum_count": len(stratum_summaries),
            "complete_stratum_count": sum(1 for row in stratum_summaries if row["complete"]),
            "selected_molecule_slots": total_slots,
            "unique_molecule_ids": len(molecule_counts),
            "duplicate_molecule_id_count": sum(1 for count in molecule_counts.values() if count > 1),
            "expected_selected_molecule_slots": len(args.splits)
            * len(args.partitions)
            * len(ROTATABLE_BINS)
            * int(args.molecules_per_stratum),
        },
        "estimated_evaluation_size": {
            "selected_molecule_slots": total_slots,
            "method_count": len(METHODS),
            "seed_count": len(SEEDS),
            "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
            "estimated_generated_conformer_count": total_slots
            * len(METHODS)
            * len(SEEDS)
            * int(args.generated_conformers_per_mol),
        },
        "training_result_readiness": {
            "expected_result_count": len(METHODS) * len(SEEDS) * len(args.splits),
            "missing_result_count": len(missing_training),
            "missing_results": missing_training,
            "evaluation_ready": len(missing_training) == 0,
        },
        "strata": stratum_summaries,
        "selected_molecules": selections,
        "skips_by_stratum": dict(skips_by_stratum),
        "output_targets": {
            "future_summary": "experiments/results/geom_full_database_final_optimized_amr_cov_subset_summary.json",
            "future_report": "reports/tables/geom_full_database_final_optimized_amr_cov_subset_summary.md",
            "future_per_molecule_jsonl": "experiments/results/geom_full_database_final_optimized_amr_cov_subset_per_molecule.jsonl",
        },
        "future_evaluation_command": {
            "local_command": (
                "python scripts/evaluate_geom_full_database_final_optimized_amr_cov_subset.py "
                "--plan-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json "
                "--summary-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_summary.json "
                "--report-path reports/tables/geom_full_database_final_optimized_amr_cov_subset_summary.md "
                "--per-molecule-jsonl-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_per_molecule.jsonl "
                "--device cpu --delta 0.75 --progress-every 25"
            ),
            "recommended_sbatch_command": (
                "sbatch --account=torch_pr_872_general "
                "scripts/slurm_geom_full_database_final_optimized_amr_cov_subset.sbatch"
            ),
            "scheduler_submission_recommended": len(missing_training) == 0,
            "scheduler_submission_requires_user_approval": True,
            "recommended_account": "torch_pr_872_general",
            "recommendation_reason": (
                "Use CPU-only torch_pr_872_general after the optimized all-method matrix is complete; "
                "the bounded subset is evaluation-only but reloads checkpoints and samples 20 conformers per selected molecule."
            ),
        },
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a bounded subset plan, not benchmark evaluation.",
            "Raw GEOM pickle files were read only to select molecules with usable reference conformer ensembles and rotatable-bond bins.",
            "Generated conformer ensembles are not stored in the training outputs; the future evaluator must reload checkpoints and sample K conformers per selected molecule.",
            "Do not run the evaluation SLURM job until the final optimized all-method matrix is complete and the user approves.",
        ],
    }
    return plan


def _load_split(path: Path, split_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_type") != split_type:
        raise ValueError(f"Unexpected split type in {path}")
    if payload.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Unexpected split scope in {path}")
    return payload


def _shuffled_ids(ids: list[str], subset_seed: int, stratum_key: str) -> list[str]:
    output = list(ids)
    seed = int(subset_seed) + sum(ord(char) for char in stratum_key) * 1009
    random.Random(seed).shuffle(output)
    return output


def _in_bin(rotatable: int, low: int, high: int | None) -> bool:
    if high is None:
        return rotatable >= low
    return low <= rotatable <= high


def _record_skip(
    skips_by_stratum: dict[str, list[dict[str, Any]]],
    stratum_key: str,
    molecule_id: str,
    reason: str,
    detail: str | None,
) -> None:
    if len(skips_by_stratum[stratum_key]) >= 50:
        return
    skips_by_stratum[stratum_key].append(
        {"molecule_id": molecule_id, "reason": reason, "detail": detail}
    )


def _missing_training_results() -> list[dict[str, Any]]:
    missing = []
    root = Path("experiments/results")
    for split_type in ("random", "scaffold"):
        for seed in SEEDS:
            for method in METHODS:
                candidates = _candidate_paths(root, split_type=split_type, seed=seed, method=method)
                if not any(item["path"].exists() for item in candidates):
                    missing.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "method": method,
                            "expected_paths": [str(item["path"]) for item in candidates],
                        }
                    )
    return missing


def _markdown_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Final Optimized AMR/COV Subset Plan",
        "",
        "Planning-only artifact for bounded final optimized AMR/COV evaluation. This is not benchmark evaluation.",
        "",
        "## Scope",
        "",
        f"- Selected molecule slots: `{plan['coverage']['selected_molecule_slots']}` / `{plan['coverage']['expected_selected_molecule_slots']}`",
        f"- Unique molecule IDs: `{plan['coverage']['unique_molecule_ids']}`",
        f"- Complete strata: `{plan['coverage']['complete_stratum_count']}` / `{plan['coverage']['stratum_count']}`",
        f"- Generated conformers per molecule: `{plan['subset_design']['generated_conformers_per_molecule']}`",
        f"- Estimated generated conformers: `{plan['estimated_evaluation_size']['estimated_generated_conformer_count']}`",
        f"- Missing optimized training results: `{plan['training_result_readiness']['missing_result_count']}`",
        f"- Evaluation ready: `{plan['training_result_readiness']['evaluation_ready']}`",
        f"- Scheduler submission recommended: `{plan['future_evaluation_command']['scheduler_submission_recommended']}`",
        "",
        "## Strata",
        "",
        "| Split | Partition | Rotatable bin | Selected | Target | Scanned | Complete |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in plan["strata"]:
        lines.append(
            f"| {row['split_type']} | {row['partition']} | {row['rotatable_bin']} | "
            f"{row['selected_molecule_count']} | {row['target_molecule_count']} | "
            f"{row['scanned_candidate_count']} | {row['complete']} |"
        )
    lines.extend(
        [
            "",
            "## Future Evaluation Command",
            "",
            "Do not run without explicit approval.",
            "",
            "```bash",
            plan["future_evaluation_command"]["local_command"],
            "```",
            "",
            "Recommended scheduler command after readiness is true:",
            "",
            "```bash",
            plan["future_evaluation_command"]["recommended_sbatch_command"],
            "```",
            "",
            "## Missing Training Results",
            "",
            "```json",
            json.dumps(plan["training_result_readiness"]["missing_results"], indent=2, sort_keys=True),
            "```",
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{plan['benchmark_result']}`",
            f"- `training_executed_by_plan`: `{plan['training_executed_by_plan']}`",
            f"- `sampling_executed_by_plan`: `{plan['sampling_executed_by_plan']}`",
            f"- `raw_data_modified`: `{plan['raw_data_modified']}`",
            f"- `environment_modified`: `{plan['environment_modified']}`",
            "",
            "No jobs were submitted and no model sampling, training, benchmark evaluation, raw-data edits, or environment changes were run by this plan.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
