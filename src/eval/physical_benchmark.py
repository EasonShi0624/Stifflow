"""Evaluation-only physical benchmark planning and diagnostics.

The default public entry points in this module write a non-benchmark plan.
The diagnostic runners are deliberately gated by scripts so they are not
accidentally used as training, preprocessing, or job-submission steps.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.train import tiny_real as tiny_real_data
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
CHECKPOINTS_DIR = ROOT / "experiments" / "checkpoints"
REPORTS_TABLE_DIR = ROOT / "reports" / "tables"
PROCESSED_ARTIFACT = ROOT / "data" / "processed" / "geom_drugs_full_database" / "molecules.json"
PROCESSED_MANIFEST = ROOT / "data" / "processed" / "geom_drugs_full_database" / "manifest.json"
RANDOM_SPLIT = ROOT / "data" / "splits" / "geom_drugs_full_database" / "random_seed0" / "split.json"
SCAFFOLD_SPLIT = ROOT / "data" / "splits" / "geom_drugs_full_database" / "scaffold_seed0" / "split.json"

DEFAULT_PLAN_JSON = RESULTS_DIR / "geom_full_database_physical_benchmark_plan.json"
DEFAULT_PLAN_MD = REPORTS_TABLE_DIR / "geom_full_database_physical_benchmark_plan.md"
DEFAULT_XTB_PLAN_JSON = RESULTS_DIR / "geom_full_database_xtb_subset_plan.json"
DEFAULT_XTB_PLAN_MD = REPORTS_TABLE_DIR / "geom_full_database_xtb_subset_plan.md"
DEFAULT_PHYSICAL_OUTPUT_DIR = RESULTS_DIR / "geom_full_database_physical_benchmark"
DEFAULT_PHYSICAL_SUMMARY_JSON = RESULTS_DIR / "geom_full_database_physical_benchmark_summary.json"
DEFAULT_PHYSICAL_SUMMARY_MD = REPORTS_TABLE_DIR / "geom_full_database_physical_benchmark_summary.md"

SPLITS = ("random", "scaffold")
METHODS = ("isotropic", "harmonic", "torsion_only", "stiffness_structured")
SEEDS = (0, 1, 2)
ROTATABLE_BINS = (
    {"label": "0", "min_inclusive": 0, "max_inclusive": 0},
    {"label": "1-3", "min_inclusive": 1, "max_inclusive": 3},
    {"label": "4-6", "min_inclusive": 4, "max_inclusive": 6},
    {"label": "7+", "min_inclusive": 7, "max_inclusive": None},
)
PARTITIONS = ("train", "val", "test")

METHOD_TO_FILE_LABEL = {
    "isotropic": "isotropic",
    "harmonic": "harmonic",
    "harmonic_prior": "harmonic",
    "torsion_only": "torsion_only",
    "stiffness_structured": "stiffness",
}
METHOD_TO_CANONICAL = {
    "isotropic": "isotropic",
    "harmonic": "harmonic",
    "harmonic_prior": "harmonic",
    "torsion_only": "torsion_only",
    "stiffness": "stiffness_structured",
    "stiffness_structured": "stiffness_structured",
}
CONFIG_BY_SPLIT_AND_METHOD = {
    ("random", "isotropic"): ROOT / "configs" / "train" / "geom_full_database_formal_isotropic.yaml",
    ("random", "harmonic"): ROOT / "configs" / "train" / "geom_full_database_formal_harmonic.yaml",
    ("random", "torsion_only"): ROOT / "configs" / "train" / "geom_full_database_formal_torsion_only.yaml",
    ("random", "stiffness_structured"): ROOT / "configs" / "train" / "geom_full_database_formal_stiffness.yaml",
    ("scaffold", "isotropic"): ROOT / "configs" / "train" / "geom_full_database_scaffold_formal_isotropic.yaml",
    ("scaffold", "harmonic"): ROOT / "configs" / "train" / "geom_full_database_scaffold_formal_harmonic.yaml",
    ("scaffold", "torsion_only"): ROOT / "configs" / "train" / "geom_full_database_scaffold_formal_torsion_only.yaml",
    ("scaffold", "stiffness_structured"): ROOT
    / "configs"
    / "train"
    / "geom_full_database_scaffold_formal_stiffness.yaml",
}
COORDINATE_FIELD_NAMES = (
    "sampled_coords",
    "sampled_coordinates",
    "generated_coords",
    "generated_coordinates",
    "predicted_coords",
    "predicted_coordinates",
)
_POSEBUSTERS_MOL_RUNNER: Any | None = None


def full_database_result_path(split: str, method: str, seed: int) -> Path:
    """Return the canonical result JSON path for a completed formal run."""

    split = _normalize_split(split)
    method = _normalize_method(method)
    split_label = "" if split == "random" else "_scaffold"
    file_label = METHOD_TO_FILE_LABEL[method]
    return (
        RESULTS_DIR
        / f"geom_formal_full_database{split_label}_train_{file_label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
    )


def full_database_checkpoint_path(split: str, method: str, seed: int, role: str = "best") -> Path:
    """Return the canonical checkpoint path for a completed formal run."""

    split = _normalize_split(split)
    method = _normalize_method(method)
    split_label = "" if split == "random" else "_scaffold"
    file_label = METHOD_TO_FILE_LABEL[method]
    if role not in {"best", "latest"}:
        raise ValueError("checkpoint role must be 'best' or 'latest'")
    return (
        CHECKPOINTS_DIR
        / f"geom_formal_full_database{split_label}_train_{file_label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}_{role}.pt"
    )


def default_full_database_run_matrix() -> list[dict[str, Any]]:
    """Describe the expected completed full-database run matrix."""

    rows = []
    for split in SPLITS:
        for method in METHODS:
            for seed in SEEDS:
                result_path = full_database_result_path(split, method, seed)
                best_checkpoint_path = full_database_checkpoint_path(split, method, seed, role="best")
                latest_checkpoint_path = full_database_checkpoint_path(split, method, seed, role="latest")
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "result_json": _path_status(result_path),
                        "best_checkpoint": _path_status(best_checkpoint_path),
                        "latest_checkpoint": _path_status(latest_checkpoint_path),
                        "config": _path_status(CONFIG_BY_SPLIT_AND_METHOD[(split, method)]),
                    }
                )
    return rows


def build_physical_benchmark_plan() -> dict[str, Any]:
    """Build a reproducible non-benchmark plan from existing outputs only."""

    run_matrix = default_full_database_run_matrix()
    tool_availability = check_tool_availability()
    coordinate_scans = [
        scan_result_for_coordinate_fields(full_database_result_path("random", "harmonic", 0)),
        scan_result_for_coordinate_fields(full_database_result_path("scaffold", "stiffness_structured", 0)),
    ]
    all_results_exist = all(row["result_json"]["exists"] for row in run_matrix)
    all_best_checkpoints_exist = all(row["best_checkpoint"]["exists"] for row in run_matrix)
    result_coordinate_fields_found = sorted(
        {
            field
            for scan in coordinate_scans
            for field in scan.get("coordinate_fields_found", [])
        }
    )
    sampled_geometry_available_in_results = bool(result_coordinate_fields_found)

    return {
        "plan_name": "geom_full_database_physical_benchmark_plan",
        "result_kind": "physical_benchmark_plan_not_benchmark",
        "benchmark_result": False,
        "generated_from_existing_outputs_only": True,
        "training_executed_by_plan": False,
        "preprocessing_executed_by_plan": False,
        "scheduler_submitted_by_plan": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "tool_availability": tool_availability,
        "planned_inputs": {
            "governing_documents_read": [
                "AGENTS.md",
                "README.md",
                "TASKS.md",
            ],
            "processed_artifact": _path_status(PROCESSED_ARTIFACT),
            "processed_manifest": _path_status(PROCESSED_MANIFEST),
            "split_files": {
                "random": _path_status(RANDOM_SPLIT),
                "scaffold": _path_status(SCAFFOLD_SPLIT),
            },
            "run_matrix": run_matrix,
        },
        "current_output_availability": {
            "all_expected_result_jsons_found": all_results_exist,
            "all_expected_best_checkpoints_found": all_best_checkpoints_exist,
            "existing_proxy_diagnostics_available": [
                "sampled_finite_coordinate",
                "sampled_mean_abs_bond_length_delta",
                "sampled_bond_count_checked",
                "sampled_rmsd_to_target",
                "rotatable_bond_count",
            ],
            "sampled_coordinates_saved_in_result_jsons": sampled_geometry_available_in_results,
            "coordinate_field_scan": coordinate_scans,
            "implication": (
                "RDKit/PoseBusters/MMFF/UFF diagnostics require an evaluation-only reload of existing "
                "checkpoints to recreate sampled coordinates in memory; the completed result JSONs do not "
                "appear to save generated coordinate arrays."
                if not sampled_geometry_available_in_results
                else "At least one coordinate-like field was detected; inspect before recomputing samples."
            ),
        },
        "diagnostic_stages": _diagnostic_stages(tool_availability),
        "diagnostics_computable_from_existing_outputs": {
            "without_checkpoint_resampling": [
                "finite-coordinate proxy rate already stored in result JSON summaries",
                "mean absolute sampled bond-length delta to target already stored in result JSON summaries",
                "rotatable-bond stratified RMSD and bond-delta summaries already stored in reports",
            ],
            "with_evaluation_only_checkpoint_reload": [
                "PoseBusters mol-config checks or RDKit fallback intramolecular validity",
                "MMFF/UFF setup availability, energies, minimization deltas, and gradient-norm proxies",
                "per-method/split/seed pass/fail summaries",
            ],
            "requires_scheduler_for_full_matrix": [
                "full random/scaffold x 4 methods x seeds 0/1/2 RDKit/PoseBusters/MMFF/UFF pass over 304339 molecules per run",
            ],
            "requires_separate_approval": [
                "any xTB/GFN2-xTB execution",
                "any SLURM job submission",
                "any full-database physical benchmark execution",
            ],
        },
        "scheduler_recommendation": {
            "full_rdkit_posebusters_mmff_uff": {
                "recommended_account": "torch_pr_872_general",
                "hardware": "cpu_only",
                "why": (
                    "This is an evaluation-only CPU chemistry workload over many existing generated structures; "
                    "it does not need A100-class GPU resources."
                ),
                "scale": "real full-database evaluation if all methods/splits/seeds are run",
                "submission_status": "not_submitted",
            },
            "xtb_subset": {
                "recommended_account": "torch_pr_226_chemistry",
                "hardware": "cpu_only",
                "why": (
                    "xTB is chemistry-heavy even on a small subset; keep it bounded and run only after the subset "
                    "manifest is reviewed."
                ),
                "scale": "small stratified subset only",
                "submission_status": "not_submitted",
            },
        },
        "xtb_subset_design": build_xtb_subset_design(samples_per_stratum=2),
        "outputs_prepared_by_this_plan": {
            "plan_json": str(_relative(DEFAULT_PLAN_JSON)),
            "plan_markdown": str(_relative(DEFAULT_PLAN_MD)),
            "xtb_subset_plan_json": str(_relative(DEFAULT_XTB_PLAN_JSON)),
            "xtb_subset_plan_markdown": str(_relative(DEFAULT_XTB_PLAN_MD)),
            "evaluation_module": "src/eval/physical_benchmark.py",
            "evaluation_script": "scripts/evaluate_geom_full_database_physical_benchmark.py",
            "xtb_subset_planning_script": "scripts/plan_geom_full_database_xtb_subset.py",
            "slurm_script_physical": "scripts/slurm_geom_full_database_physical_benchmark.sbatch",
            "slurm_script_xtb_subset_plan": "scripts/slurm_geom_full_database_xtb_subset.sbatch",
        },
    }


def _diagnostic_stages(tool_availability: dict[str, Any]) -> list[dict[str, Any]]:
    posebusters_available = bool(tool_availability["posebusters"]["available"])
    rdkit_available = bool(tool_availability["rdkit"]["available"])
    mmff_available = bool(tool_availability["rdkit"]["mmff_available"])
    uff_available = bool(tool_availability["rdkit"]["uff_available"])
    xtb_available = bool(tool_availability["xtb"]["available"])

    return [
        {
            "stage": 0,
            "name": "existing_proxy_diagnostics",
            "status": "already_available",
            "source": "completed result JSON summaries",
            "diagnostics": [
                "finite-coordinate rate",
                "mean absolute sampled bond-length delta to target",
                "sampled RMSD by split partition and rotatable-bond bin",
            ],
            "benchmark_status": "proxy_only_not_full_physical_benchmark",
        },
        {
            "stage": 1,
            "name": "coordinate_reconstruction_from_existing_checkpoints",
            "status": "prepared_not_run",
            "source": "existing processed artifact, split files, best checkpoints, and result metadata",
            "diagnostics": [
                "evaluation-only one-step sampled coordinates recreated in memory",
                "sample identity validation against stored molecule_id/split/sample_step_size where possible",
            ],
            "requires_training": False,
            "requires_preprocessing": False,
            "requires_scheduler_for_full_matrix": True,
        },
        {
            "stage": 2,
            "name": "posebusters_style_intramolecular_validity",
            "status": "prepared_not_run",
            "preferred_tool": "PoseBusters config='mol'" if posebusters_available else "RDKit fallback checks",
            "diagnostics": [
                "finite coordinates",
                "valid atom/bond graph consistency",
                "RDKit sanitization and connectedness",
                "bond length sanity",
                "bond angle sanity where PoseBusters or fallback geometry supports it",
                "stereochemistry/chirality warnings where RDKit can identify them",
                "ring geometry warnings where PoseBusters or fallback geometry supports them",
                "severe clash or abnormal-distance checks",
                "pass/fail summary per method/split/seed",
            ],
            "tool_available": posebusters_available or rdkit_available,
        },
        {
            "stage": 3,
            "name": "mmff_uff_energy_force_diagnostics",
            "status": "prepared_not_run",
            "diagnostics": [
                "MMFF optimization availability",
                "UFF optimization availability",
                "MMFF energy before/after minimization when setup succeeds",
                "UFF energy before/after minimization when setup succeeds",
                "MMFF/UFF gradient-norm proxy from CalcGrad when available",
                "failure counts and failure reasons",
                "aggregate means and quantiles per method/split/seed",
            ],
            "mmff_available_in_active_environment": mmff_available,
            "uff_available_in_active_environment": uff_available,
        },
        {
            "stage": 4,
            "name": "xtb_gfn2_subset",
            "status": "plan_only_not_run",
            "diagnostics": [
                "GFN2-xTB energy and warning/failure diagnostics on a small stratified subset only",
                "subset manifest must be written and reviewed before execution",
            ],
            "xtb_executable_available": xtb_available,
            "full_database_xtb": "forbidden_by_plan",
            "requires_separate_approval_before_execution": True,
        },
    ]


def write_physical_benchmark_plan(
    json_path: str | Path = DEFAULT_PLAN_JSON,
    markdown_path: str | Path = DEFAULT_PLAN_MD,
) -> dict[str, Path]:
    """Write the non-benchmark physical evaluation plan."""

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    plan = build_physical_benchmark_plan()
    _write_json(json_path, plan)
    _write_text(markdown_path, render_physical_benchmark_plan_markdown(plan))
    return {"json": json_path, "markdown": markdown_path}


def render_physical_benchmark_plan_markdown(plan: dict[str, Any]) -> str:
    """Render the physical plan as a compact report table."""

    tools = plan["tool_availability"]
    lines = [
        "# GEOM Full-Database Physical Benchmark Plan",
        "",
        "**Status:** planning output only; not a benchmark result.",
        "",
        "This document prepares a staged physical evaluation using completed outputs only. It does not rerun "
        "training, regenerate processed artifacts, modify raw data, install packages, submit jobs, or execute xTB.",
        "",
        "## Tool Availability",
        "",
        "| Tool | Available | Detail |",
        "|---|---:|---|",
        f"| PoseBusters | {_yes_no(tools['posebusters']['available'])} | {tools['posebusters'].get('detail', '')} |",
        f"| RDKit | {_yes_no(tools['rdkit']['available'])} | version={tools['rdkit'].get('version')} |",
        f"| RDKit MMFF | {_yes_no(tools['rdkit']['mmff_available'])} | {tools['rdkit'].get('mmff_detail', '')} |",
        f"| RDKit UFF | {_yes_no(tools['rdkit']['uff_available'])} | {tools['rdkit'].get('uff_detail', '')} |",
        f"| xTB executable | {_yes_no(tools['xtb']['available'])} | {tools['xtb'].get('path') or tools['xtb'].get('detail', '')} |",
        "",
        "## Current Output Availability",
        "",
        f"- Expected result JSONs found: {_yes_no(plan['current_output_availability']['all_expected_result_jsons_found'])}.",
        f"- Expected best checkpoints found: {_yes_no(plan['current_output_availability']['all_expected_best_checkpoints_found'])}.",
        "- Existing proxy diagnostics: finite-coordinate rate, mean absolute bond-length delta, sampled RMSD, "
        "and rotatable-bond metadata.",
        "- Saved generated coordinate arrays in result JSONs: "
        f"{_yes_no(plan['current_output_availability']['sampled_coordinates_saved_in_result_jsons'])}.",
        f"- Implication: {plan['current_output_availability']['implication']}",
        "",
        "## Staged Diagnostics",
        "",
        "| Stage | Name | Status | Main output |",
        "|---:|---|---|---|",
    ]
    for stage in plan["diagnostic_stages"]:
        diagnostics = "; ".join(stage["diagnostics"][:3])
        if len(stage["diagnostics"]) > 3:
            diagnostics += "; ..."
        lines.append(f"| {stage['stage']} | {stage['name']} | {stage['status']} | {diagnostics} |")

    lines.extend(
        [
            "",
            "## What Can Be Computed From Existing Outputs",
            "",
            "| Category | Diagnostics |",
            "|---|---|",
        ]
    )
    computable = plan["diagnostics_computable_from_existing_outputs"]
    for category, items in computable.items():
        lines.append(f"| {category} | {'; '.join(items)} |")

    xtb = plan["xtb_subset_design"]
    lines.extend(
        [
            "",
            "## xTB / GFN2-xTB Subset Plan",
            "",
            f"- Default samples per stratum: {xtb['samples_per_stratum']}.",
            f"- Planned strata: {xtb['planned_stratum_count']} split x method x rotatable-bin x partition strata.",
            f"- Maximum planned structures before duplicate removal: {xtb['maximum_structures_before_duplicate_removal']}.",
            "- Stratification axes: split, method, seed, rotatable-bond bin, and train/val/test partition.",
            "- Full-database xTB is explicitly excluded. Execution requires separate approval.",
            "",
            "## Scheduler Guidance",
            "",
            "| Workload | Recommended account | Hardware | Status |",
            "|---|---|---|---|",
        ]
    )
    sched = plan["scheduler_recommendation"]
    for workload, item in sched.items():
        lines.append(
            f"| {workload} | {item['recommended_account']} | {item['hardware']} | {item['submission_status']} |"
        )

    missing_results = [
        row
        for row in plan["planned_inputs"]["run_matrix"]
        if not row["result_json"]["exists"] or not row["best_checkpoint"]["exists"]
    ]
    lines.extend(["", "## Missing Inputs", ""])
    if not missing_results:
        lines.append("No missing expected result JSONs or best checkpoints were found in the planned 24-run matrix.")
    else:
        lines.append("| Split | Method | Seed | Missing |")
        lines.append("|---|---|---:|---|")
        for row in missing_results:
            missing = []
            if not row["result_json"]["exists"]:
                missing.append("result_json")
            if not row["best_checkpoint"]["exists"]:
                missing.append("best_checkpoint")
            lines.append(f"| {row['split']} | {row['method']} | {row['seed']} | {', '.join(missing)} |")

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Keep all outputs labeled non-benchmark until the full diagnostic pass is actually run and reviewed.",
            "- Report existing proxy diagnostics separately from RDKit/PoseBusters/MMFF/UFF diagnostics.",
            "- Do not claim physical superiority from finite-coordinate and bond-delta proxies alone.",
            "- Ask before submitting any SLURM job or running xTB.",
            "",
        ]
    )
    return "\n".join(lines)


def check_tool_availability() -> dict[str, Any]:
    """Check available physical-evaluation tools without installing anything."""

    return {
        "posebusters": _check_posebusters(),
        "rdkit": _check_rdkit_forcefields(),
        "xtb": _check_xtb(),
    }


def _check_posebusters() -> dict[str, Any]:
    try:
        from importlib.metadata import version

        from posebusters import PoseBusters

        # Instantiate only; do not run a molecule calculation here.
        PoseBusters(config="mol")
        return {
            "available": True,
            "version": version("posebusters"),
            "config": "mol",
            "detail": "PoseBusters import and mol-config construction succeeded.",
        }
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return {
            "available": False,
            "version": None,
            "config": "mol",
            "detail": f"PoseBusters unavailable or failed to initialize: {type(exc).__name__}: {exc}",
        }


def _check_rdkit_forcefields() -> dict[str, Any]:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        embed_status = int(AllChem.EmbedMolecule(mol, randomSeed=13))
        if embed_status != 0:
            raise RuntimeError(f"RDKit EmbedMolecule failed with status {embed_status}")
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        mmff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props) if mmff_props is not None else None
        uff = AllChem.UFFGetMoleculeForceField(mol)
        mmff_available = mmff is not None
        uff_available = uff is not None
        return {
            "available": True,
            "version": rdBase.rdkitVersion,
            "mmff_available": mmff_available,
            "uff_available": uff_available,
            "mmff_detail": "MMFF properties, force field, energy, and CalcGrad are available."
            if mmff_available and hasattr(mmff, "CalcGrad")
            else "MMFF force field unavailable or CalcGrad missing.",
            "uff_detail": "UFF force field, energy, and CalcGrad are available."
            if uff_available and hasattr(uff, "CalcGrad")
            else "UFF force field unavailable or CalcGrad missing.",
        }
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return {
            "available": False,
            "version": None,
            "mmff_available": False,
            "uff_available": False,
            "mmff_detail": f"RDKit force-field smoke check failed: {type(exc).__name__}: {exc}",
            "uff_detail": f"RDKit force-field smoke check failed: {type(exc).__name__}: {exc}",
        }


def _check_xtb() -> dict[str, Any]:
    path = shutil.which("xtb")
    if path is None:
        return {
            "available": False,
            "path": None,
            "version_text": None,
            "detail": "No xtb executable found on PATH.",
        }
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_text = (result.stdout or result.stderr).strip().splitlines()[:6]
        return {
            "available": True,
            "path": path,
            "version_text": version_text,
            "detail": "xtb executable found; no molecule calculation was run.",
        }
    except Exception as exc:  # pragma: no cover - executable-dependent.
        return {
            "available": True,
            "path": path,
            "version_text": None,
            "detail": f"xtb executable found, but version check failed: {type(exc).__name__}: {exc}",
        }


def scan_result_for_coordinate_fields(path: str | Path) -> dict[str, Any]:
    """Scan a result JSON for coordinate-like field names without loading it all into memory."""

    path = Path(path)
    status = _path_status(path)
    fields_found: set[str] = set()
    if not path.exists():
        return {
            "path": str(_relative(path)),
            "exists": False,
            "coordinate_fields_found": [],
            "bytes_scanned": 0,
        }
    needles = [f'"{field}"'.encode("utf-8") for field in COORDINATE_FIELD_NAMES]
    overlap = max(len(needle) for needle in needles)
    scanned = 0
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            scanned += len(chunk)
            haystack = tail + chunk
            for field, needle in zip(COORDINATE_FIELD_NAMES, needles, strict=True):
                if needle in haystack:
                    fields_found.add(field)
            tail = haystack[-overlap:]
            if len(fields_found) == len(COORDINATE_FIELD_NAMES):
                break
    return {
        "path": str(_relative(path)),
        "exists": status["exists"],
        "size_bytes": status["size_bytes"],
        "coordinate_fields_found": sorted(fields_found),
        "bytes_scanned": scanned,
    }


def build_xtb_subset_design(samples_per_stratum: int = 2) -> dict[str, Any]:
    """Build a bounded xTB subset design without selecting or executing structures."""

    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive")
    strata = []
    for split in SPLITS:
        for method in METHODS:
            for seed in SEEDS:
                for rot_bin in ROTATABLE_BINS:
                    for partition in PARTITIONS:
                        strata.append(
                            {
                                "split": split,
                                "method": method,
                                "seed": seed,
                                "rotatable_bond_bin": rot_bin["label"],
                                "partition": partition,
                                "samples_per_stratum": int(samples_per_stratum),
                            }
                        )
    return {
        "result_kind": "xtb_subset_design_not_executed",
        "benchmark_result": False,
        "xtb_execution_performed": False,
        "full_database_xtb_excluded": True,
        "samples_per_stratum": int(samples_per_stratum),
        "stratification_axes": ["split", "method", "seed", "rotatable_bond_bin", "partition"],
        "rotatable_bond_bins": list(ROTATABLE_BINS),
        "planned_stratum_count": len(strata),
        "maximum_structures_before_duplicate_removal": len(strata) * int(samples_per_stratum),
        "strata": strata,
        "selection_rule": (
            "After approval, select sampled structures from existing evaluation-only coordinate reconstruction, "
            "deduplicate by split/method/seed/molecule_id/conformer_id, and cap each stratum at samples_per_stratum."
        ),
        "execution_guardrail": "Write and review this manifest before any xTB/GFN2-xTB run.",
    }


def write_xtb_subset_plan(
    json_path: str | Path = DEFAULT_XTB_PLAN_JSON,
    markdown_path: str | Path = DEFAULT_XTB_PLAN_MD,
    samples_per_stratum: int = 2,
) -> dict[str, Path]:
    """Write the xTB subset design manifest. This does not run xTB."""

    design = build_xtb_subset_design(samples_per_stratum=samples_per_stratum)
    design["tool_availability"] = {"xtb": _check_xtb()}
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    _write_json(json_path, design)
    _write_text(markdown_path, render_xtb_subset_plan_markdown(design))
    return {"json": json_path, "markdown": markdown_path}


def render_xtb_subset_plan_markdown(design: dict[str, Any]) -> str:
    xtb = design.get("tool_availability", {}).get("xtb", {})
    lines = [
        "# GEOM Full-Database xTB Subset Plan",
        "",
        "**Status:** subset design only; xTB was not executed.",
        "",
        f"- xTB available: {_yes_no(bool(xtb.get('available')))}.",
        f"- xTB path: {xtb.get('path') or 'missing'}.",
        f"- Samples per stratum: {design['samples_per_stratum']}.",
        f"- Planned strata: {design['planned_stratum_count']}.",
        f"- Maximum structures before duplicate removal: {design['maximum_structures_before_duplicate_removal']}.",
        "- Full-database xTB is excluded. Ask before any xTB execution or job submission.",
        "",
        "## Strata",
        "",
        "| Split | Method | Seed | Rotatable-bin | Partition | Planned n |",
        "|---|---|---:|---|---|---:|",
    ]
    for row in design["strata"]:
        lines.append(
            f"| {row['split']} | {row['method']} | {row['seed']} | "
            f"{row['rotatable_bond_bin']} | {row['partition']} | {row['samples_per_stratum']} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_existing_result_physical_subset(
    result_path: str | Path,
    output_path: str | Path,
    max_molecules: int | None,
    device: str = "cpu",
    include_posebusters: bool = True,
    include_forcefields: bool = True,
    allow_full_database: bool = False,
    save_per_molecule: bool = False,
    record_cache: dict[Path, list[dict[str, Any]]] | None = None,
    split_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an explicitly gated evaluation-only physical diagnostic pass.

    This reloads an existing best checkpoint and rebuilds one-step sampled
    coordinates in memory. It does not train or modify the source artifacts.
    """

    if max_molecules is None and not allow_full_database:
        raise ValueError(
            "Refusing to evaluate the full database from the CLI without --allow-full-database. "
            "Pass --max-molecules for a bounded local check or use the prepared SLURM script after approval."
        )
    start = time.time()
    result_path = Path(result_path)
    output_path = Path(output_path)
    payload = _load_result_payload(result_path)
    split = _normalize_split(payload["dataset"]["split_type"])
    method = _normalize_method(payload["corruption_mode"])
    seed = int(payload["random_seed"])
    config = _load_evaluation_config(payload, split=split, method=method)

    artifact_path = ROOT / payload["processed_artifact_path"]
    split_path = ROOT / payload["split_path"]
    checkpoint_path = ROOT / payload.get("best_checkpoint_path", payload["checkpoint_path"])
    records = _cached_records(artifact_path, record_cache)
    if max_molecules is not None:
        records = records[: int(max_molecules)]
    for record in records:
        if record.get("rotatable_bond_count") is None:
            record["rotatable_bond_count"] = tiny_real_data._rotatable_bond_count(record, config)
    split_payload = _cached_split(split_path, split_cache)
    split_membership = _split_membership(split_payload)
    config["eval"]["eval_subset_size"] = len(records)

    rows = []
    aggregate_state = _new_aggregate_state()
    sample_count = 0
    tool_availability = check_tool_availability()
    samples = tiny_real_data._iter_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    record_by_id = {str(record["molecule_id"]): record for record in records}
    for sample in _iter_sampled_coordinates_from_checkpoint(
        samples=samples,
        checkpoint_path=checkpoint_path,
        sample_step_size=float(payload["sample_step_size"]),
        device=device,
        record_by_id=record_by_id,
    ):
        diag = evaluate_molecule_physical_diagnostics(
            record=sample["record"],
            sampled_coords=sample["sampled_coords"],
            target_coords=sample["target_coords"],
            use_posebusters=include_posebusters,
            run_forcefields=include_forcefields,
        )
        sample_count += 1
        _update_aggregate_state(aggregate_state, diag)
        if save_per_molecule:
            rows.append(
                {
                    "molecule_id": sample["molecule_id"],
                    "split_partition": sample["split"],
                    "rotatable_bond_count": sample.get("rotatable_bond_count"),
                    "diagnostics": diag,
                }
            )

    output = {
        "result_kind": "physical_benchmark_diagnostics_not_benchmark_until_reviewed",
        "benchmark_result": False,
        "training_executed": False,
        "preprocessing_executed": False,
        "scheduler_submitted_by_script": False,
        "source_result_path": str(_relative(result_path)),
        "checkpoint_path": str(_relative(checkpoint_path)),
        "processed_artifact_path": str(_relative(artifact_path)),
        "split_path": str(_relative(split_path)),
        "split": split,
        "method": method,
        "seed": seed,
        "max_molecules": max_molecules,
        "sample_count": sample_count,
        "tool_availability": tool_availability,
        "aggregates": _finalize_aggregate_state(aggregate_state),
        "per_molecule": rows if save_per_molecule else None,
        "runtime_seconds": time.time() - start,
        "notes": [
            "Evaluation-only reload of existing checkpoint.",
            "No training, preprocessing, raw-data mutation, environment mutation, xTB execution, or job submission.",
        ],
    }
    _write_json(output_path, output)
    return output


def evaluate_full_database_physical_matrix(
    output_dir: str | Path = DEFAULT_PHYSICAL_OUTPUT_DIR,
    summary_json_path: str | Path = DEFAULT_PHYSICAL_SUMMARY_JSON,
    summary_markdown_path: str | Path = DEFAULT_PHYSICAL_SUMMARY_MD,
    max_molecules_per_result: int | None = None,
    device: str = "cpu",
    include_posebusters: bool = True,
    include_forcefields: bool = True,
    allow_full_database: bool = False,
) -> dict[str, Any]:
    """Run the gated physical diagnostic pass across the 24 completed runs."""

    if max_molecules_per_result is None and not allow_full_database:
        raise ValueError("Matrix evaluation requires max_molecules_per_result or allow_full_database=True")
    start = time.time()
    output_dir = Path(output_dir)
    summary_json_path = Path(summary_json_path)
    summary_markdown_path = Path(summary_markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_cache: dict[Path, list[dict[str, Any]]] = {}
    split_cache: dict[Path, dict[str, Any]] = {}
    per_run = []
    for split in SPLITS:
        for method in METHODS:
            for seed in SEEDS:
                result_path = full_database_result_path(split, method, seed)
                output_path = output_dir / f"geom_full_database_physical_benchmark_{split}_{method}_seed{seed}.json"
                run_output = evaluate_existing_result_physical_subset(
                    result_path=result_path,
                    output_path=output_path,
                    max_molecules=max_molecules_per_result,
                    device=device,
                    include_posebusters=include_posebusters,
                    include_forcefields=include_forcefields,
                    allow_full_database=allow_full_database,
                    save_per_molecule=False,
                    record_cache=record_cache,
                    split_cache=split_cache,
                )
                per_run.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "output_json": str(_relative(output_path)),
                        "sample_count": run_output["sample_count"],
                        "aggregates": run_output["aggregates"],
                        "runtime_seconds": run_output["runtime_seconds"],
                    }
                )

    summary = summarize_physical_matrix(per_run)
    payload = {
        "result_kind": "physical_benchmark_matrix_summary_not_benchmark_until_reviewed",
        "benchmark_result": False,
        "training_executed": False,
        "preprocessing_executed": False,
        "scheduler_submitted_by_script": False,
        "xtb_executed": False,
        "device": device,
        "max_molecules_per_result": max_molecules_per_result,
        "include_posebusters": include_posebusters,
        "include_forcefields": include_forcefields,
        "run_count": len(per_run),
        "per_run": per_run,
        "summary": summary,
        "runtime_seconds": time.time() - start,
        "notes": [
            "CPU-only evaluation from existing result JSONs, processed artifact, split files, and checkpoints.",
            "No training, preprocessing, raw-data mutation, processed-artifact mutation, split mutation, checkpoint mutation, environment mutation, or xTB execution.",
        ],
    }
    _write_json(summary_json_path, payload)
    _write_text(summary_markdown_path, render_physical_matrix_summary_markdown(payload))
    return payload


def summarize_physical_matrix(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """Build method-level summaries and rankings from per-run aggregate outputs."""

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    by_split_method: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {method: [] for method in METHODS} for split in SPLITS
    }
    for row in per_run:
        method = row["method"]
        split = row["split"]
        by_method[method].append(row)
        by_split_method[split][method].append(row)

    method_summary = {method: _summarize_run_group(rows) for method, rows in by_method.items()}
    split_method_summary = {
        split: {method: _summarize_run_group(rows) for method, rows in method_rows.items()}
        for split, method_rows in by_split_method.items()
    }
    ranking = sorted(
        (
            {
                "method": method,
                "overall_intramolecular_pass_rate_mean": values["pass_rate_means"].get(
                    "overall_intramolecular_pass"
                ),
                "mmff_available_rate_mean": values["pass_rate_means"].get("MMFF_available"),
                "uff_available_rate_mean": values["pass_rate_means"].get("UFF_available"),
                "mean_abs_bond_length_delta_to_target_mean": values["numeric_means"].get(
                    "mean_abs_bond_length_delta_to_target"
                ),
            }
            for method, values in method_summary.items()
        ),
        key=lambda row: (
            _sort_desc(row["overall_intramolecular_pass_rate_mean"]),
            _sort_desc(row["mmff_available_rate_mean"]),
            _sort_asc(row["mean_abs_bond_length_delta_to_target_mean"]),
        ),
    )
    return {
        "method_summary": method_summary,
        "split_method_summary": split_method_summary,
        "ranking_by_intramolecular_pass_rate": ranking,
        "ranking_metric_note": (
            "Ranking is descriptive and uses mean overall intramolecular pass rate first, then MMFF availability, "
            "then lower mean absolute bond-length delta. Energies are not normalized by molecule size and should not "
            "be overinterpreted as a universal ranking metric."
        ),
    }


def render_physical_matrix_summary_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# GEOM Full-Database Physical Benchmark Summary",
        "",
        "**Status:** generated physical diagnostics; still labeled non-benchmark until reviewed.",
        "",
        f"- Runs evaluated: {payload['run_count']}.",
        f"- xTB executed: {_yes_no(bool(payload['xtb_executed']))}.",
        f"- Max molecules per result: {payload['max_molecules_per_result'] or 'full selected result'}.",
        f"- Runtime seconds: {payload['runtime_seconds']:.3f}.",
        "",
        "## Method Ranking",
        "",
        "| Rank | Method | Intramolecular pass rate | MMFF available | UFF available | Mean abs bond delta |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["ranking_by_intramolecular_pass_rate"], start=1):
        lines.append(
            f"| {rank} | {row['method']} | {_fmt(row['overall_intramolecular_pass_rate_mean'])} | "
            f"{_fmt(row['mmff_available_rate_mean'])} | {_fmt(row['uff_available_rate_mean'])} | "
            f"{_fmt(row['mean_abs_bond_length_delta_to_target_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"Ranking note: {summary['ranking_metric_note']}",
            "",
            "## Method Aggregates",
            "",
            "| Method | Runs | Samples | Finite coord | Overall pass | RDKit sanitize | MMFF available | UFF available | MMFF failures | UFF failures |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in summary["method_summary"].items():
        passes = values["pass_rate_means"]
        failures = values["failure_reason_counts"]
        mmff_failures = sum(count for reason, count in failures.items() if reason.startswith("MMFF:"))
        uff_failures = sum(count for reason, count in failures.items() if reason.startswith("UFF:"))
        lines.append(
            f"| {method} | {values['run_count']} | {values['sample_count']} | "
            f"{_fmt(passes.get('finite_coordinate_pass'))} | "
            f"{_fmt(passes.get('overall_intramolecular_pass'))} | "
            f"{_fmt(passes.get('rdkit_sanitize_success'))} | "
            f"{_fmt(passes.get('MMFF_available'))} | "
            f"{_fmt(passes.get('UFF_available'))} | "
            f"{mmff_failures} | {uff_failures} |"
        )
    lines.extend(
        [
            "",
            "## Split x Method Aggregates",
            "",
            "| Split | Method | Runs | Samples | Overall pass | MMFF available | UFF available | Mean abs bond delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, methods in summary["split_method_summary"].items():
        for method, values in methods.items():
            passes = values["pass_rate_means"]
            numerics = values["numeric_means"]
            lines.append(
                f"| {split} | {method} | {values['run_count']} | {values['sample_count']} | "
                f"{_fmt(passes.get('overall_intramolecular_pass'))} | "
                f"{_fmt(passes.get('MMFF_available'))} | "
                f"{_fmt(passes.get('UFF_available'))} | "
                f"{_fmt(numerics.get('mean_abs_bond_length_delta_to_target'))} |"
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "These diagnostics are physical-validity diagnostics only. They should be interpreted alongside the "
            "existing RMSD/loss summaries and should not be used to claim overall physical superiority unless "
            "coverage and failure modes support that claim.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_molecule_physical_diagnostics(
    record: dict[str, Any],
    sampled_coords: list[list[float]],
    target_coords: list[list[float]] | None = None,
    use_posebusters: bool = True,
    run_forcefields: bool = True,
) -> dict[str, Any]:
    """Compute RDKit/PoseBusters-style diagnostics for one sampled molecule."""

    diagnostics: dict[str, Any] = {
        "finite_coordinate_pass": _coords_are_finite(sampled_coords),
        "graph_consistency": _graph_consistency(record, sampled_coords),
    }
    mol_result = build_rdkit_mol_from_record(record, sampled_coords, sanitize=True)
    mol = mol_result.get("mol")
    diagnostics["rdkit_molecule"] = {
        "build_success": mol is not None,
        "sanitize_success": bool(mol_result.get("sanitize_success")),
        "failure_reason": mol_result.get("failure_reason"),
    }
    diagnostics["fallback_geometry"] = fallback_geometry_checks(record, sampled_coords, target_coords)
    diagnostics["posebusters"] = (
        run_posebusters_mol_checks(mol) if use_posebusters and mol is not None else {"executed": False}
    )
    diagnostics["forcefields"] = (
        run_rdkit_forcefield_diagnostics(mol) if run_forcefields and mol is not None else {"executed": False}
    )
    diagnostics["overall_intramolecular_pass"] = _overall_intramolecular_pass(diagnostics)
    return diagnostics


def build_rdkit_mol_from_record(
    record: dict[str, Any],
    coords: list[list[float]],
    sanitize: bool = True,
) -> dict[str, Any]:
    """Build an RDKit molecule with one conformer from a processed GEOM record."""

    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except Exception as exc:  # pragma: no cover - optional dependency.
        return {
            "mol": None,
            "sanitize_success": False,
            "failure_reason": f"rdkit_unavailable:{type(exc).__name__}:{exc}",
        }

    try:
        atom_symbols = record.get("atom_symbols") or []
        atomic_numbers = record.get("atomic_numbers") or []
        formal_charges = record.get("formal_charges") or [0 for _ in atom_symbols]
        atom_count = len(atom_symbols) or len(atomic_numbers)
        if len(coords) != atom_count:
            raise ValueError(f"coordinate_count={len(coords)} atom_count={atom_count}")
        editable = Chem.RWMol()
        for index in range(atom_count):
            if atomic_numbers:
                atom = Chem.Atom(int(atomic_numbers[index]))
            else:
                atom = Chem.Atom(str(atom_symbols[index]))
            if index < len(formal_charges) and formal_charges[index] is not None:
                atom.SetFormalCharge(int(formal_charges[index]))
            editable.AddAtom(atom)
        for bond in record.get("bonds", []):
            if len(bond) < 2:
                continue
            i, j = int(bond[0]), int(bond[1])
            if i == j or i < 0 or j < 0 or i >= atom_count or j >= atom_count:
                continue
            editable.AddBond(i, j, _rdkit_bond_type(bond[2] if len(bond) > 2 else 1, Chem))
        mol = editable.GetMol()
        conf = Chem.Conformer(atom_count)
        for index, xyz in enumerate(coords):
            conf.SetAtomPosition(index, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        mol.AddConformer(conf, assignId=True)
        sanitize_success = True
        failure_reason = None
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                sanitize_success = False
                failure_reason = f"sanitize_failed:{type(exc).__name__}:{exc}"
        return {"mol": mol, "sanitize_success": sanitize_success, "failure_reason": failure_reason}
    except Exception as exc:
        return {
            "mol": None,
            "sanitize_success": False,
            "failure_reason": f"build_failed:{type(exc).__name__}:{exc}",
        }


def run_posebusters_mol_checks(mol: Any) -> dict[str, Any]:
    """Run PoseBusters mol-config checks on an RDKit molecule when available."""

    try:
        global _POSEBUSTERS_MOL_RUNNER
        if _POSEBUSTERS_MOL_RUNNER is None:
            from posebusters import PoseBusters

            _POSEBUSTERS_MOL_RUNNER = PoseBusters(config="mol")
        report = _POSEBUSTERS_MOL_RUNNER.bust([mol], full_report=True)
        row = report.iloc[0].to_dict()
        bool_items = {str(key): _to_python_scalar(value) for key, value in row.items()}
        pass_values = [value for value in bool_items.values() if isinstance(value, bool)]
        return {
            "executed": True,
            "config": "mol",
            "all_boolean_checks_passed": all(pass_values) if pass_values else None,
            "checks": bool_items,
        }
    except Exception as exc:  # pragma: no cover - optional dependency/runtime.
        return {
            "executed": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def run_rdkit_forcefield_diagnostics(mol: Any, max_iters: int = 200) -> dict[str, Any]:
    """Compute MMFF/UFF energies and gradient-norm proxies when RDKit supports them."""

    return {
        "executed": True,
        "MMFF": _single_forcefield_diagnostics(mol, "MMFF", max_iters=max_iters),
        "UFF": _single_forcefield_diagnostics(mol, "UFF", max_iters=max_iters),
    }


def fallback_geometry_checks(
    record: dict[str, Any],
    sampled_coords: list[list[float]],
    target_coords: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Closest available RDKit-free intramolecular validity checks."""

    graph = _graph_consistency(record, sampled_coords)
    bond_lengths = _bond_length_checks(record, sampled_coords, target_coords)
    angle_checks = _bond_angle_checks(record, sampled_coords)
    clash_checks = _severe_clash_checks(record, sampled_coords)
    ring_checks = _ring_geometry_checks(record, sampled_coords)
    chirality = _chirality_warning_count(record, sampled_coords)
    return {
        "graph_consistency_pass": graph["pass"],
        "bond_length_sanity_pass": bond_lengths["pass"],
        "bond_lengths": bond_lengths,
        "bond_angle_sanity_pass": angle_checks["pass"],
        "bond_angles": angle_checks,
        "severe_clash_pass": clash_checks["pass"],
        "severe_clashes": clash_checks,
        "ring_geometry": ring_checks,
        "chirality": chirality,
        "fallback_pass": all(
            [
                graph["pass"],
                bond_lengths["pass"],
                angle_checks["pass"],
                clash_checks["pass"],
            ]
        ),
    }


def aggregate_physical_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-molecule physical diagnostics."""

    pass_keys = [
        ("finite_coordinate_pass", lambda d: d.get("finite_coordinate_pass")),
        ("overall_intramolecular_pass", lambda d: d.get("overall_intramolecular_pass")),
        ("rdkit_sanitize_success", lambda d: d.get("rdkit_molecule", {}).get("sanitize_success")),
        (
            "fallback_bond_length_sanity_pass",
            lambda d: d.get("fallback_geometry", {}).get("bond_length_sanity_pass"),
        ),
        (
            "fallback_bond_angle_sanity_pass",
            lambda d: d.get("fallback_geometry", {}).get("bond_angle_sanity_pass"),
        ),
        ("fallback_severe_clash_pass", lambda d: d.get("fallback_geometry", {}).get("severe_clash_pass")),
    ]
    values = {
        "mean_abs_bond_length_delta_to_target": [],
        "max_abs_bond_length_delta_to_target": [],
        "MMFF_energy_initial": [],
        "MMFF_energy_final": [],
        "MMFF_gradient_norm_initial": [],
        "MMFF_gradient_norm_final": [],
        "UFF_energy_initial": [],
        "UFF_energy_final": [],
        "UFF_gradient_norm_initial": [],
        "UFF_gradient_norm_final": [],
    }
    failure_reasons: dict[str, int] = {}
    for row in rows:
        diag = row["diagnostics"]
        bond = diag.get("fallback_geometry", {}).get("bond_lengths", {})
        _append_number(values["mean_abs_bond_length_delta_to_target"], bond.get("mean_abs_delta_to_target"))
        _append_number(values["max_abs_bond_length_delta_to_target"], bond.get("max_abs_delta_to_target"))
        forcefields = diag.get("forcefields", {})
        for name in ("MMFF", "UFF"):
            ff = forcefields.get(name, {})
            _append_number(values[f"{name}_energy_initial"], ff.get("energy_initial"))
            _append_number(values[f"{name}_energy_final"], ff.get("energy_final"))
            _append_number(values[f"{name}_gradient_norm_initial"], ff.get("gradient_norm_initial"))
            _append_number(values[f"{name}_gradient_norm_final"], ff.get("gradient_norm_final"))
            reason = ff.get("failure_reason")
            if reason:
                failure_reasons[f"{name}:{reason}"] = failure_reasons.get(f"{name}:{reason}", 0) + 1

    return {
        "sample_count": len(rows),
        "pass_rates": {
            key: _pass_rate([getter(row["diagnostics"]) for row in rows])
            for key, getter in pass_keys
        },
        "numeric_summaries": {key: _numeric_summary(items) for key, items in values.items()},
        "failure_reasons": failure_reasons,
    }


def _new_aggregate_state() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "pass_counts": {},
        "numeric_values": {
            "mean_abs_bond_length_delta_to_target": [],
            "max_abs_bond_length_delta_to_target": [],
            "MMFF_energy_initial": [],
            "MMFF_energy_final": [],
            "MMFF_gradient_norm_initial": [],
            "MMFF_gradient_norm_final": [],
            "UFF_energy_initial": [],
            "UFF_energy_final": [],
            "UFF_gradient_norm_initial": [],
            "UFF_gradient_norm_final": [],
        },
        "failure_reasons": {},
    }


def _update_aggregate_state(state: dict[str, Any], diag: dict[str, Any]) -> None:
    state["sample_count"] += 1
    pass_values = {
        "finite_coordinate_pass": diag.get("finite_coordinate_pass"),
        "overall_intramolecular_pass": diag.get("overall_intramolecular_pass"),
        "rdkit_sanitize_success": diag.get("rdkit_molecule", {}).get("sanitize_success"),
        "fallback_bond_length_sanity_pass": diag.get("fallback_geometry", {}).get("bond_length_sanity_pass"),
        "fallback_bond_angle_sanity_pass": diag.get("fallback_geometry", {}).get("bond_angle_sanity_pass"),
        "fallback_severe_clash_pass": diag.get("fallback_geometry", {}).get("severe_clash_pass"),
    }
    forcefields = diag.get("forcefields", {})
    for name in ("MMFF", "UFF"):
        ff = forcefields.get(name, {})
        pass_values[f"{name}_available"] = ff.get("available")
        pass_values[f"{name}_optimization_converged"] = ff.get("optimization_converged")
    for key, value in pass_values.items():
        if value is None:
            continue
        counts = state["pass_counts"].setdefault(key, {"count": 0, "passed": 0})
        counts["count"] += 1
        counts["passed"] += int(bool(value))

    bond = diag.get("fallback_geometry", {}).get("bond_lengths", {})
    _append_number(state["numeric_values"]["mean_abs_bond_length_delta_to_target"], bond.get("mean_abs_delta_to_target"))
    _append_number(state["numeric_values"]["max_abs_bond_length_delta_to_target"], bond.get("max_abs_delta_to_target"))
    for name in ("MMFF", "UFF"):
        ff = forcefields.get(name, {})
        _append_number(state["numeric_values"][f"{name}_energy_initial"], ff.get("energy_initial"))
        _append_number(state["numeric_values"][f"{name}_energy_final"], ff.get("energy_final"))
        _append_number(state["numeric_values"][f"{name}_gradient_norm_initial"], ff.get("gradient_norm_initial"))
        _append_number(state["numeric_values"][f"{name}_gradient_norm_final"], ff.get("gradient_norm_final"))
        reason = ff.get("failure_reason")
        if reason:
            label = f"{name}:{reason}"
            state["failure_reasons"][label] = state["failure_reasons"].get(label, 0) + 1


def _finalize_aggregate_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": state["sample_count"],
        "pass_rates": {
            key: {
                "count": values["count"],
                "passed": values["passed"],
                "rate": values["passed"] / values["count"] if values["count"] else None,
            }
            for key, values in sorted(state["pass_counts"].items())
        },
        "numeric_summaries": {
            key: _numeric_summary(values) for key, values in state["numeric_values"].items()
        },
        "failure_reasons": dict(sorted(state["failure_reasons"].items())),
    }


def _cached_records(
    artifact_path: Path,
    record_cache: dict[Path, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    artifact_path = artifact_path.resolve()
    if record_cache is not None and artifact_path in record_cache:
        return record_cache[artifact_path]
    records = tiny_real_data._load_tiny_real_artifact(artifact_path)["molecules"]
    if record_cache is not None:
        record_cache[artifact_path] = records
    return records


def _cached_split(
    split_path: Path,
    split_cache: dict[Path, dict[str, Any]] | None,
) -> dict[str, Any]:
    split_path = split_path.resolve()
    if split_cache is not None and split_path in split_cache:
        return split_cache[split_path]
    split_payload = tiny_real_data._load_tiny_real_split(split_path)
    if split_cache is not None:
        split_cache[split_path] = split_payload
    return split_payload


def _summarize_run_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_keys = sorted(
        {
            key
            for row in rows
            for key in row.get("aggregates", {}).get("pass_rates", {})
        }
    )
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key in row.get("aggregates", {}).get("numeric_summaries", {})
        }
    )
    failure_counts: dict[str, int] = {}
    for row in rows:
        for reason, count in row.get("aggregates", {}).get("failure_reasons", {}).items():
            failure_counts[reason] = failure_counts.get(reason, 0) + int(count)
    return {
        "run_count": len(rows),
        "sample_count": sum(int(row.get("sample_count", 0)) for row in rows),
        "pass_rate_means": {
            key: _mean_present(
                [
                    row.get("aggregates", {}).get("pass_rates", {}).get(key, {}).get("rate")
                    for row in rows
                ]
            )
            for key in pass_keys
        },
        "numeric_means": {
            key: _mean_present(
                [
                    row.get("aggregates", {}).get("numeric_summaries", {}).get(key, {}).get("mean")
                    for row in rows
                ]
            )
            for key in numeric_keys
        },
        "failure_reason_counts": dict(sorted(failure_counts.items())),
    }


def _mean_present(values: list[Any]) -> float | None:
    usable = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    return mean(usable) if usable else None


def _sort_desc(value: Any) -> float:
    return -float(value) if isinstance(value, (int, float)) and value is not None else math.inf


def _sort_asc(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and value is not None else math.inf


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


def _iter_sampled_coordinates_from_checkpoint(
    samples: Any,
    checkpoint_path: str | Path,
    sample_step_size: float,
    device: str = "cpu",
    record_by_id: dict[str, dict[str, Any]] | None = None,
):
    from src.models.checkpoint_sampling_smoke import _load_torch, _resolve_device

    torch = _load_torch()
    device = _resolve_device(torch, device=device, require_cuda=False)
    if device == "cpu":
        torch.set_num_threads(1)
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = torch.nn.Linear(4, 3)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            xt = torch.tensor(sample["xt"], dtype=torch.float32, device=device)
            t_column = torch.full((xt.shape[0], 1), float(sample["t"]), dtype=torch.float32, device=device)
            prediction = model(torch.cat([xt, t_column], dim=1))
            sampled_coords = xt + float(sample_step_size) * prediction
            yield {
                "molecule_id": sample["molecule_id"],
                "split": sample.get("split"),
                "rotatable_bond_count": sample.get("rotatable_bond_count"),
                "record": (record_by_id or {}).get(str(sample["molecule_id"]), _sample_to_record(sample)),
                "sampled_coords": _tensor_coords_to_list(sampled_coords),
                "target_coords": [[float(value) for value in xyz] for xyz in sample["x1"]],
            }


def _sample_to_record(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "molecule_id": sample["molecule_id"],
        "smiles": sample.get("smiles"),
        "atom_symbols": sample.get("atom_symbols") or [],
        "atomic_numbers": sample.get("atomic_numbers") or [],
        "formal_charges": sample.get("formal_charges") or [],
        "bonds": sample.get("bonds") or [],
        "rotatable_bond_count": sample.get("rotatable_bond_count"),
        "ring_atom_indices": sample.get("ring_atom_indices") or [],
    }


def _load_evaluation_config(payload: dict[str, Any], split: str, method: str) -> dict[str, Any]:
    config = load_config(CONFIG_BY_SPLIT_AND_METHOD[(split, method)])
    config["experiment"]["seed"] = int(payload["random_seed"])
    config["data"]["processed_artifact_path"] = payload["processed_artifact_path"]
    config["data"]["split_path"] = payload["split_path"]
    config["data"]["split_type"] = split
    config["eval"]["compute_physical_validity_diagnostics"] = True
    config["eval"]["compute_rotatable_bond_stratification"] = True
    config["eval"]["compute_rotatable_bond_count_from_smiles"] = True
    config["train"]["device"] = "cpu"
    config["train"]["require_cuda"] = False
    return config


def _load_result_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Expected benchmark_result=false in {path}")
    if payload.get("sampling_executed") is not True:
        raise ValueError(f"Expected sampling_executed=true in {path}")
    return payload


def _split_membership(split_payload: dict[str, Any]) -> dict[str, str]:
    membership = {}
    for partition in PARTITIONS:
        for molecule_id in split_payload.get(partition, []):
            membership[str(molecule_id)] = partition
    return membership


def _single_forcefield_diagnostics(mol: Any, name: str, max_iters: int) -> dict[str, Any]:
    try:
        from rdkit.Chem import AllChem

        working = _copy_mol(mol)
        if name == "MMFF":
            props = AllChem.MMFFGetMoleculeProperties(working, mmffVariant="MMFF94s")
            if props is None:
                return {"available": False, "failure_reason": "MMFF_properties_unavailable"}
            ff = AllChem.MMFFGetMoleculeForceField(working, props)
        elif name == "UFF":
            ff = AllChem.UFFGetMoleculeForceField(working)
        else:
            raise ValueError(f"Unsupported force field: {name}")
        if ff is None:
            return {"available": False, "failure_reason": "force_field_unavailable"}
        energy_initial = float(ff.CalcEnergy())
        gradient_initial = _gradient_norm(ff)
        status = int(ff.Minimize(maxIts=int(max_iters)))
        energy_final = float(ff.CalcEnergy())
        gradient_final = _gradient_norm(ff)
        return {
            "available": True,
            "optimization_status": status,
            "optimization_converged": status == 0,
            "energy_initial": energy_initial,
            "energy_final": energy_final,
            "energy_delta_final_minus_initial": energy_final - energy_initial,
            "gradient_norm_initial": gradient_initial,
            "gradient_norm_final": gradient_final,
        }
    except Exception as exc:
        return {
            "available": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def _copy_mol(mol: Any) -> Any:
    from rdkit import Chem

    return Chem.Mol(mol)


def _gradient_norm(forcefield: Any) -> float | None:
    if not hasattr(forcefield, "CalcGrad"):
        return None
    grad = forcefield.CalcGrad()
    return math.sqrt(sum(float(value) * float(value) for value in grad))


def _overall_intramolecular_pass(diagnostics: dict[str, Any]) -> bool:
    return bool(
        diagnostics.get("finite_coordinate_pass")
        and diagnostics.get("graph_consistency", {}).get("pass")
        and diagnostics.get("rdkit_molecule", {}).get("build_success")
        and diagnostics.get("fallback_geometry", {}).get("fallback_pass")
    )


def _coords_are_finite(coords: list[list[float]]) -> bool:
    return all(len(xyz) == 3 and all(math.isfinite(float(value)) for value in xyz) for xyz in coords)


def _graph_consistency(record: dict[str, Any], coords: list[list[float]]) -> dict[str, Any]:
    atom_count = len(record.get("atom_symbols") or record.get("atomic_numbers") or coords)
    invalid_bonds = []
    for index, bond in enumerate(record.get("bonds", [])):
        if len(bond) < 2:
            invalid_bonds.append({"index": index, "reason": "short_bond_record"})
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            invalid_bonds.append({"index": index, "reason": "invalid_atom_index", "bond": bond})
    return {
        "pass": len(coords) == atom_count and not invalid_bonds,
        "atom_count": atom_count,
        "coordinate_count": len(coords),
        "bond_count": len(record.get("bonds", [])),
        "invalid_bond_count": len(invalid_bonds),
        "invalid_bonds": invalid_bonds[:10],
    }


def _bond_length_checks(
    record: dict[str, Any],
    coords: list[list[float]],
    target_coords: list[list[float]] | None,
) -> dict[str, Any]:
    distances = []
    deltas = []
    abnormal = []
    for index, bond in enumerate(record.get("bonds", [])):
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
            continue
        distance = _distance(coords[i], coords[j])
        distances.append(distance)
        if distance < 0.45 or distance > 2.50:
            abnormal.append({"bond_index": index, "distance": distance, "bond": [i, j]})
        if target_coords is not None and i < len(target_coords) and j < len(target_coords):
            deltas.append(abs(distance - _distance(target_coords[i], target_coords[j])))
    return {
        "pass": not abnormal and bool(distances),
        "checked_count": len(distances),
        "abnormal_count": len(abnormal),
        "abnormal_examples": abnormal[:10],
        "mean_bond_length": mean(distances) if distances else None,
        "min_bond_length": min(distances) if distances else None,
        "max_bond_length": max(distances) if distances else None,
        "mean_abs_delta_to_target": mean(deltas) if deltas else None,
        "max_abs_delta_to_target": max(deltas) if deltas else None,
    }


def _bond_angle_checks(record: dict[str, Any], coords: list[list[float]]) -> dict[str, Any]:
    neighbors = {index: set() for index in range(len(coords))}
    for bond in record.get("bonds", []):
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)
    angles = []
    abnormal = []
    for center, items in neighbors.items():
        ordered = sorted(items)
        for idx, i in enumerate(ordered):
            for j in ordered[idx + 1 :]:
                angle = _angle_degrees(coords[i], coords[center], coords[j])
                if angle is None:
                    continue
                angles.append(angle)
                if angle < 15.0 or angle > 175.0:
                    abnormal.append({"center": center, "atoms": [i, center, j], "angle_degrees": angle})
    return {
        "pass": not abnormal,
        "checked_count": len(angles),
        "abnormal_count": len(abnormal),
        "abnormal_examples": abnormal[:10],
        "mean_angle_degrees": mean(angles) if angles else None,
        "min_angle_degrees": min(angles) if angles else None,
        "max_angle_degrees": max(angles) if angles else None,
    }


def _severe_clash_checks(record: dict[str, Any], coords: list[list[float]]) -> dict[str, Any]:
    bonded = set()
    for bond in record.get("bonds", []):
        if len(bond) < 2:
            continue
        i, j = sorted((int(bond[0]), int(bond[1])))
        bonded.add((i, j))
    clashes = []
    atomic_numbers = record.get("atomic_numbers") or []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if (i, j) in bonded:
                continue
            distance = _distance(coords[i], coords[j])
            threshold = 0.75
            if atomic_numbers and i < len(atomic_numbers) and j < len(atomic_numbers):
                threshold = 0.45 * (_vdw_radius(int(atomic_numbers[i])) + _vdw_radius(int(atomic_numbers[j])))
            if distance < threshold:
                clashes.append({"atoms": [i, j], "distance": distance, "threshold": threshold})
    return {
        "pass": not clashes,
        "checked_pairs": max(0, len(coords) * (len(coords) - 1) // 2 - len(bonded)),
        "severe_clash_count": len(clashes),
        "severe_clash_examples": clashes[:10],
    }


def _ring_geometry_checks(record: dict[str, Any], coords: list[list[float]]) -> dict[str, Any]:
    rings = record.get("ring_atom_indices") or []
    valid_rings = [
        [int(index) for index in ring if isinstance(index, int) or str(index).isdigit()]
        for ring in rings
        if isinstance(ring, list)
    ]
    return {
        "available": bool(valid_rings),
        "ring_count": len(valid_rings),
        "ring_atom_counts": [len(ring) for ring in valid_rings],
        "note": "Detailed ring flatness is delegated to PoseBusters when available.",
    }


def _chirality_warning_count(record: dict[str, Any], coords: list[list[float]]) -> dict[str, Any]:
    mol_result = build_rdkit_mol_from_record(record, coords, sanitize=True)
    mol = mol_result.get("mol")
    if mol is None:
        return {"available": False, "unassigned_chiral_center_count": None}
    try:
        from rdkit import Chem

        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        unassigned = [center for center in centers if len(center) > 1 and center[1] == "?"]
        return {
            "available": True,
            "chiral_center_count": len(centers),
            "unassigned_chiral_center_count": len(unassigned),
        }
    except Exception as exc:
        return {"available": False, "failure_reason": f"{type(exc).__name__}: {exc}"}


def _rdkit_bond_type(value: Any, chem_module: Any) -> Any:
    text = str(value).lower()
    if text in {"aromatic", "1.5"}:
        return chem_module.BondType.AROMATIC
    if text in {"double", "2", "2.0"}:
        return chem_module.BondType.DOUBLE
    if text in {"triple", "3", "3.0"}:
        return chem_module.BondType.TRIPLE
    return chem_module.BondType.SINGLE


def _vdw_radius(atomic_number: int) -> float:
    table = {
        1: 1.20,
        6: 1.70,
        7: 1.55,
        8: 1.52,
        9: 1.47,
        15: 1.80,
        16: 1.80,
        17: 1.75,
        35: 1.85,
        53: 1.98,
    }
    return table.get(int(atomic_number), 1.70)


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def _angle_degrees(a: list[float], b: list[float], c: list[float]) -> float | None:
    ba = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    bc = [float(x) - float(y) for x, y in zip(c, b, strict=True)]
    norm_ba = math.sqrt(sum(x * x for x in ba))
    norm_bc = math.sqrt(sum(x * x for x in bc))
    if norm_ba == 0.0 or norm_bc == 0.0:
        return None
    cosine = sum(x * y for x, y in zip(ba, bc, strict=True)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _append_number(items: list[float], value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        items.append(float(value))


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q05": None, "q25": None, "q75": None, "q95": None}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": mean(sorted_values),
        "median": median(sorted_values),
        "q05": _quantile(sorted_values, 0.05),
        "q25": _quantile(sorted_values, 0.25),
        "q75": _quantile(sorted_values, 0.75),
        "q95": _quantile(sorted_values, 0.95),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _pass_rate(values: list[Any]) -> dict[str, Any]:
    usable = [bool(value) for value in values if value is not None]
    if not usable:
        return {"count": 0, "passed": 0, "rate": None}
    passed = sum(1 for value in usable if value)
    return {"count": len(usable), "passed": passed, "rate": passed / len(usable)}


def _tensor_coords_to_list(tensor: Any) -> list[list[float]]:
    array = tensor.detach().cpu().tolist()
    return [[float(value) for value in xyz] for xyz in array]


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_split(split: str) -> str:
    split = str(split)
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _normalize_method(method: str) -> str:
    try:
        return METHOD_TO_CANONICAL[str(method)]
    except KeyError as exc:
        raise ValueError(f"Unsupported method: {method}") from exc


def _path_status(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    exists = path.exists()
    return {
        "path": str(_relative(path)),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _relative(path: str | Path) -> Path:
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
