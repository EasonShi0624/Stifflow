"""Read-only GEOM-Drugs raw-data preflight and preprocessing dry-run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


EXPECTED_LAYOUT_HINTS = [
    "summary_drugs.json",
    "rdkit_folder",
    "rdkit_folder/drugs",
    "drugs_crude.msgpack",
    "drugs.msgpack",
    "molecules",
    "mols",
]


def run_geom_preflight(
    raw_root: str | Path = "data/raw/geom_drugs",
    processed_root: str | Path = "data/processed/geom_drugs_subset",
    manifest_path: str | Path = "experiments/manifests/geom_drugs_preflight.json",
    report_path: str | Path = "reports/tables/geom_drugs_preflight.md",
    max_scan_entries: int = 200,
) -> dict[str, Path]:
    start = time.time()
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    manifest_path = Path(manifest_path)
    report_path = Path(report_path)

    raw_summary = inspect_raw_root(raw_root, max_scan_entries=max_scan_entries)
    manifest = {
        "run_name": "geom_drugs_preflight",
        "dataset_name": "geom_drugs",
        "dataset_kind": "raw_data_preflight",
        "execution_mode": "local_read_only_preflight",
        "result_kind": "preflight_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_written": False,
        "preprocessing_executed": False,
        "dry_run": True,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "raw_summary": raw_summary,
        "dry_run_plan": {
            "would_read_from": str(raw_root),
            "would_write_processed_records_to": str(processed_root),
            "would_create_split_files": False,
            "would_modify_raw_data": False,
            "would_write_processed_data": False,
            "next_real_step": "Add a bounded parser for the detected GEOM raw layout before processing molecules.",
        },
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a read-only preflight and preprocessing dry-run.",
            "No GEOM-Drugs molecules were parsed or processed.",
            "No raw data, processed data, split files, checkpoints, or training outputs were modified.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(manifest), encoding="utf-8")
    return {"manifest": manifest_path, "report": report_path}


def inspect_raw_root(raw_root: Path, max_scan_entries: int = 200) -> dict[str, Any]:
    exists = raw_root.exists()
    is_dir = raw_root.is_dir()
    summary: dict[str, Any] = {
        "exists": exists,
        "is_dir": is_dir,
        "status": "missing_root",
        "expected_layout_hints": EXPECTED_LAYOUT_HINTS,
        "present_layout_hints": [],
        "missing_layout_hints": EXPECTED_LAYOUT_HINTS,
        "top_level_entries": [],
        "sample_files": [],
        "sample_dirs": [],
        "extension_counts": {},
        "scanned_entry_count": 0,
        "scan_truncated": False,
    }
    if not exists:
        summary["blocking_missing_items"] = [str(raw_root)]
        return summary
    if not is_dir:
        summary["status"] = "raw_root_not_directory"
        summary["blocking_missing_items"] = [f"{raw_root} is not a directory"]
        return summary

    top_level = sorted(raw_root.iterdir(), key=lambda path: path.name)
    summary["top_level_entries"] = [_entry_payload(path, raw_root) for path in top_level[:max_scan_entries]]
    summary["scan_truncated"] = len(top_level) > max_scan_entries

    present_hints = [hint for hint in EXPECTED_LAYOUT_HINTS if (raw_root / hint).exists()]
    summary["present_layout_hints"] = present_hints
    summary["missing_layout_hints"] = [hint for hint in EXPECTED_LAYOUT_HINTS if hint not in present_hints]

    sample_files: list[str] = []
    sample_dirs: list[str] = []
    extension_counts: dict[str, int] = {}
    scanned = 0
    for path in _bounded_walk(raw_root, max_entries=max_scan_entries):
        scanned += 1
        rel = str(path.relative_to(raw_root))
        if path.is_dir():
            if len(sample_dirs) < 25:
                sample_dirs.append(rel)
        elif path.is_file():
            if len(sample_files) < 25:
                sample_files.append(rel)
            suffix = path.suffix.lower() or "<no_ext>"
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1

    summary["sample_files"] = sample_files
    summary["sample_dirs"] = sample_dirs
    summary["extension_counts"] = dict(sorted(extension_counts.items()))
    summary["scanned_entry_count"] = scanned
    summary["scan_truncated"] = summary["scan_truncated"] or scanned >= max_scan_entries

    if present_hints:
        summary["status"] = "present_candidate_layout"
        summary["blocking_missing_items"] = []
    elif sample_files or sample_dirs:
        summary["status"] = "present_unrecognized_layout"
        summary["blocking_missing_items"] = ["No known GEOM-Drugs layout hint was detected."]
    else:
        summary["status"] = "present_empty"
        summary["blocking_missing_items"] = ["Raw root exists but contains no visible entries."]
    return summary


def _bounded_walk(root: Path, max_entries: int) -> list[Path]:
    entries: list[Path] = []
    pending = [root]
    while pending and len(entries) < max_entries:
        current = pending.pop(0)
        try:
            children = sorted(current.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for child in children:
            entries.append(child)
            if len(entries) >= max_entries:
                break
            if child.is_dir():
                pending.append(child)
    return entries


def _entry_payload(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "kind": "dir" if path.is_dir() else "file" if path.is_file() else "other",
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _markdown_report(manifest: dict[str, Any]) -> str:
    raw = manifest["raw_summary"]
    lines = [
        "# GEOM-Drugs Preflight",
        "",
        "**Scope:** read-only raw-data preflight and preprocessing dry-run. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{manifest['result_kind']}`",
        f"- `benchmark_result`: `{str(manifest['benchmark_result']).lower()}`",
        f"- `raw_root`: `{manifest['raw_root']}`",
        f"- `processed_root`: `{manifest['processed_root']}`",
        f"- `status`: `{raw['status']}`",
        f"- `raw_data_modified`: `{str(manifest['raw_data_modified']).lower()}`",
        f"- `processed_data_written`: `{str(manifest['processed_data_written']).lower()}`",
        "",
        "## Layout Hints",
        "",
        f"- Present: {', '.join(raw['present_layout_hints']) if raw['present_layout_hints'] else 'none'}",
        f"- Missing: {', '.join(raw['missing_layout_hints']) if raw['missing_layout_hints'] else 'none'}",
        "",
        "## Top-Level Entries",
        "",
    ]
    if raw["top_level_entries"]:
        lines.extend(["| Path | Kind | Size Bytes |", "|---|---|---:|"])
        for entry in raw["top_level_entries"]:
            lines.append(f"| `{entry['path']}` | {entry['kind']} | {entry['size_bytes'] or 'NA'} |")
    else:
        lines.append("No top-level entries found.")

    lines.extend(
        [
            "",
            "## Dry-Run Plan",
            "",
            f"- Would read from: `{manifest['dry_run_plan']['would_read_from']}`",
            f"- Would write processed records to: `{manifest['dry_run_plan']['would_write_processed_records_to']}`",
            "- Would modify raw data: `false`",
            "- Would write processed data: `false`",
            "",
            "## Notes",
            "",
            "- This command does not parse molecules yet.",
            "- A real preprocessing implementation should be added only after the detected raw layout is confirmed.",
            "",
        ]
    )
    return "\n".join(lines)
