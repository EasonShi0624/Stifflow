"""Copy a reviewed legacy formal checkpoint into a new, version-2 run directory."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .cache_io import atomic_json, read_json
from .formal import METHODS, SEEDS, formal_training_signature, require_torch_runtime
from .official_data import validate_label_pair
from .preflight import fingerprint, provenance
from .smoke import external_directory
from .verify_resume import compare_states

LEGACY_COMMITS = (
    "e463b16543f1365baab731e4da909d6f6cc8e2e6",
    "793d502d86e521039d8f641480cd3960ef2c81d1",
    "52091a92957a3c9a2ce0e0b47428310dd0a1f1f5",
    "867e75649ac66f50f05c2831d0d7e50bd8e43301",
)
METADATA = {
    "training_signature",
    "recovery_directory",
    "execution_provenance",
    "checkpoint_migration",
}


def legacy_manifest_matches(canonical, old_record, expected):
    """Reconstruct the exact old staging serialization; never guess a missing path."""
    if old_record.get("sha256") != expected:
        raise ValueError("Legacy run/checkpoint manifest hashes differ")
    if fingerprint(canonical)["sha256"] == expected:
        return
    old_path = Path(old_record["path"])
    if not old_path.is_absolute() or old_path.name != "manifest.json":
        raise ValueError("Missing absolute legacy staging manifest path")
    manifest = copy.deepcopy(read_json(canonical))
    if old_path.parent.name != manifest["split"]:
        raise ValueError("Legacy staging split differs")
    source = str(Path(canonical).resolve().parent) + "/"
    target = str(old_path.parent) + "/"

    def rewrite(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path" and isinstance(item, str) and item.startswith(source):
                    value[key] = target + item[len(source) :]
                else:
                    rewrite(item)
        elif isinstance(value, list):
            for item in value:
                rewrite(item)

    rewrite(manifest)
    digest = hashlib.sha256(
        (json.dumps(manifest, indent=2) + "\n").encode()
    ).hexdigest()
    if digest != expected:
        raise ValueError(
            "Legacy staged manifest cannot be reconstructed from canonical data"
        )


def migrate(checkpoint, legacy_run, train, val, acceptance, compatibility, output):
    import torch

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Migration requires a fresh run directory")
    source_receipt = fingerprint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run, accepted, compat = map(read_json, (legacy_run, acceptance, compatibility))
    old = saved["training_signature"]
    cfg = old["config"]
    require_torch_runtime(accepted, cfg, compat)
    if (
        old.get("signature_version") is not None
        or cfg.get("runtime_acceptance") != "torch_runtime"
    ):
        raise ValueError("Only legacy torch-runtime formal runs need this migration")
    if compat.get("legacy_source_commits") != list(LEGACY_COMMITS):
        raise ValueError("Missing reviewed legacy source commits")
    code = old.get("code", {})
    dependencies = accepted["runtime_code"]["dependencies"]
    if (
        set(code) != {"stifflow-v1", *dependencies}
        or code["stifflow-v1"].get("head") not in LEGACY_COMMITS
        or any(item.get("diff") != "" for item in code.values())
        or any(code[name].get("head") != head for name, head in dependencies.items())
        or code
        != {
            name: {"head": item["head"], "diff": item["diff"]}
            for name, item in run["provenance"]["repos"].items()
        }
    ):
        raise ValueError(
            "Legacy code is unknown, dirty or differs from its run receipt"
        )
    method = saved["hyper_parameters"]["method"]
    seed = old["training_seed"]
    if (
        method not in METHODS
        or seed not in SEEDS
        or run["method"] != method
        or run["seed"] != seed
        or run["config"] != cfg
        or run["profile"] != "formal"
        or saved["hyper_parameters"].get("seed") != seed
        or "distributed_states" in saved
        or len(saved.get("cuda_rng", [])) != 1
        or saved.get("snapshot_kind") is not None
    ):
        raise ValueError(
            "Legacy method/seed/config or single-GPU recovery scope differs"
        )
    required = (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "stifflow_rng",
        "torch_rng",
        "gradient_queue",
        "consumed_samples",
    )
    if (
        any(key not in saved for key in required)
        or len(saved["optimizer_states"]) != 1
        or not saved["optimizer_states"][0]
    ):
        raise ValueError("A complete optimizer/RNG recovery checkpoint is required")
    step = saved["global_step"]
    if (
        not 0 < step < cfg["profiles"]["formal"]["steps"]
        or saved["consumed_samples"] != step * old["effective_batch_size"]
    ):
        raise ValueError(
            "Checkpoint is not at a resumable complete accumulation boundary"
        )
    manifests = [read_json(path) for path in (train, val)]
    validate_label_pair(*manifests)
    for split, path, manifest in zip(("train", "val"), (train, val), manifests):
        if manifest.get("schema_version") != 3:
            raise ValueError("Migration requires canonical packed manifests")
        legacy_manifest_matches(path, run[split + "_manifest"], old[split])
        if run[split + "_ids"] != [record["id"] for record in manifest["records"]]:
            raise ValueError("Legacy and canonical molecule cohorts differ")
    signature = formal_training_signature(train, val, cfg, seed, accepted)
    expected = {
        key: value
        for key, value in signature.items()
        if key not in ("signature_version", "runtime_scope")
    }
    previous = {key: value for key, value in old.items() if key != "code"}
    previous.update(train=signature["train"], val=signature["val"])
    if previous != expected:
        raise ValueError("Legacy sampling, data, seed or training budget differs")
    recovery = output / "checkpoints"
    receipt = {
        "schema_version": 1,
        "kind": "checkpoint_migration",
        "source_checkpoint": source_receipt,
        "legacy_run": fingerprint(legacy_run),
        "legacy_training_signature": old,
        "acceptance": fingerprint(acceptance),
        "io_compatibility": fingerprint(compatibility),
        "global_step": step,
        "consumed_samples": saved["consumed_samples"],
        "training_signature": signature,
    }
    migrated = dict(saved)
    migrated.update(
        training_signature=signature,
        recovery_directory=str(recovery),
        checkpoint_migration=receipt,
        execution_provenance={"migration": receipt, "provenance": provenance()},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive creation, after all identity checks have passed.
    recovery.mkdir()
    path = recovery / f"step-{step:08d}.ckpt"
    temporary = recovery / ".migration.ckpt"
    torch.save(migrated, temporary)
    written = torch.load(temporary, map_location="cpu", weights_only=False)
    compare_states(
        {k: v for k, v in saved.items() if k not in METADATA},
        {k: v for k, v in written.items() if k not in METADATA},
        exact=True,
    )
    compare_states(migrated, written, exact=True)
    if fingerprint(checkpoint) != source_receipt:
        raise ValueError("Source checkpoint changed during migration")
    temporary.replace(path)
    atomic_json(
        recovery / "owner.json", {"training": signature, "method": method, "seed": seed}
    )
    receipt["migrated_checkpoint"] = fingerprint(path)
    receipt["state_preserved_exactly"] = True
    atomic_json(output / "migration.json", receipt)
    (recovery / "latest.ckpt").symlink_to(path.name)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "legacy-run",
        "train-labels",
        "val-labels",
        "acceptance",
        "io-compatibility",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full checkpoint migration belongs in an SHHPC CPU allocation")
    receipt = migrate(
        args.checkpoint,
        args.legacy_run,
        args.train_labels,
        args.val_labels,
        args.acceptance,
        args.io_compatibility,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "global_step": receipt["global_step"],
                "state_preserved_exactly": True,
            }
        )
    )


if __name__ == "__main__":
    main()
