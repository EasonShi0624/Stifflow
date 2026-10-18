"""Copy a frozen packed cache without rewriting manifests or deleting old data."""

import argparse
import fcntl
import shutil
import subprocess
import time
from pathlib import Path

from stifflow.cache_io import atomic_json, read_json
from stifflow.packed_data import relocate_manifest
from stifflow.preflight import fingerprint


def stage(
    src_root,
    dst_root,
    splits=("train", "val"),
    verify_samples=8,
    min_free_gib=120,
    space_lock=None,
):
    source = Path(src_root).resolve(strict=True)
    target_arg = Path(dst_root).absolute()
    target = target_arg.resolve()
    if (
        not source.is_dir()
        or source == target
        or source.is_relative_to(target)
        or target.is_relative_to(source)
    ):
        raise ValueError("Source and destination must be separate directories")
    if target_arg.exists() or target_arg.is_symlink() or target.exists():
        raise FileExistsError("Existing staging destinations are protected")
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(s not in ("train", "val", "test") for s in splits)
    ):
        raise ValueError("Invalid or repeated cache split")
    if verify_samples < 1 or min_free_gib < 0:
        raise ValueError(
            "Positive verification count and nonnegative free-space limit required"
        )
    manifests, required = {}, 0
    for split in splits:
        directory = source / split
        if not directory.resolve().is_relative_to(source) or directory.is_symlink():
            raise ValueError("Source split escapes cache root")
        path = directory / "manifest.json"
        manifest = read_json(path)
        if manifest.get("split") != split:
            raise ValueError("Packed manifest split differs")
        relocated = relocate_manifest(manifest, path, target)
        files = [relocated["row_index"]]
        for kind in ("graph_shards", "factor_shards"):
            for shard in relocated[kind]:
                files.extend(shard["files"].values())
        required += int(
            subprocess.check_output(["du", "-sb", str(directory)], text=True).split()[0]
        )
        manifests[split] = path, files
    # Serialize the free-space check and copy across this user's jobs on this disk.
    lock = Path(space_lock) if space_lock else target.parent / ".stifflow-stage.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        needed = max(min_free_gib * 1024**3, required * 1.05)
        if shutil.disk_usage(target.parent).free < needed:
            raise OSError("Insufficient node-local space for this cache")
        target.mkdir(
            mode=0o700
        )  # Exclusive creation; never reuse or remove a destination.
        receipts = {}
        for split, (path, files) in manifests.items():
            started = time.monotonic()
            subprocess.run(
                ["cp", "-a", str(source / split), str(target / split)], check=True
            )
            original = fingerprint(path)
            if (
                fingerprint(target / split / "manifest.json")["sha256"]
                != original["sha256"]
            ):
                raise ValueError("Staged manifest bytes differ from canonical manifest")
            stride = max(1, len(files) // verify_samples)
            for item in files[::stride]:
                if fingerprint(item["path"])["sha256"] != item["sha256"]:
                    raise ValueError(f"Packed hash mismatch after copy: {item['path']}")
            receipts[split] = original
            print(
                f"[stage] {split}: byte-identical manifest; copy/verify {time.monotonic() - started:.1f}s",
                flush=True,
            )
        atomic_json(
            target / ".staged.json", {"source": str(source), "manifests": receipts}
        )
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val"],
    )
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=120)
    parser.add_argument("--space-lock", type=Path)
    args = parser.parse_args()
    stage(
        args.src_root,
        args.dst_root,
        args.splits,
        args.verify_samples,
        args.min_free_gib,
        args.space_lock,
    )


if __name__ == "__main__":
    main()
