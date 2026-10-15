"""Stage the frozen v3 packed label cache onto node-local storage.

Content is copied byte-for-byte (all sha256 records stay valid); only the stored
absolute paths inside each split manifest are rewritten to the staging root.
The original cache is never modified. Read-only w.r.t. the source.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def rewrite_paths(obj, src_prefix, dst_prefix):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                key == "path"
                and isinstance(value, str)
                and value.startswith(src_prefix)
            ):
                obj[key] = dst_prefix + value[len(src_prefix) :]
            else:
                rewrite_paths(value, src_prefix, dst_prefix)
    elif isinstance(obj, list):
        for value in obj:
            rewrite_paths(value, src_prefix, dst_prefix)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--verify-samples", type=int, default=8)
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        src = src_root / split
        dst = dst_root / split
        if dst.exists():
            shutil.rmtree(dst)
        t = time.time()
        subprocess.run(["cp", "-a", str(src), str(dst)], check=True)
        print(f"[stage] copied {src} -> {dst} in {time.time() - t:.1f}s", flush=True)

        manifest_path = dst / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        before = json.dumps(manifest, sort_keys=True)
        rewrite_paths(manifest, str(src) + "/", str(dst) + "/")
        after = json.dumps(manifest, sort_keys=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"[stage] {split}: manifest paths rewritten (changed={before != after}) "
            f"train_rows={len(manifest['records'])} "
            f"graph_shards={len(manifest['graph_shards'])} "
            f"factor_shards={len(manifest['factor_shards'])}",
            flush=True,
        )
        # Spot-check that content hashes still match the recorded fingerprints.
        checks = [manifest["row_index"]]
        for kind in ("graph_shards", "factor_shards"):
            for shard in manifest[kind]:
                checks.extend(shard["files"].values())
        step = max(1, len(checks) // args.verify_samples)
        for record in checks[::step]:
            actual = sha256(record["path"])
            if actual != record["sha256"]:
                raise SystemExit(f"[stage] HASH MISMATCH after copy: {record['path']}")
        print(
            f"[stage] {split}: spot-checked {len(checks[::step])} files, hashes OK",
            flush=True,
        )


if __name__ == "__main__":
    main()
