# SHHPC staging helpers

These helpers stage the frozen packed label cache onto node-local NVMe before a
formal run. They were used for the 2026-10-15 formal campaign while the shared
GPFS cache was the training bottleneck.

## Why

Measured single-process, 384 uniform draws against
`cache/qm9-v3-packed-strict/train` (123 graph + 388 factor shards):

| access path | warm steady state | at 128 samples/update |
| --- | --- | --- |
| GPFS, frozen `max_open_parts=16` | ~43 ms/sample | ~5.5 s/update |
| GPFS, LRU sized from manifest (`shhpc-reader-cache-sizes`) | ~15-19 ms/sample | ~1.87 s/update |
| node-local NVMe, same cache staged | 0.8 ms/sample | ~0.10 s/update |

Cold first pass over GPFS was ~304 ms/sample (0.9 ms per `np.load(mmap)` open
plus a full-cache SHA-256 verification per worker). GPFS mmap row access remains
the residual cost even with the LRU fix, which is why staging is the larger win
for production runs.

End to end, five concurrent jobs per GPU node reached ~0.9 s/update (~2.6 days
for 250k updates), versus ~5.5 s/update (~16 days) without staging.

## What the staging does

`stage_packed_cache.py` copies the split directories byte-for-byte with
`cp -a` (so every recorded sha256 still matches) and rewrites only the absolute
`path` fields inside each split `manifest.json` to the staging root. The source
cache is never modified. `validate_label_manifest` verifies content hashes, not
shard paths, so the relocated cache passes unchanged; `data_manifest` (the
frozen split) is left pointing at its GPFS location.

Staging the train split (91.9 GB) takes ~2 minutes at ~600 MB/s.

## Usage

```bash
export STIFFLOW_WORKSPACE=/path/to/workspace        # has stifflow-v1/{stifflow,...}
export STIFFLOW_ACCEPTANCE=/path/to/acceptance.json # matching torch_runtime receipt
export STIFFLOW_RUN_ROOT=/path/to/runs/formal-staged
sbatch --account=<acct> --partition=<part> --gres=gpu:4090d:1 \
       --cpus-per-task=6 --mem=64G --time=7-00:00:00 --array=0-8%9 \
       --output="$STIFFLOW_RUN_ROOT/slurm-%A_%a.out" \
       scripts/shhpc/formal_staged.sbatch
```

Array index maps to `methods[index/3]` x `seeds[index%3]` with
`methods=(precision alternative_a control)`, `seeds=(42 43 44)`.

The staging path is deterministic per run (`<base>/stifflow-packed-v3-<method>-seed<seed>`),
so a requeue re-creates an identical manifest and the checkpoint owner signature
still matches. Staging is removed on exit; ~120 GB free node-local disk is
required.

Note: this placement does not change the experiment (same content, same labels,
same config). It only changes where the packed shards are read from.
