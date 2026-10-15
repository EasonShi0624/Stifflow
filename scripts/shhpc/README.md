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

## 修复后的 staging 与恢复规则

`stage_packed_cache.py` 使用 `cp -a` 原样复制 split，**不再重写 manifest**。
训练始终以 GPFS 原始 manifest 的哈希标识数据，通过 `--packed-cache-root`
只在内存中重定位 packed 数组；冻结数据与审计来源仍指向原始位置。
reader 保留原有大小、哈希、shape、dtype 与 row/label 配对检查。

每次启动创建私有临时目录和唯一 attempt ID；同一 job ID 重启也会创建新的
preflight/segment。作业退出只清理其拥有的临时目录。源目标相同、相互包含、
已有目标、非法 split 或空间不足均报错，不删除已有缓存。复制阶段使用节点上
按用户和文件系统标识的锁，重新核对至少 120 GiB 空闲和完整缓存实际大小；
持久化 run 和 checkpoint 另有单写入锁。

版本 2 checkpoint 签名绑定 canonical manifest、完整配置、seed、采样与
原 GPU 验收范围；实际代码版本、差异、验收/兼容记录和 staging 路径单独保存。
原 GPU receipt 保持不变，`--io-compatibility` 只继承经过 CPU 等价检查且
哈希完全匹配的这次 I/O/恢复元数据变更；其他算法或依赖变化仍拒绝启动。

Staging the train split (91.9 GB) takes ~2 minutes at ~600 MB/s.

## Usage

```bash
export STIFFLOW_WORKSPACE=/path/to/workspace        # has stifflow-v1/{stifflow,...}
export ETFLOW_ENV=/path/to/existing/etflow          # 已安装的环境，不重新创建/安装
export STIFFLOW_ACCEPTANCE=/path/to/acceptance.json # matching torch_runtime receipt
export STIFFLOW_IO_COMPATIBILITY="$STIFFLOW_WORKSPACE/stifflow-v1/results/shhpc-io-compatibility-20261015.json"
export STIFFLOW_RUN_ROOT=/path/to/runs/formal-staged
export STIFFLOW_PACKED_SOURCE_ROOT="$STIFFLOW_WORKSPACE/cache/qm9-v3-packed-strict"
mkdir -p "$STIFFLOW_RUN_ROOT"
# 由 SHHPC 填入真实 account/partition/GPU/CPU/memory/walltime、array 并提交：
# sbatch <本集群资源参数> --output="$STIFFLOW_RUN_ROOT/slurm-%A_%a.out" scripts/shhpc/formal_staged.sbatch
```

Array index maps to `methods[index/3]` x `seeds[index%3]` with
`methods=(precision alternative_a control)`, `seeds=(42 43 44)`.

`formal_staged.sbatch` 复用 `scripts/formal.sbatch` 的启动和恢复逻辑。
也可直接使用后者读取 GPFS；指定 `STIFFLOW_TRAIN_LABELS`、`STIFFLOW_VAL_LABELS`
为原始 manifest，并设置相同 acceptance/compatibility。配置仍为单卡 FP32、
microbatch 32、accumulation 4、effective batch 128、250000 updates。

## 已开始的旧版训练：保留进度迁移

不要在旧作业运行时切换其代码，不将新代码直接接到旧 checkpoint 目录。
等待旧作业退出或由 SHHPC 在有效恢复点停止，再交接完整 commit SHA、job ID、
global step、不可变 `step-XXXXXXXX.ckpt` 和旧 segment 的 `manifest.json`。
旧版 segment manifest 在训练退出后写入；若异常退出而没有记录，需从原始
preflight/作业日志恢复真实字段，不能猜测路径、代码差异或数据哈希。

在 SHHPC 已分配的 CPU 资源内，使用已有 etflow 环境和正确 PYTHONPATH：

```bash
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
python -m stifflow.migrate_run \
  --checkpoint "$OLD_CHECKPOINT" --legacy-run "$OLD_SEGMENT/manifest.json" \
  --train-labels "$STIFFLOW_PACKED_SOURCE_ROOT/train/manifest.json" \
  --val-labels "$STIFFLOW_PACKED_SOURCE_ROOT/val/manifest.json" \
  --acceptance "$STIFFLOW_ACCEPTANCE" \
  --io-compatibility "$STIFFLOW_IO_COMPATIBILITY" \
  --output "$NEW_RUN_ROOT/precision-seed42"
```

按实际 method/seed 指定全新的输出目录。只支持本次核对的四个旧提交
`e463b165…`、`793d502d…`、`52091a92…`、`867e7564…`，且其已跟踪代码差异为空、
依赖版本与 GPU receipt 一致；其他版本需先审查。完整 SHA 保存在迁移工具和兼容记录。
旧 manifest 若经 staging 改写，必须用旧运行记录中的绝对路径精确重建旧 JSON 哈希。
配置、cohort、seed、采样、optimizer、单卡 RNG 与累积边界均须匹配。

工具只向新目录写入 checkpoint、owner、latest 和 `migration.json`，保存原 checkpoint
SHA；序列化后对所有非迁移元数据执行零容差比较，再发布 latest。原文件不变。
验证失败的输出目录保留供排查，不自动覆盖或重试写入。
将 `STIFFLOW_RUN_ROOT` 设置为 `NEW_RUN_ROOT` 后，原启动脚本自动从新 latest 接续。
九组任务分别迁移，不能用一个 method/seed 的目录恢复其他任务。

## References 修复

reference 导出只接受同一完整图的原子排列变化：先逐行核对原始 memmap/分子，
再按化学图匹配同步重排分子和坐标，保存 `reference_atom_mappings`，独立记录 stereo。
真实连接、键级、电荷变化或不可核验映射会报错，不静默过滤。
在 SHHPC CPU allocation 中向**新目录**导出，不覆盖历史 references：

```bash
python -m stifflow.formal references --manifest "$FROZEN_DATA_MANIFEST" \
  --config configs/qm9-formal-4090d.yaml --output "$NEW_REFERENCE_ROOT"
```

无需重新生成 Hessian、重新训练或增加独立 GPU 验收。恢复后的正式作业沿用现有
有限性、显存上限和样本游标检查；有异常由 SHHPC 返回实际 commit 与日志。

上述历史速度数字来自 SHHPC 原交接文档，不是本次 CPU 修复测得的新性能结论。
