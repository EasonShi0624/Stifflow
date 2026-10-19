# SHHPC staging helpers

The bounded delayed-curvature campaign is documented below under
[Stiffness locality screen](#stiffness-locality-screen).

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

## Stiffness locality screen

2026-10-19 的有界研究轮次保留正式 Alternative A。先诊断，满足门槛后最多训练
`a_delayed` 和 `a_delayed_rot_null` 各一组 seed-42/25k；`aligned_linear` 是 ET-Flow
已有机制，只作参照。项目内 smoothstep control 不等同于官方 ET-Flow。
本轮不运行 TS 搜索、test 调参或额外 seed。

交接分支及开发基线如下；运行结果的 provenance 会保存实际完整提交 SHA 和未提交差异。
SHHPC 接手后保持实验期间代码固定，不覆盖其正在写入的其它任务分支。

| worktree | 交接分支 | 开发基线 |
|---|---|---|
| core | `torch/stiffness-locality-core` | `dc3e07b650447faceeefe7abac5c7b90d82d0cc4` |
| Alternative A | `torch/stiffness-locality-alt-a` | `c56d466cdd4c309e7ce87543475a085413ac57a1` |
| benchmark | `torch/stiffness-locality-benchmark` | `01b3865f49b6d01ad91de3f79dee5631c944d9bd` |

### 输入与集群边界

所有下列物理计算、保存结果的批量处理、训练、生成和评测均由 **SHHPC allocation** 执行。
Torch 只做小型解析/合成检查。继续使用已安装 `etflow` 环境，不安装 xTB 或其它依赖。
按 SHHPC 的实际位置设置以下环境变量，不照抄 Torch 路径：

```bash
export STIFFLOW_WORKSPACE=/actual/shhpc/workspace
export ETFLOW_ENV=/actual/existing/etflow
export STIFFLOW_TRAIN_LABELS=/actual/canonical/packed-train/manifest.json
export STIFFLOW_VAL_LABELS=/actual/canonical/packed-val/manifest.json
export STIFFLOW_LOCALITY_ROOT=/actual/external/runs/locality-diagnostic
export STIFFLOW_LOCALITY_REPORT="$STIFFLOW_LOCALITY_ROOT/report.json"
export STIFFLOW_RUN_ROOT=/actual/external/runs/locality-campaign
export STIFFLOW_MONITOR_REFERENCES=/actual/existing/frozen-validation-monitor/manifest.json
```

train/val 必须是已完成、全部候选已处理的 v3 packed manifest；完整原始
`audit_receipt` 与其 hash 指向的 `.npz` 档案也必须存在。slim packed 数组本身没有
原始 Hessian，不能代替完整档案。缺失时报告失败，不重算 Hessian、不换分子。
可用 `STIFFLOW_PACKED_CACHE_ROOT` 指定现有节点本地 packed 数组，但 canonical manifest
仍标识训练数据；物理审计继续读取原始完整档案。
monitor 必须复用既有固定 512 个 validation 分子及相同 reference 身份。
已有 A/control/rot-null/trace-null 输入必须是 **25k/seed42/Euler50/2*N_ref/无后处理**，
不能用 250k 文件混入筛选。旧模型推理可读；训练恢复继续执行原签名校验，不迁移旧实验。

### 阶段一：冻结训练集诊断

在 CPU allocation 内执行：

```bash
bash "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc/research_task.sbatch" stiffness locality \
  --train-labels "$STIFFLOW_TRAIN_LABELS" \
  --config "$STIFFLOW_WORKSPACE/stifflow-v1/configs/qm9-research-25k.yaml" \
  --output "$STIFFLOW_LOCALITY_ROOT"
```

固定 SHA256 选 128 个 train 分子，每个固定一个标签行、4 个共享噪声，分别检查
`tau=0,0.25,0.5,0.75`，每个介入区间 16 个积分点。完整成功时直接能量/梯度查询
为 `128*(4*4*16+2)=33024`，不包含历史 stiffness 预处理；无新 Hessian 查询。
`records.jsonl` 保存每次查询与失败，`report.json` 保存每分子汇总、缓存身份与冻结选择。
展开式使用原始 Hessian 和真实一阶梯度；同时报告弛豫参考与原始目标之间的迁移误差。

局部有效要求能量和力变化的相对 Taylor 误差均不超过 25%，并保持指定手性。
归一化分母取真实变化、预测变化和固定下限的最大值；能量下限为 0.1 kcal/mol，
原子 RMS force 下限为 0.1 kcal/mol/angstrom。按
`quadrature_weight*gdot^2*||(L-S)z||^2` 计算有效监督比例，然后按分子等权。
只有原 A 的比例低于 80%、延迟候选达到 80%、真实 Hessian 力变化预测相对旋转 null
的配对差值 95% CI 上界小于零时，选择最小正 tau。
`selected_start=null` 表示停止这条训练分支，保留 A；不得手工填 tau 继续。
该门槛是预定筛选规则，Taylor 失配不等于原 A 的 CFM 推导失效。

对四个已有方法分别补物理审计。以 A 为例，`A25K_ENSEMBLES` 与 `A25K_EVALUATION`
分别指向真实的生成 manifest 和匹配的原几何 evaluation JSON，`A25K_PHYSICS`
指向新的外部输出目录，其余方法使用各自实际文件：

```bash
bash "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc/research_task.sbatch" stiffness physics \
  "$A25K_ENSEMBLES" --evaluation "$A25K_EVALUATION" \
  --val-labels "$STIFFLOW_VAL_LABELS" --output "$A25K_PHYSICS"
```

每模型固定同一 64 个分子、每分子前 `min(8,2*N_ref)` 个构象。
主指标是在原始生成坐标上的原子 RMS force：先构象中位数，再分子等权平均。
任何原始力计算失败都会保留失败分子、阻止通过，不计算有利的成功子集平均。
strain 只汇总收敛且弛豫未改变手性的结果，另存弛豫失败、手性改变、位移和参考力。
缺少 Hessian 只影响 Taylor 覆盖率，不过滤主物理指标；弛豫坐标不覆盖原始生成文件。

若现有 xTB 可用且 graph 明确为闭壳层，可为每个方法加 `--xtb /actual/existing/xtb`。
对固定 16 个分子计算 GFN2 单点梯度，电荷来自原 graph，显式 `--uhf 0`；开壳层状态
不猜测。使用同样的 oracle 设置处理五个比较方法，版本、二进制 hash、失败与查询成本
单列。未做或不完整时结论限定于 MMFF94，出现反向效果阻止推进。
xTB `--grad` 的 Turbomole 文件和单位依据
[官方手册](https://github.com/grimme-lab/xtb/blob/main/man/xtb.1.adoc)及
[官方 API 单位说明](https://xtb-python.readthedocs.io/en/latest/general-api.html)。

### 阶段二：两组验收与 25k

仅在诊断通过后，SHHPC 分别用其已确认的 GPU 资源配置提交
`scripts/shhpc/stiffness_train.sbatch a_delayed` 与
`scripts/shhpc/stiffness_train.sbatch a_delayed_rot_null`。
也可在各自现有 GPU allocation 内执行以下命令；不要在 Torch 运行：

```bash
bash "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc/stiffness_train.sbatch" a_delayed
# 在另一个已分配的 GPU 作业中：
bash "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc/stiffness_train.sbatch" a_delayed_rot_null
```

脚本先核验冻结报告及 train/source 身份，再运行 40 连续步、20+20 恢复并调用
`verify_resume` 检查权重、优化器、LR、样本游标与 RNG；通过后从连续 40 步继续到 25k。
有效 batch=128，seed=42，学习率周期保留 250k。每方法只允许新建目录，重复提交不覆盖。
生产中断后按既有 `stifflow.training --resume` 使用匹配签名和最新完整 checkpoint
恢复到新 segment，不重跑整个脚本。已写入的验收和结果均保留。
tau、报告 hash 与代码 hash 进入 checkpoint/训练签名，改变它们会拒绝恢复。

完成后，在 GPU allocation 对每个新模型生成，在 CPU allocation 评测和做物理审计。
以下是一个模型的完整入口；另一个只改变 `method`：

```bash
method=a_delayed
python -m stifflow_benchmark.generate "$STIFFLOW_MONITOR_REFERENCES" \
  --checkpoint "$STIFFLOW_RUN_ROOT/$method/snapshots/step-00025000.ckpt" \
  --steps 50 --seed 42 --output "$STIFFLOW_RUN_ROOT/$method/generated"
python -m stifflow_benchmark.evaluate "$STIFFLOW_RUN_ROOT/$method/generated/manifest.json" \
  --ratio 2 --allow-disconnected --output "$STIFFLOW_RUN_ROOT/$method/evaluation.json"
python -m stifflow_benchmark.stiffness_diagnostics physics \
  "$STIFFLOW_RUN_ROOT/$method/generated/manifest.json" \
  --evaluation "$STIFFLOW_RUN_ROOT/$method/evaluation.json" \
  --val-labels "$STIFFLOW_VAL_LABELS" --output "$STIFFLOW_RUN_ROOT/$method/physics"
```

在每个 allocation 先 source `scripts/shhpc_env.sh`；以上 GPU 和 CPU 命令不是同一资源作业。
若已有方法使用 `--allow-disconnected`，新方法也必须保持一致；筛选器严格核对完整协议。
使用 oracle 时同样给新模型的 physics 命令加 `--xtb`。

### 配对判据与回传

`CONTROL25K_EVALUATION/PHYSICS` 和 `ROT25K_EVALUATION/PHYSICS` 与上述 A 变量同义，
都由 SHHPC 指向已核对的真实历史文件。物理变量指目录，几何变量指 JSON：

```bash
python -m stifflow_benchmark.stiffness_compare \
  --geometry alternative_a "$A25K_EVALUATION" \
  --geometry control "$CONTROL25K_EVALUATION" \
  --geometry a_rot_null "$ROT25K_EVALUATION" \
  --geometry a_delayed "$STIFFLOW_RUN_ROOT/a_delayed/evaluation.json" \
  --geometry a_delayed_rot_null "$STIFFLOW_RUN_ROOT/a_delayed_rot_null/evaluation.json" \
  --physics alternative_a "$A25K_PHYSICS/report.json" \
  --physics control "$CONTROL25K_PHYSICS/report.json" \
  --physics a_rot_null "$ROT25K_PHYSICS/report.json" \
  --physics a_delayed "$STIFFLOW_RUN_ROOT/a_delayed/physics/report.json" \
  --physics a_delayed_rot_null "$STIFFLOW_RUN_ROOT/a_delayed_rot_null/physics/report.json" \
  --output "$STIFFLOW_RUN_ROOT/stiffness-screen.json"
```

若历史 checkpoint 绑定的 train/val manifest 因旧 staging 的绝对路径不同而 hash 不同，
为五个方法分别加 `--labels METHOD ACTUAL_TRAIN_MANIFEST ACTUAL_VAL_MANIFEST`。
每个文件必须匹配其 checkpoint 记录的 hash；只在接受行、矩阵 checksum、label 实现
及配置全部相同时接受路径差异，不能只按 split 或全局数据 hash 放宽。

通过要求：延迟真实方向的 force 相对 A 至少下降 10% 且差值 CI 上界小于零；
相对同 tau 的 null 与 control 也须 CI 上界小于零。相对 A/control 的 MAT-R/P
点估计增加各不超过 0.005 angstrom，COV-R/P@0.5 和 macro chirality 下降各不超过
0.01。五组物理 cohort 必须完整。所有 CI 使用固定 seed42 的 2000 次分子配对 bootstrap。
另报 `(delayed- delayed_null) - (A-rot_null)`；时间与监督总量仍有混杂。
旧 `research_compare` 的几何门槛不能用于延迟方法，会明确拒绝。

`advance_candidate=true` 仅表示有资格在后续预算中做多 seed 确认，不自动升级正式方案。
失败则停止，不在 val 继续扫 tau，也不据此作 ET-Flow 优越性、跨盆地或热力学正确性结论。
回传实际代码 SHA、配置、数据 hash、两项 resume receipt、几何/物理汇总、筛选 JSON，
以及训练时间/显存、生成 NFE/时间、诊断时间与查询次数。历史 stiffness 预处理成本单列，
未知时写明未知。大 checkpoint、构象、完整查询日志留在外部目录，不提交 Git。
