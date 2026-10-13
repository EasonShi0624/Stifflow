# StifFlow：官方 memmap 接入与受控 pilot

当前路线锁定：保留 main@80f82d7 的现有 SO(3) backbone，数据使用
clean-sde@2f20cadeb047b7dc7cd3a363c729373f288c2d08 对应的 **v3 QM9 scaffold**
memmap；三方法共用 128 个 train / 32 个 val 分子的全部官方构象。
Torch 实现接口与轻量检查，真实标签计算、官方生成、Lightning 验收和 pilot 由 SHHPC 执行。
本节之后的 2026-09-21 记录是历史验收，不能替代新接口的真实运行。

## 新路线的接口与验收顺序

- `stifflow.official_data` 复用官方 GEOM memmap 行布局，以 inventory 的固定偏移关联
  原始 pickle；核对元数据哈希、文件尺寸、构象行、原子序、电荷和坐标。复用 SHHPC 已完成
  的大文件 SHA256，不在 Torch 再扫描或复制完整数据。冻结清单前检查 train/val/test 的
  分子级隔离；已有 v2 random 重叠记录保持诊断用途。
- `stifflow.official_labels` 默认覆盖入选分子的全部官方构象，无 top-30 截断或原子数过滤。
  seed42 与规范分子 ID 的 SHA256 确定候选顺序；任意构象失败则整分子排除、记录原因，
  补入下一候选。`--max-candidates` 可限制审计预算；不足指定数量时退出 1 并阻止训练。
  `--audit` 只计算每个分子一条构象，用于成本审计，始终不能用作训练缓存。
- 物理定义仍使用两份正文的 `eq:hessian` / `eq:stiffness`：原始目标不变，另行弛豫并
  proper-align 的 MMFF94 参考是一项明确的曲率迁移近似，保存参考位移、残余梯度、谱和
  kcal/mol/Å² 单位；不放宽现有负曲率、梯度或差分门槛。
- 新标签 schema_version=2、dataset_kind=official_memmap_scaffold，与旧缓存区分。
  每条标签保存全局 `row_ids`，训练检查其与官方 split 的完整行集合一致；按构象行均匀
  抽样，同一 draw ID 在中断恢复后选择同一行。网络和解析 target 的接口保持不变。
- 完整 val 标签额外导出 `references/manifest.json` 及只含坐标/原子序的 NPZ；生成入口
  使用该独立目录，生成时不打开 Hessian 缓存。三方法使用同一参考清单、seed、训练预算
  和 50 步采样，每个分子生成 `2*N_ref`；官方预训练模型单独做接口验收。
- `stifflow.pilot --profile smoke` 检查官方 qm9-o3 权重严格加载及 4 分子×2 构象生成，
  然后分别对 precision / alternative_a / control 执行 50 步连续训练、25 步中断及恢复至
  50 步。`stifflow.verify_resume` 比较模型、优化器、scheduler、目标/torch/CUDA RNG、
  gradient queue 和数据签名。`--profile pilot` 要求相同数据、配置和代码版本的 smoke
  验收记录，运行三方法各 1000 steps、batch8，并调用共享生成、评测和汇总入口。

现有官方 full-data 脚本增加 allocation 检查和输出保护；原始数据、已有 split、报告及
缓存保持只读。任何输出目录已存在时拒绝覆盖。运行中的进度记录只更新本次新建目录。

## SHHPC 交接命令

在 SHHPC 已有工作区中执行；`STIFFLOW_WORKSPACE` 使用当地实际路径，先激活现有
`etflow` 环境并导出 `ETFLOW_ENV="$CONDA_PREFIX"`。本路线的 `etflow` 导入路径需指向
`ETFlow-reference` main@80f82d7，不将 clean-sde 的模型安装到它上面。
需要既有 `phase1-inventory.npz`、`phase3-validation.json`（含三份 memmap SHA256）、
完整 memmap、v3 expanded split、原始 GEOM pickle/summary，以及匹配的 qm9-o3 权重。
不用重新执行已完成的 7 小时全量预处理，也不自动下载数据、安装依赖或提交作业。

三个任务分支固定在：公共分支本次交接 commit（见 GitHub 完整 SHA），Alternative A
`55009a52c2a00b2b379275c9553e3eead3922884`，benchmark
`cbeb314bcd1f99f2a7354477592791f6151821a7`。formulation 文件另行提供，哈希沿用下方历史
记录所对应正文。先核对 worktree、git status、实际导入路径及 SHHPC 现有验收记录。

```bash
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export PYTHONDONTWRITEBYTECODE=1
export STIFFLOW_OFFICIAL_MANIFEST="$STIFFLOW_WORKSPACE/cache/qm9-v3-scaffold-new.json"
# 使用新输出路径；若已有验收通过的同版清单，直接复用。
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/official_data.sbatch"
```

清单作业完成后，先提交小型成本审计，检查失败原因、每行耗时与存储，保持配置冻结：

```bash
export STIFFLOW_LABEL_MODULE=stifflow.official_labels
export STIFFLOW_LABEL_MANIFEST="$STIFFLOW_OFFICIAL_MANIFEST"
export STIFFLOW_LABEL_OUTPUT="$STIFFLOW_WORKSPACE/cache/qm9-v3-cost-train-new"
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/labels.sbatch" --audit --split train --molecules 4 --max-candidates 16
export STIFFLOW_LABEL_OUTPUT="$STIFFLOW_WORKSPACE/cache/qm9-v3-cost-val-new"
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/labels.sbatch" --audit --split val --molecules 2 --max-candidates 8
```

审阅成本报告后，在 SHHPC 资源预算内提交完整标签；审计缓存不进入训练：

```bash
export STIFFLOW_LABEL_OUTPUT="$STIFFLOW_WORKSPACE/cache/qm9-v3-train-new"
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/labels.sbatch" --split train --molecules 128
export STIFFLOW_LABEL_OUTPUT="$STIFFLOW_WORKSPACE/cache/qm9-v3-val-new"
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/labels.sbatch" --split val --molecules 32
```

两份完整标签成功后执行 smoke；成功记录包含各命令、日志、实际版本和返回码：

```bash
export STIFFLOW_TRAIN_LABELS="$STIFFLOW_WORKSPACE/cache/qm9-v3-train-new/manifest.json"
export STIFFLOW_VAL_LABELS="$STIFFLOW_WORKSPACE/cache/qm9-v3-val-new/manifest.json"
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/pilot.sbatch" --profile smoke
# 用实际成功的 smoke job ID 替换 SMOKE_JOB_ID，再提交 pilot。
sbatch "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/pilot.sbatch" --profile pilot \
  --smoke-acceptance "$STIFFLOW_WORKSPACE/runs/pilot-SMOKE_JOB_ID/manifest.json"
```

Slurm 资源沿用已存在的 SHHPC 模板，2 小时是模板时限，不是运行耗时承诺；SHHPC 根据
成本与 smoke 记录设置当地资源/时限。失败留下 manifest 和逐命令日志，不发布成功验收。
pilot 比较产物为 `comparison.json`，另有每方法训练、生成和评测摘要。实际运行结果、
code SHA、配置、数据条件、过滤计数、NFE/耗时/显存及标签成本需一并回传小型摘要。
本阶段不运行正式 test 集、DRUGS、全量标签或超参数搜索。

2026-10-12 Torch 本轮轻量验收：三个 worktree 的 pytest **28 passed**（70.61 s），
包含合成 memmap 的完整行/原子序、分子隔离、整分子排除、缓存重排拒绝、行均匀抽样与
实际 DataLoader 后续抽样一致性，以及既有小型物理/矩阵/CPU 模型测试。
Ruff、Black、20 个 Python 文件语法和 5 个 Slurm 脚本语法通过；保留两条上游字符串转义
警告和一条刻意病态谱警告。记录位于工作区外部运行目录
`runs/memmap-pilot-cpu-20261012/tests.xml` 和 `verification.json`，不随 Git 提交。
这不包含真实 memmap 加载、128/32 完整标签、官方 GPU 生成、Lightning 恢复或 pilot；
这些均标记为 **待 SHHPC 运行验证**。

### 2026-10-13 SHHPC 执行记录（清洗变体 + pilot）

官方 v3 scaffold QM9 列表核验（立体保留身份）：112744/9785/1000 字符串；train 内 155 个
重复条目、val 内 7 个；train∩val 12 个分子（按骨架 45 个）；test 与 train/val 无重合。
另有 592/45/2 个 summary key（自由基/两性离子）无法 sanitize。按用户确认执行清洗变体：
按官方列表顺序去重、train∩val 保留 train、test 也移除与 train/val 的身份重合；同时用
“原子序敏感指纹”重建 v3→memmap 映射（避免 canonical 回退的错映射），映射统一使用
`molecule_identity` 的 sanitize=False 回退。清洗后 expanded 划分：train 111525 分子/
1649352 行、val 9658/62652、test 993/8580，行级与分子级重合均为 0。官方 manifest 为
`cache/qm9-v3-scaffold-new.json`（job 4171797）。

标签与运行（全部通过）：train 128 分子/2126 行（3 个整分子剔除，job 4171820）、
val 32/120（job 4171821，含 references）；smoke job 4171892（9m35s：preflight、官方
val 生成、三方法 50 步 + 25 步中断恢复 verify_resume 全过）；pilot job 4172000（14m09s：
三方法各 1000 步 + 32 分子生成/评测/汇总 `comparison.json`）。早期指标（不构成性能结论）：
MAT-R mean precision/alternative_a/control = 1.588/0.835/0.880 Å，COV-R@0.5 = 0/0.031/0，
四元手性一致率 2.5%/31.2%/27.2%。

环境注意事项：GPU 节点首次 `import torch/etflow` 可能超过 preflight 探针 120s 超时；
SHHPC 本地 wrapper 先做 import 预热（未改动上游脚本）后稳定通过。本次改动：
`molecule_identity` 回退与 `validate_memmap.py` 重映射/清洗逻辑；旧校验/加载报告与旧
expanded 划分以 `*-pre-clean` 保留。

## 正式 QM9 训练：Torch 实现与 SHHPC 交接

本轮从 `stifflow-v1@90d4697a90b24d22d2a0b2ee3f190fe073ececd8` fast-forward 开发。
兄弟包保持 `alternative-a@55009a52c2a00b2b379275c9553e3eead3922884`、
`benchmark-harness@cbeb314bcd1f99f2a7354477592791f6151821a7`。
正式训练接口已实现，Torch 只做小型合成 I/O 和标量 CPU 优化器检查；真实映射审查、
全量物理标签、新接口 GPU 验收、正式训练及评测均 **待 SHHPC 执行**。
按用户要求不再核查上一节 smoke/pilot 的原始日志，其运行状态采用 SHHPC README 摘要。

Torch 本轮检查：现有 `etflow` Python 3.12，实际 etflow 导入仍来自只读 `ETFlow-reference/`；
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，pytest 缓存及 XML 放在仓库外。
以下两个命令合计 **35 passed**（32 + 3），包含一名 DataLoader worker 预取和 epoch 中途恢复，
使用小型合成文件及标量优化器；不包含真实分子模型训练、全量标签或 GPU 验收。

```bash
python -m pytest tests/test_identity.py tests/test_official_data.py tests/test_formal_cache.py \
  tests/test_formal_training.py tests/test_preflight.py tests/test_model_integration.py -q
python -m pytest tests/test_formal_evaluation.py -q
```

变更 Python 文件的 Ruff、Black --check、AST 检查、五个新增脚本的 bash -n、git diff --check
以及四个入口的 --help 均通过。参考代码的旧字符串转义警告和 Lightning 的 worker/Slurm/恢复提示
保留；恢复实际样本、优化器和 RNG 状态已由小型测试比较，真实 CUDA 仍须按下文验收。

`configs/qm9-formal.yaml` 冻结 precision / alternative_a / control × seeds 42/43/44，
每次从头训练 250000 次**优化器更新**，microbatch 8、梯度累积 16，有效 batch 128。
九次共 2.88 亿次构象抽样；数据选择 seed42 固定，行抽样使用独立 RNG domain，
路径 RNG 独立保存/恢复。沿用现有 20 层/160 hidden、AdamW 7e-4、cosine 250k、
source sigma0=rho=1、普通 Euclidean CFM、网络 FP32 与路径/标签 float64。
这是共同预算的 controlled 比较，不能宣称复现了官方 ET-Flow 训练预算或原始 1000 分子 test。

### 1. 固定代码与数据映射

开始前核对三个 worktree 的分支、完整 SHA、未提交差异和实际包导入路径；九次运行保持同一版本。
SHHPC 使用已安装的 etflow 环境，不安装依赖。新增作业脚本不硬编码账号、分区、内存或墙时；
下列 `CPU_ARGS` / `GPU_ARGS` 必须由 SHHPC 按真实资源填写，GPU 任务为单卡、至少四个 CPU。
所有变量使用 SHHPC 的实际路径。`STIFFLOW_WORKSPACE`、`ETFLOW_ENV`、输出目录通过 export 传入。

```bash
cd "$STIFFLOW_WORKSPACE/stifflow-v1"
CPU_ARGS=(--account="$SHHPC_ACCOUNT" --partition="$SHHPC_CPU_PARTITION" \
  --cpus-per-task="$SHHPC_CPU_COUNT" --mem="$SHHPC_CPU_MEMORY" --time="$SHHPC_CPU_TIME")
GPU_ARGS=(--account="$SHHPC_ACCOUNT" --partition="$SHHPC_GPU_PARTITION" \
  --gres=gpu:1 --cpus-per-task=4 --mem="$SHHPC_GPU_MEMORY" --time="$SHHPC_GPU_TIME")

sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch freeze_data \
  --manifest "$STIFFLOW_OFFICIAL_MANIFEST" --scaffold-dir "$STIFFLOW_V3_SCAFFOLD_DIR" \
  --smiles-npz "$STIFFLOW_SMILES_NPZ" --output "$STIFFLOW_FREEZE_ROOT"
```

此步骤只重审已有 v3 列表→inventory 映射，复用原 phase3 的大文件验证，不重新扫描完整 memmap。
共享身份函数统一普通氢表示、保留立体/同位素/电荷；指纹保留全部显式氢及完整 indexed connectivity。
多个 UID 只有落在同一个 inventory 分子组时才可合并；跨组歧义不选第一项。
输出每个官方列表条目的 `mapping.jsonl`、`audit.json` 及 `qm9-v3-clean-v1` manifest。
全部新映射行必须与现有清理版精确一致；歧义、test 未解析或行差异会阻止冻结，保留旧划分，
由 SHHPC 回传受影响 ID/原因后处理，不能默默改变 test 数量。
train/val 原本已排除且仍未映射的条目按原因记录，不能伪装成完整官方划分。

成功后设置 `STIFFLOW_FROZEN_DATA="$STIFFLOW_FREEZE_ROOT/manifest.json"`。

### 2. 全量候选标签、可恢复分片和 mmap 合并

每个候选保留所有官方构象；任何物理标签失败仍整分子拒绝。处理完成表示所有候选均有终态，
分别记录候选数、成功数、拒绝数、原因、构象行数及预处理成本。三方法使用同一个合格训练清单。
每 1024 个候选一片、单进程/BLAS 线程1；通过 CPU 作业数组并发限制控制集群开销。
数组上界从冻结 manifest 的实际分子数计算：`ceil(molecules/1024)-1`。

```bash
export STIFFLOW_LABEL_SPLIT=train
# 可选：复用已有同 split 的完整 v2 pilot manifest；会核对原始目标、行及两种因子的矩阵恒等式。
export STIFFLOW_REUSE_LABELS="$STIFFLOW_OLD_TRAIN_LABELS"
sbatch "${CPU_ARGS[@]}" --array="0-$TRAIN_SHARD_LAST%$LABEL_CONCURRENCY" scripts/formal_labels.sbatch

export STIFFLOW_LABEL_SPLIT=val
export STIFFLOW_REUSE_LABELS="$STIFFLOW_OLD_VAL_LABELS"
sbatch "${CPU_ARGS[@]}" --array="0-$VAL_SHARD_LAST%$LABEL_CONCURRENCY" scripts/formal_labels.sbatch
```

`STIFFLOW_LABEL_ROOT` 指向新的外部缓存目录。失败/中断任务重复提交同一 shard-index 时自动 resume；
已完成候选不重算，输入配置/数据不同会拒绝恢复。I/O 故障中断任务，不冒充物理样本拒绝。
保留物理审计 NPZ 与每候选原子写入的 receipt，不在每个分子后重写全量 JSON。

```bash
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal_cache merge \
  --manifest "$STIFFLOW_FROZEN_DATA" --config configs/qm9-formal.yaml --split train \
  --shards "$STIFFLOW_LABEL_ROOT"/train/shard-*/shard.json --output "$STIFFLOW_PACKED_TRAIN"
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal_cache merge \
  --manifest "$STIFFLOW_FROZEN_DATA" --config configs/qm9-formal.yaml --split val \
  --shards "$STIFFLOW_LABEL_ROOT"/val/shard-*/shard.json --output "$STIFFLOW_PACKED_VAL"
```

合并检查完整候选覆盖、拒绝账目和全部行配对，产生 schema3 清单；按原子数最多 4096 行一片，
因子 `.npy` 用 mmap 读取，graph/source 也打包。读取器最多打开16片、缓存64个 graph/source，
样本复制后可安全淘汰 mmap。行偏移只计算一次。
`--resume` 可恢复合并，已提交的分片检查哈希后复用；中断留下的未提交产物保留，不自动清理。
`archive_bytes` 与 `packed_bytes` 分别记录归档和打包占用：二者都会保留，SHHPC 须预留总配额。

```bash
export STIFFLOW_TRAIN_LABELS="$STIFFLOW_PACKED_TRAIN/manifest.json"
export STIFFLOW_VAL_LABELS="$STIFFLOW_PACKED_VAL/manifest.json"
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal references \
  --manifest "$STIFFLOW_FROZEN_DATA" --config configs/qm9-formal.yaml --output "$STIFFLOW_REFERENCE_ROOT"
```

参考导出覆盖全部清理版 val/test，并检查图特征与参考手性；不依赖 Hessian 合格性。
val/monitor.json 是 seed42 预先冻结的512分子子集，val/manifest.json 和 test/manifest.json 是完整清单。
test 仅用于最终固定配置评测，不能拟合标签尺度或挑选超参数。

### 3. 新接口验收与九次正式训练

Torch 的小型测试不能代替真实 CUDA 运行。全量合并后，用同一 mmap 清单完成新接口验收：
三方法各“连续40更新”对比“20更新后恢复至40”，batch8×accum16，检查样本游标、权重、优化器、
scheduler、路径/torch/CUDA RNG 和梯度裁剪状态。先以300秒 import 预热，再执行原120秒探针。

```bash
sbatch "${GPU_ARGS[@]}" scripts/formal_acceptance.sbatch
# STIFFLOW_ACCEPTANCE_ROOT 必须是新目录。
export STIFFLOW_ACCEPTANCE="$STIFFLOW_ACCEPTANCE_ROOT/workflow.json"
sbatch "${GPU_ARGS[@]}" --array="0-8%$TRAIN_CONCURRENCY" scripts/formal.sbatch
```

数组0/1/2＝precision seeds42/43/44；3/4/5＝Alternative A；6/7/8＝control。
长训练强制匹配新验收的代码、配置和标签哈希；旧 smoke/pilot 原始日志不是启动条件。
按40步验收的实际稳定吞吐、内存与显存设置墙时和并发；不从14m09s pilot 摘要估算正式 GPU 核时。
预处理成本单列。每次训练3200万抽样，九次2.88亿。

`STIFFLOW_RUN_ROOT/<method>-seed<seed>/segments/<job-id>/` 保存独立运行记录和 CSV。
每500优化器更新保存恢复点，只保留最近两个，`checkpoints/latest.ckpt` 指向最新完整点；
每25k更新保存 `snapshots/step-XXXXXXXX.ckpt`（权重及元数据），最终保存完整 checkpoint。
未完成任务再次提交对应数组 ID 自动恢复，读取器从已消费 draw ID 接续，预取不计入游标；
仅在完整累积边界恢复。已完成250k的任务不要重复提交。
每5000更新在固定4096个验证抽样上记录 CFM loss；这是诊断，不比较不同方法的 loss 尺度。
代码记录优化器更新数、已消费样本、耗时、实际 Slurm 资源、版本、配置与 peak GPU memory。

### 4. 监测、最终评测与三个 seed 汇总

每25k快照监测固定512个验证分子的 MAT/COV/手性；主比较统一250k最终权重。
生成沿用 Euler50、每分子2×参考数、推理 seed42、现有 heavy-atom RMSD 协议。
GPU 生成与 CPU RMSD 评测分别申请资源，不让 Torch 代跑。

```bash
sbatch "${GPU_ARGS[@]}" scripts/formal_task.sbatch formal generate \
  --config configs/qm9-formal.yaml --checkpoint "$STIFFLOW_SNAPSHOT" \
  --references "$STIFFLOW_REFERENCE_ROOT/val/monitor.json" --stage monitor --output "$STIFFLOW_MONITOR_OUTPUT"
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal evaluate \
  --manifest "$STIFFLOW_MONITOR_OUTPUT/generated/manifest.json" --output "$STIFFLOW_MONITOR_OUTPUT/evaluation.json"

# 九次训练完成后，分别对 full val 和 frozen test 执行；TEST_CHECKPOINT 是250k快照。
sbatch "${GPU_ARGS[@]}" scripts/formal_task.sbatch formal generate \
  --config configs/qm9-formal.yaml --checkpoint "$TEST_CHECKPOINT" \
  --references "$STIFFLOW_REFERENCE_ROOT/test/manifest.json" --stage final --output "$FINAL_OUTPUT"
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal evaluate \
  --manifest "$FINAL_OUTPUT/generated/manifest.json" --output "$FINAL_OUTPUT/evaluation.json"

sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal summarize \
  --reports "$FORMAL_EVAL_ROOT"/*/evaluation.json --output "$FORMAL_EVAL_ROOT/comparison.json"
```

同一次汇总提供恰好九个**同 split**的最终报告，val 与 test 分开汇总。
完整参考清单按哈希核对，生成中止后未处理的 ID 仍以失败保留；存在失败时结果标记不完整，
不能进入主比较。报告 MAT-R/P mean/median、COV-R/P@0.5、手性、NFE、耗时和显存，
三个 seeds 汇总均值及样本标准差（ddof=1）。生成不读取 Hessian 缓存。
出现非有限 loss/gradient/坐标或配对错误立即失败；手性低是监测风险，不自动删构象或改变方法。

SHHPC 回传完整代码 SHA、实际配置/输入哈希、环境、分片和过滤数量、作业命令与资源、
新接口验收结果及指标摘要。数据、标签、checkpoint、构象和大量日志仍留在 Git 外。

## 历史 CPU 实现与验收记录

按用户后续指示，暂缓 GPU 验收，先实现可在 CPU 检查的后续部分。
现有主方案、Alternative A、MMFF94 标签和共享评测代码；训练/生成接入已写入但尚未运行。
最新检查发现现有环境已是兼容的 CUDA 12.1 构建，CPU 导入和模型单元测试通过；
本代理未执行环境修改。阶段 1 的 GPU 短生成及后续完整训练仍按用户指示暂缓。
参考仓库、原始数据及两份 formulation 均保持不变。

运行依据：`formulation/StifFlow_core_formulation_v1.tex` 正文 Version 2，2026-09-15；
Alternative A 位于兄弟 worktree 的 `formulation/StifFlow_v1_Alternative_A..tex`，2026-09-16。
核心实现使用正文公式，详见下文。诊断记录 PDF/LaTeX 哈希，但未逐页比较两种格式，未找到 companion verification 脚本。

## 环境诊断

从工作区根目录执行；使用现有环境，不做 editable 安装：

```bash
source /scratch/ys6132/programs/miniforge3/etc/profile.d/conda.sh
conda activate etflow
export STIFFLOW_WORKSPACE="$PWD"
export ETFLOW_ENV="$CONDA_PREFIX"
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export PYTHONDONTWRITEBYTECODE=1
python -m stifflow.preflight --output "runs/preflight-$(date +%Y%m%d-%H%M%S).json"
```

非零退出表示导入、radius graph 或模块路径校验未通过。JSON 保存解释器、包路径、
conda 构建、导入错误、各仓库提交和 tracked diff、untracked 文件哈希、formulation 哈希，
以及三个本地 checkpoint 与同名官方 YAML 的对应关系及哈希。已有报告拒绝覆盖。
`--gpu` 仅在 Slurm 分配内使用，同时验证 CUDA 张量运算和 CUDA radius graph。

阶段 0 历史诊断曾复现：`pytorch 2.4.1 py3.12_cpu_0` 与
`pytorch-cluster 1.6.3 py312_torch_2.4.0_cu121` 不匹配，
`torch_cluster` 报 `libc10_cuda.so` 缺失。环境另有 `cpuonly 2.0`、
`pytorch-mutex 1.0 cpu` 及 `pytorch-cuda 12.1`。

以下是阶段 0 留存的修复候选，当前兼容构建已存在，不应再次执行。原目标是在原环境保留 PyTorch 2.4.1 / Python 3.12 / CUDA 12.1，
切换 PyTorch CUDA 构建及 mutex，并移除冲突的 cpuonly 元包。先审查 solver 事务：

```bash
conda install --offline --dry-run --json --override-channels \
  -c pyg -c pytorch -c nvidia -c conda-forge -n etflow \
  'pytorch=2.4.1=*cuda12.1*' 'pytorch-mutex=1.0=cuda' 'pytorch-cuda=12.1'
```

本轮默认 channel 的离线求解报缺少 `pytorch-mutex=1.0=cuda`；按官方 channel 重试
在 60 秒时限内未返回事务（退出码 124）。未安装或修改任何包。
这是候选方案，不是已验证可执行的安装命令。如果本地缓存不足，需获得联网授权后重新
求解；若 solver 仍保留冲突或升级无关包，不能直接应用。实际环境修改须另获授权。
当前登录节点 `/tmp` 已满；本次未清理。作业使用独立的工作区 `cache/tmp/$SLURM_JOB_ID`。

## 官方短生成

`scripts/smoke.sbatch` 固定 4 个测试分子、每个 2 个构象、seed 42，默认申请一个 GPU、
4 CPU、24 GiB 内存、30 分钟。账号可在提交时替换为已允许的账号；站点若要求 partition，
在提交命令中显式指定。脚本不会自行提交作业。

未来取得本次作业提交授权后，先重新运行上述 preflight，再执行：

```bash
mkdir -p "$STIFFLOW_WORKSPACE/runs"
sbatch --output="$STIFFLOW_WORKSPACE/runs/etflow-smoke-%j.log" \
  "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/smoke.sbatch"
```

数据来自现有 `dataset/torsional_diffusion/QM9/test_mols.pkl`，按分子 ID 排序取前四个，
只用于接口检查，不拟合参数或选择方法。复用官方 `process_test_mol` 和 `EuclideanDataset`，
在全新外部 cache 中保存逐分子 `.pt`，保留所有参考构象、原子顺序和显式氢。
加载本地 `qm9-o3.ckpt` 配对 `configs/qm9-o3.yaml`，`weights_only=True` 读取，
`load_state_dict(strict=True)` 核验全部键和形状。复用 `batched_sampling`，保持官方网络、
特征、harmonic prior、50 步 ODE、归一化和 post-hoc parity；本阶段不修正基线的特征编码。
为避开上游按 SMILES 缓存坐标的问题，每次在 RDKit 图副本上重新设置当前坐标。

结果保存在全新 `runs/etflow-smoke-JOBID/`：每分子一个 NPZ（生成/参考坐标、原子序号、
原子索引、分子 ID、SMILES），以及 manifest。核验键/形状、图原子与键映射、氢、坐标有限性、
中心误差不超过 `1e-4 Å`、前向调用数及保存/读取一致性。记录 GPU 同步计时、峰值显存，
失败时留下状态和 traceback，不把失败样本静默删去。上游特征及 checkpoint 不匹配时直接失败。

通过仅表明数据—权重—生成接口可用，不代表训练、论文指标或 StifFlow 数学已验证。
GPU 生成按用户指示暂缓，该端到端验收仍为未运行。

## 低成本检查

```bash
python -m unittest discover -s stifflow-v1/tests -v
bash -n stifflow-v1/scripts/smoke.sbatch
python -m stifflow.smoke --help
```

上述 unittest 命令覆盖公共 CPU 检查；跨 worktree 的完整测试命令见下文。它们不能代替真实模型生成。缓存、日志和 checkpoint
均置于仓库外，不覆盖已有实验，不自动下载数据或权重，不调用 wandb。

## 后续 CPU 实现与边界

- `geometry.py`：Helmert 去平移正交基、proper Kabsch、按几何确定旋转秩；不做质量加权。
- `source.py`：`eq:graphsource` 的图弹簧 source，默认 sigma0=1 Å、rho=1；只读原子数和无权化学键。
- `precision.py`：正文 Version 2 的 `eq:defaults/factor/adot/trajectory`，smoothstep 均值和 precision bridge；
  内部 float64，训练使用同一 z 的解析 sample target，无谱上截断或 Hessian loss weighting。
- 兄弟包 `stifflow_alt_a.path`：默认 `A_E`，对齐因子 `L_K=T_K S`；没有 A_W、低秩、delayed gate。
- `targets.py`：独立 Gaussian z 和 Haar proper rotation；同时旋转目标与两个采样向量，
  不对齐源噪声。时间只拒绝转换为 float32 后的精确 0/1，不裁掉固定端点区间。
- `data.py`：冻结文件名顺序与分子级划分；保留 top-30 原始构象、原始权重/能量和稳定构象 ID。
  清单冻结当前原始 `glob` 顺序；该顺序下官方 1000 测试分子与 `split.npy` 的 test 数组并不重合，
  历史 train/val 文件顺序仍无法独立证明（见“数据与 MMFF94 审计命令”中的实测记录）。
- `stiffness.py`：原目标的副本做 MMFF94 弛豫，再 proper-align 至原目标；能量 kcal/mol、长度 Å，
  epsilon_E=RT=0.5924849497，kappa_min=kappa_rot=epsilon_E/(1 Å)^2，是已选试运行尺度。
  梯度中心差分 h=1e-4 Å 并检查 h/2；每次 CalcGrad 前对同坐标 CalcEnergy，防止 RDKit 复用旧距离缓存。
  梯度阈值 1e-4、Hessian 相对残差阈值 1e-4、负振动曲率阈值 -1e-3 均记录于审计，
  不放宽门槛以补齐样本。保存原始谱、floor mask、原始 Hessian、K、两方法因子及曲率迁移位移。

已执行的小型 CPU 测试覆盖非对易 SPD、因子恒等式、速度有限差分、Gaussian 协方差连续性、
完整端点误差收敛、相同协方差退化、线性/非线性旋转秩、旋转/置换、极端谱，以及 A_E 文档反例。
对角 condition(K)=1e18 压力测试保留 SciPy 病态矩阵警告；这不代表任意极端谱均已验证。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STIFFLOW_WORKSPACE/cache/cpu-check-tmp"
export TMPDIR="$STIFFLOW_WORKSPACE/cache/cpu-check-tmp"
python -m pytest -q -ra stifflow-v1/tests stifflow-v1-alternative-A/tests stifflow-v1-benchmark/tests
```

模型检查位于 `test_model_integration.py`；最新检查实际执行并通过了 40 原子邻域旋转/置换
一致性与有限梯度，以及 precision/A_E 两种方法的小型 CPU 优化器、scheduler、模型与 RNG
保存/恢复后下一步更新一致性。现有 PyTorch 为 2.4.1 py3.12_cuda12.1_cudnn9.1.0_0，
torch-cluster 为原 cu121 构建。没有使用 stub，也没有由本代理安装或修改环境。

## 数据与 MMFF94 审计命令

首次冻结清单（已有同名清单时拒绝覆盖，后续直接使用它）：

```bash
python -m stifflow.data --raw-dir dataset/rdkit_folder/qm9 \
  --split-path dataset/torsional_diffusion/QM9/split.npy \
  --official-test-smiles dataset/torsional_diffusion/QM9/test_smiles.csv \
  --output cache/qm9-split-manifest.json
python -m stifflow.stiffness --manifest cache/qm9-split-manifest.json \
  --output cache/qm9-mmff-audit-new --molecules 2 --conformers 1 --max-atoms 12
```

清单计数 106586/13323/13323。2026-09-29 在 SHHPC 复跑同一 `stifflow.data` 命令：官方 1000 个测试
分子（`test_smiles.csv` 与 `test_mols.pkl` 键一致）按官方 `prepare_data` 使用的 `rdkit_folder/qm9`
glob 映射只有 87 个落在 `split.npy` 的 test 数组，806 落 train、107 落 val，隔离校验以 913 个失败
分子退出；此前“官方 1000 测试分子全部隔离于 test”的结论无法在本机数据上复现。ET-Flow
`prepare_data.py` 实际只用 `split.npy[0]/[1]`（train/val）和 `test_mols.pkl`（test），若按其训练，
913/1000 官方测试分子会进入 train/val；确认官方划分口径前，不应把该 1000 分子视为已隔离评测集。
原始数据未改动，证据存在本集群 `runs/etflow-ref-smoke/phase1-splits.json` 与
`phase1-stifflow-data.log`（未入库）。2026-09-21 的实际审计产物在
`cache/qm9-mmff-audit-20260921/`：乙炔 `C#C` 通过，旋转秩 2，差分步长减半相对误差
5.97e-8，A_E 因子残差 2.03e-15；能量二次比值随位移减半由 1.00036 趋近 1.000022。
`C#CC#C` 以残余梯度 3.07844e-4 超阈值失败，完整失败记录保留，命令退出 1。
这不是训练分子筛选或性能实验；不能将这个含失败/部分构象的 cache 用于训练。

默认最多读取 32 个候选，只对最多 2 个小分子/每个 1 构象做标签。`--split val` 可审计验证集。
大于 4 个分子或 2 个构象的预处理必须在 Slurm 内；本轮未运行。完整训练缓存须覆盖每个入选
分子的全部 retained top-30 构象，标记 audit_partial=false，且失败已通过明确共同清单解决。
不要把 max-atoms 审计过滤直接沿用到正式 pilot。
更大规模预处理提交 `scripts/labels.sbatch`（CPU，默认 partition `parallel,debug,chem,rooster`）：
export `STIFFLOW_LABEL_MANIFEST` 与 `STIFFLOW_LABEL_OUTPUT`，其它参数透传，例如
`sbatch scripts/labels.sbatch --molecules 128 --conformers 30 --split val`；该脚本不自动提交。

## 官方 clean-sde memmap 数据方案（正式）与全量验证

上一节 main 分支 `.pt` 方案的审计记录仅适用于该方案。正式预处理与训练数据采用
**clean-sde@2f20cad + Zenodo 官方划分**（论文与已发布 checkpoint 对应的 memmap 格式）；
`ETFlow-reference` 保持 main@80f82d7 不变，另建 `ETFlow-clean-sde/` 克隆。
clean-sde 代码依赖 `torch_sparse/torch_scatter/h5py`，已按 torch 2.4.1+cu121 的 PyG wheel
装入现有 `etflow` 环境（未改 torch 版本）。

官方文件已下载并 md5 校验（`cache/etflow-official/zenodo/`）：
- v2 record 14226681：`QM9.zip`(1bed2e29…)、`DRUGS.zip`(5e9f7dc7…)、`XL.zip`(4759ae27…)
  与 `dataset/ET-Flow_zenodo/` 逐文件一致；提供 conformer 级 random 索引与 QM9 scaffold 子采样索引。
- v3 record 16551316：`scaffold_data.tar.gz`(142b4651…)，含 QM9/DRUGS scaffold 分子级 SMILES 列表
  （112744/9785/1000 与 260752/29185/1000）及 scaffold checkpoint。

预处理命令（官方 `scripts/preprocess.py`，SHHPC job 4072957，7h00m，MaxRSS 14.9GB）：

```bash
PYTHONPATH="$STIFFLOW_WORKSPACE/ETFlow-clean-sde" \
python ETFlow-clean-sde/scripts/preprocess.py -p dataset/rdkit_folder \
  -d cache/etflow-official/geom-preprocessed
```

结果与脚本常量逐项一致：437580 分子 / 33078483 构象 / 1656320500 原子 / 15904498 unique 原子；
`atomic_inputs.memmap` 33,126,410,000 B 及 `pos_index_range.memmap`、`energy.memmap`、
`smiles.npz`、`subset.npz` 齐备。校验脚本位于 `scripts/etflow_official/`（接受
`STIFFLOW_WORKSPACE` 环境变量）：`build_split_manifest.py`、`count_inventory.py`、
`validate_memmap.py`、`loading_test.py`。

校验（job 4159963）：文件尺寸/形状精确、per-molecule 边界与计数完全一致、能量有限、
400 个分层抽样分子与原始 pickle 的坐标/原子序/电荷/能量/逐构象 SMILES 完全相等。实测 split
语义（conformer 级）：v3 scaffold（主）QM9/DRUGS 的 test 行与 train/val 行级 **0 重叠**、train/val
行级 0 重叠；v2 random（对照）QM9 test 有 6457 行落入 train、578 行落入 val，DRUGS 有
18710/1615 行落入 train/val，**不能作为隔离评测**；v2 的 QM9 scaffold 索引只是 v3 分子列表的
子采样（train 少 794528 行），不得与 v3 混用。v3 分子列表回映射时 13 个分子因字符串无法解析而
未映射（QM9 train/val 4/1，DRUGS train/val 4/4；test 全部映射），已在报告中单列。

数据加载（同 job，32 workers）：`GEOM` len=33078483；随机访问 1244 items/s；featurized
DataLoader 13335 items/s；完整 QM9 scaffold train（1,654,366 构象）一个 pass 124s，峰值 RSS
6.7GB；val/test 首 batch 正常。集成 smoke（job 4160201）：clean-sde `BaseFlow` 2 步 GPU
训练，492 tensors、9,064,209 参数，checkpoint strict 重载通过。

正式锁定划分（本地生成，未入库）：`cache/etflow-official/splits/expanded/` 下
`scaffold_qm9_{train,val,test}_indices.npy`（1654366/62801/8558 行）与
`scaffold_drugs_{train,val,test}_indices.npy`（25464381/2803983/106988 行）为主，`random_*`
仅作对照；memmap sha256 见 `runs/etflow-official/phase3-validation.json`。官方 split 的
conformer 级目标规模（QM9 scaffold 约 1.65M）意味着 StifFlow K 标签需单独规划，不能直接沿用旧
top-30 manifest。

## 共享训练和生成入口（待运行）

`model.StifFlow` 继承 ET-Flow BaseFlow/BaseModel，复用 TorchMDDynamics、Lightning、优化器、
scheduler 及平方欧氏损失 `batchwise_mse_loss(reduce="sum")`。只换监督及图源；关闭桥噪声、
源到目标对齐、post-hoc parity/churn。原子特征只修正手性槽为 CIP 编码；边特征增加键类型、
E/Z stereo 和共轭，三方法共用 SO(3) 网络。radius 邻居上限按当前最大分子原子数确定。
上游距离/归一化语义保持原样。训练的稠密标签列表留在 CPU float64，只有 xt/Ut 转 float32。

`configs/qm9.yaml`：seed42、batch8，smoke=32 train/8 val/50 steps，pilot=128/32/1000 steps；
网络规模沿用官方 SO(3) 配置，仅作上述 StifFlow 必要适配。`prior_type` 是父类构造参数，
实际 source 始终由 source_config 控制；父类 sample 已禁用，避免误采其原算法。

```bash
# 仅在未来获准的计算节点作业内执行；需要完整 train/val 标签缓存。
python -m stifflow.training --method precision --profile smoke \
  --train-labels cache/train-complete/manifest.json --val-labels cache/val-complete/manifest.json \
  --output runs/precision-smoke
# 同一入口 --method alternative_a 或 control；pilot 使用 --profile pilot。
# 中断/恢复检查可用 --stop-after 25，再从 final.ckpt --resume 到同一预算。
```

`training.py` 保存 Lightning 模型/优化器/scheduler 及独立 target RNG、torch RNG、gradient queue；
图/构象抽样按 draw ID 固定，恢复时检查数据哈希、seed 和配置。小型 CPU 状态恢复测试通过；
完整 Lightning Trainer 中断/恢复及 50-step 作业尚未执行，不能把前者当作后者的验收。
`train.sbatch` 已准备，需导出 STIFFLOW_TRAIN_LABELS / STIFFLOW_VAL_LABELS，**尚未提交**。

`model.generate(model, graph, source_config, sampler_config)` 只接收图，默认 50 步左端 Euler
从 0 到 1，不查询条件 Eulerian 奇点。输出坐标和实测前向调用数、同步时间、峰值显存。
统一生成/评测入口及文件协议见兄弟 worktree `stifflow-v1-benchmark/README.md`。
阶段 4/5 的真实训练、恢复、生成、25/50/100 步比较和 DRUGS 扩展均待计算节点验证；
当前没有完整 pilot、论文指标或方法性能结论。

最终验证：跨三个 worktree 的 pytest 为 **16 passed**（56.40 s），包含小型 CPU graph-only
两步采样；保留 2 条上游字符串转义警告和 1 条刻意病态谱警告。官方 qm9-o3 checkpoint
按原 YAML 在 CPU 严格匹配全部 492 个 state-dict 项（8,337,809 参数），CPU radius graph 通过。
记录位于 `runs/stage2-3-cpu/tests-final.xml`、`checkpoint-cpu.json`、`verification.json`；
这些检查不包含官方 50 步 GPU 生成或正式训练作业。

## 另一集群上的代码布置

本集群只维护代码和轻量静态检查。数值测试、标签批量预处理、CPU/GPU intensive 的任务均迁至
另一集群运行；以下是交接命令，不代表本集群已执行这些任务。

GitHub 仓库的三个分支分别保存公共实现、Alternative A 和 benchmark，没有合并或复制公共代码：

```bash
# 在目标集群选定的同一工作区内执行。保留以下目录名，供默认路径与 provenance 使用。
git clone --branch stifflow-v1 git@github.com:EasonShi0624/Stifflow.git stifflow-v1
git -C stifflow-v1 worktree add ../stifflow-v1-alternative-A alternative-a
git -C stifflow-v1 worktree add ../stifflow-v1-benchmark benchmark-harness
# 使用只读参考代码（本次复用版本），不在其中开发。
git clone https://github.com/shenoynikhil/ETFlow.git ETFlow-reference
git -C ETFlow-reference checkout 80f82d7d142bc3872412d2966a8d71ed14462b18
# editable 安装到目标环境；--no-deps 不改动现有依赖，preflight 要求 etflow 导入路径指向该目录。
python -m pip install --no-deps -e ETFlow-reference

export STIFFLOW_WORKSPACE="$PWD"
# 激活目标集群已有、兼容的 etflow 环境后设置，不使用本集群的绝对路径。
export ETFLOW_ENV="$CONDA_PREFIX"
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export PYTHONDONTWRITEBYTECODE=1
```

目标环境的实际 `etflow` 导入路径需指向所核验的 `ETFlow-reference`（上面的 editable 安装即满足）；
先在目标集群执行 preflight，不要假设两个集群安装相同。SHHPC/nyushc 的 `scripts/*.sbatch` 已配置
account `wjg3`、GPU partition `rooster,chem`、CPU partition `parallel,debug,chem,rooster`，
`ETFLOW_ENV` 默认 `$HOME/anaconda3/envs/etflow`；其它站点必须替换这些资源参数，并按当地规则提交。

数据、checkpoint、缓存、运行结果和原先未跟踪的 formulation PDF/LaTeX **不随本次代码推送**。
它们需由用户在目标集群提供；README 中 `cache/`、`runs/` 的既有验证记录只在当前工作区，
不是 GitHub 中的文件。可选官方 checkpoint 缺失时 provenance 记录 `exists=false`；真正使用
某项数据/权重的任务仍会在读取它时明确失败，不会使用随机权重代替。

全文数学依据的目录与文件名见本 README 顶部；复现实验时另外提供原文件并记录哈希。
先运行共享测试，再解决 MMFF94 未达到 stationarity 的样本及完整共同训练清单，最后开展
Lightning smoke、恢复验收和受控 pilot。过去的 16 项 CPU 测试结果是本次迁移之前的记录，
不能代替目标集群上的验证。
