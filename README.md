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

### 2026-10-18 评估工具补丁（staged manifest / 多片段 / 增量落盘）

训练时 packed cache 被复制到节点本地并只重写 manifest 的绝对 `path`，因此九个 run 的
`training_signature.train/val` 文件哈希不同。`formal summarize` 新增
`--normalize-staged-labels`：以冻结的 `official_data` 哈希作为 canonical 数据身份，把每个 run 的
staged train/val 哈希记录在 `staged_label_manifests` 中而不参与相等性比较；其他数据/配置差异
仍然报错。benchmark 的 `summarize` 提供同名参数。

formal 评测现在允许参考集中的多片段分子（val 12 个，test 0 个），并在报告 protocol 与
`aggregate.multi_fragment_molecules` 中记录；benchmark CLI 默认仍拒绝，只有显式
`--allow-disconnected` 才评测。这样 val 不会因 12 个多片段分子被判为不完整，test 主比较数值不变。

`formal generate --resume` 复用已存在的 `generated/manifest.json`；benchmark 每
`--flush-every`（默认 50）个分子原子写一次 manifest，并校验 checkpoint、参考清单、模型/采样配置、
训练签名和代码 provenance。已完成且 NPZ 指纹完整的分子跳过，失败/中断的分子重生成。
以上补丁只涉及生成/评测编排与报告，不改变训练目标、Euler50、2×参考数、seed42 或 RMSD 协议；
训练 checkpoint 仍对应 `stifflow-v1@793d502`。

SHHPC 回传完整代码 SHA、实际配置/输入哈希、环境、分片和过滤数量、作业命令与资源、
新接口验收结果及指标摘要。数据、标签、checkpoint、构象和大量日志仍留在 Git 外。

### 2026-10-13 SHHPC 执行更新（严格映射与 freeze）

按确认的“严格映射重生成”口径，`validate_memmap.py`（已在 5e54e9f 改用
`identity.SplitMapper`）重新生成 expanded 划分：train 111457 分子/1650255 行、
val 9654/62648、test 993/8580，行级与分子级重合为 0；旧 first-match 划分与报告以
`*-firstmatch` 保留。官方 manifest 重建为 `cache/qm9-v3-scaffold-strict.json`
（job 4183794）。freeze（job 4183840）三 split `matches_existing=True`、行差 0；官方
v3 列表存在 81 条固有跨组歧义（train 77、val 4），test 无未解析，按确认把门控放宽为
“三 split 行集一致 + test 无未解析 + 全部排除项记录在 audit/mapping”
（`stifflow/freeze_data.py`；受影响条目本地见
`runs/etflow-official/freeze-ambiguous-strict.json`）。冻结清单
`cache/qm9-v3-freeze-strict/manifest.json`（variant `qm9-v3-clean-v1`）。
全量标签分片完成：train 109/109、val 10/10（job 4183844/4183845）；mmap 合并完成
（job 4184229/4184230：train 1,650,255 行 / packed 91,831,100,092 B、val 62,648 行；
`cache/qm9-v3-packed-strict/{train,val}/manifest.json`，schema3 completed）。

新接口验收通过（job 4221188，42m58s；`cache/qm9-v3-acceptance-strict/workflow.json`）：
三方法 40 更新连续 vs 20+恢复，`verify_resume` 全部一致，peak GPU memory 1.9GB。
**实测吞吐（正式配置 batch8×累加16）**：precision ≈9.9 s/update、alternative_a ≈7.9、
control ≈4.1；换算 250k updates 每次约 28.7/22.7/11.8 天，9 次合计约 190 GPU-天。
按用户决定**先不启动正式训练**，将吞吐与预算分析回传 Torch 调整正式配置后再执行。

未完成的阻塞项：`formal references` 在 `official_data.load_molecule` 的跨构象签名检查上
失败。第一类为分离片段物种（例 `qm9:C=C1ON=C(N)C1=O.N`），已把连通性检查移出
`load_molecule`（连通性是 stiffness 标签域约束，references 需覆盖全部冻结分子）；
第二类为同一分子不同构象的**键列表顺序**不同（例 `qm9:NCC=C([N][O])[C@H]1CO1`，
position 3428，26 行，bonds 2 种变体），需要 Torch 决定是否改为无序/集合比较。
本地已应用“立体容忍 + 移除连通性检查”补丁；由于该补丁改变代码 diff，现有
acceptance 签名将失效，Torch 调整后需重新跑 `formal_acceptance`。

### Torch 训练速度审查与小规模 GPU 测试

本轮审查以 SHHPC `stifflow-v1@756637e954213f1ef5b340b606dd7b2288597df2`
为基准。远端只提交了上述 README 摘要，没有原始 `workflow.json`、逐步计时或作业
日志，因此 9.9/7.9/4.1 s/update 是 SHHPC 报告值。`formal.acceptance` 的命令耗时包含
解释器/模型初始化、数据加载及最终 checkpoint 保存；正式读取器又会在每个 worker
首次打开分片时计算整文件 SHA256。40 更新验收不能证明 250k 更新稳定吞吐，约
190 GPU-天的外推需用 SHHPC 全量缓存的稳定区间重新核对。

本轮保留所有逐次 loss/gradient/parameter 有限性检查，将 gradient/parameter 的
CUDA 结果合并成一次 host 判断；precision 监督直接应用 `A_t z` 和 `A_dot_t z`，
保持 Version 2 的非对易顺序、float64、同一噪声与 RNG 流。完整因子与标签生成实现
保留，已有物理标签无需重建。训练入口支持 profile 的 `pin_memory` 和
`prefetch_factor`，默认值与原配置相同；网络仍接收图、坐标及时间，标签矩阵留在 CPU。

references 使用规范化的成对键端点/键型比较，避免 RDKit 键顺序差异导致误拒绝；
`OfficialDataset.load_molecule(..., allow_stereo_variation=False)` 默认检查立体条件。
仅 references 显式容忍并记录 `stereo_variation_rows`，使用第一官方构象作为条件图，
保留所有参考构象；连通性仍由 stiffness 标签生成检查。

`stifflow.speed_benchmark` 复用现有 StifFlow、Lightning、mmap 读取器和优化器。
输入是 Torch 本地已冻结 legacy 训练划分的 16 个真实 QM9 分子，每分子取原始
构象 0；另检查与官方 test 的规范化身份隔离。小样本覆盖不同原子规模，物理标签
复用现有 MMFF94/Hessian 实现，保留 float32 官方坐标舍入后的原始目标。
缓存标记 `speed_only`/`audit_partial`，禁止进入正常训练/评测入口；不复制或扫描
SHHPC 91.8 GB 正式缓存。预处理成本单列。

测试固定正式 20 层/160 hidden 模型和学习率。FP32 比较有效 batch 128 的
8×16、16×8、32×4、64×2、128×1，以及有效 batch 256/512；再比较 worker 0/2/4、
prefetch 2/4、pinned memory 和独立 BF16 变体。每格 2 次预热+6 个计时更新，候选
追加 12 更新复测；GPU 同步在 optimizer 更新边界执行，记录 samples/s、atoms/s、
更新耗时分布、启动和显存。最大分子压力检查保留至少 20% 显存余量；profiling、
固定监督梯度对照和 4 更新 checkpoint 恢复检查单独报告。

按用户授权，Torch 使用单卡最多 1 GPU 小时；55 分钟停止新增测试，自动重排关闭。
Torch 当前 GPU 提交应使用 `--gres=gpu:1 --constraint=l40s` 等型号约束，由 scheduler
分配分区；普通账户的 A100 仅可通过 `--comment='preemption=yes;requeue=false'` 的
抢占访问，具体 GPU/账户以实际 allocation 记录为准。固定完整代码 SHA 后提交：

```bash
export STIFFLOW_WORKSPACE=/scratch/ys6132/project/0_stiff_flow_matching/26Fall_DL
export ETFLOW_ENV=/scratch/ys6132/programs/miniforge3/envs/etflow
export STIFFLOW_SPEED_ROOT="$STIFFLOW_WORKSPACE/runs/torch-speed"
export STIFFLOW_SPEED_CACHE="$STIFFLOW_WORKSPACE/cache/torch-speed"
export STIFFLOW_EXPECTED_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$STIFFLOW_SPEED_ROOT"
sbatch --account=torch_pr_1034_general --gres=gpu:1 --constraint=l40s \
  --cpus-per-task=8 --mem=32G \
  --output="$STIFFLOW_SPEED_ROOT/slurm-%j.out" scripts/torch_speed.sbatch
```

实际测试状态、GPU、代码 SHA 和结果由作业的 `summary.json` 记录；脚本实现和 CPU
测试通过不表示 GPU 已运行。当前正式配置仍为 batch8×累积16、FP32、250k 更新；
较大有效 batch 和 BF16 仅用于独立速度探索，不据短训练选择学习率或宣称生成质量。
等样本预算为 128×250k = 256×125k = 512×62.5k = 3200 万样本。
Torch 小缓存吞吐不能代替 SHHPC strict-v3 全量 I/O；最终公共配置、完整 commit SHA
和 small result summary 回传后，SHHPC 需重跑匹配的新接口/恢复验收及稳定吞吐核查。

本轮相关 CPU 检查：`test_targets`、`test_model_integration`、`test_official_data`、
`test_formal_cache`、`test_formal_training`、`test_formal_evaluation`、`test_speed`、
`test_paths`，41 passed，0 failed/0 skipped；线程数 1，轻量 worker 最多 1。

### 2026-10-14 Torch 同卡实测与 SHHPC 下一步

Torch job `19784028` 已完成，退出 `0:0`，`gl004` 的单张 NVIDIA L40S（47.696 GB）
实际分配 9m17s，约 0.155 GPU 小时；未自动重排。运行代码为
`12edaa5a926a353626dc99be1a91a0b469e9f8d4`，tracked diff 为空。
Alternative A 为 `55009a52c2a00b2b379275c9553e3eead3922884`，benchmark 为
`cbeb314bcd1f99f2a7354477592791f6151821a7`。小型结果记录见
[results/torch-speed-20261014-l40s.json](results/torch-speed-20261014-l40s.json)，
完整本地报告为 `runs/torch-speed/19784028/summary.json`，日志为
`runs/torch-speed/slurm-19784028.out`；缓存和 checkpoint 均在仓库外。

同卡 baseline 保留旧 dense precision target、逐参数 host 有限性判断和阻塞图传输，
其余模型/训练设置共用。baseline 计时 6 更新；32×4 候选复测 12 更新；各自预热 2 更新。
下表固定有效 batch 128、FP32；候选使用 tiny-cache 最快的 worker0，baseline 为 worker2。

| 方法 | baseline 8×16 秒/更新 | 候选 32×4 秒/更新 | 候选 samples/s | 同卡加速 |
| --- | ---: | ---: | ---: | ---: |
| precision | 1.699 | 0.969 ± 0.037 | 132.16 | 1.75× |
| Alternative A | 1.726 | 0.966 ± 0.037 | 132.57 | 1.79× |
| control | 1.707 | 0.963 ± 0.037 | 132.91 | 1.77× |

同为 worker2 时，precision 的 microbatch 8/16/32/64/128 分别为
1.642/1.086/0.993/1.029/1.084 秒/更新；32×4 的显存保留量约 7.89 GB，
本 cohort 最大 25 原子压力检查约 8.24 GB。microbatch256 的两次独立压力探针均 OOM，
脚本记录失败并释放显存；其余 34 个 case 完成。真实全量数据的最大原子数仍须在 SHHPC 核查。

有效 batch256 的 32×8 为 130.55 samples/s，batch512 的 32×16 为 128.39 samples/s；
把 microbatch 提至128反而下降至119.13/117.56 samples/s，本轮没有增大有效 batch 的依据。
worker0/2/4 在32×4下为131.27/129.13/129.51 samples/s；pinned memory 为131.15，
收益不足以支持全量 I/O 改为 worker0。主测试最优为 worker0；补充 job `19786218` 在
`gl019` 的一张 L40S 上复用相同缓存，worker2、32×4、每格2预热+12计时更新，
prefetch2/4 分别为129.28/129.60 samples/s，差0.25%，小于约0.04秒的更新标准差。
保留 prefetch2。补充运行代码为 `c3288ceb7dc60b6768d2257adf7279ddab19c4bd`，
退出 `0:0`，分配1m41s；两次作业合计10m58s，0.183 GPU 小时。
后续完整测试脚本始终包含一次非零 worker 的 prefetch4 格。

BF16 的三方法分别为126.81/129.60/129.98 samples/s，显存保留量约6.20 GB，
比 FP32 略慢；batch256/512 的 BF16 为127.16/126.33 samples/s。
固定监督 BF16 loss 相对差1.38e-4、预测相对 L2 差2.37%、梯度相对 L2 差3.28%，均有限；
保持正式 FP32。FP32 的16×1与4×4固定监督梯度相对 L2 差2.35e-6，通过一致性检查。
GPU 连续4更新与2+恢复2更新的模型、优化器、scheduler、路径/torch/CUDA RNG、
样本游标及裁剪状态完全一致，已消费512样本。

独立 profiler 的 host `aten::item` 次数9537→1062，stream synchronization 次数
12156→2789；float64 target 的 CPU 区间18.01→10.75 ms/更新。此对照同时改变
microbatch/累积与检查实现，不能把全部收益归给某一项；profiling 有额外开销，未计入吞吐表。
物理标签预处理0.296秒、整个小缓存准备4.174秒，单列于结果。共享临时目录导致 worker
清理时出现 EBUSY 警告，未阻断测试；Torch 脚本改用节点本地临时目录，补充作业使用相同
目录规则并运行 worker2，两格均无 EBUSY/Traceback。日志也显示频繁裁剪较大的梯度，
个别裁剪前 norm 约5e6，但 loss/梯度保持有限；沿用 clip100，正式验收应核查其分布与
损失稳定性，短速度测试不足以判断收敛。

上轮计划中的 SHHPC 独立 GPU 重新验收，由下述用户确认的本地验收方案取代。
Torch 的小缓存吞吐不代表 SHHPC strict-v3 的全量 I/O 或生成质量。

### 2026-10-14：4090D 正式配置与本地最大分子验收

单卡 job `19803828` 已完成（gl031 / NVIDIA L40S，退出0:0，535 GPU秒，无重排）。
实际运行代码为 `b5ab58509fb347d1ae466f8e63df03580e8a6e89`，tracked diff为空；
后续 `7bffe00e1388bbd816f9a6c382cf9812ebdc15a1` 仅关闭速度测试中的 Lightning 自动重排。
共享算法文件哈希未变，真实报告已通过 `formal.require_torch_runtime` 检查。
验收摘要见 [results/torch-runtime-acceptance-20261014.json](results/torch-runtime-acceptance-20261014.json)。

正式配置采用 [configs/qm9-formal-4090d.yaml](configs/qm9-formal-4090d.yaml)：
FP32，microbatch32×累积4，有效batch128，workers2 / prefetch2 / pin_memory=false，
20层 / hidden160，AdamW 7e-4，250k更新，seeds42/43/44；物理标签和监督仍为float64。
显存以24 GiB的90%为优选线、95%为硬上限，计入设备非PyTorch占用；
实际训练设allocator上限并检查forward/backward和validation的显存。

本地冻结legacy train中选取 `CCCCCCCCC` 和 `CCCCCCC(C)C` 的原始构象：
均为29原子 / 9重原子，达到[GEOM QM9记录的最大原子数](https://www.nature.com/articles/s41597-022-01288-4)。
两条物理标签共耗时0.212秒，完整准备60.981秒（含导入、身份核对和写入），保持speed_only。
缓存为 `cache/torch-speed/max29-20261014/manifest.json`；每个压力batch全部由29原子分子构成。
三方法均完成2次预热+6次计时更新：峰值allocated10.05 GiB、reserved10.25 GiB、
整卡实际占用10.81 GiB，约24 GiB预算的45.03%，没有OOM，microbatch保留32。
本集群没有4090D；用L40S限制24 GiB作显存代理，不能据此给出4090D的实际吞吐或通信性能。

三方法还用原有16分子缓存完成正式stream RNG的连续40更新与20+恢复20更新，
模型、优化器、scheduler、目标/torch/CUDA RNG、样本游标及gradient queue均在
rtol1e-6 / atol1e-7下通过比较；每次完整运行消费5120样本，loss/gradient/parameter均有限。
相关CPU检查14项通过，随后加入正式门禁和运行显存检查的13项复测通过；
Ruff、Black、脚本语法、实际环境导入和真实报告门禁检查通过。

单卡小缓存、workers0、global-draw监督下，三方法稳定吞吐为130.76 / 131.21 / 131.51 samples/s。
precision的有效batch128/256/512分别为130.76/131.98/130.48 samples/s：增大后+0.93% / -0.21%，
没有明确效率收益，因此保持有效batch128；不据这些短训练改变学习率或样本预算。

两卡DDP仅评测，正式训练本轮继续单卡。每卡32×累积2、全局batch128、workers0；
按全局draw ID生成相同监督，不使用会补齐重复样本的DistributedSampler。
正式单卡默认stream RNG和32×4配置保持不变。

#### 双卡恢复debug与复测（2026-10-14）

原作业 `19804304`（代码 `7bffe00e1388bbd816f9a6c382cf9812ebdc15a1`，gl031）
退出1:0，实际278秒×2 GPU。三方法显存、稳定吞吐已完成，precision / Alternative A
恢复通过，但control的40更新恢复比较失败。前20更新loss完全一致，恢复后第22更新
才出现约2.08e-8相对loss差，最终模型最大绝对差1.714e-3，489个张量超过原比较门槛。
样本游标、数据/配置签名、目标/torch RNG和scheduler一致；优化器及裁剪历史有差异。

诊断 job `19807600`（gl045，153秒×2，退出0:0）在固定
`dbeb803d11846e2bb17bb93e87df1654121e95f8` 上独立重跑旧control连续40更新，
与原连续训练的模型差值为0，逐更新loss完全相同。只关闭 `gradient_as_bucket_view`，
再运行连续40与20+恢复20：模型差值为0，优化器、scheduler、每rank RNG、游标及
裁剪历史均通过原检查。诊断脚本及哈希记录在结果摘要中，原日志/checkpoint未覆盖。

修复 commit `0818c67df984bedde64a43d948960666bfbd0a8e` 将DDP梯度bucket别名默认关闭，
设置写入case和checkpoint签名，防止混用不同策略的恢复记录；单卡行为不变。
[PyTorch说明](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py)
指出该选项使用通信bucket的梯度视图，视图在首轮后生效；本次实测确认该优化会影响
本配置恢复后的浮点梯度存储/裁剪/更新。比较容差仍为rtol1e-6 / atol1e-7，未放宽。
失败信息现保留具体参数路径、张量误差统计或标量差值；初始梯度对照先于长恢复测试保存。
运行脚本允许 `STIFFLOW_RUNTIME_SECONDS` 缩短内部截止时间，仍受原CLI上限约束。

复测 job `19808783` 固定上述修复commit，gl066 / 两张NVIDIA L40S，退出0:0，
实际298秒×2 GPU；allocation上限8分钟、内部390秒截止、关闭requeue。
三方法2次预热+12次稳定更新的结果：

| 方法 | 单卡32×4 samples/s | 两卡每卡32×2 samples/s | 墙钟加速 | 每样本GPU时间比 |
| --- | ---: | ---: | ---: | ---: |
| precision | 130.76 | 227.83 | 1.74× | 1.148 |
| Alternative A | 131.21 | 229.76 | 1.75× | 1.142 |
| control | 131.51 | 241.87 | 1.84× | 1.087 |

三方法的单/双卡初始loss相同，裁剪前梯度相对L2差4.63e-8至4.73e-8（门槛1e-4）。
三方法连续40与20+恢复20全部通过；每次完整运行消费5120样本。29原子最大分子压力
检查全部通过，每卡峰值设备占用11.18 GiB，约24 GiB预算46.57%，未OOM或出现非有限值。
同一样本预算下，双卡减少墙钟时间，但GPU时间增加约9%–15%。单卡与复测双卡位于
不同节点，不能把旧/新双卡速度差完全归因于bucket设置；tiny-cache吞吐不代表全量I/O。

相关CPU检查 `tests/test_runtime.py`、`tests/test_speed.py` 和既有checkpoint比较测试，
9 passed / 2 warnings（85.17秒）；Ruff、Black、bash语法、实际导入路径和原单卡验收
门禁检查通过。共享数学/标签/模型/生产训练文件未修改，已有标签和单卡GPU验收继续可用。

完整本地报告为 `runs/torch-runtime/0818c67-ddp-fixed/summary.json`，日志为
`runs/torch-runtime/slurm-19808783.out`；诊断为 `runs/torch-runtime/ddp-debug-20261014/19807600/summary.json`。
可同步摘要见 [results/torch-runtime-ddp-20261014.json](results/torch-runtime-ddp-20261014.json)，
包括实际版本、环境、数据哈希、命令、各case、debug数值差和Slurm用量。
累计GPU时间：既有658秒 + 单卡535秒 + 原失败双卡556秒 + 诊断306秒 + 复测596秒
= **2651秒（44分11秒）**；原3600秒预算剩949秒（15分49秒），不自动追加测试。

#### 4090D定向batch复测

用户追加1 GPU小时，累计预算上限改为7200 GPU秒；已有2651秒用量，
新测试可用4549秒（75分49秒），启动、失败和复测均按实际allocation计入。
定向模式保留FP32、有效batch128和三方法相同的250k正式更新预算；
候选外置配置将显存优选线设为92%、硬上限95%，正式配置只在运行验收后更新。

`stifflow.runtime_benchmark --optimize-batch` 比较32×4与64×2，workers2 / prefetch2 /
pin_memory=false，TF32关闭。每个方法三轮，每轮5次预热+40次计时更新，第二轮交换顺序。
三方法都在全29原子的batch上检查两组配置，选中配置另外连续100次更新。
64×2须每个方法合并吞吐提升至少5%，且每轮都更快；否则选32×4。
固定监督梯度门槛1e-4、loss容差rtol/atol1e-5，正式stream的40与20+恢复20比较
仍用原容差rtol1e-6 / atol1e-7。缺失或失败的检查不能生成可用正式验收记录。

运行封装支持 `STIFFLOW_RUNTIME_CONFIG` 指向仓库外候选配置，
`STIFFLOW_RUNTIME_OPTIMIZE_BATCH=1` 启用定向模式，并要求显式设置
`STIFFLOW_RUNTIME_SECONDS`（最多3210秒）；单卡allocation最多55分钟，预留至少
20分49秒作必要诊断。原单/双卡验收模式和默认截止时间继续可用。
既有16分子与两个29原子speed-only缓存复用，不重新生成标签或改变生产数学实现。
作业 `19815618` 已完成（gl010 / 单张NVIDIA L40S，退出0:0，无重排），
实际allocation为1959 GPU秒（32分39秒）；固定运行代码
`25f09e803abc3b09360456a80a7dedb507f2bd97`，tracked diff为空。
42个case全部完成；合并三轮计时结果如下，全部使用正式workers2及stream监督：

| 方法 | 32×4 samples/s | 64×2 samples/s | 64×2变化 |
| --- | ---: | ---: | ---: |
| precision | 125.75 | 121.09 | -3.71% |
| Alternative A | 125.83 | 120.86 | -3.95% |
| control | 126.72 | 121.90 | -3.80% |

三方法每一轮都支持32×4，正式配置保留该batch/累积组合；显存优选线更新为92%，
硬上限仍为95%。全29原子压力下，32×4峰值allocated / reserved / device-used
为10.05 / 10.25 / 10.81 GiB（设备占用45.03%）；64×2为19.90 / 20.39 / 20.95 GiB
（设备占用87.29%）。64×2容量通过，但没有速度收益。选中32×4的三方法另外各完成
100次最大分子更新（12800样本），设备峰值保持10.81 GiB。

固定监督的32/64梯度相对L2差分别为1.42e-6 / 3.58e-7 / 4.07e-7，loss差均通过
原1e-5容差。三方法正式stream连续40与20+恢复20全部通过原恢复容差，完整运行消费
5120样本。日志无Traceback、OOM、EBUSY或非有限值错误；频繁梯度裁剪仍需SHHPC
在正式训练中观察，速度检查不代表收敛或生成质量。

第一轮precision / microbatch32混合吞吐中，原ET-Flow的 `GatedEquivariantBlock`
提示1个原子的 `vec1_buffer` 含零分量：原掩码使用 `all(dim=1)`，该原子的vec1范数
支路未求导，其余支路仍运行。该提示不能单独证明原子的真实邻域为空；警告默认过滤
可能去重，日志未给出总触发频率。对应case完成，梯度有限；本轮没有改变backbone或
cutoff，SHHPC正式训练遇到该提示时应回报实际样本及日志。

相关轻量CPU检查11 passed（102.05秒）；配置选择、显存回退、验收匹配、原速度缓存及
既有checkpoint比较检查通过。Ruff、Black、bash语法及git diff检查通过；新摘要与实际
正式配置已通过 `formal.require_torch_runtime`，核对真实GPU摘要的算法哈希、配置、
最大分子显存及三方法恢复记录。配置更新后的正式预算/验收门禁针对性复测2 passed
（63.85秒）；没有重复运行未修改数学实现的整套测试。
完整报告为 `runs/torch-runtime/25f09e8-batch-opt/summary.json`，日志为
`runs/torch-runtime/slurm-19815618.out`；同步摘要为
[results/torch-runtime-batch-4090d-20261014.json](results/torch-runtime-batch-4090d-20261014.json)。
旧单卡/DDP摘要保留其原配置和预算历史，不能代替匹配92%优选线的新验收摘要。
累计用量为2651+1959 = **4610 GPU秒（1小时16分50秒）**，7200秒上限还剩
**2590秒（43分10秒）**；测试已完成，不为耗尽预算追加作业。
L40S的24 GiB代理和小缓存不代表真实4090D或SHHPC strict-v3全量I/O性能。

#### SHHPC下一步

使用该配置和已同步的Torch验收记录，不再单独提交 `formal_acceptance.sbatch`。
`runtime_acceptance: torch_runtime` 门禁核对算法文件哈希、配置、三方法正式stream恢复及
29原子压力结果；完整生产缓存、划分、标签哈希和resume签名检查保留。
默认旧配置的SHHPC验收门禁不变。SHHPC遇到问题时返回实际commit、日志和设备占用，
本轮不把DDP切入正式训练，也不把tiny-cache吞吐外推到91.8 GB的正式缓存。

在SHHPC现有工作区快进到本次交接commit，复用已完成的strict-v3 train/val标签。
references用新代码在新目录CPU重导出，核对val/test覆盖和 `stereo_variation_rows`。
环境、数据路径及CPU_ARGS / GPU_ARGS由当地现有配置提供；以下命令不代表Torch已提交SHHPC作业。

```bash
export STIFFLOW_FORMAL_CONFIG="$STIFFLOW_WORKSPACE/stifflow-v1/configs/qm9-formal-4090d.yaml"
export STIFFLOW_ACCEPTANCE="$STIFFLOW_WORKSPACE/stifflow-v1/results/torch-runtime-batch-4090d-20261014.json"
# STIFFLOW_FROZEN_DATA、TRAIN_LABELS、VAL_LABELS使用SHHPC已验收的完整缓存；REFERENCE_ROOT使用新目录。
sbatch "${CPU_ARGS[@]}" scripts/formal_task.sbatch formal references \
  --manifest "$STIFFLOW_FROZEN_DATA" --config "$STIFFLOW_FORMAL_CONFIG" \
  --output "$STIFFLOW_REFERENCE_ROOT"
# references完成后，RUN_ROOT使用新目录；每个array任务单卡，N取当地允许的并发数。
sbatch "${GPU_ARGS[@]}" --array=0-8%"$N" scripts/formal.sbatch
```

Alternative A仍为 `55009a52c2a00b2b379275c9553e3eead3922884`，benchmark仍为
`cbeb314bcd1f99f2a7354477592791f6151821a7`，参考为 `80f82d7d142bc3872412d2966a8d71ed14462b18`。
全量SHHPC训练/生成与4090D实卡运行未执行；GPU速度测试不提供收敛或生成质量结论。

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
