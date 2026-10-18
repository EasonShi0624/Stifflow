# 正式实验后的研究工具与 SHHPC 交接

本轮实现 P0 诊断、P1 负对照、P2 线性耦合和可选 P3 时间重参数化的代码、配置及轻量验证。**没有提交作业、训练新模型、运行正式生成或拟合真实数据的 clock。新方法的性能未知。**

## 版本和改动边界

- core 工作分支 `torch/shhpc-training-safety`，起点 `860a040ceac30d374ddb9258f35ff9b4f1d79bbd`。
- benchmark 工作分支 `benchmark-harness`，起点 `cbeb314bcd1f99f2a7354477592791f6151821a7`。
- Alternative A 算法实现继续基于 `55009a52c2a00b2b379275c9553e3eead3922884`；本次在该分支补充归档既有 PDF/LaTeX 及文件哈希，算法源文件未改动。
- 正式训练代码依据 `793d502d86e521039d8f641480cd3960ef2c81d1`；结果依据 `shhpc-formal-eval@09e7824b75a6bc293bbca2b59f75452e7f0bc92e`。
- benchmark 的多片段分子显式开关沿用 `shhpc-formal-eval-benchmark@1fd8a8581bd310f6a7620e3ff47a31666e2db11f` 的已审阅行为；普通评测默认仍拒绝多片段，正式协议需加 `--allow-disconnected`。没有整分支 merge，也没有覆盖 SHHPC 的断点生成和暂存缓存修复。
- 本轮通过 core、benchmark 和 Alternative A 三个独立分支交接；完整提交 SHA 和获取说明见 [发布记录](shhpc-research-release-20261018.md)。SHHPC 接收时应在其现有修复上整合本轮文件差异并核对冲突，保留既有增量生成能力，不能直接覆盖目录。
- `ETFlow-reference/`、共享数据、已有缓存/checkpoint、原始 formulation 和 Alternative A 源文件均未修改。没有重算标签；原三种方法继续使用原配置、路径公式、随机数流和推理 Euler 设置。

## 实现及其可证伪问题

| 优先级 | 入口 / 方法 | 要检验的问题 | 产物与判断 |
|---|---|---|---|
| P0 | `stifflow_benchmark.research_diagnostics ensembles` | 错误是否主要是镜像、局部手性翻转或参考映射问题？ | 逐中心 CIP、全反转/部分反转/缺失分类、错误样本原始/镜像最近参考 RMSD、参考自身手性、macro/micro；可选原始数据映射核对 |
| P0 | `stifflow_benchmark.research_sampling` | 首步场误差、积分误差是否足以解释 Precision 的损失？ | 同一噪声下 Euler50/100、精确首步 Euler50、RK4 200/400；端点位移、各自 COV/MAT/手性、轨迹尺度、调用数 |
| P0 | `stifflow_benchmark.research_diagnostics logs` | 是否频繁触发梯度裁剪？ | 原日志行号、裁剪前 norm/阈值；仅在提供可靠更新数时计算已记录事件比例 |
| P0 | `stifflow.research_path_statistics` | 训练路径和闭环采样的尺度是否显著不同？ | 同一组训练行、同一 z 在时间网格上的坐标/速度 RMS 和分子总平方范数 |
| P1 | `a_rot_null` | A 的收益是否依赖 Hessian 方向与构象的对应关系？ | 25k、seed42，与 A 的同预算结果配对比较 |
| P1 | `a_trace_null` | 是否仅有噪声总尺度有用？ | 保留终端协方差 trace 的各向同性因子对照；不能称为能量完全匹配的对照 |
| P2 | `linear`、`aligned_linear` | 改善路径/耦合是否比 stiffness 更有效？ | 25k、seed42，与 Control、A 的 25k snapshot 配对比较 |
| P3（条件性执行） | `precision_clock` | Precision 的早期速度能量集中是否可通过全局时钟缓解？ | 训练集固定 clock，目标导数同步变换；仍使用普通 Euclidean CFM |

P0 结果用于解释机制。若精确首步干预无效、Euler 与收敛参考接近，才削弱相应数值解释；不能据此证明所有数值误差均不存在。若 A 与旋转负对照持平，不能继续声称收益来自方向性物理信息。若线性/对齐路径获益，应优先研究耦合和流的可学习性。

## 新 formulation 与实现对应

原数学来源保持不变：core 的 `formulation/StifFlow_core_formulation_v1.tex` 正文 Version 2，`eq:factor/adot/trajectory`；Alternative A 的 `../stifflow-v1-alternative-A/formulation/StifFlow_v1_Alternative_A..tex`，`eq:factor/offlineL/coding`。以下是明确命名的新实验，不冒充原 formulation 的实现修复。

令 Q 是去平移基、S 是图先验因子，h=t²(3−2t)，s=1−t，g=t²(1−t)。A 路径为

`X_t = h x1 + Q [s S + g (L_K − S)] z`。

1. **旋转负对照**：独立 proper Haar R，`W = Qᵀ(I_N ⊗ R)Q`，`L_null = W L_K Wᵀ`。图先验空间各向同性使 WS=SW，因此 `A_null=W A Wᵀ`，并且 `A_dot_null=W A_dot Wᵀ`。每个 t 的噪声位置能量、速度能量及协方差谱均保留，目标构象不跟随这个相对 R 旋转。该标签是非物理的科学负对照。实现只做因子对向量的作用，避免每条样本构造旋转后的稠密矩阵。
2. **trace 负对照**：`L_null = (||L_K||F / ||S||F) S`。保留 `tr(L_null L_nullᵀ)`，不保留整个路径的噪声能量或源/目标因子的交叉项。由此产生的差异不能单独归因为方向性。
3. **独立线性耦合**：独立 `ξ=QS z` 与 x1，`X_t=(1−t)ξ+t x1`，`U_t=x1−ξ`。
4. **对齐线性耦合**：先对 ξ 做到 x1 的 proper Kabsch 对齐得到 ξ*，随后用独立全局 Haar H 同时旋转两端。`X_t=H[(1−t)ξ*+t x1]`，`U_t=H[x1−ξ*]`。对齐只改变 ξ 的空间取向；独立 Haar 恢复每条形状旋转轨道的均匀取向，故源边缘仍是各向同性图先验。目标边缘仍是经 SO(3) 增广的数据。它改变的是耦合，不能把 ξ* 代入原 precision bridge 的独立条件公式。
5. **全局 clock**：训练行均匀抽样，解析计算 `e(t)=E[||U_t||²]=h_dot² E||x1||²+E||A_dot||F²`，拟合 `q(t) ∝ sqrt(e(t))`，`τ(t)=∫₀ᵗq(s)ds`。训练 τ 均匀，`X̃_τ=X_t(τ)`，`Ũ_τ=U_t/q(t)`。这使 `∫ e(t)/q(t) dt` 在正、归一化 q 中最小，但不保证降低不可约条件方差或生成误差。代码用正的分段线性 q、分段二次 CDF 的解析反解，因子与监督应用同一链式法则。全表及训练 manifest 身份随 checkpoint 保存，生成只接收图和 τ。

`research_paths.py` 的辅助相对旋转使用独立哈希随机域，不推进原 t/z/全局 Haar 的 RNG。负对照与原 A 共用 t、z、全局旋转；线性方法共用原始噪声。路径内相对 R、z 固定。矩阵和监督运算为 float64，网络仍为 float32，损失仍是 Euclidean CFM。`precision_clock` 消耗相同均匀随机数，但将它解释为 τ，这是该变体有意改变的时间分布。

clock 默认只取 512 个训练行，最大 4096；early-time 网格包含 0、1e−6…0.1 的对数点及 0.1…1 的线性点。该表只是小样本近似，需在 SHHPC 检查网格/样本稳定性；不得用 val/test 调整它。该变体必须重新训练，不能只把原 checkpoint 的推理时间替换成 τ。

## P0 运行方式（仅交接命令，本轮未执行）

在 SHHPC 已分配资源中使用已有环境。设置实际 `STIFFLOW_WORKSPACE`、`ETFLOW_ENV`，以及下列输入变量；不要把示例解释为已存在的新产物。

```bash
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
RESEARCH_CONFIG="$STIFFLOW_WORKSPACE/stifflow-v1/configs/qm9-research-25k.yaml"
# STIFFLOW_TRAIN_LABELS / STIFFLOW_VAL_LABELS：canonical version-3 完整标签 manifest
# STIFFLOW_DATA_MANIFEST：冻结的官方数据 manifest
# STIFFLOW_MONITOR：原验证 monitor/reference manifest；绝不使用 test 选方法
# STIFFLOW_GENERATED：已保存的某方法验证生成 manifest
# STIFFLOW_CHECKPOINT：该方法指定版本、指定训练 seed 的 checkpoint
# STIFFLOW_RESEARCH_OUTPUT：仓库外、全新的本实验输出根目录
```

CPU，先分析已有样本，不重新生成：

```bash
python -m stifflow_benchmark.research_diagnostics ensembles "$STIFFLOW_GENERATED" \
  --limit 64 --data-manifest "$STIFFLOW_DATA_MANIFEST" \
  --output "$STIFFLOW_RESEARCH_OUTPUT/p0-ensembles.json"
python -m stifflow_benchmark.research_diagnostics logs "$STIFFLOW_TRAIN_LOG" \
  --optimizer-updates 250000 --output "$STIFFLOW_RESEARCH_OUTPUT/p0-clipping.json"
python -m stifflow.research_path_statistics \
  --train-labels "$STIFFLOW_TRAIN_LABELS" --config "$RESEARCH_CONFIG" \
  --method precision --rows 128 --output "$STIFFLOW_RESEARCH_OUTPUT/p0-paths.json"
```

更换方法时使用新的输出名。同一输入全集按 SHA256(id) 排序取 64 个分子，选择与结果无关；输入 manifest 必须具有同一组 ID。`--data-manifest` 会从受信任的原始 GEOM 文件核验当前 canonical 对应关系，与保存参考逐坐标比较，并报告原始行号、拓扑/立体化学变动及映射；仅报告差异，不改写旧 NPZ。省略该参数则只做坐标手性诊断，不能据此排除同元素原子排列错误。

日志分母 250000 仅适用于完整的单次、单 rank 正式训练日志。拼接的恢复日志、只保留后半段日志或关闭 INFO 时不要提供该分母；没有匹配日志不表示没有裁剪。

GPU，P0 第二步：

```bash
python -m stifflow_benchmark.research_sampling "$STIFFLOW_MONITOR" \
  --checkpoint "$STIFFLOW_CHECKPOINT" --limit 64 --seed 42 \
  --reference-steps 200 --output "$STIFFLOW_RESEARCH_OUTPUT/p0-solvers"
```

所有变体共用每个分子的初始噪声。原三种路径在 t=0 的真实边缘场均为 −x，因此报告 `||vθ(x,0)+x||²/||x||²`，并仅将 Euler 首步替换为 −x。RK4 的 200/400 步分别需要 800/1600 NFE；连同三组 Euler 和首场检查，每个 batch 约 2600 次调用，明显贵于普通 50 步采样。先用 `--limit 2` 做运行验收，再执行预定 64 个分子；不是 Torch 轻量测试。若最大每构象端点位移超过 1e−3 Å，则不标记数值参考收敛。即使达到阈值，也不代表模型分布正确。

## P1 / P2 的训练、恢复和公平比较

`qm9-research-25k.yaml` 保留正式模型、源、优化器、数据 seed 42、row RNG domain 2、microbatch 32、累积 4。只将训练预算改为 25000；**LR cycle 仍为 250000**，以匹配原模型的 25k snapshot。训练种子先用 42。新路径只能使用 `--profile research`，不能进入旧 `formal` 接口。

先完成 SHHPC 的 40 步连续/20+20 步恢复检查，再继续 25k。每次输出目录必须新建；恢复沿用 checkpoint 的 recovery_directory，不新建竞争写入者。

```bash
python -m stifflow.training --config "$RESEARCH_CONFIG" --profile research \
  --method a_rot_null --seed 42 --train-labels "$STIFFLOW_TRAIN_LABELS" \
  --val-labels "$STIFFLOW_VAL_LABELS" --stop-after 40 \
  --output "$STIFFLOW_RESEARCH_OUTPUT/continuous40"
python -m stifflow.training --config "$RESEARCH_CONFIG" --profile research \
  --method a_rot_null --seed 42 --train-labels "$STIFFLOW_TRAIN_LABELS" \
  --val-labels "$STIFFLOW_VAL_LABELS" --stop-after 20 \
  --output "$STIFFLOW_RESEARCH_OUTPUT/partial20"
python -m stifflow.training --config "$RESEARCH_CONFIG" --profile research \
  --method a_rot_null --seed 42 --train-labels "$STIFFLOW_TRAIN_LABELS" \
  --val-labels "$STIFFLOW_VAL_LABELS" --stop-after 40 \
  --resume "$STIFFLOW_RESEARCH_OUTPUT/partial20/checkpoints/latest.ckpt" \
  --output "$STIFFLOW_RESEARCH_OUTPUT/resumed40"
python -m stifflow.verify_resume \
  "$STIFFLOW_RESEARCH_OUTPUT/continuous40/final.ckpt" \
  "$STIFFLOW_RESEARCH_OUTPUT/resumed40/final.ckpt" \
  --output "$STIFFLOW_RESEARCH_OUTPUT/resume-check.json"
```

确认有限 loss/gradient、恢复比较通过、吞吐与显存可接受后，使用相同命令去掉 `--stop-after`，从连续 40 步的 checkpoint 恢复到 25k。P1 更换 `--method a_trace_null`；P2 分别使用 `linear`、`aligned_linear`。不要一次提交所有后续阶段。研究 profile 的验收记录与旧 formal 的 runtime acceptance 分开，不能宣称旧验收自动覆盖新路径。

若已有 node-local packed 数组，可加 `--packed-cache-root`，继续传 canonical manifest；不重生成、不改写 manifest。训练签名记录配置、数据、路径实现哈希和 clock；代码或表改变会拒绝从旧研究 checkpoint 恢复。

验证生成复用 benchmark 入口，原 A / Control 的 25k snapshot 同样通过此入口生成，确保三个方法使用同一 monitor manifest、生成 seed 和评测协议：

```bash
python -m stifflow_benchmark.generate "$STIFFLOW_MONITOR" \
  --checkpoint "$STIFFLOW_RESEARCH_CHECKPOINT" --steps 50 --seed 42 \
  --output "$STIFFLOW_RESEARCH_OUTPUT/generated"
python -m stifflow_benchmark.evaluate "$STIFFLOW_RESEARCH_OUTPUT/generated/manifest.json" \
  --allow-disconnected --output "$STIFFLOW_RESEARCH_OUTPUT/evaluation.json"
python -m stifflow_benchmark.research_compare \
  "$STIFFLOW_CANDIDATE_EVALUATION" "$STIFFLOW_BASELINE_EVALUATION" \
  --output "$STIFFLOW_RESEARCH_OUTPUT/paired-screen.json"
```

比较工具只接受 validation、相同种子、完整且无失败的 25k 报告；检查模型/优化器/LR、源、数据、microbatch/累积、参考 ID/manifest 和采样/评测协议。按分子 bootstrap，不把构象或多个种子混成独立观测。原有 `formal summarize` 仍只接受三方法九组 250k 实验，研究结果不塞入该汇总。

若旧暂存标签 manifest 哈希与 canonical 不同，提供 `--label-manifests CAND_TRAIN BASE_TRAIN CAND_VAL BASE_VAL`。四份归档 JSON 必须各自匹配 checkpoint 已记录的哈希；工具去除存储路径后比较接受行列表、图/因子分片 checksum、row index、标签实现和源/物理配置。它核对已记录的内容身份，不重新扫描大矩阵。缺少归档证据时拒绝比较，不仅凭 official_data 相同绕过过滤差异。

进入第二种子的预定筛选条件：MAT-P 点估计降低至少 5%，其配对差值 95% CI 上界小于 0；MAT-R 增加不超过 0.005 Å；COV-R@0.5 和 macro 手性下降均不超过 0.01。缺少手性证据不通过门槛。上述是预算决策规则，不是已观察到的收益，也不是跨训练种子的显著性结论。P1 应同时检查 A 对两个 null、各 null 对 Control 的差值；“null 未击败 Control”本身不能证明 A 使用了物理方向信息。

## P3 的独立、可选入口

仅在 P0 支持早期路径能量分配问题仍值得检验时执行：

```bash
python -m stifflow.research_clock --train-labels "$STIFFLOW_TRAIN_LABELS" \
  --config "$RESEARCH_CONFIG" --rows 512 --seed 42 \
  --output "$STIFFLOW_RESEARCH_OUTPUT/precision-clock.json"
python -m stifflow.training --config "$RESEARCH_CONFIG" --profile research \
  --method precision_clock --clock "$STIFFLOW_RESEARCH_OUTPUT/precision-clock.json" \
  --seed 42 --train-labels "$STIFFLOW_TRAIN_LABELS" --val-labels "$STIFFLOW_VAL_LABELS" \
  --stop-after 40 --output "$STIFFLOW_RESEARCH_OUTPUT/clock40"
```

后续恢复和验证步骤同上。clock 必须属于同一训练 manifest 和源配置；不接受 val/test 拟合表。推理时无需额外 `--clock` 或标签文件，表已封装于 checkpoint。

## 验证范围与交接产物

Torch 的测试仅使用小型合成分子/矩阵、单层小网络、标量恢复 fixture 和假 Trainer：解析导数与有限差分、端点/中心化、旋转/置换、负对照谱/能量不变量、对齐的 proper 性、随机流与恢复、clock 反解/链式法则、模型参数初始化一致性和 checkpoint 重建、缓存标签一致性、预算/累积参数传递、镜像/部分翻转、失败记账、数据身份和配对筛选。

2026-10-18 实际执行结果：**69 passed，51.47 秒**。解释器为既有 `/scratch/ys6132/programs/miniforge3/envs/etflow/bin/python`（Python 3.12.14）；numpy 1.26.4、torch 2.4.1、RDKit 2026.03.6；etflow 导入自本工作区只读 `ETFlow-reference/etflow/__init__.py`。从工作区根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/stifflow-research-mpl PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=stifflow-v1:stifflow-v1-alternative-A:stifflow-v1-benchmark \
/scratch/ys6132/programs/miniforge3/envs/etflow/bin/python -m pytest -q -p no:cacheprovider \
  stifflow-v1/tests/test_research_paths.py stifflow-v1/tests/test_research_training.py \
  stifflow-v1/tests/test_formal_cache.py stifflow-v1/tests/test_model_integration.py \
  stifflow-v1/tests/test_formal_training.py stifflow-v1/tests/test_formal_evaluation.py \
  stifflow-v1/tests/test_paths.py stifflow-v1/tests/test_targets.py \
  stifflow-v1-alternative-A/tests/test_alternative_a.py \
  stifflow-v1-benchmark/tests/test_research.py stifflow-v1-benchmark/tests/test_evaluate.py
```

相关改动的 `ruff check --no-cache`、两个 worktree 的 `git diff --check`、新 sbatch 的 `bash -n` 及七个 CLI 的 `--help` 均通过。测试中的警告来自上游文档转义、Lightning 的运行/恢复提示，以及既有极端谱测试刻意构造的病态矩阵；恢复测试实际比较了模型、优化器、调度器、RNG 和数据游标。没有运行整个仓库的所有测试，也没有据此认证完整 GPU 训练链路。

没有用这些测试代替真实 checkpoint 验收；没有新训练曲线、COV/MAT、GPU 显存/吞吐或科学收益结论。SHHPC 返回每次实际 commit SHA、未提交差异、配置和数据身份、环境、seed、预算、NFE、时间/显存、失败记录及本工具输出。P0 诊断不修改现有结果；如参考映射确有问题，应另建产物并对所有方法统一重评。

两阶段局部 Hessian refiner 仍是研究方向，未实现：先需要 P0/P1 证明方向标签有增益，再检验它是否预测局部能量误差。当前不扩展 backbone、不加 Hessian loss weighting、不消耗新一轮完整训练预算。
