# 特定扭转结构：分子级验证与 SHHPC 交接

目标：复用正式实验的 **GEOM-QM9 v3 scaffold validation** 保存结果，检查原 Alternative A
相对 control 的收益是否随具体扭转结构变化。precision 仅为可选诊断参照。本轮没有新训练、
GPU 推理、DRUGS 扩展、新标签生成或结构优化。真实数据清点、批量统计及能量/梯度查询
全部由 **SHHPC CPU allocation** 执行；Torch 的合成测试不能替代这些运行。

## 版本与输入

开发分支：`torch/stiffness-locality-benchmark`；开发基线
`e4d9603e32cbe2e910788bafcdc557a4ac8b431d`。入口是新增的
`python -m stifflow_benchmark.torsion_screen`。交接时记录实际完整 commit SHA、未提交差异
及文件哈希；本文件的存在不表示提交已推送或 SHHPC 已验收。

所需输入均来自 SHHPC 的现有归档，不复制到 Torch：

- 完整冻结的 val reference manifest（正式报告为 9654 分子，不能用 512 monitor 或 test）；
  配套数据 manifest、mapping receipt、每分子的 reference NPZ。
- A/control 各训练 seed 42、43、44 的 **250000 updates** 生成 manifest 与匹配 evaluation，
  共六对文件。可选 precision 也必须三种子齐全。Euler50、生成 seed42、`2*N_ref`、无后处理。
- 物理检查需要 canonical v3 val label manifest，以及 `audit_receipt` 指向的完整原始档案。
  packed 数组不含完整原始 Hessian，不能代替档案。
- 若历史 staging 改写过标签路径，提供每个 run 在训练时归档的 train/val manifest；其 SHA
  必须匹配 checkpoint 元数据。使用现有 `label_identity` 比较去路径后的接受行、标签实现、
  图/因子 checksum 等内容。缺少归档证据时拒绝比较，不仅凭数据集相同放宽。

先在 SHHPC 已确认的 CPU allocation 中设置实际路径，使用已有环境：

```bash
export STIFFLOW_WORKSPACE=/actual/shhpc/workspace
export ETFLOW_ENV=/actual/existing/etflow
export STIFFLOW_VAL_REFERENCES=/actual/frozen/val/manifest.json
export STIFFLOW_TORSION_ROOT=/actual/external/runs/torsion-screen
export STIFFLOW_VAL_LABELS=/actual/canonical/val/manifest.json
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
python -m stifflow_benchmark.torsion_screen --help
```

不更改现有环境，不在本脚本猜测 account、partition 或 walltime。所有输出必须是仓库和
数据目录外的**新目录**；不覆盖旧分析，输入文件只读。

## 第一阶段：在查看方法差值之前冻结分子

```bash
python -m stifflow_benchmark.torsion_screen select \
  --references "$STIFFLOW_VAL_REFERENCES" \
  --output "$STIFFLOW_TORSION_ROOT/selection"
```

在去氢图上使用显式 `Strict` rotor 定义；保存原始全原子索引映射。非环二面角复用 RDKit
`CalculateTorsionLists(maxDev='equal', symmRadius=2, ignoreColinearBonds=True)`，只保留中央
单键。共轭组优先：存在可测非环共轭单键时归入该组；其余按 Strict 数量 0/1/>=2 分类。
酰胺/酯的 Strict 计数与可测二面角数量分别记录。相邻仅表示中央键共享原子，不预先断言
动力学耦合。非芳香环单独标记，零 rotor 不等于完全刚性。

完整 val 的结构清点输出为 `inventory.json`、`inventory.csv`。面板限定连通、7–9 个重原子、
`N_ref<=32`；零 rotor 组 `N_ref>=1`，其余三组 `N_ref>=3`。每组按
`SHA256('42:' + id)` 排序取三分子，最多十二个，输出 `panel.json` 和标注原始原子索引的
结构 SVG。候选不足则记录 shortfall，不扩大条件、不按模型效果或 Hessian 可用性补选。
面板保存参考/清点文件 SHA、RDKit 版本及规则；分析时更改输入、选择或版本会被拒绝。

## 第二阶段：先两个分子验收，再分析整个面板

由 SHHPC 设置下列十二个变量，分别指向**实际已有**的六对归档文件；变量名不规定存储布局：

```bash
run_args=(
  --run "$A42_GENERATED" "$A42_EVALUATION"
  --run "$A43_GENERATED" "$A43_EVALUATION"
  --run "$A44_GENERATED" "$A44_EVALUATION"
  --run "$C42_GENERATED" "$C42_EVALUATION"
  --run "$C43_GENERATED" "$C43_EVALUATION"
  --run "$C44_GENERATED" "$C44_EVALUATION"
)
label_args=()
# 标签 manifest 的 hash 因旧 staging 不同，才提供以下整组六对归档证据：
# label_args=(
#   --labels alternative_a 42 "$A42_TRAIN_ARCHIVE" "$A42_VAL_ARCHIVE"
#   --labels alternative_a 43 "$A43_TRAIN_ARCHIVE" "$A43_VAL_ARCHIVE"
#   --labels alternative_a 44 "$A44_TRAIN_ARCHIVE" "$A44_VAL_ARCHIVE"
#   --labels control 42 "$C42_TRAIN_ARCHIVE" "$C42_VAL_ARCHIVE"
#   --labels control 43 "$C43_TRAIN_ARCHIVE" "$C43_VAL_ARCHIVE"
#   --labels control 44 "$C44_TRAIN_ARCHIVE" "$C44_VAL_ARCHIVE"
# )
python -m stifflow_benchmark.torsion_screen analyze \
  --panel "$STIFFLOW_TORSION_ROOT/selection/panel.json" \
  "${run_args[@]}" "${label_args[@]}" \
  --physics --val-labels "$STIFFLOW_VAL_LABELS" --limit 2 \
  --output "$STIFFLOW_TORSION_ROOT/acceptance2"
```

两个分子从冻结面板中确定，优先包含一个可测扭转案例；不重新选分子。验收报告标记
`acceptance_only=true`，仍核对全部六个 run 的元数据和已有逐分子几何统计。确认有效输入、
完整物理查询、无意外失败后，在同一类 CPU allocation 使用相同命令**去掉 `--limit 2`**，
输出改为 `$STIFFLOW_TORSION_ROOT/full`。可不带 `--physics` 和 `--val-labels` 先分析保存产物；
这会报告物理检查未执行，不能替代完整物理验收。加入 precision 时追加其三对 `--run`，
如使用 `--labels` 也需追加对应归档；precision 不额外调用参考势。

Torch 基线的 `stiffness_diagnostics physics` 限定 25k 和固定 monitor。SHHPC 的
`shhpc-physics-budget-audit@69c38f8d6a902be45a85c7b1cef0bdbba574012c` 已允许通过
`--budget 250000` 审计 250k 模型，但仍要求固定的 512 分子 monitor；`research_compare`
仍限定 25k。本轮入口复用底层函数，不将新面板塞进旧 advancement gate。

## 输出、判读与成本

- `report.json` 的 `geometry` 使用完整 val 的既有 MAT-R/P、COV-R/P@0.5、macro 手性。
  同时按组、精确重原子数、指定手性有无、非芳香环有无分层。主要差值为 A−control MAT-P。
  每个分子先平均三个训练 seed 的配对差值，再做 seed42/2000 次分子 bootstrap；各训练 seed
  的差值单列。无指定中心者不进入手性分母。失败会阻止对应总体均值/CI，不计算有利成功子集。
- `records` 保留面板每分子/方法/seed 的输入 SHA、TFD-MAT-R/P、物理结果和失败原因；
  `paired_panel` 保存三 seed 的逐分子差值与分组汇总。非环 TFD 使用等权 torsion 及 RDKit
  等价末端原子处理；无有效 torsion 时为 `not_applicable`，不填零。
- `*-angles.png` 和 `*-joint-*.png` 使用固定 24 个周期角度区间。等价四元组等权展示；
  相邻角只在同一构象内配对。图中各 seed 等权合并仅用于展示，统计单位仍是分子。
  某方法未齐全三个 seed 时不绘制该方法的成功子集，遗漏方法列于 `plot_omitted_methods`。
- 物理检查复用 MMFF94，原子 RMS force 单位为 kcal/mol/angstrom。A/control 各 seed 每分子
  前 `min(8,2*N_ref)` 个原始生成构象、前两个原始参考构象计算力。生成坐标不优化、不覆盖。
- 每个面板分子按 row ID 取最多两个**已有**标签。用档案中的弛豫参考和原始 `H_raw` 检查
  `0, ±2, ±5, ±15°` 的逐键刚性扭转，proper 对齐后比较能量/力 Taylor 变化，保留实际一阶项。
  依据 core 正文 Version 2 的 `eq:hessian/rotspace/calibration`；方向导数为 float64、去平移、
  去刚体旋转的非质量加权 Cartesian 计算。单独报告原始/regularized 方向曲率、落入 floored
  子空间的切向比例、原始目标与弛豫参考的偏移及各自 force。无新 Hessian、无谱绝对值。
- `queries.jsonl` 保存实际查询坐标、能量/梯度、上下文和失败，`energy_gradient_queries` 与
  `seconds` 记录本轮成本。正常情况下每个已有标签行为 2 个锚点查询加每 torsion 7 次查询；
  原 stiffness 预处理成本应另引历史记录，未知则写未知。

`status=completed` 只表示处理结束。`geometry_complete` 表示完整 val 的几何统计完整性，
`all_saved_runs_valid` 表示所选面板的保存产物/TFD有效性，`physics_complete` 表示物理检查
完整性；缺 Hessian 保留分子，仍可计算 TFD/原始力，物理
完整性为 false。命令对几何/保存产物失败、请求但不完整的物理检查返回非零并保留报告。
`panel_complete=false` 也可能只是候选不足或两分子验收；不可称为十二分子面板验证完成。

必须回传全部四组，包括无收益和失败；区分“候选适用结构”“未见收益”“证据不足”。
小面板的 bootstrap 不提供跨训练种子总体显著性，不能据此挑组后声称确认性收益。
局部曲率有效不等于网络利用了方向信息，MMFF94 校准不等于 GEOM 参考理论级别正确；
此处刚性小角度扫描不是约束优化势垒、TS 搜索或热力学采样证据。不据此自动启动训练。

回传实际代码 SHA/差异、环境、配置、数据/归档 hash、命令、两分子与完整报告、结构图和
代表角分布、查询次数/耗时、历史标签成本及失败。大构象和查询日志留在外部目录，Git 只同步
代码、文档及小型摘要。未推送和另一端未取得的版本不得报告为“已同步”。

## Torch 已执行的轻量检查

既有 etflow 环境，OMP/OpenBLAS/MKL 单线程，正确的三个 worktree PYTHONPATH，禁用字节码
与 pytest 缓存写入。实际执行 `python -m pytest -q -x -p no:cacheprovider
--junitxml=.../runs/torsion-screen-torch-20261101/pytest.xml stifflow-v1-benchmark/tests`：
**48 passed，53.04 秒**，其中新增 23 项；保留一条 ET-Flow 上游 docstring 的转义警告。
ruff、Black 检查、两个 CLI `--help` 和 whitespace 检查通过。完整命令、环境、源码哈希及
未运行项见 [小型检查记录](../results/torch-torsion-screen-20261101.json)。

测试仅用小型 RDKit 构象、合成归档、解析二次势和现有单分子 MMFF 单元检查；integration
fixture 缩小 bootstrap 重复数，生产协议仍固定 2000。这些结果不包含真实分子面板选择、
GEOM 批量统计或 SHHPC 物理检查，也不验证已有 checkpoint 的科学收益。
