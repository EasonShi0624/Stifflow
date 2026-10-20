# 25k 消融筛选结果（2026-10-19，SHHPC）

范围：阶段 A–D。四个 25k 候选在冻结 validation monitor（512 分子、Euler50、
seed42、2×参考、`--allow-disconnected`）上生成并评测，按研究交接的预注册门槛
配对筛选。**test 未参与任何环节。**

## 作业

| 阶段 | 作业 |
|---|---|
| 生成+评测 | `4350061`（a_rot_null）/ `4350062`（a_trace_null）/ `4350063`（linear）/ `4350064`（aligned_linear），全部 COMPLETED，512/512 ok |
| 配对比较 | 登录节点 CPU（`run_compare.sh`） |
| 探索比较 + 汇总 | 登录节点 CPU（`stage_d_summary.py`） |
| MMFF94 strain | 登录节点 CPU（`stage_d_strain.py`，5×5482 构象全部收敛） |
| ensembles 审计 | `4350093` COMPLETED（64 分子/方法，映射全部干净） |

## 门槛判定（预注册，分子配对 bootstrap 2000 次）

| 候选 | 配对基线 | MAT-P Δ (CI95) | MAT-R Δ (CI95) | chirality Δ | 通过 |
|---|---|---|---|---|---|
| a_rot_null | StifFlow-25k | +0.0061 (+0.0035…+0.0137 外限见 JSON) | +0.0106 (+0.0037…+0.0175) | +0.0150 | **否** |
| a_trace_null | StifFlow-25k | +0.0120 (+0.0035…+0.0200) | +0.0185 (+0.0085…+0.0298) | −0.0055 | **否** |
| linear | control-25k | −0.0174 (−0.0269…−0.0084) | −0.0152 (−0.0276…−0.0028) | +0.0452 | 否（未达 5% MAT-P） |
| **aligned_linear** | control-25k | **−0.0959 (−0.1121…−0.0796)** | **−0.0615 (−0.0803…−0.0434)** | **+0.1805** | **是（5/5 全过）** |

- aligned_linear 相对 MAT-P 变化 **−18.2%**（门槛 −5%），COV-R +0.0617，
  所有 5 项判据全部通过，是唯一进入 seed43 提名的候选。
- 两个 null 都未击败 StifFlow（MAT-P/MAT-R 均变差）：破坏 Hessian 方向
  （保留谱）或只保留 trace 都不改善——**方向性与整体刚度信息在当前
  StifFlow 实现中都没有被转化为生成收益**，与 biweekly2 的机制结论
  （监督信号仅差 6–10%）一致。

## 探索性比较（非门槛）

- aligned_linear − StifFlow-25k：MAT-R −0.0519（CI −0.0692…−0.0340），
  chirality +0.1750 —— aligned_linear 也大幅优于 StifFlow 本身。
- a_rot_null − control：MAT-R +0.0011（CI 跨零）；a_trace_null − control：
  +0.0089 —— nulls 与 control 持平，确认 StifFlow 的 stiffness 条件整体
  处于“无净收益”状态。
- linear − StifFlow：MAT-R −0.0056（CI 跨零），方向弱正向。

## 与 biweekly-2 机制发现的交叉解释

1. **耦合 > stiffness**：aligned_linear（对齐 ξ* 到 x1 的 Kabsch + 独立
   Haar 恢复各向同性边缘）是四个候选中唯一大幅胜出的，且胜出幅度远大于
   任何 stiffness 变体与 control 的差异。这与“对齐调度 g(t)=t²(1−t) 抑制
   stiffness 项、监督仅差 6–10%”的机制分析互证：**当前瓶颈在条件耦合的
   可学习性，不在刚度信息的方向性**。
2. **chirality 定位**：StifFlow−control 宏手性在 25k 为 +0.0055（CI 跨
 零），250k 才出现 −0.012 的落后——StifFlow 的手性代价是训练后期演化
   出来的，不是路径公式的固有缺陷。aligned_linear 在 25k 的手性大幅领先
   （+0.18 vs control）提示对齐耦合对手性学习有直接帮助。
3. **物理质量（MMFF94，诊断用）**：25k 各方法 strain 中位数 StifFlow 9.45
   < trace_null 10.64 < aligned_linear 13.41 < rot_null 13.86 < linear
   15.98 kcal/mol；aligned_linear 的 RMSD/chirality 优势伴随更高的
   relaxable strain——早期训练中它换来的是更准的全局构象与手性，但局部
   力学质量未同步领先。此为 25k 单点观察，非最终结论。
4. **ensembles 审计**：四候选的参考映射全部干净（拓扑/立体化学/原始
   行零差异），无评测口径问题。

## 建议下一步（未执行，等待授权）

1. **aligned_linear seed43/44 训练**（每方法 6–9 h GPU，`--profile
   research`，同一冻结协议）验证跨种子稳健性；这是唯一进入预注册第二
   种子阶段的候选。
2. seed43 通过后，考虑 aligned_linear 的 250k 完整预算训练 + 冻结 test
   单次比较（新预算需另行授权）。

## 产物

- 门限比较：`compare/{a_rot_null,a_trace_null}-25k-vs-alternative_a-25k-paired.json`、
  `compare/{linear,aligned_linear}-25k-vs-control-seed42-step025000-paired.json`
- 探索比较：`compare/{a_rot_null,a_trace_null}-25k-vs-control-25k-paired.json`、
  `compare/{linear,aligned_linear}-25k-vs-alternative_a-25k-paired.json`
- 汇总：`compare/screening-summary.json`
- 物理质量：`compare/strain-25k-candidates.json`
- ensembles：`compare/ensembles-{method}-25k.json`
- 生成/评测：`generated/{method}-25k/`
- 脚本：`scripts/stage_d_{strain.py,summary.py,ensembles.sbatch}`
- 作业日志：`logs/gen-*-435006*.out`、`logs/ensembles-25k-4350093.out`
