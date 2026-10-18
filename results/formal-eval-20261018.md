# StifFlow 正式生成评测报告（SHHPC，2026-10-18）

## 0. 结论摘要

- 9/9 test、9/9 val 训练权重全部完成生成与评测；test 993/993、val 9654/9654 全部 `ok`，0 failed；val 的 12 个多片段分子全部正常评测。
- **在相同 250k updates、50 NFE、Euler、2×参考数、seed42 下，precision 明显差于 stiffness-free control。**
  - test：precision MAT-R 0.2285 Å vs control 0.1292 Å；COV-R@0.5 0.8369 vs 0.9274；chirality 0.449 vs 0.951。
  - val：precision MAT-R 0.2387 Å vs control 0.1239 Å；COV-R@0.5 0.8353 vs 0.9419；chirality 0.440 vs 0.900。
  - 逐分子配对：test precision−control MAT-R **+0.0993 Å**（95% CI +0.0852…+0.1141，84.7% 分子更差）；val **+0.1148 Å**（CI +0.1104…+0.1194，88.3% 分子更差）。
- **Alternative A 与 control 基本持平**：test A−control MAT-R −0.0004 Å（CI −0.0101…+0.0090）；val +0.0024 Å（CI +0.0007…+0.0041），差异 <0.003 Å，实际可视为无差异。没有观察到 stiffness 条件带来的生成质量收益。
- 诊断指向 **precision 的低 t 速度场**：其总 MSE 的 **74.3% 来自 t<0.1**（target energy 在该区间远大于 A/control），而 A/control 只有 3.5%；100 NFE 与 50 NFE 几乎相同，说明不是 Euler 步数不足。P2 显示 precision 收敛明显更慢：250k 时大致只达到 control 25k 的水平。
- 本次结果不支持“再加 seed/增加采样步数”作为下一步；应先做方法级评审，尤其是 precision path 在 t→0 的 `Adot` 项、时间条件化与训练预算。

## 1. Provenance

### 训练代码与依赖

- `stifflow-v1`：`793d502d86e5`（`shhpc-references-topology-tolerance`）
- `stifflow-v1-alternative-A`：`55009a52c2a0`（`alternative-a`）
- `stifflow-v1-benchmark`：`cbeb314bcd1f`（`benchmark-harness`）
- `ETFlow-reference`：`80f82d7d142b`

### 评估补丁（本集群，已推送，远端已验证）

- `stifflow-v1`：分支 `shhpc-formal-eval`
  - 评估补丁 `b8989dd`：staged train/val manifest 哈希归一化、`formal generate --resume`、formal summary CLI 与测试；
  - 结果摘要 `76b92e3`：`results/formal-eval-20261018.json`；本报告随后加入同一分支。
- `stifflow-v1-benchmark`：分支 `shhpc-formal-eval-benchmark` @ `1fd8a85`
  - evaluator 多片段支持；
  - generation 增量落盘/恢复；
  - summarize 归一化与测试。
- 远端核对（2026-10-18）：`origin/shhpc-formal-eval` 包含 `b8989dd`、`76b92e3` 及本报告提交；`origin/shhpc-formal-eval-benchmark = 1fd8a85`。具体 head 以 `git ls-remote` 为准。

这些补丁只影响生成/评测编排与报告，不改变训练目标、Euler50、2×参考数、seed42 或 RMSD 协议；训练 checkpoint 仍对应 `793d502`。

### 数据与参考

- 配置：`stifflow-v1/configs/qm9-formal-4090d.yaml`，sha256 `363d608dbc84a99b49ac61d08bcdbc94b79d2b9c2fc5dff170cb0ed8079cd9f9`
- 冻结数据清单：`cache/qm9-v3-freeze-strict/manifest.json`，sha256 `5bb7f2615455a38a779dd95166165421f3e6622512be9cbb91565bb85b66cf52`
- 参考：
  - val：`cache/qm9-v3-references-4090d/val/manifest.json`，sha256 `dc50f76413eff97751c91c903e8765e120c0b40850daf657c483cf8c0f412d9d`，9654 分子 / 62,648 构象
  - test：`.../test/manifest.json`，sha256 `a1608b921174e52b1537d172b01127be3d030703f226c188dd52b28a2cc55252`，993 / 8,580
  - monitor：`.../val/monitor.json`，sha256 `557bcd437ee0959fca5ff76cb0d53b4e94796a70712485d806bccaf862485ae2`，512 / 2,741
- packed labels：train `a0bbf33c4b50ca76…`，val `3022258abc1e24e4…`
- 训练时 packed cache 被复制到节点本地并只重写绝对 `path`；内容哈希保留。summary 以 `official_data` 为 canonical 数据身份，逐 run staged train/val 哈希记录在 `staged_label_manifests`。

### 作业与资源

| 阶段 | Job ID | 结果 | 耗时 |
|---|---|---|---|
| test generate / evaluate / summarize | `4334706` / `4334715` / `4334716` | 全部 COMPLETED | 27–30 min/run；eval 20 s |
| val generate / evaluate / summarize | `4335526` / `4335535` / `4335536` | 全部 COMPLETED | 3.7–4.2 h/run；eval ~3 min |
| monitor precision 250k | `4334285` / `4334286` | COMPLETED | 13 min；eval 3 s |
| NFE 100 precision | `4335545` / `4335546` | COMPLETED | 26 min；eval 3 s |
| monitor A/control | `4336109` / `4336111` | COMPLETED | 13 min/run |
| P1 per-t | `4341370` | COMPLETED | 5.5 min |
| P2 snapshots | `4341371` / `4341376` | COMPLETED | 12–14 min/run；eval 11 s |
| P3 chirality | 登录节点 CPU | COMPLETED | ~1 min |

- 集群：SHHPC `rooster` 分区，NVIDIA 4090D；每 run 1 GPU / 4 CPU / 32 GB。
- 生成峰值显存约 **0.46 GiB**；NFE=50；无 postprocessing。
- 全部作业退出码 0；无 OOM、无 NaN、无 failed record。

## 2. 评测协议

- 生成：Euler 左端积分 0→1，50 NFE；每分子 `2 × reference_count` 个样本；推理 seed 42；模型 eval；无 force-field 优化。
- 评测：heavy-atom `Chem.RemoveHs` + RDKit `GetBestRMS`；COV 阈值 0.05–3.00；MAT-R/P mean/median；手性为指定 tetrahedral center 的 CIP 一致率；全部构象数固定为 2×参考数。
- test 为主比较，val 为次比较；两者分开汇总，不混用。
- evaluator 允许参考集中的多片段分子（val 12，test 0），并在 `protocol.connectivity` 与 `aggregate.multi_fragment_molecules` 中记录。
- 所有生成只读图与参考坐标；不读取 Hessian 缓存。test 未用于选择配置或超参数。

## 3. 主结果

### 3.1 test（993 分子，主比较）

| 方法 | MAT-R mean | MAT-R median | COV-R@0.5 | MAT-P mean | COV-P@0.5 | chirality |
|---|---:|---:|---:|---:|---:|---:|
| precision | 0.2285 ± 0.0046 | 0.1524 ± 0.0081 | 0.8369 ± 0.0048 | 0.4696 ± 0.0135 | 0.5651 ± 0.0183 | 0.4493 ± 0.0556 |
| alternative_a | 0.1270 ± 0.0071 | 0.0639 ± 0.0029 | 0.9302 ± 0.0040 | 0.2187 ± 0.0086 | 0.8480 ± 0.0082 | 0.9482 ± 0.0039 |
| control | 0.1292 ± 0.0040 | 0.0642 ± 0.0013 | 0.9274 ± 0.0052 | 0.2153 ± 0.0038 | 0.8503 ± 0.0042 | 0.9514 ± 0.0030 |

### 3.2 val（9654 分子，次比较）

| 方法 | MAT-R mean | MAT-R median | COV-R@0.5 | MAT-P mean | COV-P@0.5 | chirality |
|---|---:|---:|---:|---:|---:|---:|
| precision | 0.2387 ± 0.0076 | 0.1420 ± 0.0081 | 0.8353 ± 0.0111 | 0.4773 ± 0.0142 | 0.5677 ± 0.0189 | 0.4395 ± 0.0464 |
| alternative_a | 0.1263 ± 0.0033 | 0.0543 ± 0.0029 | 0.9392 ± 0.0013 | 0.2168 ± 0.0084 | 0.8529 ± 0.0068 | 0.8944 ± 0.0040 |
| control | 0.1239 ± 0.0009 | 0.0543 ± 0.0014 | 0.9419 ± 0.0006 | 0.2094 ± 0.0012 | 0.8593 ± 0.0010 | 0.9004 ± 0.0012 |

### 3.3 逐分子配对（bootstrap 95% CI，按分子重采样）

| 对比 | split | MAT-R 差值 | 95% CI | 更差比例 | COV-R@0.5 差值 |
|---|---|---:|---|---:|---:|
| precision − control | test | +0.0993 | +0.0852…+0.1141 | 84.7% | −0.0904 |
| precision − control | val | +0.1148 | +0.1104…+0.1194 | 88.3% | −0.1066 |
| alternative_a − control | test | −0.0004 | −0.0101…+0.0090 | 48.5% | −0.0006 |
| alternative_a − control | val | +0.0024 | +0.0007…+0.0041 | 57.1% | −0.0026 |

- precision 的差距在 test/val 一致，且不是 seed 噪声（3 seeds 同方向、量级相近）。
- A/control 的差异 <0.003 Å，远小于 precision 的 ~0.1 Å；无实际意义。
- `nearest_reference_rmsd` 也显示 precision 的生成构象整体离参考更远（MAT-P 0.47 vs 0.21）。

## 4. 诊断

### 4.1 P1：per-t 误差分解（固定 4096 条 val 抽样，seed42）

复现的总体 MSE 与训练 CSV 的 val loss 完全一致（precision 51.3225、A 33.9810、control 34.5972），说明数据/target/RNG 复现正确。

| 方法 | 总体 MSE | 总体 R² | t<0.1 MSE 贡献 | t<0.1 占比 | t=0–0.05 target energy | t=0–0.05 R² | t=0.5–0.55 R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| precision | 51.32 | 0.798 | 38.14 | **74.3%** | 1640.8 | 0.623 | 0.971 |
| alternative_a | 33.98 | 0.789 | 1.20 | 3.5% | 29.1 | 0.896 | 0.935 |
| control | 34.60 | 0.783 | 1.20 | 3.5% | 27.3 | 0.889 | 0.913 |

- precision 在中间/高 t 的 R² 甚至高于 A/control（例如 t=0.5–0.55：0.971 vs 0.935/0.913），但 **t<0.1 的绝对误差极大**，占总 MSE 的 74.3%。
- precision 在 t=0–0.05 的 target energy 为 1640.8，而 A/control 约 27–29；其 `Adot` 在 t 很小时包含随 stiffness `b` 增大的项，target velocity 很快变大。模型在该区间的预测范数只有 target 的 ~0.80，残差同时有平行和垂直分量。
- 这解释了 100 NFE 无效：误差来自学到的低 t 速度场，而不是积分截断；也解释了为什么整体 val loss 高但 R² 不差——低 t 的 target 尺度把 MSE 放大。
- 生成从 t=0 开始的 Euler 前几步正落在该区间，低 t 的绝对误差很可能驱动了后续的过散/全局构象偏离。

### 4.2 P2：precision 训练轨迹（monitor 512，seed42）

| 模型 | step | MAT-R | COV-R@0.5 | MAT-P | COV-P@0.5 | chirality |
|---|---:|---:|---:|---:|---:|---:|
| precision | 25k | 0.3406 | 0.7875 | 0.5641 | 0.5067 | 0.3366 |
| precision | 50k | 0.3127 | 0.7839 | 0.5468 | 0.5146 | 0.3508 |
| precision | 100k | 0.2779 | 0.8133 | 0.5237 | 0.5373 | 0.3694 |
| precision | 175k | 0.2622 | 0.8025 | 0.5055 | 0.5432 | 0.3956 |
| precision | 250k | 0.2425 | 0.8267 | 0.4840 | 0.5636 | 0.4278 |
| control | 25k | 0.2791 | 0.8205 | 0.5263 | 0.5397 | 0.3821 |
| control | 250k | 0.1188 | 0.9410 | 0.2330 | 0.8365 | 0.8939 |

- precision 从 25k 到 250k 持续改善，但速度明显慢于 control；250k 时大致只相当于 control 25k 的水平（MAT/COV/chirality 均接近），而 control 250k 已远好于 precision。
- 这不是训练后期崩溃；precision 仍在改善，但在 250k 预算内没有追平 control。更长训练是另一个实验，需要单独预算和对照。
- 低 t 误差（P1）与慢收敛一致：precision 需要先学好低 t 的大速度场，剩余预算用于中高 t。

### 4.3 P3：手性失败与局部几何（test，seed42）

| 方法 | 含指定中心分子 | 构象 | 正确 | 错误 | 正确率 | 正确构象 nearest RMSD | 错误构象 nearest RMSD | 正确/错误 bond dev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| precision | 722 | 14,184 | 5,989 | 8,195 | **42.2%** | 0.187 Å | 0.771 Å | 0.0094 / 0.0092 Å |
| alternative_a | 722 | 14,184 | 13,475 | 709 | 95.0% | 0.134 Å | 0.705 Å | 0.0041 / 0.0071 Å |
| control | 722 | 14,184 | 13,451 | 733 | 94.8% | 0.141 Å | 0.702 Å | 0.0037 / 0.0056 Å |

- precision 的手性错误不是局部键长畸变：正确/错误构象的 bond deviation 几乎相同（0.0094 vs 0.0092 Å），Rg deviation 也接近（0.060 vs 0.059 Å）。
- 错误构象的 nearest-reference RMSD 平均 0.77 Å，正确构象 0.19 Å；说明 precision 生成的是**远离参考的全局构象**，其中手性中心自然不正确。
- precision 的正确构象也比 control 的正确构象离参考更远（0.187 vs 0.141 Å），与 MAT-P 的差距一致。
- 结论：手性差距是全局构象质量/过散的一部分，不是单独的手性标签或局部几何 bug。

### 4.4 NFE 敏感性

precision-seed42 monitor：

| NFE | MAT-R | COV-R@0.5 | MAT-P | COV-P@0.5 | chirality | 生成耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.2425 | 0.8267 | 0.4840 | 0.5636 | 0.4278 | 12.8 min |
| 100 | 0.2437 | 0.8229 | 0.4866 | 0.5648 | 0.4296 | 25.5 min |

100 NFE 无收益，排除 Euler 截断/积分步数不足。

### 4.5 几何与集合多样性

test 240 分子分层子集（seed42）：

| 方法 | 键长偏差 | Rg 偏差 | 生成/参考集合 spread（均值/中位数） |
|---|---:|---:|---:|
| precision | 0.0122 Å | 0.0353 Å | 3.65× / ~1.3× |
| alternative_a | 0.0063 Å | 0.0312 Å | 1.56× |
| control | 0.0054 Å | 0.0301 Å | 1.85× |

- precision 的局部键长略差，但主要差距在全局构象；spread 均值受少数分子影响较大，中位数约 1.3×，说明过散不是所有分子都同样严重，但确实存在。
- 过散与 P1 的低 t 绝对误差、P3 的全局构象偏离一致。

### 4.6 stiffness 分层（monitor 501 个有标签分子）

- precision−control 的 MAT-R 差距与 `b_min`、`log_contrast` 基本不相关（Spearman 0.02 / 0.03）。
- 与整体刚度 `mean_log_b` 有弱负相关（MAT-R −0.19，MAT-P −0.24）：soft 分子略更明显（最低四分位 MAT-P 差距 0.305 vs 最高四分位 0.133）。
- 没有单一 stiffness 子群能解释全部差距；precision 的问题是全区间的。

## 5. 解释

1. **precision 不是简单欠拟合**：整体 target-normalized R²（0.798）与 A/control（0.789/0.783）接近，中高 t 的 R² 甚至更好。
2. **低 t 速度场是主要异常**：precision 的 `Adot` 在 t→0 时对 stiff modes 产生很大的 target velocity；t<0.1 占总 MSE 74.3%，target energy 1640 vs 27–29。模型在该区间的绝对误差大，预测范数偏低（~0.80×），残差有垂直分量。
3. **生成从 t=0 开始**：Euler 前几步直接使用该低 t 速度场；误差会进入轨迹，造成全局构象偏离/过散，表现为 MAT-P、COV-P、chirality 差。
4. **100 NFE 无效**：不是积分误差；是学到的速度场误差。
5. **precision 收敛慢**：P2 显示 250k 仅相当于 control 25k 的水平；仍在改善，但 250k 预算内未追平。更长训练是另一个实验。
6. **Alternative A ≈ control**：A_E 的对齐因子没有带来可测的生成收益；stiffness 条件在当前设置下没有被有效利用。
7. **不是数据/评测 bug**：checkpoint/config/data/reference 哈希一致；P1 复现 CSV val loss 到 1e-4 以内；test/val 一致；evaluator 多片段补丁验证；无 failed record。

## 6. 限制

- 仅 QM9-v3；test 993 分子，val 9654；没有 test Hessian 标签，stiffness 分层只能在 val/monitor 做。
- 没有干净的 ET-Flow 官方 checkpoint 同 split 基线（官方 split 与本 test 存在重叠），未做外部 pretrained 对比。
- 未做 force-field postprocessing；所有指标基于原始生成坐标。
- NFE 只测了 50/100；未测 25/200。
- precision 的更长训练（>250k）未运行；当前结论严格限定在 250k、50 NFE、相同架构/预算。
- val 的 12 个多片段分子已评测，但其相对片段构象可能受模型能力限制；test 无多片段。
- `precision` 的 R² 是总体 target-normalized 诊断，不替代生成指标。

## 7. 建议的下一步

### 方法评审（优先，交回 Torch）

1. **低 t `Adot` 项**：核对 `0.5*s*alphadot*(b-1)/d` 在 t→0 的量级与条件数；确认这是 formulation 的本意，而不是实现/符号/缩放问题。若本意如此，讨论是否需要时间重参数化、preconditioned target 或更细的低 t 时间采样。
2. **时间条件化**：检查网络在 t<0.1 的 t-embedding 分辨率与训练样本覆盖；uniform t 下低 t 只占 5%，但 target 尺度极大。任何改变 loss 权重/时间采样的方案必须作为独立变体。
3. **慢收敛**：P2 显示 precision 250k 仍明显未收敛；可考虑更长训练（如 500k/1M）作为独立预算实验，同时给 control/A 相同预算，不能在 test 上调参。
4. **Alternative A**：A≈control；评审 `L_K`/alignment 是否在生成端被有效利用，或是否需要不同的 A_W/变体。
5. **不要**通过增加 seed、挑选 snapshot、在 test 上选 NFE/超参来“改善”结果。

### 可选后续实验（需单独授权/预算）

- precision 500k/1M updates，同架构、同数据、同 50 NFE，对照 control/A 同预算。
- 低 t 加权/重参数化变体（独立命名，重新训练，冻结 test）。
- 在 val/monitor 上做 25/100 NFE 敏感性（仅诊断）。
- 若方法修正后，再评估 test；test 仍只用于最终固定配置。

## 8. 产物路径

- 主汇总：`runs/formal-eval-20261018/summaries/test-comparison.json`、`val-comparison.json`
- 配对分析：`runs/formal-eval-20261018/analysis/paired-test.json`
- P1：`runs/formal-eval-20261018/analysis/per-t-error.json`
- P2：`runs/formal-eval-20261018/analysis/snapshot-trajectory.json`
- P3：`runs/formal-eval-20261018/analysis/chirality-geometry.json`
- NFE：`runs/formal-eval-20261018/diagnostics/nfe100/precision-seed42/`
- stiffness：`runs/formal-eval-20261018/analysis/stiffness-monitor.json`
- 生成/评测：`runs/formal-eval-20261018/{test,val,monitor,monitor-snapshots}/...`
- 日志：`runs/formal-eval-20261018/logs/`
- 预检：`runs/formal-eval-20261018/preflight-generation.json`
- 训练 checkpoint：`runs/formal-4090d-staged/<method>-seed<seed>/checkpoints/step-00250000.ckpt`
- 快照：`.../snapshots/step-XXXXXXXX.ckpt`

## 9. 复现命令（示例）

```bash
export STIFFLOW_WORKSPACE=/gpfsnyu/scratch/ys6132/stifflow
export ETFLOW_ENV=/gpfsnyu/home/ys6132/anaconda3/envs/etflow
export STIFFLOW_EVAL_ROOT=$STIFFLOW_WORKSPACE/runs/formal-eval-20261018
export STIFFLOW_RUN_ROOT=$STIFFLOW_WORKSPACE/runs/formal-4090d-staged
export STIFFLOW_REFERENCE_ROOT=$STIFFLOW_WORKSPACE/cache/qm9-v3-references-4090d

# 生成（test 示例；val 用 val/manifest.json，monitor 用 val/monitor.json + --stage monitor）
python -m stifflow.formal generate \
  --config "$STIFFLOW_WORKSPACE/stifflow-v1/configs/qm9-formal-4090d.yaml" \
  --checkpoint "$STIFFLOW_RUN_ROOT/precision-seed42/checkpoints/step-00250000.ckpt" \
  --references "$STIFFLOW_REFERENCE_ROOT/test/manifest.json" --stage final \
  --output "$STIFFLOW_EVAL_ROOT/test/precision-seed42"

# 评测
python -m stifflow.formal evaluate \
  --manifest "$STIFFLOW_EVAL_ROOT/test/precision-seed42/generated/manifest.json" \
  --output "$STIFFLOW_EVAL_ROOT/test/precision-seed42/evaluation.json"

# 汇总
python -m stifflow.formal summarize \
  --reports "$STIFFLOW_EVAL_ROOT"/test/*/evaluation.json \
  --normalize-staged-labels \
  --output "$STIFFLOW_EVAL_ROOT/summaries/test-comparison.json"
```
