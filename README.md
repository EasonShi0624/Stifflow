# StifFlow：共享实现与 CPU 验证

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
