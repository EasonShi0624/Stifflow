# SHHPC 研究实验固定版本交接：2026-10-18

本记录给出本轮代码与原始 formulation 的固定版本。Torch 已完成实现和轻量验证；真实 checkpoint 接入、短训练/恢复、P0 诊断及 25k 实验由 SHHPC 执行。没有启动或提交这些实验作业。

## 分支与完整提交

仓库：<https://github.com/EasonShi0624/Stifflow>。

| worktree / 分支 | 固定实现提交 | 本轮内容 |
|---|---|---|
| `stifflow-v1` / `torch/shhpc-training-safety` | `09aad129935caff602657ec543e67392039129db` | 新路径/监督、clock、路径统计、25k 配置、运行脚本、研究说明、core PDF/LaTeX |
| `stifflow-v1-benchmark` / `benchmark-harness` | `f26cbc5ddde472f1192767c605ef8bff4710a5af` | P0 诊断、共享初始噪声的 solver 研究、配对筛选及测试、多片段评测开关 |
| `stifflow-v1-alternative-A` / `alternative-a` | `c56d466cdd4c309e7ce87543475a085413ac57a1` | 原有 Alternative A 算法及新增 PDF/LaTeX 原始归档；算法未改动 |

本记录自身作为后续仅文档提交发布在 core 分支上；它的父提交是上表 core 实现版本。因此可使用上表精确实现版本，或包含本记录的 core 分支后续文档提交；运行时仍需保存实际 `git rev-parse HEAD` 和未提交差异，不用“最新代码”作为实验版本。

相对起点的提交范围：

- core：`860a040ceac30d374ddb9258f35ff9b4f1d79bbd..09aad129935caff602657ec543e67392039129db`。
- benchmark：`cbeb314bcd1f99f2a7354477592791f6151821a7..f26cbc5ddde472f1192767c605ef8bff4710a5af`。
- Alternative A：`55009a52c2a00b2b379275c9553e3eead3922884..c56d466cdd4c309e7ce87543475a085413ac57a1`。

## 获取与已有 SHHPC 修复的关系

在现有共享 Git 仓库中获取三个分支即可取得上述对象：

```bash
git fetch origin torch/shhpc-training-safety benchmark-harness alternative-a
git show 09aad129935caff602657ec543e67392039129db:docs/research-next-stage.md
```

本次未更新 `shhpc-formal-eval` 或 `shhpc-formal-eval-benchmark`。它们分别已有正式结果、生成断点续跑等不同提交，不能将三个工作分支直接覆盖到仍在使用的正式实验目录，也不能假定一定能 fast-forward。SHHPC 应先保留现有工作区和运行版本，在其独立研究分支中审阅上表提交范围，处理与其修复的重叠，继续保留增量生成和暂存缓存兼容行为。整合后的实际 commit 也应另行记录。

本轮 core 以训练安全修复 `860a040...` 为起点，包含参考原子映射修复；不会把 SHHPC 旧 references 自动转换成新版本。P0 的映射诊断只报告差异。若决定重评，应另建统一的参考/生成产物并对所有方法应用同一协议。

三份工作树和参考包应来自匹配版本；已有 `scripts/shhpc_env.sh` 会核对实际导入路径。先检查 `git status` 和包路径，再执行入口。没有要求 SHHPC 重装环境、复制完整数据或重新生成 Hessian 标签。

## 配置、数据与执行顺序

详细数学定义、输入变量和命令见 [研究交接](research-next-stage.md)。核心配置为 `configs/qm9-research-25k.yaml`：训练预算 25000，effective batch 128（32×4），LR cycle 仍为 250000，dataset seed 42、row RNG domain 2，首先使用训练 seed 42。

需要 SHHPC 已有的冻结官方数据 manifest、canonical version-3 完整 train/val 标签、验证 monitor/reference manifest、正式 checkpoint 与日志；路径由 `STIFFLOW_WORKSPACE`、`ETFLOW_ENV` 和交接文档中的输入变量提供。测试集不用于选择研究方法或 clock。输出均放在仓库外的新目录。

建议按下列顺序跟进：

1. CPU 分析既有验证生成和日志：手性/镜像、参考映射、裁剪记录、训练路径尺度。
2. GPU 对 P0 solver 入口先做 2 个分子的运行验收，再执行预定 64 个分子的数值诊断。记录 RK4 加密收敛是否满足判据。
3. 对要运行的新路径完成 40 步连续与 20+20 步恢复检查，确认有限 loss/gradient、状态一致、显存与吞吐后再继续 25k。不能用旧 formal acceptance 代替新路径验收。
4. 优先开展两个 stiffness null 和两种线性耦合；用同一验证集合、seed、NFE 与 A/Control 的 25k snapshot 配对比较。P3 `precision_clock` 根据 P0 证据再决定执行，其表只能用训练集拟合。
5. 按预定门槛决定第二种子，返回实际版本、配置、数据身份、失败记录、指标、NFE、时间及显存。

通用脚本入口为 `scripts/shhpc/research_task.sbatch`。资源、account 和 partition 由 SHHPC 按实际情况提供；脚本不会自行调用 `sbatch`。它支持 `train`、`clock`、`paths`、`diagnostics`、`sampling`、`generate`、`evaluate`、`compare`，参数与对应 Python 入口一致。完整模型恢复对比使用 `python -m stifflow.verify_resume`。

## 原始 formulation 与验证状态

两份既有 PDF/LaTeX 均已纳入上表对应提交，原文件名和文件内容保留。core 正文为 Version 2 / 2026-09-15，Alternative A 正文为 2026-09-16；各自 `formulation/README.md` 保存四个文件的完整 SHA256。新研究变体的定义位于研究交接文档，没有改写历史 formulation。

相关实现的上一轮实际轻量回归为 **69 passed / 51.47 s**；命令、环境和范围已记录在研究交接文档。本次发布仅补充归档和交接文档，没有改变已测算法代码，未重复相同的计算检查。推送前核对三份工作树的 staged diff、提交文件清单、`git diff --check`，并确认未纳入数据、checkpoint、缓存或凭据。

未运行真实 GPU 训练/生成、正式指标重算或 LaTeX 重编译。现有 PDF 已按原文件归档，但本次没有逐页重新核对。代码可获取和轻量测试通过不构成新方法性能获益的证据。
