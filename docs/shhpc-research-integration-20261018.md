# SHHPC 研究分支整合记录：2026-10-18

本记录描述 SHHPC 在正式 QM9 评估完成后，将 Torch 的研究交接分支整合到本集群
工作树的结果。只记录代码、冲突解决和轻量验证；没有提交训练、生成或评测作业。

## 分支与提交

| worktree | 整合后分支 | 整合前 | 合并的远端提交 | 整合后 HEAD |
|---|---|---|---|---|
| `stifflow-v1` | `shhpc-research-20261018` | `shhpc-formal-eval@09e7824` | `torch/shhpc-training-safety@7bc68e4` | `f749abc5b54a671ab8c359f41d9193dd69188f3` |
| `stifflow-v1-benchmark` | `shhpc-research-20261018-benchmark` | `shhpc-formal-eval-benchmark@1fd8a85` | `benchmark-harness@f26cbc5` | `01b3865f49b6d01ad91de3f79dee5631c944d9bd` |
| `stifflow-v1-alternative-A` | `alternative-a` | `55009a5` | `alternative-a@c56d466`（fast-forward） | `c56d466cdd4c309e7ce87543475a085413ac57a1` |

core 合并基于 `shhpc-formal-eval`，保留正式 QM9 评估结果、断点生成和 staged
manifest 归一化提交。benchmark 合并基于 `shhpc-formal-eval-benchmark`，保留多片段
评测、增量生成和 summarize 修复。

## 冲突解决

core 合并无冲突。benchmark 合并有两处冲突：

- `README.md`：同时保留 staged label 归一化说明和 post-formal 研究工具说明。
- `stifflow_benchmark/evaluate.py`：保留 SHHPC 的多片段 `allow_disconnected` 文档、
  `multi_fragment` 记账和 CLI 帮助文本，同时采用 Torch 的
  `stifflow.research_paths.METHODS` 方法集合。

冲突文件通过 `py_compile` 和完整测试验证，没有覆盖任一方的已审阅行为。

## 环境修正

交接文档中的解释器路径 `/scratch/ys6132/programs/miniforge3/envs/etflow/bin/python`
在本机不存在。SHHPC 实际使用：

- `ETFLOW_ENV=/gpfsnyu/home/ys6132/anaconda3/envs/etflow`
- Python 3.12.14
- numpy 1.26.4、torch 2.4.1、rdkit 2026.03.6、scipy 1.17.1
- `etflow` 导入自 `/gpfsnyu/scratch/ys6132/stifflow/ETFlow-reference/etflow/__init__.py`

## 轻量验证

在工作区根目录执行交接中的 69 项检查并补入 benchmark 冲突相关测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/stifflow-research-mpl PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=stifflow-v1:stifflow-v1-alternative-A:stifflow-v1-benchmark \
/gpfsnyu/home/ys6132/anaconda3/envs/etflow/bin/python -m pytest -q -p no:cacheprovider \
  stifflow-v1/tests/test_research_paths.py \
  stifflow-v1/tests/test_research_training.py \
  stifflow-v1/tests/test_formal_cache.py \
  stifflow-v1/tests/test_model_integration.py \
  stifflow-v1/tests/test_formal_training.py \
  stifflow-v1/tests/test_formal_evaluation.py \
  stifflow-v1/tests/test_paths.py \
  stifflow-v1/tests/test_targets.py \
  stifflow-v1-alternative-A/tests/test_alternative_a.py \
  stifflow-v1-benchmark/tests/test_research.py \
  stifflow-v1-benchmark/tests/test_evaluate.py
```

结果：**74 passed，12 warnings，69.51 s**。`py_compile`、`git diff --check`、
`research_task.sbatch` 的 `bash -n` 以及八个研究/评测 CLI 的 `--help` 均通过。

`ruff check --no-cache` 在仓库范围内仍有 13 个既有 `I001` import 排序提示，集中在
历史测试和 `scripts/etflow_official/validate_memmap.py`；本次合并没有新增运行时代码
错误。未运行完整仓库测试、GPU 训练或真实数据生成。

## 后续实验边界

- 只使用 validation monitor 做研究筛选；test 不参与方法选择。
- P0 诊断、四方法 40 步/20+20 恢复验收、25k 筛选按
  [研究交接](research-next-stage.md) 执行。
- 25k 作业必须依赖对应方法的验收通过（Slurm `afterok`），不跳过验收。
- GPU 作业使用 `chem` 或 `rooster` 的 4090D；记录实际分区、节点、时间和显存。
- 旧 `runs/formal-*` 结果不覆盖；新产物放在 `runs/research-20261018/`。
