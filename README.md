# StifFlow Alternative A

实现 `formulation/StifFlow_v1_Alternative_A..tex` 正文 **2026-09-16** 的
Euclidean `A_E`，依据 `eq:T`、`eq:factor`、`eq:offlineL`、`eq:coding`。
原文件（含双点文件名）保持不变；PDF 尚未逐页核对。代码导入公共 `stifflow`，
不复制 backbone、训练器或 source，不安装 editable 包。

```python
from stifflow_alt_a import prepare_label, sample_target

label = prepare_label(source, K, epsilon_E)
x_t, U_t = sample_target(x1, t, z, source, label)
```

`source` 是公共 `Source(Q, S, C0)`；`K` 为去平移正交基中的 float64 SPD
能量刚度，`x1` 为 `[N,3]` Å 坐标，`z` 为同一条路径固定的标准正态向量。
`label["L_K"]` 使用 `S^{-1}(S C_K S)^{1/2}`；其余字段记录协方差残差及条件数。
`factor(t, source, label)` 返回 `(A_t, A_dot_t)`；精确端点用于测试，训练取内部时间。
标签只能用于监督，生成不得读取 `K` 或 `L_K`。

CPU 测试命令（工作区根目录）：

```bash
PYTHONPATH="$PWD/stifflow-v1:$PWD/stifflow-v1-alternative-A" \
  /scratch/ys6132/programs/miniforge3/envs/etflow/bin/python -m pytest -q \
  stifflow-v1-alternative-A/tests
```

检查包括非对易 SPD 的 OT 因子、有限差分速度、Gaussian 协方差连续性、
端点及完整误差二阶矩收敛、旋转/置换/平移、极端谱、文档中违反方差次序的反例。
极端谱案例为可精确识别的小型对角实例，不代表任意病态稠密矩阵都可靠。
协方差相对残差大于 `1e-8` 时明确报错；不会静默裁剪谱。
仅实现 `A_E`；`A_W`、free-start、近似稠密因子、训练与 GPU 生成不在本模块内。
