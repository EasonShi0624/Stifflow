# Alternative A formulation 原始归档

本目录保存本项目已用于实现和数值核验的原始 PDF/LaTeX，正文日期为 **16 September 2026**。本次仅将现有文件纳入版本控制，没有改写正文或重新生成 PDF。

| 文件 | SHA256 |
|---|---|
| `StifFlow_v1_Alternative_A..tex` | `7ca92270d16e992aa4c2162a14628b7398f906a31e2a571339034d006ea148f0` |
| `StifFlow_v1_Alternative_A.pdf` | `ecbbc2a76e269bd75774642f1e81997c88330f0379c2c1f2621aa9612111ef82` |

LaTeX 文件名中的连续两个点是现有原文件名，按原样归档。默认代码使用文档的 Euclidean `A_E` 路径，因子为 `L_K = T_K S`；实现依据包含 `eq:factor/offlineL/coding`。`A_W` 等其它数学变体不因此成为已实现、已验证的实验。

SHHPC 可使用 `sha256sum formulation/StifFlow_v1_Alternative_A..tex formulation/StifFlow_v1_Alternative_A.pdf` 核对文件。正文中的 companion verification 文件不在本次本地归档中；实际已运行的轻量核验以仓库 `tests/test_alternative_a.py` 和 core 的研究交接记录为准。本次未执行 LaTeX 编译验证。

## 独立研究扩展：延迟曲率

`StifFlow_A_delayed_curvature.tex`（2026-10-19）单独推导固定 control 前段后的 Hermite
延续：`tau=0` 恢复原 A，端点校准不变；噪声加速度代价乘 `1/(1-tau)`，总 stiffness
监督差异积分乘 `1-tau`。最优性只针对新增前段约束下的可行路径，不取代原 A 的结论。
原始两份档案及其上述 hash 保持不变；原 `stifflow_alt_a/path.py` 也保持不变，以复用标签缓存。
研究方法由 core 的独立 research 接口接入，实际生成质量待 SHHPC 检验。

本次 Torch 没有可用的 `pdflatex`/`latexmk`，因此未编译新文档。可在已有 LaTeX 环境中
将输出写到仓库外：`pdflatex -halt-on-error -output-directory=/existing/external/tex-output formulation/StifFlow_A_delayed_curvature.tex`。
