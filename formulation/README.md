# Alternative A formulation 原始归档

本目录保存本项目已用于实现和数值核验的原始 PDF/LaTeX，正文日期为 **16 September 2026**。本次仅将现有文件纳入版本控制，没有改写正文或重新生成 PDF。

| 文件 | SHA256 |
|---|---|
| `StifFlow_v1_Alternative_A..tex` | `7ca92270d16e992aa4c2162a14628b7398f906a31e2a571339034d006ea148f0` |
| `StifFlow_v1_Alternative_A.pdf` | `ecbbc2a76e269bd75774642f1e81997c88330f0379c2c1f2621aa9612111ef82` |

LaTeX 文件名中的连续两个点是现有原文件名，按原样归档。默认代码使用文档的 Euclidean `A_E` 路径，因子为 `L_K = T_K S`；实现依据包含 `eq:factor/offlineL/coding`。`A_W` 等其它数学变体不因此成为已实现、已验证的实验。

SHHPC 可使用 `sha256sum formulation/StifFlow_v1_Alternative_A..tex formulation/StifFlow_v1_Alternative_A.pdf` 核对文件。正文中的 companion verification 文件不在本次本地归档中；实际已运行的轻量核验以仓库 `tests/test_alternative_a.py` 和 core 的研究交接记录为准。本次未执行 LaTeX 编译验证。
