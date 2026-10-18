# Core formulation 原始归档

本目录保存本项目已用于实现和数值核验的原始 PDF/LaTeX。正文为 **Version 2，15 September 2026**；文件名中的 `v1` 不是正文版本号。本次仅将现有文件纳入版本控制，没有改写正文或重新生成 PDF。

| 文件 | SHA256 |
|---|---|
| `StifFlow_core_formulation_v1.tex` | `422efa327fdb6284ee37b61500f41d109263cc77ed81759d0f772a5b5f390d42` |
| `StifFlow_core_formulation_v1.pdf` | `5374c6f2cb600ef800f12f1840d1e4495c371a8fd1877bbe32809ef2a154a988` |

代码公式依据包含 `eq:source/graphsource`、`eq:factor/adot/trajectory` 和 `eq:joint/samplinglaw`。原始数学方案不等于已验证的性能结论；新负对照、线性耦合与 clock 的定义见 [研究交接](../docs/research-next-stage.md)，没有把新变体写回旧 formulation。

SHHPC 可使用 `sha256sum formulation/StifFlow_core_formulation_v1.tex formulation/StifFlow_core_formulation_v1.pdf` 核对文件。LaTeX 为独立文档；如需重建 PDF，正文建议使用两次 pdfLaTeX 或 `latexmk -pdf`，但本次未执行编译验证。
