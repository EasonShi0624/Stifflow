# Final Claims And Limitations

This summary states the paper/presentation claim boundary implied by the complete COV/AMR lambda evaluation.

## Claim Summary Table

| Method / lambda | validation/RMSD role | precision role | recall role | final interpretation |
|---|---|---|---|---|
| harmonic prior | Strong global-prior baseline; weaker than hybrid lambda=5.0 on optimized loss/RMSD. | AMR-P weaker than all tested hybrid lambdas in the lambda comparison. | Best AMR-R within the harmonic-vs-hybrid lambda comparison. | Recall anchor for the harmonic/hybrid family; not dominated by hybrid on ensemble metrics. |
| hybrid lambda=0.025 | Low-modulation hybrid; not selected by RMSD/validation loss. | Small AMR-P improvement over harmonic, far weaker than lambda=5.0. | Best hybrid COV-R and AMR-R; balanced by rank aggregation. | Best hybrid recall/balanced setting, but not a full dominance result over harmonic. |
| hybrid lambda=5.0 | Strongest optimized validation loss and single-sample RMSD among harmonic/hybrid settings. | Precision-optimal by AMR-P; COV-P tied at 1.0. | Worse COV-R and AMR-R than harmonic and low-lambda hybrids. | RMSD/precision-favorable setting; not ensemble-optimal. |
| stiffness structured | Better than isotropic/torsion-only on tracked RMSD, weaker than harmonic/hybrid. | Improves AMR-P over harmonic but not as strongly as hybrid lambda=5.0. | Does not beat harmonic on AMR-R in the all-method COV/AMR summary. | Local stiffness alone helps precision but does not replace harmonic/global prior. |
| isotropic | Weak optimized baseline on loss/RMSD. | Poor AMR-P relative to harmonic and hybrids. | Not recall-leading in the final comparison. | Naive Cartesian source baseline remains weaker. |
| torsion only | Very weak optimized loss/RMSD. | Poor AMR-P and lower COV-P, indicating weak precision. | Recall-biased COV-R/AMR-R behavior, but not balanced. | Diversity/recall-favorable but precision-poor; useful as a tradeoff contrast. |

## Supported Claims

- hybrid_harmonic_stiffness_v1 lambda=5.0 improves validation loss, single-sample RMSD, and AMR-P precision relative to harmonic_prior.
- hybrid_harmonic_stiffness_v1 lambda=0.025 is the best tested hybrid lambda for recall/balanced COV/AMR.
- The hybrid lambda sweep forms a precision-recall Pareto frontier.

## Not Supported

- A single hybrid lambda is universally best.
- lambda=5.0 is ensemble-optimal.
- Any tested hybrid lambda dominates harmonic_prior on both precision and recall COV/AMR metrics.

## Limitations

- This is still labeled non-benchmark reporting.
- COV/AMR evaluates ensemble matching, while the training selection originally used validation loss and single-sample RMSD.
- The final narrative must distinguish harmonic-vs-hybrid conclusions from torsion_only's recall-biased but precision-poor behavior.

## Safety

- `benchmark_result`: `False`
- `training_executed_by_summary`: `False`
- `raw_data_modified`: `False`
- `environment_modified`: `False`
