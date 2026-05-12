# Concise Results And Discussion Narrative

## Results

Under the optimized AdamW lr=1e-3 protocol, the high-modulation hybrid `hybrid_harmonic_stiffness_v1` with `lambda=5.0` remains the strongest setting on validation loss and single-sample RMSD among the harmonic/hybrid source designs. Its cross-split sampled RMSD mean is 0.089909, compared with 0.116301 for harmonic_prior.

The complete val/test COV/AMR lambda evaluation changes the interpretation. `lambda=5.0` is precision-optimal, with AMR-P mean 0.059748 and COV-P tied at 1.0, but it is not ensemble-optimal because its COV-R and AMR-R are worse than lower-lambda settings.

The best hybrid recall/balanced setting is `lambda=0.025`, with COV-R mean 0.184890 and AMR-R mean 1.490074. Harmonic_prior remains the AMR-R recall anchor within the harmonic-vs-hybrid lambda comparison at 1.489857. No tested hybrid lambda beats harmonic_prior on both precision and recall COV/AMR metrics.

## Discussion

The honest conclusion is not that one hybrid lambda is universally best. Instead, stiffness modulation turns the harmonic prior into a tunable precision-recall frontier: larger lambda improves AMR-P precision, while smaller lambda preserves more recall by COV-R/AMR-R.

For presentation and paper framing, `lambda=5.0` should be described as RMSD/precision-favorable, not COV/AMR-optimal. `lambda=0.025` should be described as the best tested hybrid for recall/balanced ensemble metrics. The final claim should emphasize controllable tradeoffs and should explicitly state that harmonic_prior is not dominated in ensemble-level COV/AMR.

The all-method comparison adds context: torsion_only can look recall-favorable on AMR-R/COV-R, but its precision and tracked RMSD are poor. That supports reporting a Pareto frontier rather than collapsing the story to a single scalar winner.

## Recommended Claim Wording

Stiffness-modulated harmonic source priors improve precision-oriented metrics and create a tunable precision-recall frontier, but the optimized hybrid does not uniformly dominate harmonic_prior on ensemble COV/AMR metrics.
