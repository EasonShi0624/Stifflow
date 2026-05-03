# Full-Database Results Report: Source-Corruption Comparisons for Molecular Conformation Flow Matching

## 1. Executive summary

- The current supported message is a hierarchy: harmonic > stiffness_structured > isotropic > torsion_only on the tracked core metrics. Stiffness_structured is a robust improvement over isotropic and torsion_only, but it does not beat harmonic.
- In formal A/E random-split multiseed runs, stiffness_structured improves best validation loss from 0.063082 to 0.046168 and sampled RMSD overall from 0.2161 to 0.1829.
- In formal A/E scaffold-split multiseed runs, stiffness_structured improves best validation loss from 0.063043 to 0.049749 and sampled RMSD overall from 0.2161 to 0.1829.
- Across both random and scaffold A/E summaries, stiffness_structured is lower than isotropic on all 6 tracked metrics and all 18 seed-metric checks.
- In attribution, the random split ordering is harmonic < stiffness_structured < isotropic < torsion_only; the scaffold split ordering is harmonic < stiffness_structured < isotropic < torsion_only.
- Harmonic is stronger than stiffness_structured on the tracked loss and sampled RMSD metrics, so the evidence does not support a stiffness-wins-overall claim.
- The qualitative ordering does not flip between random and scaffold splits.
- AMR-R, COV-R, AMR-P, and COV-P are not present and are not computable honestly from the current completed outputs.
- Physical-validity evidence is currently proxy-level: finite-coordinate rates and mean absolute bond-length deltas are available, but PoseBusters, MMFF/UFF, xTB, and COV/MAT are not present in the supplied summaries.

## 2. Experimental scope

- Dataset scope: `geom_drugs_full_database_one_conformer`.
- Splits: full-database random split and Bemis-Murcko scaffold split.
- Seeds: 0, 1, and 2.
- Formal A/E comparison: `isotropic` vs `stiffness_structured`.
- Attribution setting: A/B/D/E modes `isotropic`, `harmonic`, `torsion_only`, and `stiffness_structured`.
- Tracked core metrics: best validation loss, reloaded validation loss, sampled RMSD overall, and sampled RMSD on train/val/test.
- Reporting scope: this bundle reads existing summaries only. It does not run training, preprocessing, sampling, scheduler jobs, or environment modification.
- Compact numeric table: `reports/tables/full_database_presentation_key_table.md`.
- AMR/COV availability table: `reports/tables/full_database_amr_cov_metric_availability.md`.
- Machine-readable presentation summary: `experiments/results/geom_full_database_presentation_results_summary.json`.

### 2.1 AMR/COV availability audit

The requested AMR/COV ensemble metrics are unavailable in the completed full-database outputs. The current processed artifact records `304339` molecules and `304339` conformers with `max_conformers_per_mol=1` and `one_conformer_per_molecule=True`. The completed result JSONs store one scalar `sampled_rmsd_to_target` per sampled molecule plus proxy validity fields, not generated coordinate ensembles or full reference conformer sets.

See `reports/tables/full_database_amr_cov_metric_availability.md` for the explicit metric-by-metric AMR/COV status.

## 3. Main results

### 3.1 Random split: isotropic vs stiffness_structured

On the random split, stiffness_structured is lower than isotropic on every tracked A/E metric. Best validation loss improves from 0.063082 to 0.046168; reloaded validation loss has the same means. Sampled RMSD overall improves from 0.2161 to 0.1829. The train/val/test RMSD means also move in the same direction.

![Core full-database comparison](figures/full_database_core_random_vs_scaffold.png)

Caption: Core full-database A/E comparison for random and scaffold splits. Bars are multiseed means and black dots are seed-level values. The figure covers best validation loss, reloaded validation loss, sampled RMSD overall, and sampled RMSD by train/val/test. Source file: `reports/figures/full_database_core_random_vs_scaffold.png`.

### 3.2 Scaffold split: isotropic vs stiffness_structured

On the scaffold split, the same direction holds. Best validation loss improves from 0.063043 to 0.049749; sampled RMSD overall improves from 0.2161 to 0.1829. The split-specific RMSD means remain lower for stiffness_structured, including validation and test.

![Seed-level comparison](figures/full_database_seed_level_comparison.png)

Caption: Seed-level deltas for stiffness_structured minus isotropic. Negative values indicate lower loss/RMSD for stiffness_structured. All plotted seed-level deltas are negative in the available random and scaffold A/E summaries. Source file: `reports/figures/full_database_seed_level_comparison.png`.

### 3.3 Attribution on random split

The random attribution summary gives the same qualitative ordering across tracked core metrics: harmonic < stiffness_structured < isotropic < torsion_only. Stiffness_structured is better than isotropic and torsion_only, but harmonic is lower than stiffness_structured on best/reloaded validation loss and sampled RMSD metrics.

![Random attribution comparison](figures/full_database_attribution_random.png)

Caption: Random-split attribution comparison. Each metric row is normalized to the best method for that metric, so 1.0 is best and larger values are worse. This makes the ordering visible despite the large torsion_only values. Source file: `reports/figures/full_database_attribution_random.png`.

### 3.4 Attribution on scaffold split

The scaffold attribution summary preserves the same qualitative ordering: harmonic < stiffness_structured < isotropic < torsion_only. This is important because the scaffold split does not reverse the random-split story. The current evidence still places harmonic above stiffness_structured on the tracked loss/RMSD metrics.

![Scaffold attribution comparison](figures/full_database_attribution_scaffold.png)

Caption: Scaffold-split attribution comparison with metric-wise normalization. The qualitative ordering is stable relative to the random split. Source file: `reports/figures/full_database_attribution_scaffold.png`.

### 3.5 Physical-validity and rotatable-bond diagnostics

Physical-validity diagnostics are available for the A/B/D/E modes on both splits, but they are proxies. Finite-coordinate rate is 1.0 for all available mode/split means. Mean absolute bond-length delta orders as harmonic < stiffness_structured < torsion_only < isotropic on random and harmonic < stiffness_structured < torsion_only < isotropic on scaffold. This supports stiffness_structured improving over isotropic and torsion_only on this proxy, while harmonic remains the strongest method on bond-length preservation.

![Physical-validity proxy diagnostics](figures/full_database_physical_validity.png)

Caption: Physical-validity proxy diagnostics from existing sampled outputs. The left panel reports mean absolute bond-length delta; the right panel reports finite-coordinate rate. These are not full PoseBusters, MMFF/UFF, or xTB checks. Source file: `reports/figures/full_database_physical_validity.png`.

Rotatable-bond stratification is also available from existing sampled metadata. Across bins, the combined split means range from 0.1158 to 0.1169 for harmonic, 0.1794 to 0.1867 for stiffness_structured, 0.2156 to 0.2165 for isotropic, and 1.4077 to 1.8502 for torsion_only. The qualitative ordering persists across flexibility bins.

![Rotatable-bond stratified RMSD](figures/full_database_rotatable_bond_stratified.png)

Caption: Rotatable-bond-count stratified one-step sampled RMSD for random and scaffold splits. The y-axis is logarithmic to keep torsion_only visible without hiding the other methods. This is a stratified RMSD diagnostic, not a conformer-diversity benchmark. Source file: `reports/figures/full_database_rotatable_bond_stratified.png`.

## 4. Interpretation

The evidence supports stiffness_structured as a robust improvement over isotropic source corruption under the tracked objective and sampling diagnostics. The improvement is not a one-seed artifact: it appears on all tracked A/E metrics and all seed-level comparisons in both random and scaffold summaries.

The attribution setting also rules out a simple torsion_only explanation for the observed improvement. Stiffness_structured is consistently better than torsion_only on the tracked core metrics and on the available bond-length-delta proxy.

The evidence does not support stiffness_structured beating harmonic. Harmonic is lower than stiffness_structured on the tracked loss/RMSD metrics and on the bond-length-delta proxy on both splits. The honest conclusion is therefore a hierarchy: harmonic > stiffness_structured > isotropic > torsion_only on the tracked core metrics, while stiffness_structured remains a robust improvement over isotropic and torsion_only.

## 5. Limitations

- The source artifacts are explicitly labeled non-benchmark/reporting-only; this report preserves that label.
- The full-database outputs use a one-conformer-per-molecule setup and one-step sampled RMSD diagnostics, not a full conformer-ensemble benchmark.
- AMR-R, COV-R, AMR-P, and COV-P are not present in the supplied summary/result files and are not computable from the current one-conformer/one-sample artifacts.
- PoseBusters, MMFF/UFF energies, MMFF/UFF force norms, and xTB/GFN2-xTB checks are not present in the supplied summary files.
- Physical validity is limited to finite-coordinate coverage and mean absolute bond-length delta proxies.
- The rotatable-bond analysis is a stratified RMSD diagnostic, not a diversity or coverage analysis.
- No new training, sampling, preprocessing, or scheduler execution was performed for this bundle.

## 6. Suggested presentation message

The clean slide message is that stiffness_structured source corruption is a reliable improvement over isotropic and torsion_only under the fixed current setup, but harmonic remains the stronger tracked baseline. The value of stiffness_structured is robustness and attribution clarity, not an overall win over harmonic.

- Use the hierarchy headline: harmonic > stiffness_structured > isotropic > torsion_only.
- State that stiffness_structured beats isotropic on all tracked random and scaffold A/E seed-metric comparisons.
- State that stiffness_structured beats torsion_only in the attribution setting on both splits.
- State that harmonic is stronger than stiffness_structured on the tracked loss/RMSD metrics.
- Separate supported results from unavailable metrics: AMR/COV, COV/MAT, and full physical-validity checks are not in the current summaries.
