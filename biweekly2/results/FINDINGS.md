# Biweekly Report 2 — key findings and report-ready conclusions

SHHPC, 2026-10-19. Code: `stifflow-v1@shhpc-research-20261018` (dc3e07b),
`stifflow-v1-benchmark@shhpc-research-20261018-benchmark` (01b3865),
`stifflow-v1-alternative-A@alternative-a` (c56d466). All new experiments use the
frozen 512-molecule validation monitor; test appears only through the frozen
formal protocol. The stiffness-free baseline is labelled **ET-Flow-backbone
control** — it is *not* the pretrained ET-Flow model.

## Supported findings

1. **Evaluation caliber is sound; no re-evaluation was needed.** Across val
   (9,654), test (993) and monitor (512) reference manifests, zero molecules
   have non-empty `reference_atom_mappings`; the 5 topology-variant val
   molecules were handled at export. The P0 raw-data recheck (64 molecules per
   method, verified against the trusted GEOM pickles) shows
   `conditioning_topology_matches`, `conditioning_stereochemistry_matches` and
   `matches_current_canonical_coordinates` true for every checked molecule with
   zero different rows. The 793d502 mapping fix therefore did not perturb any
   existing result. Uncertainty is now reported in two separated forms: seed
   standard deviation (training stochasticity, n = 3 seeds) and molecule-paired
   bootstrap CIs (sampling uncertainty of the molecule set, pooled over the 3
   seeds).

2. **The conditional Gaussian of StifFlow is empirically exact.** On 32
   real validation molecules (fixed x1, fixed L_K, fixed orientation), the
   analytic identity `A_t A_t^T = f²C₀ + g²C_K + fg(SL_Kᵀ + L_KSᵀ)` holds to
   ≤ 1.3e-15 in float64. With M draws at fixed t, the empirical covariance of
   X_t converges to the analytic Σ_t at the expected ~1/√M rate (median
   relative error 0.19 at M = 1024, 0.09 at M = 4096, t = 0.5), and whitened
   samples are standard Gaussian: all eigenvalues of the whitened covariance
   lie inside the Marchenko–Pastur bounds at every molecule, time and M.

3. **The trained StifFlow model is SE(3)- and permutation-equivariant,
   and sampling is consistent under transformed noise.** With the formal 250k
   checkpoint (and the control as contrast), max velocity equivariance errors
   are 1.8e-6 (rotation) and 1.2e-6 (atom permutation) over 32 molecules —
   at the float32 noise floor. Euler-50 generation from rotated/permuted
   initial noise reproduces the transformed endpoint to max RMSD 4.2e-7 Å
   (rotation) and 7.5e-4 Å (permutation) over 16 molecules. Any quality
   difference between StifFlow and control is therefore *not* attributable to
   broken symmetry handling.

4. **Parity between StifFlow and the ET-Flow-backbone control is real,
   and now explained.** Test MAT-R: StifFlow − control = −0.0022 Å (pooled
   3-seed molecule-paired CI [−0.0081, +0.0038]); val: +0.0024 Å
   ([+0.0007, +0.0041]) — statistically detectable on val but physically
   negligible (< 0.003 Å). The new learning curves (StifFlow and control
   snapshots at 25k/50k/100k/175k/250k on the monitor) show the two methods
   track each other within 0.01 Å MAT-R at *every* budget, so parity is not a
   budget artifact. Mechanism, from the shared-z path analysis on 128 training
   rows: StifFlow's supervision velocities differ from control's by a median
   of only 6–10% over t ∈ (0, 1) (90th pct ≤ 13%), even though the terminal
   factor differs strongly from the graph prior (‖L_K − S‖/‖S‖ median 0.83;
   S⁻¹L_K spectrum median 0.07, max 1.24). The alignment schedule g(t) = t²(1−t)
   suppresses the stiffness contribution exactly where it is largest. Labels
   are trustworthy (curvature-calibration ratios ≈ 1.000; only 14.4% of
   molecules have any floored mode; MMFF-relaxed reference offset median
   0.153 Å), so *weak signal utilization*, not label error, is the most
   consistent explanation. Whether a coupling change (rather than a stiffness
   factor) helps remains the open question the linear/aligned-linear 25k
   screening is designed to answer.

5. **Physical quality matches the control; no stiffness benefit, no deficit.**
   On the monitor subset, raw generated conformers of StifFlow and control have
   indistinguishable MMFF94 relaxable strain (medians 5.47 vs 5.52 kcal/mol;
   references 5.28; paired StifFlow − control mean +0.03 kcal/mol, CI
   [−0.23, +0.35]) and heavy bond-length deviations (0.0046 vs 0.0038 Å).
   Force-field relaxation changed stereochemistry in ~1% of conformers for
   every method (58/5482 StifFlow, 58 control) and 28/2741 references —
   recorded, never used to replace raw coordinates. Chirality is the one metric
   where StifFlow is *slightly but consistently* behind control (pooled Δ:
   −0.007 test [−0.0135, −0.0003], −0.012 val [−0.0138, −0.0093]);
   per-molecule strain deltas stratify with chirality (StifFlow−control: −0.12
   kcal/mol on all-correct molecules, +0.19 on partially-wrong ones), hinting
   the small chirality deficit, not overall geometry, is where any residual
   difference lives.

## Still unknown / next stage

- Whether a coupling-level change (linear / aligned-linear) beats control at
  full budget. The 25k screening is complete: `aligned_linear` passed all five
  preregistered gates, while `linear` and both null variants did not. Cross-seed
  and full-budget confirmation remain open; see
  `screening-25k/SCREENING-25k.md`.
- QM9-only evidence; GEOM-drugs-scale stiffness conditioning is untested.
- No clean external ET-Flow pretrained baseline on this split.

## Figure captions (report-ready)

**Figure 1 — Mathematical verification and mechanism.** (a) Relative error of
the empirical covariance of X_t against the analytic A_tA_tᵀ for StifFlow
on 32 real validation molecules (fixed target, Hessian factor and orientation;
median and 10–90% band), converging at the expected 1/√M finite-sample rate.
(b) Max velocity-field equivariance errors of the trained 250k checkpoints
(rotation solid, permutation hatched; 32 molecules) with Euler-50 sampling
consistency from transformed initial noise (16 molecules) printed below; the
dotted line marks the float32 noise level. (c) Relative velocity difference
between StifFlow and the stiffness-free control along the shared-noise
conditional path (128 training rows). (d) Mean conditional target velocity
energy ‖U_t‖²: StifFlow and control coincide.

**Figure 2 — Generation performance and physical quality.** (a) Frozen test
protocol (993 molecules, 250k updates, 3 seeds): MAT-R (solid, left axis) and
macro chirality (hatched, right axis), error bars = seed SD; the annotation
gives the pooled molecule-paired bootstrap CI of StifFlow − control MAT-R.
(b) Monitor learning curves (512 molecules, seed 42): StifFlow and control are
within 0.01 Å at every budget. (c) MMFF94 relaxable strain of raw generated
conformers (5,482 conformers per method; references = 2,741 official
conformers; clipped at 40 kcal/mol for display). (d) Heavy bond-length
deviation vs chirality fraction: StifFlow and control sit at the same point.

**Table (compact).** Generation metrics on the frozen test/val protocols with
both uncertainty types; MMFF94 physical quality; numerical-verification
summary; mechanism quantities.
