# Scientific corrections on master

> Implement directly in the authorized master branch; verify before updating its ref.

**Goal:** Repair the non-identification example, align inference with the posterior-adjusted coupled estimator, and remove unsupported general claims.

**Baseline:** b536ee7bbbdbedd02168611d1fa694e68eac57bf. Supplementary source files were checked against the master Git blob hashes. Historical manuscript lineages remain historical; current appendix overrides their defective passages.

1. Add numerical regression checks for the coupled estimating equation, centered HAC, occupation cross-covariance, and assumption-consistent observational equivalence. Observe failures on the baseline.
2. Correct canonical estimator scores, matching bootstrap moments, occupation-weighted variance, and uncertainty labels. Preserve existing point-estimation modes.
3. Replace the non-identification example; rewrite current inference for posterior-adjusted nuisances under explicit sufficient dependence and empirical-process conditions. State what the implementation does and does not estimate.
4. Remove misspecification lower-bound and diagnostic identification overclaims. Keep empirical results and registered abstract unless recalculation changes their evidence.
5. Run regression checks and existing verification, rebuild the appendix, PDF and supplement, inspect the rendered result, and commit the tested changes to master with a head lease.

New learned-representation benchmarks and a general feasible latent-bias correction are future research, not fabricated as part of these repairs.
