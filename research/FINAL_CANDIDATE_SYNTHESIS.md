# Final Candidate Contribution Synthesis — Pre-Final

This document compresses the exploratory branch into a small candidate contribution set. It is intentionally stricter than the research README.

## Contribution 1 — Phantom causal resolution

**Candidate status: strongest novel phenomenon.**

Generated latent states can optimistically distort downstream causal-information geometry. State ambiguity mixes high-information observations into weak inferred regimes, inflating weak eigenvalues and reducing apparent condition number. This can persist asymptotically as a stable pseudo-geometry.

Key evidence:
- K=3 weak-state experiment: soft/hard lambda_min inflation 11.4x/21.2x under poor separation;
- proxy condition number collapses from ~153 to ~5.9/4.0;
- proxy geometry becomes more statistically stable with N while remaining wrong;
- two-state calculation proves structural inflation under symmetric misclassification.

Scientific importance: a standard conditioning diagnostic can become *more reassuring because representation recovery is worse*.

## Contribution 2 — Self-canceling reliability and task-relative representation error

**Candidate status: strong empirical/theoretical consequence.**

If representation uncertainty H rises while the same representation inflates proxy lambda_min, a composite H/lambda_proxy can suppress its own warning. Controlled result:
- entropy AUC 0.640;
- H/lambda_proxy AUC 0.470;
- H/lambda_oracle AUC 0.994.

More generally, downstream damage depends on where/direction representation errors occur relative to causal leverage. Identical global calibration can hide ~3000x task-weighted error and ~50x causal-geometry damage; representation rankings reverse ~100x across tasks.

Broad task-aware calibration is prior art, so novelty must be tied specifically to latent causal information geometry and phantom resolution.

## Contribution 3 — Conservative causal-geometry diagnosis for generated states

**Candidate status: methodological framework, pieces vary in novelty.**

Naive posterior-mean plug-in is wrong for latent second moments. For one-hot H:
    E[HH'|I]=diag(q).
If the posterior information set contains variables entering the diagnostic moment, posterior second-moment completion can recover oracle information in the correctly specified model.

But completion is highly misspecification-sensitive. Therefore the recommended operational object is not a naked completed lambda_min but:
- completed target-specific spectrum;
- model sensitivity envelope;
- perturbation uncertainty set;
- target-specific worst-case bias support function;
- validation/adjudication targeted by causal sensitivity.

This is a conservative framework rather than a single solved estimator.

## Contribution 4 — Dual causal resolution and target-specific reliability

**Candidate status: synthesis/connection, not broad novelty.**

Rich latent resolution may be needed for confounding adjustment while the causal target should be only as fine as scientifically requested and informationally supported. The correct local target perturbation is A J^{-1} b_gamma; lambda_min alone can be optimistic through phantom resolution or arbitrarily pessimistic through irrelevant weak directions.

Strongly identified functionals of weakly identified objects are prior art, so novelty should not be claimed for target-specific identification itself. The contribution is its integration with generated latent-state error and causal-resolution diagnostics.

## Contribution 5 — Representation orthogonality as future method

**Candidate status: promising but not mature enough for a main contribution yet.**

Toy results show first-order representation debiasing can change O(epsilon) to O(epsilon^2), potentially relaxing rate requirements from N^-1/2 to N^-1/4 under strong information. However operational correction requires identifying the latent perturbation tangent through validation, repeated proxies, or structural restrictions.

Unless a genuine operational estimator/theorem is completed, present this as future work rather than a headline claim.

## Negative results that strengthen credibility

Preserve:
- original task-conditioning hump was nuisance leakage;
- proxy D score often fails to beat entropy;
- generic geometry disagreement AUC ~0.486;
- simple task-weighted Brier improves only modestly;
- posterior completion can fail badly under treatment-model misspecification;
- correction of wrong representation directions can leave nearly all target bias;
- first-order operator breaks down under severe weak-information curvature;
- oracle-estimator fidelity does not guarantee scientific validity.

## Recommended paper-level thesis if this became a new paper

> Generated latent representations can systematically falsify the apparent information geometry of downstream causal problems. Reliability is therefore not a property of representation accuracy alone: it is target- and direction-relative, and must account for how representation uncertainty is transported through the causal information operator.

The cleanest empirical hook is phantom resolution; the cleanest methodological consequence is conservative target-specific causal-geometry diagnosis.
