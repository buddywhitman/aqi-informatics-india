# Resolution Divergence Theorem: More Data Can Increase State Certainty While Destroying Causal Resolution

**Status:** exact asymptotic construction + simulation plan. Not part of submission.

## Construction

Let S in {0,1} be an observed/latent microstate with fixed occupancy pi in (0,1).

### Observation/emission channel
    Z | S=s ~ N(mu_s, 1),
with fixed separation Delta_Z=|mu_1-mu_0|>0.

The per-observation KL/Jensen-Shannon information distinguishing the two emission components is a positive constant. Hence evidence favoring a correctly specified two-state observation model over a one-state collapsed model accumulates at order

    O(N)

(up to the usual O(log N) BIC penalty). State classification/recovery can therefore become increasingly certain as N grows.

### Causal channel
Within the same states, residual treatment variation shrinks:
    T_tilde | S=s ~ N(0, N^{-2 alpha})
for the contrast of interest, and
    Y_tilde = theta_s T_tilde + U.

Compare a common-effect model theta_0=theta_1 against a fixed nonzero effect contrast Delta_theta.

The causal KL information for that contrast is order

    N * Delta_theta^2 * N^{-2 alpha}
      = O(N^{1-2 alpha}).

If alpha>1/2, this tends to zero.

## Resolution divergence

For alpha>1/2:

    observational/state evidence -> infinity,
    causal-effect evidence -> 0.

Thus increasing N simultaneously makes the latent-state distinction easier to establish and its causal-effect distinction harder to learn.

This is stronger than saying observational and causal resolutions differ. They can move in **opposite asymptotic directions on the same data-generating sequence**.

## Consequence

A generative model-selection procedure can become arbitrarily confident that a state split is real while the corresponding fine-state causal contrast becomes information-theoretically unlearnable.

Therefore:
- "the state is real" does not imply "the state-specific causal effect is estimable";
- increasing sample size does not necessarily reconcile generative and causal model selection;
- selecting latent granularity solely by likelihood can asymptotically over-resolve the causal target.

## Caveat

The observation and treatment channels are deliberately constructed to scale differently with N. This is a local asymptotic sequence, not a claim that treatment variation generally shrinks in ordinary fixed-DGP sampling. Its relevance is to settings where increasing data resolution/refinement creates progressively narrower treatment support within finer states.

## Testable signature

For alpha=.75 and fixed emission separation:
- BIC advantage for K=2 over K=1 should grow roughly linearly with N;
- state classification error should remain low/decrease with better parameter estimation;
- causal Wald noncentrality scales N^(1/2-alpha)=N^-1/4 and power should fall toward test size.

This is the cleanest synthetic demonstration of observational-causal resolution divergence found so far.
