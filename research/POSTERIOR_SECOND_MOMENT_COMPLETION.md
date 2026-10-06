# Posterior Second-Moment Completion for Oracle Causal Geometry

**Status:** exact identity in a simplified latent-state moment model + Monte Carlo validation. Not part of submission.

## Problem

Proxy geometry often substitutes posterior means for latent one-hot states, producing moments based on gamma gamma'. PHANTOM_CAUSAL_RESOLUTION.md shows that this can badly distort weak causal information.

The key observation is that for one-hot H,

    H H' = diag(H).

Therefore, for any information set I and q=P(S|I),

    E[H H' | I] = diag(q),

not q q'.

The missing term is exactly posterior covariance:

    diag(q) - q q'.

## Exact moment-completion identity

Let W be any scalar/matrix weight measurable with respect to I. Then by iterated expectation,

    E[ W H H' ]
      = E[ W E(HH'|I) ]
      = E[ W diag(q) ].

Thus latent-state second moments can be recovered without hard labels or confusion-matrix inversion **if the posterior conditions on every variable in W that carries state information** and the posterior model is correct.

## Treatment-information example

Three balanced states have:
    Z|S=k ~ N(mu_k,1),
    T|S=k ~ N(0,sigma_k^2),
with sigma=(1,.3,.08).

The oracle state-specific treatment-information matrix is

    J_oracle = E[T^2 H H'].

If q=P(S|Z,T), then

    J_complete = E[T^2 diag(q)]

equals J_oracle in population.

By contrast:
- gamma=P(S|Z) omits treatment information about the state;
- E[T^2 diag(gamma)] can be grossly biased because T^2 is not measurable with respect to Z and is not conditionally independent of S given Z;
- gamma gamma' is not the conditional second moment of H even when gamma is calibrated.

## Monte Carlo

Across N=600,1200,2400,4800 and emission separations .5,1,2:
- treatment-updated posterior completion recovers oracle lambda_min to within about 0.5-0.7% on average;
- Z-only diagonal completion inflates the weak information by roughly 5x to 47x depending on separation.

Examples:
N=1200:
    sep=.5: completed/oracle = .996; Z-only completion/oracle = 45.8
    sep=1.0: completed/oracle = 1.000; Z-only completion/oracle = 26.3
    sep=2.0: completed/oracle = 1.004; Z-only completion/oracle = 5.49

## Interpretation

There are two distinct errors in naive generated-state geometry:

1. **posterior-mean substitution:** q q' is not E[HH'|I];
2. **insufficient posterior information set:** a posterior based only on representation variables Z cannot correctly weight a moment involving T^2 when T contains additional information about S.

The first is fixed by posterior second-moment completion. The second requires conditioning the diagnostic posterior on the variables entering the information moment.

## Important causal caveat

Using T to infer S inside the **causal estimator itself** can alter the estimating equation and may create post-treatment/selection problems depending on the causal graph. The identity here is for recovering/diagnosing latent-state information moments. It does not automatically justify treatment-updated state weights for outcome-effect estimation.

A safe architecture may use:
- representation posterior P(S|Z) for the causal estimator under its identification assumptions;
- a separate diagnostic posterior P(S|Z,T,X) for estimating the causal information spectrum, with strict exclusion of Y/outcomes.

Whether this separation preserves valid operational inference requires theory.

## Connection to double ill-posedness

This offers a possible alternative to explicit confusion-matrix inversion. Rather than:
    proxy moments -> invert confusion -> oracle moments,
one computes conditional latent second moments directly under a joint state/treatment model.

But this moves the burden to model specification. Misspecified P(S|Z,T,X) can still give false geometry, so sensitivity/robustness remains necessary.

## Reproduction

- research/posterior_second_moment_completion.py
- research/results/posterior_second_moment_completion_summary.csv
