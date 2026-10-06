# Granularity-Induced Weak Identification

**Status:** exact rate calculation + simulation plan. Not part of submission.

## Motivation

Previous local sequences made within-state treatment variation shrink. But fine latent representations can also destroy downstream information simply by fragmenting the sample into an increasing number of effect parameters, even when overlap within every state remains healthy.

## Growing-state construction

Let the fine latent state S_N take K_N states with approximately equal occupancy:

    K_N = N^kappa,
    P(S_N=k) ~ 1/K_N.

Assume residual treatment variance within every state is bounded away from zero:

    Var(T_tilde | S_N=k) = sigma^2 > c > 0.

Then each state receives about

    n_k ~ N/K_N = N^(1-kappa)

observations.

A state-specific residual slope therefore has standard error

    se(theta_hat_k)
      ~ 1/sqrt(n_k sigma^2)
      = O(N^((kappa-1)/2)).

Thus:
- kappa<1: fine-state effects are estimable but slower than root-N;
- kappa=1: each state receives O(1) causal information and state-specific error does not vanish;
- kappa>1: most states receive vanishing/no observations and the full fine-state target is impossible to learn.

Yet a common/task-pooled effect can still use O(N) total treatment information and remain root-N.

This is a **granularity-induced weak-identification** phenomenon with no deterioration in within-state overlap.

## Unified boundary with weak overlap and local heterogeneity

Now let:
    K_N = N^kappa,
    residual treatment SD sigma_N = N^-alpha,
    effect separation delta_N = N^-beta.

Per-state causal KL information scales as

    (N/K_N) * sigma_N^2 * delta_N^2
      = N^(1 - kappa - 2 alpha - 2 beta).

Therefore the unified causal-resolution boundary is

    kappa + 2 alpha + 2 beta = 1.

Detectable:
    kappa + 2 alpha + 2 beta < 1.

Local/nontrivial:
    kappa + 2 alpha + 2 beta = 1.

Undetectable:
    kappa + 2 alpha + 2 beta > 1.

The earlier alpha+beta=1/2 result is the special case kappa=0.

## Interpretation

There are at least two independent ways a high-fidelity latent representation can outrun causal resolution:

1. **support collapse:** treatment variation shrinks within a state (alpha>0);
2. **granularity fragmentation:** the representation creates more state-specific target parameters (kappa>0).

Both consume the same finite causal-information budget.

This yields a simple "resolution budget":

    granularity cost + overlap cost + local-effect difficulty
      = kappa + 2 alpha + 2 beta.

## Why this is potentially important

Modern representation learners often increase effective state granularity with data/model capacity. Even if every learned state is real and has good treatment overlap, asking for a separate causal effect in every increasingly fine state can create an incidental-parameter-like failure.

The generative representation may become richer with N while the downstream causal target becomes harder.

## Planned simulation

Use perfectly observed K_N states with constant residual treatment variance. Compare:
- average state-specific estimation error;
- task-pooled/common-effect error;
- power for a fine-state contrast.

Sweep kappa and N. Then add alpha to verify collapse against the combined exponent 1-kappa-2alpha-2beta.
