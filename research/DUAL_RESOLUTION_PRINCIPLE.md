# Dual-Resolution Principle: Fine for Adjustment, Coarse for the Target

**Status:** theoretical correction / exploratory note. Not part of the submission.

## Why this correction is necessary

The phrase "a finer representation is worse" is too broad. A fine latent state can be useful or necessary for confounding adjustment even when estimating a separate causal-effect parameter for every fine state is statistically disastrous.

The important distinction is between:

1. **adjustment resolution**: how much latent-state detail is retained in nuisance functions needed for identification;
2. **target resolution**: how many distinct causal-effect parameters are estimated.

These resolutions need not be the same.

## Setup

Let fine state S=(A,M), where A is a task-level macrostate and M is a microstate. Assume

    T = m_S(X) + sigma_S V
    Y = theta_A T + g_S(X) + U,

with E[V|X,S]=E[U|X,S,V]=0. Fine microstates may differ arbitrarily in treatment mean m_S and outcome baseline g_S, so coarsening S to A inside the nuisance functions can reintroduce confounding.

If nuisance residualization conditions on the **fine state** S,

    T_tilde = T - E[T|X,S] = sigma_S V
    Y_tilde = Y - E[Y|X,S] = theta_A sigma_S V + U.

Now estimate one theta_A by pooling residual score information across all microstates M belonging to A:

    theta_hat_A =
      sum_{i:A_i=A} T_tilde_i Y_tilde_i /
      sum_{i:A_i=A} T_tilde_i^2.

If at least one microstate in macrostate A has positive asymptotic occupancy and residual treatment variance bounded below, the denominator is O_p(N), so theta_hat_A is root-N even if another microstate has sigma_N=N^{-alpha}.

By contrast, estimating every theta_{A,M} separately and then occupancy-averaging gives nonvanishing weight to the weak-state estimate, whose standard error is O(N^{alpha-1/2}). This reproduces the refinement-rate penalty.

## Why naive confounder coarsening can be biased

If one instead residualizes only on A,

    T_bar = T - E[T|X,A],
    Y_bar = Y - E[Y|X,A],

then unresolved microstate differences remain. Even when theta is common within A, the score contains terms involving

    Cov(m_S(X), g_S(X) | X,A),

so macro-only adjustment is generally confounded unless the microstate is irrelevant to treatment/outcome nuisance structure conditional on A.

Therefore the safe operation is not necessarily "throw away the fine state." It is:

> retain fine latent resolution for nuisance/confounding control, but pool only those target-effect parameters whose distinctions are unsupported or task-irrelevant.

## Candidate theorem: dual-resolution pooling rescue

Under fine-state ignorability and regular nuisance estimation, suppose theta_{A,M}=theta_A for all M in A. If the pooled macrostate contains at least one microstate with occupancy bounded away from zero and residual treatment variance bounded away from zero, then the fine-adjustment/coarse-target estimator is root-N, even if another microstate in A has residual treatment variance N^{-2alpha} with alpha>1/2.

The fully refined target estimator that estimates theta_{A,M} separately and then occupancy-averages can have O_p(N^{alpha-1/2}) error.

## Consequence

The strongest defensible principle is not "coarser latent representations are better." It is:

> **Identification resolution and estimand resolution should be decoupled. Use enough latent detail to remove confounding; estimate only as much effect heterogeneity as the downstream information supports.**

This is closely related to the classical distinction between adjustment covariates and effect modifiers / subgroup parameters. Any novelty claim must therefore rest on the latent-state, sequential, weak-overlap, and data-adaptive target-resolution aspects rather than this distinction in isolation.

## Planned experiment

Create microstates with different treatment means and outcome intercepts (genuine microstate confounding), common effects within macrostate, and local weak residual treatment variance. Compare:

1. fine nuisance + fine target;
2. fine nuisance + task-coarse target;
3. coarse nuisance + coarse target.

Expected pattern:
- (1) suffers the weak-state rate penalty;
- (2) remains root-N and unbiased;
- (3) is asymptotically biased when unresolved microstate treatment/outcome baselines covary.

This is a stronger and more honest experiment than simply deleting microstates from the adjustment set.


## Exact three-way asymptotic separation in the committed experiment

The committed `dual_resolution_experiment.py` uses, within each macrostate, two equally likely microstates with treatment means (-1,+1), outcome baselines (-2,+2), common causal slope theta_A, and residual treatment SDs (1,N^{-alpha}).

For macro-only residualization,
```
plim theta_hat_coarse
 = theta_A + Cov(a_M,b_M|A) /
   [Var(a_M|A) + E(sigma_M^2|A)].
```
Here Cov(a,b)=2, Var(a)=1, and E(sigma^2)=(1+N^{-2alpha})/2, so
```
Bias_N = 2 / [1 + (1+N^{-2alpha})/2]
       -> 4/3.
```
Thus coarse adjustment has persistent asymptotic bias even though the causal slope is identical across the two microstates.

For fine adjustment + fine target, the weak microstate slope has SD O(N^{alpha-1/2}), so the occupancy-averaged macro effect inherits that rate.

For fine adjustment + coarse target, the pooled residual-score denominator contains O(N) information from the strong microstate, while the weak microstate contributes nonnegative information. Therefore the pooled macro effect is root-N.

This produces an exact **resolution trilemma**:
- over-coarse adjustment -> asymptotic confounding bias;
- over-fine target parameterization -> weak-identification variance/rate failure;
- fine adjustment + coarse target -> identification and root-N efficiency when the pooled effects are truly equal.

This correction is more precise than saying that the latent representation itself should simply be coarsened.
