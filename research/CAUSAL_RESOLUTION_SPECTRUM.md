# Causal Resolution Spectrum

**Status:** exploratory theoretical synthesis. Not part of submission.

## 1. Directional information, not just lambda_min

Let theta be a K-dimensional vector of fine-state causal effects and suppose the local residual-score experiment has information/Gram matrix J_N. For a unit direction v, compare

    H0: theta = theta_0
    H1: theta = theta_0 + delta_N v.

In a regular Gaussian/local-quadratic experiment, the KL/noncentrality scale is

    N * delta_N^2 * v^T J_N v.

Therefore the smallest direction-v effect separation that can be resolved is of order

    delta_res(v) = 1 / sqrt(N * v^T J_N v).

If J_N has eigenpairs (lambda_j, q_j), then each eigendirection has a **causal resolution scale**

    delta_j = 1 / sqrt(N lambda_j).

This yields a full resolution spectrum, rather than a single worst-case lambda_min.

## 2. Causal effective rank

For a scientifically meaningful effect scale delta, define

    r_causal(delta)
      = #{ j : N delta^2 lambda_j(J_N) >> 1 }.

This is the number of causal-effect directions resolvable at scale delta. A generative latent model may have K well-separated/recoverable states while r_causal(delta) << K.

This formalizes "observational state resolution > causal resolution."

## 3. Refinement adds contrast directions

Splitting one task-equivalent state into two microstates introduces a new contrast direction, approximately

    v_contrast = (1,-1)/sqrt(2)

inside that macrostate.

If one microstate has vanishing residual treatment information, the information eigenvalue associated with this contrast can collapse. Generative refinement therefore adds a real latent distinction but also adds a causal parameter direction whose resolution scale may diverge.

The state-refinement paradox is a special case: lambda_contrast ~ N^{-2 alpha}, so

    delta_res ~ N^{alpha-1/2}.

Exactly the earlier rate.

## 4. State merging and spectral regularization are two forms of resolution control

**Hard task abstraction / equality constraint.**
Merging two effect parameters imposes theta_1=theta_2 and deletes the contrast direction from the target space.

**Spectral regularization.**
Ridge/truncated inversion keeps the fine parameterization but shrinks low-information eigen-directions.

Thus state merging and spectral regularization can be understood as discrete and continuous versions of the same operation:

> suppress causal directions below the data's resolution limit.

This unifies two previously separate strands of the branch.

## 5. Why lambda_min is insufficient

lambda_min reports only the worst direction. Two systems can have identical lambda_min but very different:
- number of weak directions;
- alignment of the target/representation error with those directions;
- scientific effect scale of interest.

This explains the K=3 orientation experiment where eps/lambda_min was almost unchanged but downstream error differed six-fold.

A better diagnostic should retain at least:
- the eigenvalue spectrum;
- projection of score/representation perturbation onto eigenvectors;
- the target effect scale.

## 6. Resolution-adaptive target

For a chosen scientific resolution delta, one can define a spectral target subspace

    V_delta = span{q_j : N delta^2 lambda_j >= c}.

Estimate/project theta only onto V_delta, or impose structured state-equality constraints approximating that subspace.

This produces an estimand that explicitly says which effect contrasts are supported by the data at scale delta.

## 7. Connection to inference

Standard root-N asymptotics implicitly require relevant information eigenvalues bounded away from zero. When lambda_j,N -> 0, the correct local rate in direction q_j is

    1 / sqrt(N lambda_j,N),

not 1/sqrt(N).

This suggests reporting direction-specific resolution limits instead of nominal confidence intervals for near-unidentified fine-state effects.

## 8. Research implications

Potential new methodological program:

1. estimate J and its uncertainty;
2. compute a causal resolution spectrum;
3. identify scientifically relevant effect contrasts;
4. suppress/merge contrasts below resolution;
5. retain fine latent state for nuisance adjustment;
6. perform inference on the selected/resolution-adapted target.

Open issues:
- eigenvalue/eigenvector estimation under generated latent states;
- threshold choice c and scientific delta;
- post-selection inference;
- structured partitions versus arbitrary spectral subspaces;
- time-varying/local J_t;
- whether resolution-adapted estimands should be reported instead of unstable fine-state effects.

## 9. Novelty discipline

This is closely related mathematically to weak identification, Fisher information, inverse problems, spectral regularization, and estimability. The potentially distinctive contribution is the application to **latent-state refinement and task-specific causal-effect resolution**, and the connection between generative state granularity, causal effective rank, state merging, and spectral regularization. A dedicated literature review is required.
