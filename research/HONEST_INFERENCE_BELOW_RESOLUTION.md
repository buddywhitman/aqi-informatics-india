# Honest Inference Below the Causal Resolution Limit

**Status:** exploratory theory/design note. Not part of submission.

## Problem

A causal-resolution diagnostic can identify directions with information
    I_j = N lambda_j(J)
too small to resolve effects at a scientifically relevant scale. But setting those directions to zero changes the estimand and can create large approximation bias.

The honest alternative is to separate:
1. an estimable projection of the target;
2. unresolved causal contrasts.

## Directional confidence geometry

For an approximately Gaussian score experiment with information J, direction q_j has standard error
    se_j ~ 1/sqrt(N lambda_j).

A fine-state confidence ellipsoid therefore expands dramatically along weak eigendirections. If lambda_{j,N}=N^{-2 alpha}, its half-width scales
    N^{alpha-1/2}.

Thus for alpha>1/2, an honest confidence interval for that fine contrast must widen with N. This is counterintuitive but unavoidable: the triangular-array experiment itself contains less causal information as N grows.

## Resolution-aware report

Given a scientific effect scale delta and threshold c:
- resolved subspace R_delta = span{q_j: N delta^2 lambda_j >= c};
- unresolved subspace U_delta = orthogonal complement.

Report:
1. projected estimate P_R theta_hat with ordinary uncertainty on resolved directions;
2. for each scientifically named contrast a, its resolution index
       RI(a,delta)=N delta^2 * (a' J a)/(a'a);
3. if RI is below threshold, label the contrast unresolved rather than null;
4. optionally report a sensitivity interval after imposing an externally specified bound |q_j' theta|<=B_j on unresolved directions.

Without external bounds, the full target may have unbounded/very wide uncertainty along unresolved directions.

## Important implication

Under local weak overlap, "more data -> narrower confidence intervals" is false. Honest intervals can widen because information per observation collapses faster than N grows.

This is not a pathology of the CI method. It is the inferential manifestation of the KL resolution boundary.

## Planned validation

Simulate nominal 95% directional intervals under lambda_N=N^{-2 alpha}. Compare:
- correct information-aware interval;
- naive root-N interval that assumes fixed information;
- ridge interval centered on shrunken estimate without accounting for shrinkage bias.

Expected:
- honest interval retains coverage but widens for alpha>1/2;
- naive root-N coverage collapses;
- naive ridge intervals can look precise while badly undercovering nonzero unresolved effects.
