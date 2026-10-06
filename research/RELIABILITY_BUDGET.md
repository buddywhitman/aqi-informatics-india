# Target Reliability Budget

Status: local asymptotic synthesis; not submission material.

The directional reliability operator describes representation-induced target displacement, but total uncertainty also includes ordinary sampling noise.

A local linearization is

    theta_hat - theta ~= J^-1 b_gamma + J^-1 xi / sqrt(N),

where b_gamma is systematic score perturbation from the generated representation and xi has score covariance Sigma.

For scientific contrast a:

Representation component:
    B_rep(a) = |a' J^-1 b_gamma|.

Sampling component:
    V_samp(a) = a' J^-1 Sigma J^-1 a / N.

A natural local reliability budget is therefore

    R_a^2 ~= B_rep(a)^2 + V_samp(a)

when cross terms are negligible/orthogonal.

This separates two regimes:
- sampling-limited: representation bias is below sampling noise;
- representation-limited: more observations shrink sampling error but leave representation-induced displacement dominant.

## Important consequence

A fixed nonvanishing representation perturbation can become *more important* with larger N even if its absolute bias does not grow, because sampling uncertainty shrinks. Thus a representation adequate at N=1,000 may become the inferential bottleneck at N=100,000.

For valid root-N inference, the target-projected representation perturbation needs to satisfy roughly
    a'J^-1 b_gamma = o(N^-1/2),
not merely be small in an absolute/global representation metric.

This yields a target-specific representation accuracy requirement.

## Connection to prior results

- causal resolution controls J^-1;
- directional/task calibration controls b_gamma;
- sample size controls only the stochastic term;
- target map a determines which components matter.

This may be the cleanest unifying local risk equation found on the branch so far.
