# Scope of the Directional Reliability Operator

The integrated Markov latent-state experiment revealed an important distinction.

For a linear moment estimator with proxy moment system
    J_p theta_p = S_p
and oracle solution theta_o, defining
    b = S_p - J_p theta_o
implies algebraically
    theta_p - theta_o = J_p^{-1} b.

Therefore the near-zero prediction error in the integrated DGP is an **identity**, not empirical validation of a causal theory. This is useful because it shows the operator is the exact decomposition of proxy-versus-oracle displacement for linear moment systems, but it cannot by itself establish that theta_o equals a structural causal effect or that b is operationally estimable.

The research burden therefore splits:

1. Algebraic decomposition:
   exact for linear proxy moment systems.

2. Causal interpretation:
   requires identification assumptions linking oracle moment target to the desired causal estimand.

3. Operational diagnosis:
   requires estimating/bounding b and the relevant oracle/target geometry without observing latent S.

4. Nonlinear estimators:
   require Taylor expansion and curvature/remainder control; the nonlinearity number work addresses this boundary.

This correction prevents circular validation claims.

A new nonlinear-outcome experiment deliberately misspecifies the structural response while retaining a linear slope estimator. It tests whether the operator can perfectly explain proxy-versus-oracle *projection* displacement while both proxy and oracle projections differ from the structural effect. If so, it demonstrates that reliability relative to an oracle estimator and causal validity relative to the scientific estimand are separate axes.
