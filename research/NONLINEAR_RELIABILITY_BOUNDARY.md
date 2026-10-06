# Nonlinear Reliability Boundary

Status: exact scalar nonlinear analysis; not submission material.

The integrated linearized operator predicts empirical RMSE almost exactly when the data-generating experiment is itself linearized. That is a consistency check, not a difficult validation. The important falsification is nonlinear.

Consider the scalar nonlinear moment
    lambda d + c d^2 = b,
where d is target displacement, lambda is local information, c is curvature, and b is representation-induced score perturbation.

The linear reliability operator predicts
    d_lin = b/lambda.

The exact positive root is
    d = lambda/(2c) * (sqrt(1+4z)-1),
where the dimensionless number
    z = c b / lambda^2.

Therefore relative breakdown of the first-order operator is governed by z, not by b or lambda separately.

For z << 1, Taylor expansion gives
    d = (b/lambda)(1-z+O(z^2)),
so relative linearization error is O(z).

When z is order one or larger, the linear operator can badly overstate displacement. Weak information is especially dangerous because z scales as 1/lambda^2.

This adds a second reliability axis:
1. first-order amplification: J^-1 b;
2. local-validity/curvature number: roughly curvature x perturbation / information^2.

A large directional reliability operator may therefore signal both large error and failure of its own linear approximation.

## Empirical scalar examples

At curvature 1:
- lambda=1, b=.01: relative linearization error ~1%;
- lambda=.2, b=.01: ~21%;
- lambda=.05, b=.01: ~156%;
- lambda=.01, b=.01: ~951%.

Thus the same representation perturbation can move from safely local to profoundly nonlinear solely through information collapse.

## Implication

Any theorem built around A J^-1 b needs a remainder condition that scales with inverse information strongly enough to control curvature. Reporting only the leading term is unsafe near severe weak identification.

Reproduction:
research/operator_nonlinearity_breakdown.py
research/nonlinear_reliability_number.py
