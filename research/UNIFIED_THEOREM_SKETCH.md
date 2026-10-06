# Unified Local Reliability Theorem — Working Sketch

Status: theorem-development note, not a proved submission theorem.

Let theta solve an oracle moment M(theta,eta,gamma0)=0, and let a generated-representation estimator solve a perturbed empirical moment with gamma_hat. Let J be the derivative with respect to theta at the oracle target, A a fixed scientific target map, and define the population representation score perturbation b_gamma.

Under local invertibility, stochastic equicontinuity, nuisance orthogonality, and a second-order remainder bound,

    A(theta_hat-theta0)
      = -A J^{-1} b_gamma
        - A J^{-1} G_N psi
        + A J^{-1} R_gamma
        + o_p(N^-1/2),

with signs depending on score convention.

The scientific reliability problem therefore has four separable components:

1. Target sensitivity:
       A J^{-1}.

2. Representation perturbation:
       b_gamma.

3. Sampling noise:
       G_N psi / sqrt(N), with long-run covariance under dependence.

4. Nonlinear remainder:
       R_gamma, whose scale can deteriorate faster than J^{-1} near weak information.

## Candidate sufficient conditions

For ordinary first-order representation sensitivity:
    ||A J^{-1} b_gamma|| = o(N^-1/2).

For representation-orthogonal scores with b_gamma=O(epsilon_gamma^2):
    ||A J^{-1}|| epsilon_gamma^2 = o(N^-1/2),
subject to tangent-space and remainder assumptions.

For directional information lambda_N=N^-s and first-order b_N=N^-r:
    r > (1+s)/2
in the stylized information-equals-score-variance scaling.

## Curvature condition

Scalar analysis suggests local validity requires a dimensionless quantity analogous to
    curvature * perturbation / information^2
to vanish. In vector problems this should become an operator/Hessian condition such as
    ||J^{-1}|| * ||H|| * ||J^{-1}b||
small, preferably target restricted rather than global.

## Target-family version

For A in declared family F, replace fixed-target norms by
    sup_{A in F} ||A J^{-1} b_gamma||
and simultaneous stochastic bounds. This interpolates fixed-target and global guarantees.

## Generated-state geometry warning

J must correspond to the oracle/completed scientific moment, not blindly to a posterior-mean plug-in Gram. Proxy state mixing can inflate weak eigenvalues and make J_proxy anti-conservative.

## Identification warning

This expansion concerns error around the oracle moment target. A separate condition is required for the oracle target to equal the desired structural/scientific estimand.

## What remains to prove

- precise tangent space for posterior/simplex perturbations;
- dependent-data empirical-process conditions;
- cross-fitted nuisance remainder;
- generated-posterior estimation rate;
- target-restricted curvature bound;
- operational bound/estimation for b_gamma;
- weak-identification/nonregular cases where local inversion itself fails.

The final theorem should not claim validity in the nonregular regime; that regime needs weak-ID robust inference or partial identification.
