# Directional Reliability Operator

Status: exact linear-algebraic refinement; not submission material.

For a local moment perturbation b induced by representation error, first-order target displacement is

    delta_theta = J^{-1} b.

The scalar quantity ||b||/lambda_min(J) is only the operator-norm upper bound

    ||J^{-1}b|| <= ||b|| / lambda_min(J).

It discards the orientation of b.

If b lies in eigenvector q_j of J, then exact amplification is

    ||delta_theta|| = ||b|| / lambda_j.

Thus equal-magnitude representation perturbations can have arbitrarily different causal effects when they align with different information eigendirections, while the scalar worst-case bound assigns them the same difficulty.

This gives a natural directional reliability spectrum:
    R_j = |q_j' b| / lambda_j,
and total first-order displacement
    ||delta_theta||^2 = sum_j (q_j'b)^2 / lambda_j^2.

For a scientific contrast a, the relevant scalar is
    |a' J^{-1} b|,
not generally ||b||/lambda_min.

## Consequences

1. lambda_min is appropriate for worst-case guarantees, not instance-specific reliability.
2. posterior error magnitude must be projected through the downstream sensitivity operator.
3. task-weighted/directional calibration estimates the numerator components q_j'b.
4. the causal-resolution spectrum supplies the denominators lambda_j.
5. the combination yields a full reliability operator rather than a single heuristic ratio.

This unifies the K=3 orientation counterexample, directional calibration spectrum, and causal-resolution spectrum.

The methodological target is to estimate or conservatively bound b's projections without oracle states.

Reproduction:
research/directional_reliability_operator.py
