# Scientific-Target Subspace Reliability

Status: exact linear algebra + exploratory experiment; not submission material.

A vector causal model may be globally ill-conditioned because of a weak effect direction that the scientific question never asks about.

Let theta be K-dimensional and let the reported scientific target be A theta, where rows of A span a d-dimensional contrast subspace. First-order representation perturbation b induces target error

    A J^{-1} b.

The correct worst-case amplification for this target is therefore

    ||A J^{-1}||_op,

not

    1/lambda_min(J).

If the weakest eigendirections of J lie in null(A), they are irrelevant to the reported target and can make global condition numbers arbitrarily alarming without affecting scientific reliability.

Conversely, if A aligns with weak directions, the global bound can be tight.

This gives a hierarchy:
- global worst-case reliability: ||J^-1||;
- target-subspace reliability: ||A J^-1||;
- fixed-contrast reliability: ||a' J^-1||;
- realized representation displacement: |a'J^-1 b|.

Each step uses more task information and is weakly less pessimistic.

## Consequence

A paper reporting lambda_min(J) alone can simultaneously:
- over-warn about irrelevant weak directions;
- under-warn if proxy geometry hides a weak target-relevant direction.

The scientifically meaningful diagnostic must specify the target contrast/subspace.

Reproduction:
research/subspace_reliability.py
