# Target-Subspace Resolution Paradox

Status: exact triangular-array construction; not submission material.

A causal model can become arbitrarily ill-conditioned globally while a fixed scientific target becomes increasingly precise.

Let
    J_N = diag(1, N^{-2 alpha})
and suppose the scientific target is only the first effect coordinate:
    A = [1,0].

Then
    lambda_min(J_N) = N^{-2 alpha} -> 0,
    condition number = N^{2 alpha} -> infinity,
so every global conditioning diagnostic says the problem is worsening.

But
    ||A J_N^{-1}|| = 1
for every N, and the target's sampling error remains O(N^{-1/2}).

For alpha>1/2, a global standard-error scale based on lambda_min behaves as
    N^{2 alpha - 1/2}
and diverges, while the actual scientific target standard error shrinks as N^{-1/2}.

Thus global causal ill-conditioning and target-specific estimability can move in opposite directions.

## Consequence

There are two distinct failure modes of lambda_min:
1. phantom optimism: proxy state mixing can inflate lambda_min and hide a target-relevant weak direction;
2. irrelevant pessimism: a genuinely weak direction outside the scientific target can drive lambda_min to zero while the target remains perfectly regular.

Therefore lambda_min alone is neither sufficient nor necessary for scientific-target reliability. The correct object must combine the target map A, the oracle/completed information operator J, and representation perturbation direction b.

## Reproduction
research/target_subspace_resolution_paradox.py
