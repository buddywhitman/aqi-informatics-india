# Self-Canceling Reliability Diagnostics

**Status:** controlled synthetic finding. Not part of submission.

## Question

The repository's operational difficulty score couples posterior uncertainty with downstream conditioning, schematically

    D_proxy = H(gamma) / lambda_min(J_proxy).

PHANTOM_CAUSAL_RESOLUTION.md shows that state ambiguity can inflate lambda_min(J_proxy). Therefore the same representation degradation may:
- increase H(gamma), correctly warning about uncertainty;
- increase lambda_min(J_proxy), incorrectly suggesting healthier causal geometry.

Dividing the two can cancel the warning.

## Controlled experiment

K=3 balanced states. Treatment residual SDs are (1,.3,weak), with weak varied over (.04,.06,.08,.12,.2). Emission separation varies from .35 to 3. A Gaussian mixture produces soft posteriors.

Define the mechanistic oracle difficulty

    D_oracle = eps_gamma / lambda_min(J_true),

where true states are available only in simulation.

Compare:
- entropy H;
- H / lambda_min(J_proxy);
- H / lambda_min(J_true);
- true posterior L1 error eps_gamma;
- proxy eigenvalue inflation lambda_min(J_proxy)/lambda_min(J_true).

Across 3,500 worlds/runs:

Diagnostic                         Spearman vs D_oracle    AUC top-20% D_oracle
entropy                                  0.3295                  0.6402
H / proxy lambda_min                     0.0934                  0.4700
H / oracle lambda_min                    0.9861                  0.9941
posterior L1 error                       0.3633                  0.6711
lambda_min inflation                     0.9162                  0.9444

## Main finding

The proxy-conditioned composite is **worse than entropy alone and slightly worse than chance** for detecting the hardest oracle-difficulty cases.

This is not evidence against coupling representation uncertainty with task geometry in principle. The oracle-conditioned score is nearly perfect. It is evidence that using a geometry estimate contaminated by the same representation error can create a self-canceling diagnostic.

## Mechanism

As emission separation deteriorates:
1. posterior uncertainty rises;
2. inferred states mix strong- and weak-treatment-information observations;
3. proxy lambda_min rises;
4. the denominator of H/lambda_min grows at the same time as the numerator;
5. the coupled score suppresses its own uncertainty warning.

The eigenvalue-inflation factor itself has AUC 0.944 for the oracle-hard cases, confirming that optimistic spectral distortion is tightly linked to failure.

## Connection to existing repository results

This provides a plausible mechanism for why the operational coupled score is not uniformly better than entropy in the existing multidomain/cross-city experiments, especially Bengaluru and Kolkata. It does not prove that phantom resolution is the cause in those datasets because oracle latent states are unavailable there.

## Methodological implication

A task-coupled reliability score should avoid using an information denominator that is optimistically distorted by the same uncertain representation.

Candidate fixes:
1. conservative lower bound on oracle lambda_min;
2. posterior/confusion sensitivity envelope for lambda_min;
3. joint score using entropy plus an uncertainty penalty on spectral inflation rather than raw proxy lambda_min;
4. sample/representation splitting where possible;
5. partial-identification interval for oracle causal resolution.

## Reproduction

- research/phantom_diagnostic_cancellation.py
- research/results/phantom_diagnostic_cancellation_summary.csv

## Caveats

- D_oracle is a simulation-defined mechanistic target, not observed causal error.
- The experiment uses iid states and Gaussian mixtures to isolate the mechanism.
- The posterior L1 error requires label alignment and oracle states; it is not operational.
- AUC below .5 here means the proxy composite reverses ranking for this designed family, not that it must fail in every application.
