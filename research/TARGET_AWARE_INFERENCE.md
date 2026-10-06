# Target-Aware Inference, Selection, and Regularization

Status: exploratory synthesis; not submission material.

The target-subspace paradox implies that global conditioning should not automatically drive model selection or regularization.

## Target-aware model selection

For a declared scientific target A theta, compare candidate latent/causal models using target amplification ||A J^-1|| rather than lambda_min(J) alone.

A model can have a tiny irrelevant eigenvalue and therefore terrible global condition number while being optimal for A theta. Conversely, a globally well-conditioned model may provide less information in the target direction.

The script target_aware_model_selection.py constructs this reversal and measures selection regret.

## Target-aware regularization

Isotropic ridge regularizes every effect direction:
    theta_lambda = (J + lambda I)^-1 S.
If ill-conditioning lies outside the scientific target subspace, isotropic ridge introduces unnecessary bias in the target.

A target-aware spectral penalty can regularize weak irrelevant directions while leaving target-relevant strong directions unpenalized. In the diagonal construction, selective regularization stabilizes the nuisance/irrelevant coordinate with exactly zero induced bias in the scientific first-coordinate target, while isotropic ridge biases it whenever lambda>0.

This suggests regularization should be aligned with the target subspace and causal-resolution spectrum, not applied uniformly because global lambda_min is small.

## General principle

The same target map A should enter:
- reliability diagnostics: ||A J^-1||;
- representation evaluation: task-induced calibration/leverage;
- model/state-granularity selection;
- spectral regularization;
- confidence intervals.

A target-agnostic pipeline can therefore be suboptimal at every stage even when each individual component is statistically reasonable.

## Caveat

Target-aware procedures sacrifice protection for unreported contrasts. If exploratory science later changes A, previously suppressed directions may matter. Rich latent/nuisance representations should therefore be retained even when target-specific estimation/regularization is coarse.
