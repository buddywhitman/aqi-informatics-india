# Double Ill-Posedness: Representation Deconfusion x Causal Inversion

**Status:** exploratory derivation + Monte Carlo. Not part of submission.

## Motivation

PHANTOM_CAUSAL_RESOLUTION.md shows that state misclassification can inflate weak causal-information eigenvalues by mixing high-information observations into inferred weak states.

A natural correction is to deconvolve state confusion. That creates another inverse problem.

## Two-state moment deconvolution

Let true state-specific treatment-information moments be

    m = (m_0,m_1)'.

Let hard inferred labels have confusion matrix

    C = [[1-p, p],
         [p, 1-p]]

under symmetric misclassification. Observed label-specific information moments satisfy

    m_obs = C' m.

If p is known and p<1/2,

    m = C^{-T} m_obs.

The eigenvalues of C are 1 and 1-2p, so

    ||C^{-1}||_2 = 1/(1-2p).

As p -> 1/2, correcting state confusion becomes arbitrarily ill-conditioned.

## Downstream causal inversion

The causal target itself is obtained from a moment system

    theta = J^{-1} S.

Thus generated-state causal inference can contain **two sequential inversions**:

    observed proxy moments
       -- C^{-1} --> deconfused state moments
       -- J^{-1} --> causal effects.

First-order perturbations can therefore be amplified on the order of

    ||J^{-1}|| * ||C^{-1}||

(up to problem-specific Jacobians and alignment).

This gives a sharper mathematical interpretation of the original representation x task interaction:
- representation ambiguity controls the conditioning of the **deconfusion operator**;
- weak overlap/task geometry controls the conditioning of the **causal operator**;
- the two ill-posedness mechanisms can multiply.

## Monte Carlo stress test

Balanced true states. Residual treatment SDs are (1.0,0.1), so the second state's information is about 100x weaker. Hard labels are symmetrically flipped with probability p.

The naive inferred-state minimum information is dramatically inflated because the weak inferred group receives high-variance treatment observations from the strong state.

At N=20,000:
- p=.10: naive lambda_min inflation ~10.9x;
- p=.20: ~20.8x;
- p=.30: ~30.7x;
- p=.40: ~40.6x;
- p=.45: ~45.6x;
- p=.49: ~49.4x.

Deconfusion is unbiased in the ideal known-C model but unstable. At N=20,000:
- corrected weak-information RMSE rises from ~.0033 at p=.10 to ~.2148 at p=.49;
- corrected weak information is negative in ~47% of runs at p=.49.

At N=1,000 the instability is worse: even p=.10 yields negative corrected weak information in ~38% of runs because the true weak moment is tiny relative to deconvolution noise.

## Interpretation

There is no free correction:
- ignoring confusion creates **phantom causal resolution**;
- inverting confusion can expose the true weak geometry but amplifies sampling noise;
- causal inversion then amplifies the remaining error again.

This suggests that an operational method needs **joint regularization of representation deconfusion and causal inversion**, rather than correcting the representation and causal stages independently.

## Potential methodological direction

Estimate a confusion/soft-channel operator A linking latent-state moments to proxy moments, then solve a jointly regularized inverse problem such as

    min_{m,theta}
       ||m_proxy - A m||^2
       + tau R_repr(m)
       + ||S(m)-J(m)theta||^2
       + lambda R_causal(theta),

with uncertainty propagated across both stages.

Alternatively avoid explicit deconvolution and derive partial-identification bounds on oracle information given bounds on state misclassification/calibration.

## Caveats

- Known symmetric C is an idealization. Estimated/asymmetric confusion is harder.
- Negative deconvolved information moments are finite-sample artifacts indicating instability, not physically negative information.
- General soft posteriors require an integral/channel operator rather than a finite hard-label confusion matrix.
- Product conditioning is a first-order/worst-case statement; alignment can make realized amplification smaller or larger in particular directions.

## Reproduction

See:
- research/confusion_deconvolution_instability.py
- research/results/confusion_deconvolution_summary.csv
