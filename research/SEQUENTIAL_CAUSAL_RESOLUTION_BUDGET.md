# Sequential Causal Resolution Budget

**Status:** exact effective-information calculation under a stylized dependent Gaussian experiment. Not part of submission.

## Add temporal dependence to the resolution budget

Previous calculation:
    K_N=N^kappa,
    treatment SD=N^-alpha,
    effect contrast=N^-beta.

For independent observations, per-state causal information is
    N^(1-kappa-2alpha-2beta).

Now suppose the relevant score sequence has AR(1)-like correlation rho_N approaching one:
    rho_N = 1 - c N^-eta,   0<=eta<1.

For estimating a mean/score with AR(1) correlation, the long-run variance inflation is approximately
    (1+rho_N)/(1-rho_N) ~ C N^eta,
so the effective independent sample size is
    N_eff ~ N^(1-eta).

After dividing across K_N approximately balanced states, effective per-state information becomes

    N^(1-eta-kappa) * N^-2alpha * N^-2beta
      = N^(1 - eta - kappa - 2alpha - 2beta).

Hence the sequential causal-resolution boundary is

    eta + kappa + 2 alpha + 2 beta = 1.

Interpret the terms as an information budget:
- eta: **dependence/persistence tax**;
- kappa: **granularity tax**;
- 2 alpha: **overlap/treatment-information tax**;
- 2 beta: **effect-resolution tax**.

## Consequence

Even if each factor alone looks benign, their combination can cross the resolution boundary.

Example:
    eta=.25, kappa=.25, alpha=.15, beta=.10
gives total cost
    .25 + .25 + .30 + .20 = 1.00,
exactly at the local-resolution boundary.

Thus a moderately persistent sequence, moderately fine representation, modestly weakening treatment variation, and moderately subtle heterogeneity can jointly make fine-state causal contrasts nonregular even though no single diagnostic appears catastrophic.

## Relation to the current paper's ingredients

This unifies four quantities that appeared separately in the repository:
- persistent Markov/temporal dependence;
- number/granularity of latent regimes;
- downstream score conditioning/overlap;
- target effect heterogeneity scale.

The existing scalar difficulty epsilon_gamma/lambda_min addresses generated-state error and worst-case conditioning. The sequential resolution budget instead concerns whether the **target contrast itself is statistically resolvable** before representation error is added.

## Caveats

- The AR(1) effective-sample-size expression is exact only in stylized Gaussian/mean-like settings and asymptotic for large N.
- General mixing processes require long-run variance/spectral-density arguments, not a single rho.
- Markov state persistence does not automatically reduce effect information if residual score innovations remain independent; eta applies to dependence in the relevant score/influence sequence.
- Local-to-unity asymptotics are classical. The candidate contribution is the combined latent-granularity/causal-resolution interpretation, not the dependence calculation itself.

## Planned simulation

Generate score innovations with rho_N=1-cN^-eta, K_N states, treatment scale N^-alpha, and contrast N^-beta. Verify that test noncentrality/power organizes by
    1-eta-kappa-2alpha-2beta
rather than by N alone.
