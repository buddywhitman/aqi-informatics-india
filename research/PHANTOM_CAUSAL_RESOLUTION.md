# Phantom Causal Resolution from Imperfect State Recovery

**Status:** new exploratory finding. Not part of submission.

## Finding

Generated/inferred latent states can make the downstream score geometry look **better conditioned than the oracle true-state geometry**.

This is the opposite of the naive expectation that representation uncertainty merely adds noise or reduces information.

## Three-state experiment

True states are balanced. Residual treatment SDs are

    sigma = (1.0, 0.3, 0.08),

so one true state has very weak causal information. State emissions are Gaussian with means (-Delta_Z,0,+Delta_Z). A K=3 Gaussian mixture is fit to emissions and produces either:
- soft posterior weights gamma;
- hard MAP assignments.

Define the true-state Gram matrix

    J_true = E[ H H' T_tilde^2 ]

and the generated-state matrix

    J_proxy = E[ gamma gamma' T_tilde^2 ]

(or hard assignments analogously).

At N=1200 over 100 replications:

Delta_Z=0.5:
    true lambda_min ~= 0.0022
    soft proxy lambda_min ~= 0.0246   (11.4x inflation)
    hard proxy lambda_min ~= 0.0457   (21.2x inflation)

    true condition number ~= 153
    soft apparent condition ~= 5.9
    hard apparent condition ~= 4.0

Even at Delta_Z=3:
    soft lambda_min inflation ~= 1.54x
    hard inflation ~= 1.90x.

Thus state ambiguity can **manufacture apparent overlap / causal resolution** by mixing observations from high-information states into inferred weak states.

## Simple two-state hard-misclassification calculation

Suppose true states are equally likely with residual treatment variances a>b>0. Let a symmetric hard classifier flip labels with probability p.

The inferred-state treatment-information diagonals are proportional to

    ((1-p)a + p b)/2
and
    (p a + (1-p)b)/2.

For 0<p<1/2, the smaller inferred information is

    [p a + (1-p)b]/2
      > b/2,

so the apparent smallest information eigenvalue is strictly **larger** than the oracle weak-state eigenvalue whenever a>b.

At p=1/2, both inferred states have the same averaged information and the apparent condition number is 1, even though the true causal information can be arbitrarily ill-conditioned.

Hence the phenomenon is structural, not a finite-sample GMM artifact.

## Why this matters

A diagnostic based on lambda_min(J_proxy) can be anti-conservative:
- true weak causal directions can be hidden by state mixing;
- apparent condition number can improve as state recovery worsens;
- hard assignment can be worse than soft weighting;
- a generated-state "effective causal rank" can exceed the oracle true-state rank at a chosen effect scale.

This creates a **phantom causal resolution** problem.

## Relation to current graceful-degradation assumptions

Any theorem connecting proxy-weighted J_gamma to oracle J requires more than small average posterior error. In particular, a one-sided lower bound on lambda_min(J_gamma) is not evidence that oracle information is healthy: mixing can lift weak eigenvalues.

Useful theory should distinguish:
1. perturbation error ||J_gamma-J||;
2. direction/eigenvector rotation;
3. whether proxy mixing inflates or deflates weak information;
4. a conservative lower confidence bound for oracle causal information.

Weyl gives
    lambda_min(J_true)
      >= lambda_min(J_proxy) - ||J_proxy-J_true||_op,
but J_true is unobserved, so operational use requires an estimable bound on the operator perturbation from posterior uncertainty/misclassification.

## New methodological question

Can posterior uncertainty be used to **de-bias the information geometry** rather than only the causal score?

For hard assignments with known confusion matrix C, observed state-specific treatment-information moments are mixtures of true moments. In principle one can invert C (with regularization) to recover oracle information, but this itself becomes ill-conditioned when state recovery is poor.

This suggests a second inverse problem:
    representation confusion -> information-geometry deconvolution -> causal inversion.

The conditioning of both stages may multiply.

## Reproduction

See:
- research/generated_state_resolution_inflation.py
- research/results/generated_state_resolution_inflation_summary.csv

## Caveats

- The committed experiment uses iid latent states/Gaussian mixture emissions to isolate the phenomenon; temporal HMM dynamics are not required for it.
- True residual treatment variances are known only in simulation.
- Label permutations do not affect eigenvalues, so alignment is irrelevant for the spectral result.
- This does not prove the current empirical OR-DML lambda_min is wrong; it shows that proxy-weighted conditioning need not be conservative and should be stress-tested.


## Asymptotic false reassurance: the phantom geometry becomes more stable with more data

A fixed-separation experiment (Delta_Z=.5, weak-state treatment SD=.08) was repeated over N=300..4800. The oracle lambda_min remains about .0021, while the soft-proxy lambda_min converges near .0242-.0263, an approximately 11x optimistic pseudo-geometry.

Crucially, the proxy estimator's sampling SD shrinks:
- N=300: SD=.01299, bias=1.74 proxy SDs;
- N=600: SD=.01089, bias=2.21 SDs;
- N=1200: SD=.00713, bias=3.09 SDs;
- N=2400: SD=.00511, bias=4.37 SDs;
- N=4800: SD=.00360, bias=6.11 SDs.

Thus more data do not repair the proxy geometry under persistent representation ambiguity. They make the optimistic pseudo-geometry **more precisely estimated**. A naive bootstrap/stability analysis centered on the proxy estimand can therefore report increasing confidence in a systematically inflated causal-resolution diagnostic.

This is analogous to classical misspecification: resampling quantifies sampling uncertainty around the wrong pseudo-parameter, not discrepancy from the unobserved oracle geometry.

Reproduction:
- research/phantom_resolution_concentration.py
- research/results/phantom_resolution_concentration_summary.csv
