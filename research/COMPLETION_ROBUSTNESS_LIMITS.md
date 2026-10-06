# Robustness Limits of Posterior Moment Completion

Posterior second-moment completion is an exact conditional-expectation identity **only under a calibrated joint posterior for the information set entering the moment**.

The treatment-information experiment makes this dependence explicit. True treatment SDs are (1,.3,.08). If the diagnostic posterior q(S|Z,T) uses the correct treatment model, completed/oracle weak information is essentially 1. If the treatment model is misspecified:

At N=2400:
- common treatment SD for all states: inflation 46.1x / 26.2x / 5.39x at emission separation .5/1/2;
- weak-state SD assumed 2x too large: 2.37x / 2.05x / 1.49x;
- weak-state SD assumed 4x too large: 7.84x / 5.96x / 2.91x.

Thus moment completion can recreate phantom resolution under model misspecification.

## Consequence

The correction trades one problem for another:
- posterior-mean plug-in has structural moment bias;
- exact completion requires a correct/calibrated joint state-treatment model.

A usable diagnostic needs model checks targeted to the exact moment being completed.

Potential falsification:
1. posterior predictive checks of state-conditional treatment residual variance;
2. cross-fitted held-out treatment log likelihood/calibration;
3. sensitivity envelope over plausible state-treatment variance models;
4. compare Z-only, ZT-completed, and hard-label geometries; large disagreement is itself a warning;
5. avoid outcome Y in model checking unless a valid joint causal model is specified.

## Robust target

Rather than a point estimate of oracle lambda_min, report a sensitivity interval
    [ inf_{M in model set} lambda_min(J_complete(M)),
      sup_{M in model set} lambda_min(J_complete(M)) ].
If the lower end is near zero, causal resolution should be treated as weak regardless of the nominal completed estimate.

This is closer to robust partial identification than plug-in diagnostics.

Reproduction:
- research/completion_misspecification.py
- research/results/completion_misspecification_raw.csv (when full run is executed)
- research/results/completion_misspecification_selected.csv
