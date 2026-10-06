# Leverage-Calibration Impossibility

**Status:** exact counterexample + automated Monte Carlo. Not submission material.

## Statement

No scalar calibration criterion that depends only on the unweighted distribution/magnitude of posterior errors can uniformly control downstream moment error over tasks whose leverage functions are unrestricted.

Let posterior error magnitude be e(x)^2 and let downstream task leverage be w(x)>=0. Global squared calibration error is

    C = E[e^2].

Task-weighted error is

    C_w = E[w e^2] / E[w].

Fix any c>0 and any leverage ratio R. Construct two error fields with the same E[e^2]=c:
- representation L places its errors on a set where w=l;
- representation H places the same error mass on a set where w=L=R l.

Ignoring normalization effects outside that set, their task-weighted errors differ by order R while global calibration is identical. Letting R grow gives an arbitrarily large downstream gap at fixed global calibration.

Therefore global ECE/Brier/NLL improvement cannot, without assumptions linking error location to task leverage, guarantee improvement in a leverage-weighted downstream functional.

## Monte Carlo

The committed experiment puts the same 20% posterior perturbation on either the lowest- or highest-treatment-leverage quintile. Global squared posterior error is exactly 0.016 in both cases.

At N=20,000:
- high-leverage task-weighted error = 0.06903;
- low-leverage task-weighted error = 2.286e-5;
- ratio ~= 3019x.

Causal-geometry log error:
- high leverage = 0.4605;
- low leverage = 0.00931;
- ratio ~= 49.5x.

The ratios are stable from N=1,000 through 20,000, confirming that this is structural rather than sampling noise.

## Interpretation

Calibration has a measure. Ordinary calibration is under the observational measure P. A task with leverage w induces

    dP_w = w dP / E[w].

A representation may be well calibrated under P and badly calibrated under P_w. Different tasks induce different measures, so universal task-agnostic calibration is impossible without restrictions on the family of leverage functions.

## Relation to prior findings

This explains why:
- temperature scaling can improve ECE/NLL without improving causal reliability;
- posterior errors concentrated around high-score-leverage observations are disproportionately damaging;
- task dependence enters not only through latent-state abstraction but through the measure under which representation quality should be assessed.

## What this does NOT show

A simple weighted Brier score is not a universal solution. In the broad temperature sweep it only improves correlation with geometry error from 0.470 to 0.505. The theorem/counterexample establishes necessity of task awareness, not sufficiency of a particular weighting scheme.

## Reproduction

- research/adversarial_calibration_concentration.py
- research/results/adversarial_calibration_concentration_summary.csv
- research/task_weighted_calibration.py
- research/results/task_weighted_calibration_summary.csv
