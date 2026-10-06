# Exploratory Research: Task-Relative Reliability

This directory is deliberately **outside the AISTATS submission evidence chain**. Nothing here modifies `paper/`, the canonical supplementary archive, or submitted claims. The purpose is adversarial follow-up research: test where the current scalar difficulty idea generalizes, where it is loose, and which extensions are worth developing.

## Studies and current findings

### R1. Finite-sample scaling of the scalar difficulty
We independently vary proxy error, residual treatment information, and sample size. In the exploratory run (10 replications/cell), Spearman correlation between causal error and `eps/lambda_min(J)` increases from 0.783 at N=300 to 0.915 at N=4800, and exceeds either constituent at every N. A log-log regression gives an exploratory slope 0.74 on log difficulty and only -0.081 on log N (R2=0.858). The simple `sqrt(N)*difficulty` scaling does **not** improve rank correlation. This is evidence for the difficulty ordering, but not evidence for a universal root-N phase boundary.

### R2. Direction matters: the scalar bound can be very loose
At the population score level, fix `||b||`, fix the spectrum of J, and rotate the perturbation b relative to the weakest eigenvector. The exact propagation is `||J^{-1} b||`, while `||b||/lambda_min(J)` is the worst-case scalar bound. With condition number 100, the scalar bound is tight when b aligns with the weak direction but can be 100x loose when b is orthogonal. This strongly motivates a sharper **directional task-relative representation error**
`R_task = ||(J+lambda I)^{-1} b_gamma||`.
This is a mathematical stress test, not an independent empirical discovery.

### R3. K>=3 exposes a major limitation of the scalar difficulty ratio
A three-state experiment holds average posterior L1 error and $\lambda_{\min}(J)$ in roughly the same range while changing **which states receive the posterior perturbation**. At perturbation level $a=0.03$, the three designs have $\varepsilon_\gamma\approx0.020$ and $\lambda_{\min}(J)\approx0.0125$--$0.0133$, hence scalar difficulty about 1.5--1.6, but mean causal L2 error ranges from 0.109 (weak-state perturbation) and 0.127 (strong-state perturbation) to **0.667** (balanced cyclic perturbation). Across all 1,500 runs, Spearman correlation of error with $\varepsilon_\gamma/\lambda_{\min}$ is only 0.424. The score-aware worst-case bound $\|b_\gamma\|/\lambda_{\min}$ improves this to 0.712, while the directional propagation $\|J^{-1}b_\gamma\|$ exactly tracks the unregularized population moment error by construction.

This is the most important new finding: the current scalar ratio is a useful worst-case/order diagnostic in the two-state experiments, but it is **not a sufficient task-relative representation metric in higher-dimensional latent spaces**. Error orientation relative to the full spectrum matters.

### R4. Regularization helps, but scalar difficulty does not identify the optimal lambda
Across 48 synthetic settings (20 replications each), oracle choice over a fixed ridge grid reduces MSE substantially: median gains are 81.8%, 87.3%, and 85.5% for N=600,1200,2400. However, Spearman correlation between oracle lambda and scalar difficulty is only 0.393. Thus the current difficulty score is useful for detecting danger but appears insufficient by itself for tuning regularization. This is an important negative result and argues for a richer risk estimator involving perturbation direction, noise level, and target magnitude.

### R5. Posterior error is transition-localized
For a correctly specified two-state Gaussian HMM using causal filtering, posterior L1 error is concentrated near true state transitions. At emission separation 2.0, mean L1 error is 0.577 within 0-1 steps of a transition versus 0.134 at distance >=9; at separation 3.0 it is 0.270 versus 0.056. This suggests that aggregate posterior metrics may hide a small set of temporally localized, high-impact representation failures.

### R6. The same transition phenomenon appears qualitatively in the four-city sensor data
Using a standardized two-state HMM on the six exogenous meteorological features, filter-vs-smoother disagreement is sharply concentrated around inferred transitions. In Delhi, mean L1 disagreement is 0.923 within 0-1 hours of an inferred transition versus 0.009 at >=25 hours. Mumbai shows 0.413 versus 0.017. Bengaluru/Kolkata switch much more frequently, so long interior segments are scarce. Because inferred transitions are defined by the same fitted model, this is a descriptive diagnostic, not ground-truth validation.

### R7. Frozen-sensor sensitivity is a material empirical limitation, especially in Mumbai
Removing complete-analysis rows belonging to NO2 constant-value runs of length >=6 hours leaves Delhi unchanged but removes 885/4,180 Mumbai rows (21.2%), 163/4,093 Bengaluru rows (4.0%), and 110/1,209 Kolkata rows (9.1%). Re-fitting the same smoothed OR-DML specification changes Mumbai's overall estimate from 1.59 to 2.92, regime estimates from (-0.03, 3.30) to (3.86, 2.25), and lambda_min from 0.317 to 0.178. Bengaluru changes little; Kolkata remains extremely ill-conditioned and uncertain. This is not evidence that the filtered estimate is "truer"—removing long constant runs may discard legitimate periods—but it shows that the Mumbai empirical decomposition is materially telemetry-sensitive and should not be treated as a stable physical effect without source-level sensor QA.

### R8. Real-data conclusions are sensitive to the chosen number of latent regimes
Re-fitting the identical observational specification with K=2,3,4 leaves Delhi's overall ATE relatively stable (0.374, 0.428, 0.422) and Bengaluru near zero (-0.026, 0.064, 0.038), but Mumbai changes materially (1.59, 2.83, 5.99). At K=4 Mumbai's lambda_min falls to 0.029 and one regime coefficient reaches 23.0, a clear warning of weak identification rather than credible fine-grained heterogeneity. Kolkata remains near-singular for every K. The fitted HMM BIC decreases monotonically from K=2 through K=5 in all four cities, so K=2 is an interpretability/coarse-graining choice rather than a BIC-selected latent-state count. This is a substantive model-selection limitation and a promising research direction: task-aware state aggregation should trade emission fit against downstream identifiability.

A useful positive control: five different HMM random seeds (1,7,19,42,97) produced identical reported fits for all four cities under the current implementation, so the observed K sensitivity is not merely random initialization noise.

## Reproduction

Run:
```bash
python research/run_extended_reliability_studies.py
python research/frozen_sensor_sensitivity.py
python research/latent_state_count_sensitivity.py
```

The script writes raw/summary CSVs under `research/results/`. Increase `PHASE_REPS`, `REG_REPS`, and `TRANSITION_REPS` for publication-grade Monte Carlo precision.

## Interpretation discipline

Promising:
- task-relative **directional** sensitivity is a stronger theoretical object than the scalar worst-case ratio;
- transition-localized uncertainty is a plausible sequential extension;
- scalar difficulty robustly ranks error across sample sizes.

Negative/non-promising as currently formulated:
- `eps/lambda_min` alone does not reliably determine the MSE-optimal regularization parameter;
- no evidence from these runs supports a simple universal `sqrt(N) eps/lambda_min` phase boundary;
- real-data transition diagnostics do not establish true latent transitions.

## Next rigorous tests

1. Derive/estimate `b_gamma` and compare `||(J+lambda I)^-1 b_gamma||` against the scalar bound on held-out synthetic worlds.
2. Hold `||b_gamma||` fixed while rotating its direction in K>=3 latent-state DGPs, not only score-level algebra.
3. Test whether transition-weighted proxy error predicts downstream causal error better than global ECE/NLL/entropy.
4. Learn lambda from training worlds using directional risk features and evaluate on held-out worlds; do not tune on target causal error.
5. Stress wrong-K, semi-Markov durations, non-Gaussian emissions, and time-varying transition matrices.
