# Results: Synthetic and Semi-Synthetic Evidence

## 7.1 The closed form reproduces the factorial experiment

Proposition 1 predicts the coupled-score error without fitting any parameter. Across the 42 cells of the factorial design (4,900 runs) the ratio of observed to predicted median error has median 0.999 (interquartile range 0.993–1.008, full range 0.98–1.02). The error peaks sit where the law places them (Figure {{fig:phantom}}a), and the generated-state $\lambda_{\min}$ inflates up to 70-fold (Figure {{fig:phantom}}b). The failure score $\varepsilon_\gamma/\lambda_{\min}(\hat J)$ achieves Spearman 0.995 with causal error, but, as Section 5.3 showed, partly *because* $\lambda_{\min}$ inflates.

## 7.2 Learned posteriors break the scalar ratio

When posteriors are learned by EM from noisy proxies (720 runs, 36 cells), the picture changes.

@tex scores | tab_scores.tex | Spearman correlation between failure scores and causal error. "Synthetic proxy" is the factorial design; "learned" uses EM posteriors; "within $\Delta_T$" averages correlations at a fixed treatment shift.

* The ratio $\varepsilon_\gamma/\lambda_{\min}(\hat J)$ falls from **0.995 to 0.517**.
* Restoring the *oracle* geometry recovers 0.750, so the damage is in the generated-state Gram matrix.
* Restoring instead the numerator scale that the laws predict, using the plug-in regime shifts $\hat\Delta_T, \hat\Delta_Y$, gives 0.695 (0.863 across cells), and 0.664 with entropy in place of $\varepsilon_\gamma$, which is *fully observable*.
* With the same posteriors, hard fixed effects beat the posterior-weighted design in 100% of cells, while posterior-adjusted OR-DML beats hard fixed effects in 72% of cells (median error ratio 0.97; 0.69 against the weighted design). All its losses occur at the smallest treatment innovation, where the $a^2 v$ term dominates $J$.

> The actionable replacement for the scalar rule is therefore a *product*: representation uncertainty times the regime shifts it multiplies, divided by conditioning (Chapter 11 develops this "dual-factor" view).

## 7.3 Observability and persistence frontier

Table {{tab:main}} (left) varies proxy separation $\Delta_Z$ in the data-generating process of Theorem 1; a persistence sweep adds regime-0 stay probabilities 0.70 and 0.97 (900 datasets in all). Standard and blocked DML are stuck at the omitted-regime bias (8.44 and 8.47 against a true ATE of 1.53). All proxy-based estimators fail at $\Delta_Z \le 1$ and approach the oracle at $\Delta_Z \ge 2$.

@tex main | tab_main_comparison.tex | Posterior-adjusted OR-DML is the most accurate regime-aware estimator, but no estimator removes residual-state bias. Left: synthetic frontier with learned HMM posteriors (100 datasets per $\Delta_Z$, $N=1{,}200$, persistence 0.88, true ATE 1.53). Right: real-data-calibrated semi-synthetic benchmark on the corrected hourly grid (30 datasets per city). Bold: best non-oracle within each block. | sm

@fig est | fig3_estimators_hi.png | **Conditioning nuisances on the posterior, rather than weighting them by it, fixes the coupled estimator.** (a) Mean absolute ATE bias and (b) median regime-effect error along the observability frontier with learned posteriors. (c) Regime-effect error in the real-data-calibrated benchmark: the same ordering in every city. | w

Among the 7 cells with $\Delta_Z \ge 1$, OR-DML is best or tied for the ATE in 7 of 7 and for regime effects in 7 of 7. At $\Delta_Z = 2$ its ATE bias is 0.129 (hard fixed effects 0.141, weighted design 0.147) with coverage 91%; at $\Delta_Z = 1$ its regime-effect error is 2.954 against 3.652 for hard fixed effects and 6.107 for the weighted design. Persistence helps exactly through the posterior: at $\Delta_Z = 1$, raising the stay probability from 0.70 to 0.97 lowers $\varepsilon_\gamma$ from 0.090 to 0.026 and OR-DML's bias from 2.725 to 0.960. Under a continuous latent mixture that no two-state model describes, OR-DML beats hard fixed effects in 100% of 100 replications, while the weighted design beats them in 0%.

## 7.4 Real-data-calibrated semi-synthetic benchmark

The right half of Table {{tab:main}} keeps everything real but the causal effects. With posteriors that are exactly calibrated and 95–96% accurate, every proxy-based estimator retains large bias and zero interval coverage. For regime effects OR-DML is the most accurate regime-aware estimator in **every** city (Delhi 0.59, against 0.77 for hard fixed effects and 1.21 for the weighted design). For the pooled ATE it is best in Delhi and Kolkata, while posterior-feature DML is best in Mumbai and Bengaluru: scalar pooling avoids estimating weak regime contrasts. OR-DML's advantage here concerns *heterogeneous regime effects*, not universal scalar-ATE dominance. The weighted design is worse than standard DML outside Delhi.

The plug-in law, computed from the posterior, the fitted treatment shift and the injected outcome shift, ranks the four cities' biases correctly with magnitudes within 7–37%. The generated-state $\lambda_{\min}$ exceeds its oracle value by 1.7–2.4 times in Mumbai, Bengaluru and Kolkata (1.04 times in Delhi): phantom resolution is present in real-data-calibrated series too.

## 7.5 Representation quality is not task reliability

Across 15 shifted worlds and four encoders (HMM, GRU, Transformer, linear state-space model), adding posterior negative log-likelihood to state-recovery $F_1$ raises leave-one-world-out prediction of downstream reliability from $R^2 = 0.113$ to 0.221. Temperature scaling reduces calibration error of the GRU and Transformer by 47.9% and 60.3%, yet does not improve failure discrimination ($\Delta$AUC $-0.0035$ and $-0.0075$). By Theorem 4, recalibration changes the *estimate* $\hat v$ but not the residual information that the downstream task actually has. A better-calibrated encoder is not a more reliable causal input.

## 7.6 Spectral regularization

Regularization trades bias for variance. On the committed path (Table {{tab:reg}}) increasing $\lambda$ from 0 to 0.1 cuts variance fivefold, from 0.0687 to 0.0143, and mean-squared error from 0.785 to 0.014; beyond $\lambda = 0.1$ bias returns. The best penalty is data dependent, which is why the implementation exposes an adaptive optimizer with an oracle risk bound.

@tex reg | tab_regularization.tex | Spectral regularization path ($\lambda$ against bias, variance and mean-squared error).

## 7.7 Scalability in the number of regimes

@tex scal | tab_scalability.tex | Scalability check for $K = 2$ to 5 regimes: ATE bias, regime-effect error, smallest eigenvalue and condition number.

Regime-effect error grows with $K$ (0.112 at $K=2$ to 0.389 at $K=5$) as $\lambda_{\min}$ falls (0.758 to 0.238), consistent with Theorem 3; ATE bias stays below 0.08.

## 7.8 What the synthetic evidence does not show

Below $\Delta_Z \approx 1$ every proxy-based estimator, OR-DML included, has large bias and no coverage. Posterior-adjusted OR-DML does not remove residual-state bias; it is the most accurate *regime-aware* estimator in the benchmarks reported, with ties at high separation. These limits are restated in Chapter 11.
