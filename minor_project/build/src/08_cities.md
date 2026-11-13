# Observational Stress Test: Four Indian Cities

## 8.1 Purpose and framing

The stress test applies OR-DML and its sensitivity analysis to NO₂ $\to$ PM₂.₅ in Delhi, Mumbai, Bengaluru and Kolkata (OpenAQ/CPCB stations and Open-Meteo weather, February 2025–June 2026) on the corrected hourly grid. There is **no known truth** here. All numbers are *exposure associations*, and the purpose is to show what a careful analysis can and cannot conclude from real sensors. The framing is a stress test of the reporting rules developed in Chapter 5.

## 8.2 Results

@tex cities | tab_cities.tex | Four-city stress test on the corrected hourly grid (OR-DML, HAC lag 24 h). $\lambda_{\min}$ is raw, $\lambda^{\rm std}_{\min}$ is scale-free. Lead placebo: smallest $p$-value over leads $h \in \{-6, -3, -1\}$ h. $b^\star$: robustness value in SD of $\tilde Y$. Screened: runs of $\ge 6$ identical NO₂ readings removed. | sm

@fig bstar | bl_fig2_cities_hi.png | **Robustness values $b^\star$ by city** (log scale; dashed line = 1 SD). Delhi's association would need an outcome regime shift of 82 SD (90% interval 54–144) to be explained away through the HMM-regime channel; the other three cities are weak.

Table {{tab:cities}} separates four axes that the reliability rule conflates.

**Scale.** Raw $\lambda_{\min}$ ranges from 0.0012 to 63.2, but the standardized value only from 0.20 to 0.46. Raw conditioning reflects units and sensor resolution, not collinearity.

**Data quality.** Kolkata's NO₂ takes only 16 distinct values, and Mumbai's longest constant run is 277 hours; screening all qualifying runs moves the Mumbai ATE from 3.02 to 0.38 ($N$ 8,113 $\to$ 6,256). The screen removes 1,857 complete cases (22.9%) across all flagged runs, not one episode. This is a sensitivity exclusion; it does not establish hardware failure or validate the remaining calibration.

**Temporal validity.** Bengaluru and screened Kolkata fail lead placebos, the latter strongly ($p = 0.0003$), so the apparently significant screened Kolkata effect (3.32) is *not interpretable*.

**Residual confounding.** Delhi's association is positive in both regimes (0.430 and 0.102; ATE 0.234, SE 0.029) and would need an outcome shift of 82 SD to be explained away through the HMM-regime channel. Its one-hour lead placebo nonetheless rejects (coefficient 0.072, $p = 0.012$, about a third of the contemporaneous coefficient), pointing to short-horizon confounding outside that channel.

## 8.3 Placebo falsification, aligned to the information set

@tex plc | tab_placebo.tex | Lead and lag placebo coefficients (asterisk: $p<0.05$). Leads $h<0$ should be indistinguishable from zero if the temporal ordering is causal. | sm

Placebo leads are aligned to the outcome's own information set. Without this alignment the one-hour lead is mechanically zero, because the lagged outcome is a control; earlier drafts had that flaw (Chapter 10). In Delhi the lead coefficients at $-6$, $-3$ h are near zero but $-1$ h rejects; in Bengaluru the $-6$ and $-3$ h leads reject.

## 8.4 Sensitivity of inference to the HAC lag

@tex hac | tab_hac.tex | Regime effects (standard error; $p$-value) under HAC bandwidths of 12, 24, 72 and 168 hours. | sm

Delhi's regime-1 effect (0.430) stays significant at every bandwidth up to one week (standard error 0.050 $\to$ 0.065), as does regime 2 (0.102, $p$ from 0.002 to 0.011). Kolkata's regime-2 effect is significant at every lag but rests on 2,322 observations and 16 distinct NO₂ values, so it should not be over-read. Mumbai is never significant.

## 8.5 Reading the results

The honest report is short.

* **Delhi** is an exposure association, positive in both regimes, robust to residual *regime* confounding. It is not an emissions effect: the lead placebo rejects.
* **Mumbai, Bengaluru, Kolkata** are inconclusive: data quality, failed placebos or wide intervals prevent interpretation.

NO₂ is also chemically related to the formation of secondary PM₂.₅, proxies overlap controls, and lagged pollutants may be mediators. Chemistry-aware or instrument-based designs would be needed for emission-policy claims. A further point recorded in the manuscript is that five EM restarts span 1,182 log-likelihood units in Mumbai; neither optimizer multimodality nor misspecification establishes the direction of error in $\hat v$ or $b^\star$.

## 8.6 Why this section matters for the project

An applied study that had reported these four numbers as "the causal effect of traffic-related NO₂ on PM₂.₅" would have been wrong in three of four cities and incomplete in the fourth. The project's contribution is partly the refusal to do so, and the tooling (aligned placebos, screening, robustness values, standardized conditioning) that makes the refusal principled rather than ad hoc.
