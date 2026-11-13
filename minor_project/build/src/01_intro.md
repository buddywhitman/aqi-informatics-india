# Introduction

## 1.1 Background and motivation

Air pollution is the largest environmental health risk in India, and the question that policy makers ask of it is causal rather than predictive: *if emissions of a precursor fall, by how much will fine particulate matter (PM₂.₅) fall?* A forecasting model, however accurate, does not answer that question. A model that predicts tomorrow's PM₂.₅ from today's NO₂ will happily report a strong positive association even when most of the association is produced by a third factor, the state of the atmosphere, that raises both pollutants together.

The atmosphere is the textbook example of a persistent, unmeasured state. In a continental basin such as Delhi, winter nights alternate between *stagnation* (a shallow inversion traps everything near the ground) and *ventilation* (wind clears the basin). In a coastal city such as Mumbai, sea and land breezes alternate. Each regime lasts hours to days, so the state is autocorrelated, and each regime shifts both the emission-driven part of NO₂ and the background accumulation of PM₂.₅. Adjusting for observed temperature or wind does not remove this confounding if the regime is only imperfectly captured by those variables.

The same structure appears far outside air quality: the liquidity regime of a market that moves both an execution policy and its price impact, the physiological phase of a patient that moves both a treatment and an outcome, the operating mode of a plant. The statistical problem is general. We call it **causal estimation under latent Markov confounding**.

## 1.2 Problem statement

We observe a time series $(Y_t, T_t, X_t, Z_t)_{t=1}^N$ in which a latent first-order Markov regime $S_t$ affects both treatment $T_t$ and outcome $Y_t$, the regime is observed only through proxies $Z_t$, and the target is the regime-specific effect of $T$ on $Y$ and its stationary average. The prevailing approach is a two-stage recipe: fit a hidden Markov model to $Z$ to obtain posterior regime probabilities, then feed them into a regime-aware Double Machine Learning (DML) estimator. Practitioners judge the result with a reliability rule that has two parts: *trust the estimate when the posterior is confident and the score Jacobian is well conditioned*.

The project asks three questions.

1. **Is the reliability rule sound?** When the state is *generated* (estimated) rather than observed, does a confident posterior together with a well-conditioned Jacobian certify the causal estimate?
2. **If not, why, and what should the estimator look like?** Can the failure be characterized exactly, and does the characterization dictate a better design?
3. **What can be said about real sensor data?** On Indian megacity sensor networks, what can an honest analysis conclude, and what must it refuse to conclude?

## 1.3 Objectives

The objectives agreed at the start of the minor project, and their outcome, are summarized below.

@tab obj | Project objectives and where each is addressed
| # | Objective | Outcome | Chapter |
|---|---|---|---|
| O1 | Assemble and audit a multi-city hourly pollution and weather dataset | Four cities, 12 pollutant/weather series, time-grid defect found and corrected | 6, 8 |
| O2 | Characterize how latent-state error propagates into causal error | Two exact laws; phantom causal resolution discovered | 3, 5 |
| O3 | Design and implement a reliable regime-aware estimator | OR-DML with posterior-adjusted nuisances, purged cross-fitting, spectral regularization, HAC inference | 4 |
| O4 | Validate against synthetic, semi-synthetic and real data | 4,900-run factorial, 900-dataset frontier, 4-city calibrated benchmark, audited stress test | 7, 8 |
| O5 | Produce a peer-reviewed research submission | AISTATS 2027 manuscript with 8-page main body and appendices A–Q | 10 |
| O6 | Make every claim reproducible and machine-checked | Single-script number pipeline, verifier, CI | 9 |

## 1.4 Summary of contributions

* **A failure and its limits.** An exact decomposition of the omitted-regime bias of standard DML, including bias amplification (adding observed controls can make it worse), and a non-identification result when proxies carry no state information.
* **A mechanism.** Exact laws for how state confusion enters coupled and posterior-conditioned estimators. They explain *phantom causal resolution* and the non-monotone error, and why a posterior helps as an adjustment feature but hurts as a nuisance weight.
* **An estimator.** OR-DML: posterior-adjusted nuisances, purged block cross-fitting, spectral regularization, HAC inference. It comes with exact orthogonality, a degradation bound that separates nuisance error from state-recovery error, and asymptotic normality that includes occupation variance.
* **Evidence.** A factorial design verified against the closed form, learned-posterior experiments that remove its circularity, a real-data-calibrated semi-synthetic benchmark on four Indian cities, and an audited stress test with aligned placebos and robustness values.
* **A research artifact.** A reproduction pipeline in which every number in the manuscript is generated from committed files, and a documented audit trail of every claim that changed along the way (Chapter 10).

## 1.5 Scope and honesty statement

The report separates three kinds of statements: *theorems* (exact, under stated assumptions), *simulation evidence* (known truth, so errors are measurable), and *observational findings* (no known truth, so they are associations, not effects). The Delhi, Mumbai, Bengaluru and Kolkata analyses are in the third group. We do not claim that NO₂ causes PM₂.₅ with a stated magnitude, and we do not claim that OR-DML dominates every competitor on every metric. Where posterior-feature DML beats OR-DML (the pooled ATE in two cities) or where every estimator fails (weak proxy separation) the report says so, as does the manuscript.

## 1.6 Organization of the report

@svg org | pipeline | Method pipeline followed throughout the report: sensor data are summarized by an HMM, the posterior enters the nuisances as a feature, cross-fitting is purged for temporal dependence, and a regularized coupled score gives regime effects together with a reliability report.

Chapter 2 surveys the literature. Chapter 3 formalizes the model and proves the bias and non-identification results. Chapter 4 presents OR-DML and its implementation. Chapter 5 develops the theory: orthogonality, graceful degradation, the exact laws, and inference. Chapter 6 describes data and experimental design. Chapter 7 reports synthetic and semi-synthetic results. Chapter 8 reports the observational stress test. Chapter 9 documents engineering, reproducibility and verification. Chapter 10 tells the story of the project, including the audit that redirected it. Chapter 11 concludes. The annexures hold team roles, the AI-use statement, code listings, reproduction commands, additional tables, and a map of the manuscript's appendices.
