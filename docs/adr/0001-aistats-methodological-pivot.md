# ADR 0001: Methodological Pivot from Multi-City Applied Benchmarking to Regime-Conditional Causal Inference for AISTATS 2027

## Status
Accepted

## Context
The repository previously presented an extensive 7-city empirical air quality study combining deep neural networks (CNN-LSTM, TFT), tree ensembles (RandomForest, LightGBM), and causal inference (EconML LinearDML, CausalForestDML). 

A critical audit revealed foundational vulnerabilities that preclude publication at top-tier machine learning/statistics venues (e.g., AISTATS):
1. **Decorated Arithmetic in Policy Simulation**: In `src/policy_simulation_exhaustive.py`, downstream health and economic metrics relied on an unsubstantiated scaling factor (`deaths_saved = abs(pm25_reduction) * (pop / 1e6) * 15`) and arbitrary VSL valuations ($0.5M multiplier for $1.2B headline claims). Furthermore, taking `abs(pm25_reduction)` inverted negative treatment effects (pollution increases) into positive lives saved (e.g., in Mumbai and Bengaluru).
2. **Internal Inconsistencies & AI Artifacts**: The 30-page draft manuscript contained duplicate headings (e.g., sections 1.3 and 1.4 with leftover stage directions like `(Existing text continues...)`), and Table 1 mixed normalized MSE (e.g., Bengaluru 0.0005) with raw RMSE (4.2 µg/m³) without unit reconciliation.
3. **Absence of Methodological Contribution**: The paper executed off-the-shelf models across 7 cities without proposing a new estimator, theoretical identification guarantee, or formal ablation. Furthermore, simple tree baselines (RandomForest R²=0.97) outperformed the proposed deep hybrid architecture without explanation.

AISTATS enforces an 8-page main text limit, desk-rejects padded/unfocused prose, requires rigorous methodological novelty, and demands reproducible, mathematically sound artifacts.

## Decision

We pivot the core objective, codebase, and manuscript structure as follows:

1. **Single Methodological Contribution**: 
   - Reframe the paper around **Regime-Conditional Double Machine Learning (RC-DML)** for continuous environmental treatments under non-stationary atmospheric confounding.
   - Formalize the identification problem: Standard DML fails on atmospheric observational time series due to violation of i.i.d. exchangeability, unobserved boundary layer height / synoptic stagnation regimes acting as time-varying confounders, and co-emission confounding.
   - Propose an estimator that conditions Neyman-orthogonal scores on latent Markov-switching atmospheric regimes with block temporal cross-fitting.

2. **Purge Decorated Arithmetic & Unsubstantiated Health Claims**:
   - Completely remove the heuristic `* 15` multiplier, `abs()` inversion, and the unsubstantiated "$1.2B" macro-economic claims from the causal inference pipeline and primary manuscript.
   - Confine causal claims strictly to statistical estimands: Average Treatment Effects (ATE), Conditional Average Treatment Effects (CATE), and policy counterfactual trajectories $\mathbb{E}[Y(t) \mid \text{do}(T = t_0), S_t = s]$.
   - If epidemiological impact is referenced in discussion, it must be formulated strictly via peer-reviewed log-linear Concentration-Response Functions (WHO 2021) with explicit uncertainty bounds.

3. **Reconcile Predictive Baselines (Tree vs. Deep Sequential)**:
   - Explicitly acknowledge and analyze the performance of tree-based baselines (RandomForest, LightGBM) relative to deep recurrent architectures (CNN-LSTM, TFT).
   - Leverage tree models in their proper statistical role: optimal non-parametric nuisance estimators ($\hat{q}(X)$, $\hat{m}(X)$) within the orthogonal DML framework.

4. **8-Page AISTATS Manuscript Architecture**:
   - Abandon the 30-page padded manuscript and build a clean 8-page LaTeX document adhering strictly to AISTATS formatting and scope.

## Consequences

- **Positive**: Eliminates immediate desk-rejection triggers (AI stage directions, metric contradictions, fabricated constants).
- **Positive**: Shifts narrative from weak empirical benchmarking to a genuine statistical ML contribution (causal inference under non-stationarity).
- **Effort Required**: Refactoring `src/policy_simulation_exhaustive.py` to output valid statistical estimands, correcting README claims, and structuring the 8-page AISTATS manuscript.
