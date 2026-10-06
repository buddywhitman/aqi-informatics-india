# ADR 0003: Overlap-Aware Regime Double Machine Learning (OR-DML) Pivot

## Status
Accepted

## Context
A rigorous mathematical and empirical audit of the AISTATS submission (documented in `docs/critique.md` and `docs/pivot.md`) revealed critical foundational defects in the initial RC-DML framework:

1. **Identification Failure**: The SCM defines $S_t \to (X_t, T_t, Y_t, Z_t)$, meaning the true conditional expectation satisfies $\mathbb{E}[Y_t \mid X_t, T_t, Z_{1:N}] = \sum_k \theta_k^* q_{tk} T_t + \sum_k g(X_t, k) q_{tk}$ where $q_{tk} = P(S_t = k \mid X_t, T_t, Z_{1:N})$. The previous paper mistakenly substituted the marginal auxiliary posterior $\gamma_{tk} = P(S_t = k \mid Z_{1:N})$, implicitly asserting an unproven sufficiency condition.
2. **Algebraic Disproof of Key Theorems**:
   - **Theorem 5 (Universal Gateaux Orthogonality)**: The claim that $\sum_k \theta_k^* \nabla_\Lambda \gamma_{tk}^* = 0$ via simplex conservation is algebraically false whenever treatment effects are heterogeneous ($\theta_j^* \ne \theta_k^*$).
   - **Theorem 6 (Diagonal Cross-Talk Bias)**: The claimed bias formula $\sum_{j \ne k} \frac{J_{kj}}{J_{kk}}(\theta_j^* - \theta_k^*)$ fails for symmetric positive-overlap matrices (e.g. $J = \begin{pmatrix} 1 & c \\ c & 1 \end{pmatrix}, \theta^* = (1,1)^\top$ produces bias $c$, while the formula yields 0).
   - **Theorem 7 (Oracle Equivalence Rate)**: The $o_P(N^{-1/4})$ average $L_1$ posterior rate leads to a diverging $o_P(N^{3/8})$ score difference, not $o_P(1)$.
   - **Efficiency Claim**: Complete-data oracle scores were substituted for the observed-data latent model.
3. **Data Pipeline Anomalies & Window Truncation**:
   - `data/processed_hourly/combined_hourly_with_regimes.csv` contained corrupted values (negative PM2.5/PM10 down to -242,451, temperature down to -4237°C from unconstrained polynomial imputation).
   - The empirical script arbitrarily truncated the series to the first 2,500 hours (104 monsoon days in June–September 2024), while asserting claims about winter stagnation.
4. **Simulation Setup**: Only 45 replications were run, baseline coverages were assigned to 0.0 without calculation, and Gaussian emissions had near-zero posterior ambiguity.

## Decision

We pivot the core scientific narrative, mathematical foundation, and empirical pipeline to **Overlap-Aware Regime Double Machine Learning (OR-DML)**:

### 1. New Scientific Objective
Shift from claiming "universal elimination of regime bias" to characterizing the fundamental statistical object:
> **"How does causal identification degrade when the confounding state is latent, temporally persistent, and only imperfectly observed?"**

### 2. Corrected Theoretical Framework
1. **Theorem 1 (Non-Identification under Unconstrained Overlap)**: Prove that predictive state probabilities $\boldsymbol{\gamma}_t$ alone do not point-identify $\boldsymbol{\theta}^*$ without proxy-separation or completeness restrictions.
2. **Theorem 2 (Graceful Degradation Bound)**: Establish an exact finite-proxy-error bound:
   $$\|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^*\|_2 \le \frac{C_{\text{overlap}} \cdot \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2})$$
   where $\varepsilon_\gamma = \mathbb{E}\|\boldsymbol{\gamma}_t - \boldsymbol{H}_t\|_1$ is the proxy error and $\lambda_{\min}(\boldsymbol{J})$ is the minimum eigenvalue of the regime coupling matrix.
3. **Theorem 3 (Spectral Regularization & Bias-Variance Frontier)**: For ill-conditioned $\boldsymbol{J}$ under weak overlap, introduce the spectral regularized estimator $\hat{\boldsymbol{\theta}}_\lambda = (\hat{\boldsymbol{J}} + \lambda \boldsymbol{I})^{-1} \hat{\boldsymbol{S}}$ and prove its non-asymptotic bias-variance tradeoff.
4. **Theorem 4 (Corrected Omitted Regime Bias)**: Re-derive the omitted regime bias decomposition with correct algebra, formalizing the conditions for Frisch-Waugh amplification.
5. **Theorem 5 (Dependent Data Asymptotics)**: Establish asymptotic normality under stationary geometrically $\alpha$-mixing Markov regime dynamics using purged block cross-fitting with embargo $\tau \ge C \log N$.

### 3. Clean Data Engineering
1. Rebuild the preprocessing pipeline (`src/data_pipeline_clean.py`) directly from raw hourly CPCB and Open-Meteo observations.
2. Enforce strict physical boundary validations ($PM_{2.5}, PM_{10}, NO_2, SO_2 \ge 0$, Wind speed $\ge 0$, Temperature in valid Celsius $[0, 50]^\circ\text{C}$).
3. Use a continuous multi-season annual window across Delhi, Mumbai, Bengaluru, and Kolkata.

### 4. Rigorous Experimental Validation
1. **$\ge 500$ Monte Carlo Replications**: Replace 45-rep toys with large-scale replicated simulations reporting true empirical coverage for all estimators.
2. **Latent-Confounding Difficulty Frontier**: Vary regime separation $\Delta_Z$ from near-oracle ($\Delta_Z = 4.0$) down to severe overlap ($\Delta_Z = 0.2$), plotting Bias and RMSE as functions of $\lambda_{\min}(\boldsymbol{J})$.
3. **Dynamic Causal Effects**: Estimate regime-specific impulse response functions $h = 0, \dots, 24$ hours.
4. **Filtering vs. Smoothing**: Contrast retrospective smoothing with online forward filtering to address operational policy targeting.

### 5. Scope & Compliance Refactoring
1. Remove premature lives-saved, VSL, and macroeconomic claims.
2. Unify the AI Use Statement into a single, compliant disclosure before the references.
3. Remove legacy SIP-padding scripts and ensure one-command reproducibility.

## Consequences
- Transforms a vulnerable paper with disprovable theorems into a mathematically sound, award-caliber paper focused on the real challenge of latent confounding under weak overlap.
- Restores 100% scientific integrity to the data pipeline and empirical results.
