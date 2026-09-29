# ADR 0005: Oracle Score Difference Architecture and Theoretical Rigor Audit

## Status
Accepted

## Context
Following a rigorous mathematical audit of the manuscript at commit `cc3cc6b`, several theoretical gaps and conceptual alignments were resolved:
1. **P0 Theoretical Blocker in Theorem 2 Proof (Appendix B)**: The previous outcome decomposition in Appendix B ($\tilde{Y}_{tk}^\gamma = \sum_j H_{tj}\theta_j^*\tilde{T}_{tj}^\gamma + \dots$) omitted cross-regime treatment terms of the form $\sum_j H_{tj}(\theta_k^* - \theta_j^*)m_k^\gamma(X_t)$, which cannot be bounded state-by-state by $|H_{tj} - \gamma_{tj}|$.
2. **Oracle Score Difference Reformulation**: To achieve an exact algebraic identity without residual terms, the proof must be structured around the difference between the proxy score/Gram system $(\boldsymbol{S}^\gamma, \boldsymbol{J}^\gamma)$ and the oracle score/Gram system $(\boldsymbol{S}^*, \boldsymbol{J}^*)$, utilizing the exact identity $\boldsymbol{S}^* = \boldsymbol{J}^*\boldsymbol{\theta}^*$.
3. **Step A vs. Step B Conceptual Architecture**: DML Neyman orthogonality at implemented targets $(m_k^\gamma, \mu_k^\gamma)$ protects against first-order errors in estimating continuous non-parametric nuisances ($\mathcal{O}_P(N^{-1/2})$), but does *not* protect against latent proxy misspecification ($\boldsymbol{H}_t \neq \boldsymbol{\gamma}_t$). The irreducible asymptotic bias $\|\boldsymbol{b}_\gamma\|_2 \le C\varepsilon_\gamma$ degrades inversely with task conditioning $\lambda_{\min}(\boldsymbol{J})$.
4. **Assumption 3 (Latent Proxy Sufficiency)**: Renamed from "Latent Proxy State-Mean Sufficiency" to "Latent Proxy Sufficiency", stating the screening-off property $P(S_t = k \mid X_t, T_t, Z_{1:N}) = P(S_t = k \mid Z_{1:N}) \equiv \gamma_{tk}$, clarifying it as an identifying restriction where $Z_{1:N}$ renders $(X_t, T_t)$ conditionally independent of $S_t$.
5. **Proposition 4 (Surrogate Entropy-Difficulty Bridge)**: Replaced "empirically confirming that $\overline{c}_J$ is finite" with "providing empirical support for the spectral-comparability assumption with finite $\overline{c}_J$", and clarified the rate scope $\hat{\varepsilon}_{\gamma, N} = \varepsilon_\gamma + \mathcal{O}_P(N^{-1/2})$ for population posteriors $\boldsymbol{\gamma}_t(\boldsymbol{\Lambda}^*)$ (preserved under $\sqrt{N}$-consistent $\hat{\boldsymbol{\Lambda}}$).
6. **Mode 2 Forward Filtering Leakage Isolation**: Explicitly stated that calibration windows strictly precede evaluation windows ($\mathcal{T}_{\mathrm{cal}} < \mathcal{T}_{\mathrm{eval}}$) for Mode 2 training-fitted forward filtering.
7. **Proposition 6 Cross-Covariance**: Changed "if and only if" to a sufficient condition: "A sufficient condition for $\boldsymbol{\Sigma}_{\theta\pi} = \mathbf{0}$ is that the long-run cross-covariance vanishes: $\operatorname{LRCov}(\boldsymbol{\phi}_\theta, \boldsymbol{\phi}_\Lambda) = \mathbf{0}$".
8. **Regularization Penalty Tradeoff**: Clarified in Section 3 that the finite-sample risk-optimal penalty ($\lambda \approx 0.10$) is intentionally distinct from the vanishing-penalty regime ($\lambda = o(N^{-1/2})$) required in Theorem 5 for centered asymptotic inference.
9. **Representation Learning Precision**: Aligned wording to "posterior log-loss provides incremental explanatory information beyond state classification accuracy (within-$R^2$ increase of $+0.0924$)".

## Decision
1. Overhauled the Theorem 2 proof in Appendix B to derive $\boldsymbol{S}^* = \boldsymbol{J}^*\boldsymbol{\theta}^*$ identically from unconfoundedness, and bounded $\boldsymbol{b}_\gamma \equiv (\boldsymbol{S} - \boldsymbol{S}^*) - (\boldsymbol{J} - \boldsymbol{J}^*)\boldsymbol{\theta}^*$ via telescoping expansions in $\varepsilon_\gamma \equiv \E\|\boldsymbol{\gamma}_t - \boldsymbol{H}_t\|_1$, strictly establishing $\|\boldsymbol{\theta}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C\varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})}$.
2. Separated Step A (Neyman nuisance robustness) from Step B (proxy misspecification bias bound).
3. Updated Assumption 3, Proposition 4, Proposition 6, Section 2.1, Section 3, Appendix E, and Appendix F to incorporate all theoretical refinements.
4. Maintained the **strict 24-page budget** (Pages 1–8: Main text; Page 9: References & AI Use Statement; Page 10: Checklist; Pages 11–24: Appendices A–I).

## Consequences
- Theorem 2 proof is 100% algebraically exact with zero unaddressed residuals or invalid decompositions.
- Establishes a watertight theoretical foundation connecting DML nuisance robustness, HMM proxy error degradation, and task conditioning.
- All 25 verification checks in `verify_artifacts.py` pass with 0 errors.
