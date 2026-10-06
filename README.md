# Reliable Causal Estimation under Latent Markov Confounding

Official code repository and reproducibility artifact suite for the manuscript **"Reliable Causal Estimation under Latent Markov Confounding"** (AISTATS 2027).

---

## 🌟 Overview & Scientific Problem

Causal inference from dependent observational time series is complicated when an unobserved, persistent state jointly affects treatment assignment and outcomes. Standard Double Machine Learning (DML) controls for observed covariates through orthogonal residualization, but does not, by itself, resolve confounding induced by an imperfect proxy for latent state dynamics.

We study causal estimation under latent Markov confounding and characterize how latent-state uncertainty and regime overlap propagate into causal estimation error. We develop a regime-aware orthogonal estimator that combines probabilistic state inference, temporally purged cross-fitting, and regularized inversion of the cross-regime score Jacobian.

### Key Theoretical Contributions
1. **Non-Identification under Unconstrained Overlap (Theorem 1)**: Proves that when proxies contain zero information distinguishing latent regimes, regime-specific causal effects cannot be point-identified from observables.
2. **Graceful Degradation Error Bound (Theorem 3)**: Separates posterior proxy recovery error $\varepsilon_\gamma$ from downstream task conditioning $\lambda_{\min}(\boldsymbol{J})$, establishing the multiplicative error bound:
   $$\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \cdot \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2}).$$
3. **Surrogate Entropy-Difficulty Bridge (Proposition 1)**: Proves that under Bayesian calibration, posterior entropy bounds proxy error in binary regimes ($\E\|\boldsymbol{\gamma}_t - \boldsymbol{H}_t\|_1 \le C_K H(\boldsymbol{\gamma}_t)$), yielding the deployable operational difficulty index $\mathcal{D}_{\mathrm{operational}}(t) = \frac{H(\boldsymbol{\gamma}_t)}{\lambda_{\min}(\boldsymbol{J}_t)}$.
4. **Spectral Regularization & Leading-Order Risk Bound (Theorem 4)**: Establishes that ridge-regularized coupled inversion $(\boldsymbol{J} + \lambda \boldsymbol{I})^{-1}$ trades finite-sample bias for bounded variance, sharpening risk when $\lambda_{\min}(\boldsymbol{J}) \to 0$. We develop an adaptive 1D convex plug-in risk minimizer $\hat{\lambda}^*_{\mathrm{risk}} = \arg\min_{\lambda \ge 0} \hat{R}(\lambda)$.
5. **Dependent Asymptotic Normality & Delta-Method Variance (Theorem 5 & Proposition 2)**: Establishes centered asymptotic normality under purged block cross-fitting with embargo buffers $\tau^* \ge C \log N$, and derives the joint delta-method HAC sandwich covariance accounting for Markov persistence inflation ($\frac{1+\rho}{1-\rho} \approx 15.7\times$).

---

## 🌿 Branch Topology & Comparative Lineage

To ensure full transparency and scientific provenance, the evolution of this research program across repository branches is structured as follows:

| Branch | Status | Scope & Key Distinctions |
| :--- | :--- | :--- |
| **`origin/master`** | Pre-Pivot Legacy | Contains initial Regime-Conditional DML (RC-DML) exploration, heuristic policy simulations, and preliminary empirical regressions. Lacked formal treatment of regime overlap, resulting in proxy leakage and ungrounded identification assumptions. |
| **`origin/v2`** | Submission Refinement | Refined submission branch. Eliminated flawed mechanism experiments (old Exp 29/30 with unresidualized regime mean leaks), instituted the strictly compliant 24-page format with 8-page main text, introduced the corrected 7x7 factorial design (`EXP-10`), and validated the representation-to-task reliability triangle. |
| **`feature/or-dml-phase2-pivot`** *(Current Authoritative)* | Canonical Research Suite | Unified, authoritative research codebase. Contains the complete mathematical derivations (Theorems 1–5, Propositions 1–2), strict 24-page manuscript, dual-layer automated verifiers (`verify_science.py`, `verify_artifacts.py`), adaptive spectral optimizer, and three deep empirical exploration frontiers (`EXP-11`, `EXP-12`, `EXP-13`) documenting unvarnished scientific trade-offs. |

---

## 🔬 Core Empirical Discoveries & Unvarnished Research Findings

Our investigations reveal decisive insights into the mechanics of sequential causal estimation under latent confounding, avoiding cherry-picked narratives:

### 1. The Multicollinearity Trade-Off in Soft vs. Hard Weighting
* **The Mechanism**: In soft regime estimation (OR-DML), posterior beliefs $\gamma_{tk} \in (0, 1)$ induce off-diagonal Gram cross-terms $J_{01} = \frac{1}{N}\sum_t \gamma_{t0}\gamma_{t1}\tilde{T}_{t0}\tilde{T}_{t1} > 0$. When regimes exhibit substantial baseline shifts ($g_1 - g_0 = 30$, $b_1 - b_0 = 4$), even slight posterior misclassification leaks a large product into the score vector. Hard clustering (`Regime FE DML`) sets hard indicator assignments $\hat{S}_t \in \{0, 1\}$, forcing $\gamma_{t0}\gamma_{t1} \equiv 0$ and diagonalizing $\boldsymbol{J}$ by decree.
* **The Trade-Off**: Hard clustering achieves lower finite-sample point bias on continuous mixture benchmarks (Hard FE bias $2.81$ vs. Soft OR-DML $3.71$). However, hard clustering **destroys asymptotic distribution theory**, invalidates sandwich covariance under temporal dependence, and exhibits poor coverage. OR-DML preserves Bayesian belief uncertainty and admits valid delta-method HAC sandwich inference, resolving Gram ill-conditioning via spectral shrinkage $\lambda > 0$.

### 2. Empirical Validation of Theorem 4 (Adaptive Spectral Regularization)
* When $\lambda_{\min}(\boldsymbol{J}) \le 0.015$ and condition number $\kappa(\boldsymbol{J}) \ge 70$, unregularized coupled estimation suffers variance explosion ($\Var = 0.032 - 0.045$, $\text{RMSE} > 0.25$, coverage dropping to $40\%$).
* Spectral OR-DML with adaptive plug-in penalty $\hat{\lambda}^*_{\mathrm{risk}} \in [0.01, 0.05]$ reduces estimation variance by **$2.8\times$ to $9.8\times$**, halving RMSE and lifting empirical coverage to $80\%$, confirming the theoretical leading-order risk bound.

### 3. Empirical Validation of Theorem 3 (Graceful Degradation)
* Across controlled representation perturbations with correctly specified regime-aware residualization, causal estimation error scales monotonically with proxy recovery error $\varepsilon_\gamma$:
  * Pearson correlation: **$r = 0.9614$** ($p < 10^{-250}$).
  * Spearman rank correlation: **$\rho = 0.9870$**.
* This proves that Theorem 3 holds strictly once regime mean shifts are prevented from contaminating nuisance residuals.

### 4. Selective Estimation & Abstention Frontier (Frontier A)
* Using the coupled task difficulty index $\mathcal{D}_t = \bar{H}/\lambda_{\min}(\boldsymbol{J})$ as a rejection threshold, selective causal estimation reduces estimation error by **$23.8\times$** ($3.286$ at 100% coverage $\to 0.138$ at 50% coverage).
* Crucially, filtering by task conditioning alone ($1/\lambda_{\min}$) achieves **zero error reduction**, proving that reliable selective inference requires joint representation uncertainty and task geometry.

### 5. Real-Data-Calibrated Megacity Semi-Synthetic Benchmark (`EXP-13`)
* Evaluated on $>14,000$ real hourly meteorological covariates from Delhi, Mumbai, Bengaluru, and Kolkata with calibrated latent transition dynamics and injected ground-truth causal effects ($\theta_0^* = 0.50, \theta_1^* = 2.00$).
* Confirms estimator behavior across diverse empirical airsheds: Delhi Basin ($\lambda_{\min} = 111.8$) separates cleanly; Mumbai Coastal ($\lambda_{\min} = 0.32$) exhibits severe collinearity requiring spectral stabilization; Bengaluru Plateau ($\lambda_{\min} = 4.22$) demonstrates well-conditioned regime contrasts.

---

## 📁 Repository Structure

```text
├── paper/
│   ├── main.tex                    # Authoritative LaTeX manuscript (exact 8-page main text, strictly 24 pages total)
│   ├── main.pdf                    # Compiled PDF submission (24 pages, 0 warnings)
│   ├── aistats2027.sty             # Official AISTATS conference style file
│   └── plots/                      # Embedded publication figures (Figs 1–5)
├── src/                            # Active, self-contained Python codebase
│   ├── or_dml.py                   # Canonical Overlap-Aware Regime DML (OR-DML) estimator
│   ├── adaptive_spectral_optimizer.py # Theorem 4 1D bounded convex plug-in risk minimizer
│   ├── factorial_reliability_experiment.py # Corrected 7x7 representation-by-task mechanism experiment
│   ├── deep_empirical_evaluations.py # Deep evaluations: boundaries, ill-conditioning, representations
│   ├── advanced_methodological_frontiers.py # Selective estimation (Frontier A), Type I error (B), persistence stress (C)
│   ├── real_megacity_semisynthetic_benchmark.py # Semi-synthetic benchmark on 14,122 real megacity observations
│   ├── test_soft_vs_hard_continuous_mixture.py # Comparative test of soft vs. hard weighting under continuous mixtures
│   ├── calibration_intervention_zoo.py # Post-hoc temperature scaling on Representation Zoo
│   ├── empirical_evaluation.py     # 4-city sensor evaluation, conditioning diagnostics, dynamic IRFs
│   ├── empirical_falsification_checks.py # Pre-treatment lead placebo falsification suite (h in {-6, -3, -1})
│   ├── train_real_representation_zoo.py # HMM, GRU, Transformer, SSM benchmark under distribution shifts
│   ├── hierarchical_reliability_regression.py # 15-world fixed-effects regressions & LOWO cross-validation
│   ├── financial_regime_transfer.py # Multi-domain synthetic transfer & selective abstention policy
│   ├── data_pipeline_clean.py      # Clean data engineering pipeline for 14,122 hourly records
│   ├── update_difficulty_figure.py # Generates publication Figure 2 from factorial data
│   ├── build_supplementary_archive.py # Builds canonical 93-file supplementary ZIP
│   └── requirements.txt            # Minimal pip environment dependencies
├── reports/                        # Synchronized CSV report artifacts cited in manuscript
├── plots/                          # Generated high-resolution publication figures
├── data/
│   └── processed_clean/            # Cleaned analysis dataset (14,122 hourly rows)
├── docs/
│   ├── RESEARCH_INVESTIGATION_METHODOLOGY_AND_FINDINGS.md # Detailed methodology & empirical derivations
│   └── adr/                        # Architectural Decision Records (ADRs 0001–0007)
├── experiments_manifest.json       # Machine-readable experiment and theorem manifest (EXP-01 through EXP-13)
├── SUPPLEMENT_ROADMAP.md           # Authoritative supplementary roadmap and table of contents
├── verify_artifacts.py             # Automated artifact and byte-for-byte manuscript synchronization check
└── verify_science.py               # Automated verification of 10 core mathematical/algebraic invariances
```

---

## 🚀 Reproducibility Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Execution Pipeline

```bash
# Step 1: Preprocess raw atmospheric records into clean analysis format (14,122 rows)
python src/data_pipeline_clean.py

# Step 2: Run corrected 7x7 representation-by-task factorial mechanism experiment
python src/factorial_reliability_experiment.py

# Step 3: Run real-data-calibrated megacity semi-synthetic benchmark
python src/real_megacity_semisynthetic_benchmark.py

# Step 4: Run deep empirical evaluations (Theorems 3 & 4 validation)
python src/deep_empirical_evaluations.py

# Step 5: Run advanced methodological frontiers (selective estimation, persistence stress)
python src/advanced_methodological_frontiers.py

# Step 6: Run empirical megacity stress test & dynamic exposure projections
python src/empirical_evaluation.py

# Step 7: Run pre-treatment lead placebo falsification checks
python src/empirical_falsification_checks.py

# Step 8: Run multi-domain transfer and selective abstention policy
python src/financial_regime_transfer.py

# Step 9: Regenerate difficulty frontier publication figure
python src/update_difficulty_figure.py
```

### 3. Automated Dual-Layer Verification

The repository enforces strict automated continuous verification:

```bash
# Layer 1: Artifact & Manuscript Synchronization
# Validates existence of all CSVs and figures, and verifies byte-for-byte agreement with paper/main.tex
python verify_artifacts.py

# Layer 2: Mathematical & Scientific Invariances
# Asserts normal equations, oracle recovery, K=1 reduction, permutation covariance, simplex, and temporal disjointness
python verify_science.py
```

### 4. Compiling the Manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

* **Page Budget**: Strictly 8 pages of main text; exactly 24 pages total (`\@abspage@last{24}`) including AI statement, references, checklist, and Appendices A–I.
* **Compilation Status**: 0 errors, 0 warnings.

### 5. Packaging Supplementary Material

```bash
python src/build_supplementary_archive.py
```
Packages all 93 canonical files into `AISTATS2027_OR_DML_Supplementary_Material.zip` (9.33 MB).
