# Reliable Causal Estimation under Latent Markov Confounding

Official code repository and reproducibility artifact suite for the manuscript **"Reliable Causal Estimation under Latent Markov Confounding"** (AISTATS 2027).

---

## 🌟 Overview & Scientific Problem

Causal inference from dependent observational time series is challenging when an unobserved, persistent state jointly affects treatment assignment and outcomes. Standard Double Machine Learning (DML) controls for observed covariates through orthogonal residualization, but does not resolve confounding induced by an imperfect proxy for latent state dynamics.

We study causal estimation under latent Markov confounding and characterize how latent-state uncertainty and regime overlap propagate into causal estimation error:
1. **Non-Identification under Unconstrained Overlap (Theorem 1)**: Proves that when proxies contain zero information distinguishing latent regimes, regime-specific causal effects cannot be point-identified from observables.
2. **Graceful Degradation Error Bound (Theorem 2)**: Separates posterior proxy error $\varepsilon_\gamma$ from task conditioning $\lambda_{\min}(\boldsymbol{J})$, establishing an explicit error bound:
   $$\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \cdot \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2}).$$
3. **Surrogate Entropy-Difficulty Bridge (Proposition 4)**: Demonstrates that under Bayesian calibration, posterior entropy bounds latent-state error, yielding the computable operational difficulty index $\mathcal{D}_{\mathrm{operational}}(t) = \frac{H(\boldsymbol{\gamma}_t)}{\lambda_{\min}(\boldsymbol{J}_t)}$.
4. **Spectral Regularization & Asymptotics (Theorems 3 & 5, Proposition 6)**: Derives leading-order risk bounds for ridge-regularized inversion $(\boldsymbol{J} + \lambda \boldsymbol{I})^{-1}$ and establishes centered asymptotic normality with purged block cross-fitting under dependent data.
5. **Representation Zoo vs. Downstream Reliability**: Evaluates HMM, GRU, Causal Transformer, and Linear State-Space Models across 15 independent benchmark worlds under disentangled transition and emission shifts, proving that state classification accuracy ($F_1$) alone does not guarantee downstream reliability, whereas posterior calibration provides significant incremental predictive value.
6. **Real-World Observational Stress Test**: Evaluates four Indian megacities ($14,122$ complete analysis hours across Delhi, Mumbai, Bengaluru, and Kolkata), demonstrating how disparate task geometries illuminate distinct operating regimes on the difficulty frontier.

---

## 📁 Repository Structure

```text
├── paper/
│   ├── main.tex                    # Authoritative LaTeX manuscript (exact 8-page main text, strictly 24 pages total)
│   ├── main.pdf                    # Compiled PDF submission (24 pages, 0 warnings)
│   ├── aistats2027.sty             # Official AISTATS conference style file
│   └── plots/                      # Publication figures embedded in manuscript
├── src/                            # Active, self-contained Python codebase
│   ├── or_dml.py                   # Canonical Overlap-Aware Regime DML (OR-DML) implementation
│   ├── calibration_intervention_zoo.py # Post-hoc Platt/temperature scaling evaluation on Representation Zoo
│   ├── empirical_falsification_checks.py # Pre-treatment lead placebo falsification suite (h in {-6, -3, -1})
│   ├── synthetic_dgp_benchmark.py  # 500-draw Monte Carlo difficulty frontier simulation
│   ├── empirical_evaluation.py     # 4-city sensor evaluation, conditioning diagnostics, dynamic IRFs
│   ├── train_real_representation_zoo.py # HMM, GRU, Transformer, SSM benchmark under distribution shifts
│   ├── hierarchical_reliability_regression.py # 15-world fixed-effects regressions & LOWO cross-validation
│   ├── financial_regime_transfer.py # Multi-domain synthetic transfer & selective abstention policy
│   ├── data_pipeline_clean.py      # Clean data engineering pipeline for 14,122 hourly records
│   ├── generate_paper_figures.py   # Publication figures generator (Figures 1–5)
│   ├── regime_intelligence.py      # Meteorological regime validation and profiling
│   └── requirements.txt            # Minimal pip environment dependencies
├── reports/                        # Synchronized CSV report artifacts cited in manuscript
├── plots/                          # Generated high-resolution publication figures
├── data/
│   └── processed_clean/            # Cleaned analysis dataset (14,122 hourly rows)
├── docs/
│   └── adr/                        # Architectural Decision Records (ADRs 0001–0007)
├── experiments_manifest.json       # Machine-readable experiment and theorem manifest
├── SUPPLEMENT_ROADMAP.md           # Authoritative supplementary roadmap and table of contents
├── verify_artifacts.py             # Automated artifact and byte-for-byte manuscript synchronization check
└── verify_science.py               # Automated verification of 9 core mathematical/algebraic invariances
```

---

## 🔬 Canonical Implementation: OR-DML (`src/or_dml.py`)

The primary estimator is implemented in [`src/or_dml.py`](./src/or_dml.py):
* **Class**: `OverlapAwareRegimeDML`
* **Posterior Modes**: Supports causal forward filtering ($\boldsymbol{\gamma}_t = P(S_t \mid \mathcal{F}_t)$) and retrospective smoothing ($\boldsymbol{\gamma}_t = P(S_t \mid Z_{1:N})$).
* **Nuisance Estimation**: Cross-fitted with purged temporal blocks and temporal embargo buffers (Ridge regressors in the 4-city observational study; Ridge and Gradient Boosting regressors in the synthetic benchmarks).
* **Inversion**: Spectrally regularized coupled Jacobian inversion $\hat{\boldsymbol{\theta}}_\lambda = (\hat{\boldsymbol{J}} + \lambda \boldsymbol{I})^{-1} \hat{\boldsymbol{S}}$.
* **Conditioning Diagnostics**: Automatically computes $\lambda_{\min}(\hat{\boldsymbol{J}})$, condition number $\kappa(\hat{\boldsymbol{J}})$, and sample state occupancy.

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

### 2. End-to-End Pipeline Execution

To reproduce all numerical results, figures, and benchmark tables from scratch:

```bash
# Step 1: Preprocess raw atmospheric records into clean analysis format (14,122 rows)
python src/data_pipeline_clean.py

# Step 2: Run 500-draw Monte Carlo difficulty frontier benchmark
python src/synthetic_dgp_benchmark.py

# Step 3: Run real-world megacity evaluation and dynamic impulse-response functions
python src/empirical_evaluation.py

# Step 4: Run representation zoo evaluation (HMM, GRU, Transformer, SSM)
python src/train_real_representation_zoo.py

# Step 5: Run hierarchical reliability regressions across 15 benchmark worlds
python src/hierarchical_reliability_regression.py

# Step 6: Run multi-domain transfer and selective abstention policy
python src/financial_regime_transfer.py

# Step 7: Run post-hoc calibration interventions on representation zoo
python src/calibration_intervention_zoo.py

# Step 8: Run pre-treatment lead placebo falsification checks
python src/empirical_falsification_checks.py

# Step 9: Generate all publication figures (Figures 1-5)
python src/generate_paper_figures.py
```

### 3. Automated Dual-Layer Verification

The repository includes two automated verification layers to ensure reproducibility and scientific soundness:

```bash
# Layer 1: Artifact & Manuscript Synchronization
# Checks report existence, figure existence, absence of mock tokens, and byte-for-byte synchronization
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
* **Page Budget**: Strictly 8 pages of main text; exactly 24 pages total including AI statement, references, checklist, and Appendices A–I.
* **Compilation Status**: 0 errors, 0 warnings.
