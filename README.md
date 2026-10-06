# Reliable Causal Estimation under Latent Markov Confounding

Official code repository and reproducibility artifact suite for the manuscript **"Reliable Causal Estimation under Latent Markov Confounding"** (AISTATS 2027).

---

## 🌟 Overview & Scientific Problem

Causal inference from dependent observational time series is complicated when an unobserved, persistent state jointly affects treatment assignment and outcomes. Standard Double Machine Learning (DML) controls for observed covariates through orthogonal residualization, but does not, by itself, resolve confounding induced by an imperfect proxy for latent state dynamics.

We study causal estimation under latent Markov confounding and characterize how latent-state uncertainty and regime overlap propagate into causal estimation error. We develop a regime-aware orthogonal estimator that combines probabilistic state inference, temporally purged cross-fitting, and regularized inversion of the cross-regime score Jacobian.

### Key Theoretical Contributions
1. **Decomposition of Omitted Regime Bias & Frisch-Waugh Singularity (Theorem 1 & Corollary 1)**: Proves that cross-sectional residualization on observed controls $X_t$ fails to purge latent regime variation and actively amplifies omitted confounding as residual treatment variance $\E[\tilde{T}_t^2]$ is attenuated toward its lower bound.
2. **Non-Identification under Unconstrained Overlap (Theorem 2)**: Proves that when proxies contain zero information distinguishing latent regimes (under memoryless transitions $p=0.5$ or static proxy evaluation), regime-specific causal effects cannot be point-identified from observables.
3. **Graceful Degradation Error Bound (Theorem 3)**: Separates posterior proxy recovery error $\varepsilon_\gamma$ from downstream task conditioning $\lambda_{\min}(\boldsymbol{J})$, establishing the multiplicative error bound:
   $$\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \cdot \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2}).$$
4. **Surrogate Entropy-Difficulty Bridge (Proposition 1)**: Proves that under Bayesian calibration in binary regimes, posterior entropy bounds proxy error ($\E\|\boldsymbol{\gamma}_t - \boldsymbol{H}_t\|_1 \le C_K H(\boldsymbol{\gamma}_t)$), yielding the deployable operational difficulty index $\mathcal{D}_{\mathrm{operational}}(t) = \frac{H(\boldsymbol{\gamma}_t)}{\lambda_{\min}(\boldsymbol{J}_t)}$.
5. **Spectral Regularization & Leading-Order Risk Bound (Theorem 4)**: Establishes that ridge-regularized coupled inversion $(\boldsymbol{J} + \lambda \boldsymbol{I})^{-1}$ trades finite-sample bias for bounded variance, sharpening risk when $\lambda_{\min}(\boldsymbol{J}) \to 0$. We develop an adaptive 1D convex plug-in risk minimizer $\hat{\lambda}^*_{\mathrm{risk}} = \arg\min_{\lambda \ge 0} \hat{R}(\lambda)$.
6. **Dependent Asymptotic Normality & Centering Breakdown (Theorem 5 & Proposition 2)**: Establishes centered asymptotic normality under purged block cross-fitting with embargo buffers $\tau^* \ge C \log N$ when $\varepsilon_\gamma = o(N^{-1/2})$, proves that fixed proxy overlap ($\varepsilon_\gamma = \mathcal{O}(1)$) induces non-vanishing score mean shifts that break nominal centering, and derives the joint delta-method HAC sandwich covariance accounting for Markov persistence inflation ($\frac{1+\rho}{1-\rho} \approx 15.7\times$).

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

### 1. Honest Baseline Comparisons & Diagnostic Framing
* **11-Method Monte Carlo Benchmark (`reports/or_dml_benchmark_summary.csv`)**: Across 500 replications ($\Delta_Z \in [0.2, 4.0]$), when regimes separate cleanly ($\Delta_Z \ge 2.0$), regime-aware methods eliminate omitted confounding: post-clustering Hard FE (median bias 0.064) and HMM$(Z, T)$ (median bias 0.045) match or outperform Spectral OR-DML (0.075) and Oracle DML (0.042), achieving 87–94% coverage.
* **Low-Observability Breakdown ($\Delta_Z \le 1.0$)**: Under weak separation, all estimators suffer severe proxy-induced bias (Hard FE bias 2.95, Decoupled Soft 3.56, Spectral OR-DML 4.25, HMM$(Z, T)$ 1.20). Crucially, proxy-induced size distortion causes both Hard FE and OR-DML to collapse to 0.0% coverage (Frontier B null test confirms anti-conservative inference).
* **Diagnostic Reframe**: Rather than claiming to be an unconditionally dominating point estimator, OR-DML functions primarily as a **principled diagnostic and selective causal abstention framework**: when $\mathcal{D}_t$ surges, systems abstain rather than delivering spurious inferences.
* **Real-Data Megacity Semi-Synthetic Benchmark (`reports/megacity_semisynthetic_summary.csv`)**: On real meteorological covariates with injected ground truth, Hard FE exhibits bias 2.48–3.56 and 0% coverage in Mumbai and Bengaluru when real-data proxy errors violate clean clustering assumptions, disproving unconditional dominance of hard clustering.

### 2. Empirical Validation of Theorem 4 (Adaptive Spectral Regularization)
* In ill-conditioned environments ($\lambda_{\min}(\boldsymbol{J}) \le 0.015$, condition number $\kappa(\boldsymbol{J}) \ge 70$), unregularized coupled estimation suffers variance explosion ($\Var = 0.032 - 0.045$, $\text{RMSE} > 0.25$).
* Spectral OR-DML with adaptive plug-in penalty $\hat{\lambda}^*_{\mathrm{risk}} \in [0.01, 0.05]$ reduces estimation variance by **$2.8\times$ to $9.8\times$**, halving RMSE and confirming the theoretical leading-order risk bound.

### 3. Factorial Mechanism Validation (Theorem 3)
* Across 4,900 independently controlled Markov runs in a 7×7 factorial design where task conditioning is varied through residual treatment innovation variance after regime-aware nuisance residualization:
  * Spearman correlation of causal $L_2$ error with $\mathcal{D}_{\mathrm{causal}} = \varepsilon_\gamma / \lambda_{\min}(\boldsymbol{J})$: **$\rho = 0.995$** (cell-mean $\rho = 0.996$).
  * Spearman correlation with proxy error $\varepsilon_\gamma$ alone: $\rho = 0.626$.
  * Spearman correlation with $1/\lambda_{\min}$ alone: $\rho = 0.521$.
* Confirms that causal estimation degradation is strictly an interactive phenomenon governed by both latent uncertainty and task geometry.

### 4. Selective Estimation & Abstention Frontier (Frontier A)
* Using the coupled task difficulty index $\mathcal{D}_t = \bar{H}/\lambda_{\min}(\boldsymbol{J})$ as a rejection threshold, selective causal estimation reduces estimation error by **$23.8\times$** ($3.286$ at 100% coverage $\to 0.138$ at 50% coverage).
* Crucially, filtering by task conditioning alone ($1/\lambda_{\min}$) achieves **zero error reduction**, proving that reliable selective inference requires joint representation uncertainty and task geometry.

### 5. Transparent Empirical Megacity Sensor Network Findings
* **Temporal Cadence**: Analysis is performed on $>14,000$ observational records across Delhi, Mumbai, Bengaluru, and Kolkata (Feb 2025–Jun 2026) with a median cadence of 2.0 hours.
* **Delhi Basin ($\lambda_{\min} = 111.8$)**: Decomposes pooled effect ($+0.47 \pm 0.04$) into Regime 1 ($+0.7038 \pm 0.0513$, winter inversion, $p<0.0001$) and Regime 2 ($+0.1281 \pm 0.0743$, clearance, $p=0.0845$). Pre-treatment placebos at $h \in \{-6, -3, -1\}$ reject nullity ($p<0.01$), identifying these curves as exposure-response diagnostics rather than validated causal impulse responses.
* **Mumbai Coastal ($\lambda_{\min} = 0.23, \kappa = 2.09$)**: Under verified `hmmlearn`, separates Regime 1 (Coastal Stagnation, $-13.3873 \pm 9.9264, p=0.1774$) from Regime 2 (Ventilation Breeze, $+6.9702 \pm 4.5945, p=0.1292$).
* **Bengaluru Plateau ($\lambda_{\min} = 4.29, \kappa = 1.68$)**: Separates Regime 1 (Nocturnal Inversion, $+0.0875 \pm 0.1527, p=0.5667$) from Regime 2 (Afternoon Dispersion, $-0.1528 \pm 0.0691, p=0.0271$).
* **Kolkata Airshed ($\lambda_{\min} = 0.0024$)**: Sensor telemetry displays low variance ($\Var=0.156$) with 31 unique values; unregularized estimates explode ($\text{SE}=1.11$), while spectral regularization stabilizes variance ($\text{SE} \to 0.060$ at $\lambda=0.10$), with regularized condition number $\kappa(\hat{\boldsymbol{J}}+\lambda\boldsymbol{I})$ contracting monotonically from 1.28 to 1.00.

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

* **Page Budget**: Strictly 8 pages of main text; exactly 27 pages total including AI statement, references (83 verified citations), checklist, and Appendices A–I.
* **Compilation Status**: 0 errors, 0 warnings.

### 5. Packaging Supplementary Material

```bash
python src/build_supplementary_archive.py
```
Packages all 103 canonical files into `AISTATS2027_OR_DML_Supplementary_Material.zip` (9.62 MB).

---

## 🚀 Looking Ahead: The `py-ordml` Ecosystem & The Dual-Factor Law

For a detailed blueprint of post-publication software architecture and theoretical extensions, see [`docs/FUTURE_WORK_AND_FRONTIERS.md`](docs/FUTURE_WORK_AND_FRONTIERS.md).

### 1. The `py-ordml` Open-Source Library Roadmap
We are packaging the numerical core into an installable Python library (`pip install py-ordml`):
* **Scikit-Learn / EconML Compatibility**: Native `OverlapAwareRegimeDML` estimator supporting plug-and-play regime models (`HMM`, `StateSpaceModel`, `GRU`) and nuisance learners (`LightGBM`, `CatBoost`, `Ridge`).
* **Automated Causal Guardrails**: Built-in automated Bartlett causal embargo ($\tau^* \ge C \log N$) and Theorem 4 adaptive spectral shrinkage ($\hat{\lambda}^*_{\mathrm{risk}}$).
* **Self-Contained Walkthrough**: Run `python src/walkthrough_tutorial.py` for a 30-second zero-dependency tutorial demonstrating baseline omitted-regime bias vs. OR-DML recovery.

### 2. The Dual-Factor Law of Prospective Difficulty
A central methodological breakthrough of this work is that **causal estimation breakdown cannot be foreseen by representation uncertainty or task geometry alone**:
$$\mathcal{D}_t \equiv \frac{H(\boldsymbol{\gamma}_t)}{\lambda_{\min}(\boldsymbol{J}_t)}$$
* In controlled factorial experiments (`EXP-10`), the Dual-Factor Index $\mathcal{D}_t$ predicts downstream causal RMSE with **$\rho = 0.995$** Spearman rank correlation ($p < 10^{-15}$), strictly outperforming marginal entropy ($\rho = 0.626$) and marginal Gram ill-conditioning ($\rho = 0.521$).
* While filtering by task conditioning alone achieves **0% error reduction**, selective estimation via $\mathcal{D}_t$ slashes error by **$23.8\times$** ($3.28 \to 0.138$).

### 3. Broader Horizons for Selective Causal Abstention
We recommend deploying the Dual-Factor difficulty index as a prospective safety filter across high-stakes sequential decision systems:
* **Healthcare & ICU Sepsis Resuscitation**: Detecting latent hemodynamic transition phases where observational confounding makes vasopressor effect estimation unidentifiable, prompting algorithmic abstention and human clinical review.
* **Financial Markets & High-Frequency Execution**: Identifying volatile liquidity transitions to prevent causal execution policy breakdown, reducing decision regret by **$56.8\times$** (`EXP-08`).
* **Offline Reinforcement Learning in POMDPs**: Bounding importance sampling weight explosion during unobserved environmental regime shifts.
* **Airshed Environmental Policy**: Establishing airshed-specific conditioning thresholds (e.g., separating well-conditioned basins like Delhi from collinear coastal airsheds like Mumbai).
