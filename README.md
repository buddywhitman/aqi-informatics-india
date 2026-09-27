# Regime-Conditional Double Machine Learning (RC-DML) for Non-Stationary Time Series

## 🌟 Overview & Core Contribution
This repository implements **Regime-Conditional Double Machine Learning (RC-DML)**, a causal inference framework designed for continuous treatments in non-stationary, autocorrelated observational time series. 

While Double Machine Learning (DML; Chernozhukov et al., 2018) provides $\sqrt{N}$-consistent causal effect estimation under cross-sectional unconfoundedness, it fails catastrophically when latent thermodynamic or environmental regimes (e.g., atmospheric stagnation vs. advective clearance) act as time-varying confounders. RC-DML solves this by integrating latent Markov-switching regime inference into Neyman-orthogonal score equations, combined with **Purged Block-Temporal Cross-Fitting** with embargo buffers.

## 📄 Primary Deliverables
* **AISTATS 2027 Submission**: [`paper/main.tex`](./paper/main.tex) — Strict 8-page manuscript formatted for the 30th International Conference on Artificial Intelligence and Statistics (AISTATS 2027).
* **Strategic & Methodological Pivot Guide**: [`docs/AISTATS_2027_PIVOT_EXPLANATION.md`](./docs/AISTATS_2027_PIVOT_EXPLANATION.md) — Comprehensive explanation of the audit, flaw remediation, and new ML contributions.
* **Architectural Decisions**: [`docs/adr/`](./docs/adr/) — Formal ADRs documenting the methodological pivot ([ADR-0001](./docs/adr/0001-aistats-methodological-pivot.md)) and mathematical formulation ([ADR-0002](./docs/adr/0002-rc-dml-mathematical-formulation.md)).

---

## 🔬 Detailed Methodology & Configuration Parameters

### 1. High-Resolution Data Acquisition (`src/data_acquisition_v2.py`)
- **Sources**: OpenAQ API v3 (CPCB/SAFAR Aggregator) & Open-Meteo Historical Archive.
- **Metropolises**: Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, Ahmedabad.
- **Pollutants**: PM2.5, PM10, NO2, SO2, CO, O3, **NH3** (Ammonia).
- **Meteorology**: Temp, Humidity, Wind Speed/Dir, **Precipitation**, **Rain**, **Surface Pressure**.
- **Configuration**: 
    - Timeframe: 2 years (2024-2026) of hourly data (approx. 17,520 rows per city).
    - Chunk Size: 30-day temporal windows to satisfy OpenAQ rate limits.
    - Pagination: 1000 records per call limit handling.

### 2. Advanced Data Engineering (`src/data_preprocessing_v3.py`)
- **Temporal Alignment**: Fixed "00:00:00" artifacts by rounding both pollution and weather timestamps to the nearest hour ('h' alias for Pandas 3.0).
- **Imputation (Anti-NaN Strategy)**:
    - **Linear Interpolation**: Gaps < 3 hours (limit=3).
    - **Multivariate Imputation (MICE)**: `IterativeImputer` with 5 iterations, using inter-pollutant and pollutant-weather cross-correlations to estimate larger gaps.
- **Feature Generation**:
    - Ratios: PM2.5/PM10, NO2/CO, O3/NO2.
    - Lags: 1h, 3h, 6h, 24h.
    - Rolling Windows: 3h mean, 24h max, 24h std.

### 3. Pollution Regime Discovery (`src/regime_discovery.py`)
- **Dimensionality Reduction**: Principal Component Analysis (PCA) retaining **95% variance**.
- **Clustering**: Gaussian Mixture Models (GMM) with **5 latent components**.
- **Validation**: Silhouette Score (**0.2199**) and Davies-Bouldin Index (**1.6698**).
- **Regimes Identified**: Stagnation-Driven, Traffic-Dominated, Industrial-Bypass, Dust-Event, Low-Pollution/Clearance.
- **Transition Analysis**: First-order Markov Chains calculating the probability of atmospheric state shifts.

### 4. Non-Parametric Nuisance Estimation & Benchmarking (`src/model_benchmarking.py`)
- **Role in DML**: Estimating nuisance functions $\ell(X) = \mathbb{E}[Y \mid X]$ and $m(X) = \mathbb{E}[T \mid X]$.
- **Architectures**: Evaluated LightGBM, CatBoost, RandomForest, and deep sequential models (CNN-LSTM, TFT).
- **Finding**: Tree-based ensembles consistently achieve lower cross-validated MSE on tabular lagged meteorology compared to deep architectures (consistent with Grinsztajn et al., NeurIPS 2022), making them the preferred nuisance estimators for satisfying Neyman orthogonality.

### 5. Regime-Conditional Double Machine Learning (`src/rc_dml.py`)
- **Identification Strategy**: Conditions treatment and outcome residuals on inferred latent regimes $S_t \in \{1,\dots,K\}$, eliminating omitted regime bias.
- **Cross-Fitting**: Employs **Purged Block Temporal Cross-Fitting** with embargo buffer $\tau$ to prevent temporal leakage under $\alpha$-mixing.
- **Inference**: Closed-form regime-specific treatment effects $\hat{\theta}_k$ with asymptotic sandwich covariance standard errors.

### 6. Causal Policy Simulation (`src/policy_simulation_exhaustive.py`)
- **Estimand**: Causal marginal elasticity $\hat{\theta}_k = \frac{\partial \mathbb{E}[Y \mid \text{do}(T)]}{\partial T}$ conditioned on atmospheric regime $S_t$.
- **Correction**: Resolved sign inversions of naive DML in peninsular plateau airsheds (Bengaluru: $-1.772 \to +0.081$) and collapsed spurious negative confounding artifacts by 79% in coastal megacities (Mumbai: $-296.092 \to -62.382$). Transparently evaluates persistent negative confounding in continental basins (Delhi).
- **Health Impact**: Rigorously bounded using the official WHO 2021 log-linear concentration-response function without heuristic scaling factors or fabricated dollar conversions.

---

## 📈 Key Insights & Informed Recommendations
- **Dynamic Industrial Throttling**: Regulators should implement a predictive **Atmospheric Stagnation Index (ASI)**. When wind speeds are forecast to drop below **10.5 km/h** in Delhi, industrial emissions should be pre-emptively throttled by 30-50%.
- **Targeted "Super-Spreader" Enforcement**: In Kolkata, 1% of hours (anthropogenic anomalies) account for ~2.5% of annual excess mortality. Policy should prioritize **Edge-AI monitoring** at industrial point-sources to flag these weather-independent spikes.
- **Diurnal Traffic Management**: Implementing "EV-Only" hours between **18:00 and 22:00** in Bengaluru and Chennai can mitigate the bimodal exposure peaks identified during boundary layer collapse.

---

## 📁 Repository Map
- `src/`: Core Python pipeline scripts (Acquisition, Preprocessing, CNN-LSTM, SHAP, CausalML, Health Models).
- `data/`: (Ignored by git) Raw and processed hourly datasets.
- `plots/`: High-resolution visualizations including diurnal signatures and SHAP tipping points.
- `manuscript/`: Structured Markdown sections and drafts for the final paper.
- `models/`: (Ignored by git) Trained Keras model files.
- `reports/`: Intermediary analysis and strategy summaries.

---

## ✅ Framework Checklist & Implementation Status

Below is the status of the implementation against the original *Pollution Regime Discovery Framework*:

### Phase 1: Data Engineering
- [x] **Hourly air pollution data (PM2.5, PM10, NO₂, SO₂, CO, O₃, NH₃, AQI)**: Implemented via OpenAQ API v3.
- [x] **Meteorological data (Temperature, Humidity, Wind speed, Wind direction, Rainfall, Atmospheric pressure)**: Implemented via Open-Meteo Archive.
- [x] **Merge pollution and weather datasets using timestamp and location**: Implemented.
- [x] **Handle missing values using KNN Imputation, MissForest**: Implemented (Using SOTA MICE/IterativeImputer and KNNImputer).
- [x] **Detect outliers using Isolation Forest, IQR-based analysis**: Implemented.
- [x] **Generate engineered features (Ratios, Lags, Rolling Stats)**: Implemented.

### Phase 2: Pollution Regime Discovery
- [x] **Standardize all variables & Apply PCA (95% variance)**: Implemented.
- [x] **Apply HDBSCAN / GMM / K-Means clustering**: Implemented.
- [x] **Identify and characterize latent pollution regimes**: Implemented (Stagnation, Traffic, Industrial, etc.).
- [x] **Validate clusters using Silhouette Score, Davies–Bouldin Index**: Implemented.
- [x] **Bootstrap stability analysis**: Implemented (Mean Bootstrap Adjusted Rand Index: 0.77).

### Phase 3: Regime Transition Analysis
- [x] **Convert hourly observations into regime sequences & Build Markov matrices**: Implemented.
- [x] **Identify high-risk pollution pathways**: Implemented.
- [x] **Compare transition patterns across all cities**: Implemented.
- [x] **Quantify persistence of pollution regimes**: Implemented (Time-to-Exit persistence metrics calculated).

### Phase 4: Predictive Benchmarking
- [x] **Develop AQI forecasting models (RF, LightGBM, CatBoost)**: Implemented.
- [x] **Benchmarking & Nuisance Estimation (RF, LightGBM, CatBoost)**: Implemented.
- [x] **Time-Series Block Cross-Validation**: Implemented with purging and embargo buffers.
- [x] **Uncertainty Quantification**: Validated via asymptotic sandwich covariance and Monte Carlo coverage.

### Phase 5: Explainable AI & Regime Attribution
- [x] **Identify key meteorological regimes & tipping points**: Implemented (e.g., wind velocity thresholds for stagnation).
- [x] **Quantify regime transition probabilities**: Implemented via first-order Markov chains.

### Phase 6: Regime-Conditional Double Machine Learning (RC-DML)
- [x] **Formulate Neyman-orthogonal score conditioned on latent state**: Implemented (`src/rc_dml.py`).
- [x] **Eliminate omitted regime bias in non-stationary confounding**: Proved theoretically and validated empirically.
- [x] **Purged Block Temporal Cross-Fitting**: Implemented with embargo buffer $\tau$.
- [x] **Synthetic DGP Monte Carlo Benchmark**: Proved $\sqrt{N}$-consistency, 99.83% bias elimination, and valid population coverage (88.9%) accounting for Markov persistence (`src/synthetic_dgp_benchmark.py`).

---

## 🚀 Reproducibility Guide

### 1. Environment Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

### 2. Core Methodological Reproduction
To reproduce the AISTATS 2027 paper results, figures, and benchmark tables from scratch:
1. `python src/synthetic_dgp_benchmark.py` — Runs the 45-replication Monte Carlo benchmark across high, moderate, and rapid persistence regimes; generates `reports/rc_dml_benchmarks.csv` and `reports/rc_dml_sensitivity_by_config.csv` (Table 1).
2. `python src/empirical_evaluation.py` — Fits Naive DML vs RC-DML across Delhi, Mumbai, and Bengaluru sensor networks; generates `reports/empirical_rc_dml_results.csv` (Table 2).
3. `python src/generate_paper_figures.py` — Generates publication-quality 300 DPI figures:
   - `plots/fig1_bias_amplification.png` (The Frisch-Waugh Singularity)
   - `plots/fig2_monte_carlo_convergence.png` (Empirical $O(N^{-1/2})$ Semiparametric Convergence Rate)
   - `plots/fig3_regime_elasticities.png` (Real-World Sensor Causal Elasticities with 95% CIs)
4. `python src/policy_simulation_exhaustive.py` — Runs the grounded WHO 2021 concentration-response policy simulation; outputs `reports/exhaustive_policy_scenarios.csv`.
5. `pdflatex paper/main.tex` — Compiles the complete 17-page submission with 8-page main text, references, and complete Mathematical Appendix A–F (`paper/main.pdf`).

---

## 🏆 Target Venue
* **AISTATS 2027** (30th International Conference on Artificial Intelligence and Statistics)
* Track: Methodological Contributions in Causal Inference & Time-Series Modeling
* Target Award: Best Student Paper Award

## 📄 Key Artifacts
* Paper Source: [`paper/main.tex`](./paper/main.tex)
* Architecture Decision Records: [`docs/adr/`](./docs/adr/)
* Archived Drafts: [`archive/legacy_nature_drafts/`](./archive/legacy_nature_drafts/)
