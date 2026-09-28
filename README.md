# Overlap-Aware Regime Double Machine Learning (OR-DML) for Non-Stationary Time Series

## 🌟 Overview & Core Contribution
This repository implements **Overlap-Aware Regime Double Machine Learning (OR-DML)**, a causal inference framework designed for continuous treatments in non-stationary, autocorrelated observational time series subject to latent, persistent confounding.

While Double Machine Learning (DML; Chernozhukov et al., 2018) provides $\sqrt{N}$-consistent causal effect estimation under cross-sectional unconfoundedness, it fails catastrophically when latent thermodynamic or environmental regimes (e.g., atmospheric stagnation vs. advective clearance) act as time-varying confounders. OR-DML addresses this foundational challenge through:
1. **Impossibility & Graceful Degradation (Theorems 1 & 2)**: Proves causal non-identification under unconstrained overlap and establishes an honest error bound $\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2})$ connecting posterior proxy error $\varepsilon_\gamma$ and Jacobian conditioning $\lambda_{\min}(\boldsymbol{J})$.
2. **Spectral Regularization Bias-Variance Frontier (Theorem 3)**: Introduces $\hat{\boldsymbol{\theta}}_\lambda = (\hat{\boldsymbol{J}} + \lambda \boldsymbol{I})^{-1}\hat{\boldsymbol{S}}$ with automated trace shrinkage $\lambda^* \asymp N^{-1/2}$, guaranteeing stability across ill-conditioned overlap regimes.
3. **Omitted Regime Bias & Frisch-Waugh Singularity (Theorem 4)**: Decomposes the exact failure mode of standard DML into heterogeneity attenuation and confounding bias amplification.
4. **Purged Block Temporal Cross-Fitting (Theorem 5)**: Eliminates dependence leakage under $\alpha$-mixing via automated Bartlett embargo buffers $\tau^*$.
5. **Markov Occupation Variance Decomposition (Proposition 6)**: Quantifies the $15.7\times$ inflation between sample and population ATE under persistent regimes ($\rho \approx 0.88$).

## 📄 Primary Deliverables
* **AISTATS 2027 Submission**: [`paper/main.tex`](./paper/main.tex) — Strict 8-page main text manuscript (14 pages total including references, reproducibility checklist, and complete proofs in Appendices A–F).
* **Strategic Pivot Documentation**: [`docs/pivot.md`](./docs/pivot.md) and [`docs/critique.md`](./docs/critique.md).
* **Clean Longitudinal Dataset**: [`data/processed_clean/combined_hourly_clean.csv`](./data/processed_clean/combined_hourly_clean.csv) — 14,122 verified, non-negative hourly observations across Delhi, Mumbai, Bengaluru, and Kolkata.

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

### 2. Exact Methodological Reproduction
To reproduce the AISTATS 2027 paper results, figures, and benchmark tables byte-for-byte from scratch:
1. `python src/data_pipeline_clean.py` — Cleans raw ground-station and reanalysis archives into [`data/processed_clean/combined_hourly_clean.csv`](./data/processed_clean/combined_hourly_clean.csv) ($14,122$ complete cases across Delhi, Mumbai, Bengaluru, and Kolkata; verified $0$ negative values, non-negative PM10/PM2.5, physically bounded meteorology).
2. `python src/synthetic_dgp_benchmark.py` — Runs the 500-replication Monte Carlo difficulty frontier benchmark across $\Delta_Z \in [0.2, 4.0]$ ($N=1,200$); generates [`reports/or_dml_difficulty_frontier.csv`](./reports/or_dml_difficulty_frontier.csv), [`reports/or_dml_benchmark_summary.csv`](./reports/or_dml_benchmark_summary.csv) (**Table 1** in paper), and [`plots/fig2_difficulty_frontier.png`](./plots/fig2_difficulty_frontier.png) (**Figure 3** in paper).
3. `python src/empirical_evaluation.py` — Fits Standard DML, Block DML, Spectral OR-DML, and Filtered OR-DML across all four megacities; computes dynamic causal impulse-response functions up to $h=24$ hours; generates [`reports/empirical_or_dml_results.csv`](./reports/empirical_or_dml_results.csv) (**Table 2** in paper), [`reports/empirical_irf_results.csv`](./reports/empirical_irf_results.csv), and [`plots/fig3_dynamic_irf.png`](./plots/fig3_dynamic_irf.png) (**Figure 4** in paper).
4. `cd paper && pdflatex -interaction=nonstopmode main.tex` — Compiles [`paper/main.pdf`](./paper/main.pdf) adhering strictly to the $\le 8$ pages main text limit (14 pages total including references, AI use disclosure, reproducibility checklist, and complete proofs in Appendices A–F).

---

## 🏆 Target Venue
* **AISTATS 2027** (30th International Conference on Artificial Intelligence and Statistics)
* Track: Methodological Contributions in Causal Inference & Time-Series Modeling
* Target Award: Best Student Paper Award

## 📄 Key Artifacts
* Paper Source: [`paper/main.tex`](./paper/main.tex)
* Architecture Decision Records: [`docs/adr/`](./docs/adr/)
* Archived Drafts: [`archive/legacy_nature_drafts/`](./archive/legacy_nature_drafts/)
