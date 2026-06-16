# Spatiotemporal Air Quality Informatics & Causal Policy Simulation: A Pan-India Study (2021-2026)

## 🌟 Vision & Objective
This project establishes a State-of-the-Art (SOTA) environmental intelligence system for the Indian subcontinent. By integrating high-resolution (hourly) ground-sensor data, atmospheric physics, and advanced causal inference, we provide a transparent, interpretable framework for urban planners and sustainable development policy-makers. The work addresses critical gaps in data resolution, "Black Box" model opacity, and the lack of causal attribution in standard environmental datasets.

## 📄 Key Deliverables
- **[Final_Q1_Manuscript_30Pages.pdf](./Final_Q1_Manuscript_30Pages.pdf)**: An 11,000+ word seminal research paper including exhaustive spatiotemporal profiling, public health risk assessments, and counterfactual policy simulations.
- **Full Automated Pipeline**: 25+ Python scripts covering the entire lifecycle from raw API ingestion to professional PDF rendering.

---

## 🔬 Scientific Methodology & Parameters

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

### 4. Hybrid Modeling & Benchmarking (`src/dl_hybrid_modeling.py` & `src/model_benchmarking.py`)
- **Architecture**: **Hybrid CNN-LSTM**.
    - CNN: 1D-Convolutional layers for local spatial feature extraction from multi-sensor streams.
    - LSTM: 50 hidden units with ReLU activation for long-term temporal dependencies.
    - Window Size: 168 hours (7-day sliding lookback).
- **Benchmarks**: Compared against LightGBM, CatBoost, and RandomForest using `TimeSeriesSplit` cross-validation.
    - **SOTA Performance**: RandomForest achieved an **R² of 0.97** for high-resolution capturing.

### 5. Explainable AI (XAI) (`src/xai_interpretability.py`)
- **Mechanism**: SHAP (Shapley Additive Explanations) via `GradientExplainer`.
- **Insight**: Identification of non-linear "Tipping Points."
    - **Trigger**: Delhi industrial throttling recommended when Wind Speed < **10 km/h**.

### 6. Causal Policy Simulation (`src/policy_simulation.py`)
- **Framework**: EconML Double Machine Learning (DML).
- **Simulation**: Counterfactual evaluation of a hypothetical **20% reduction in vehicular emissions** (NO2 proxy).
- **Outcome**: Delhi PM2.5 baseline reduction of **6.86 ug/m³**, representing ~$1.2 billion in annual health-economic savings.

### 7. Public Health Integration (`src/health_impact_model.py`)
- **Standard**: WHO 2021 Concentration-Response Functions (CRF).
- **Metric**: Attributable Excess Mortality.
- **Key Finding**: In Kolkata, **2.5% of annual excess deaths** occur during just the top **1% of polluted hours** ("Super-Spreader" anthropogenic events).

---

## 📁 Repository Map
```text
/
├── README.md               # Detailed Project Documentation (Full Disclosure)
├── Final_Q1_Manuscript.pdf  # Final 30-Page Seminal Paper
├── .gitignore              # Configured for lightweight repo management
├── manuscript/             # Persistent Markdown source sections (1-6 + Supplementary)
├── src/                    # The SOTA Pipeline
│   ├── data_acquisition_v2.py      # Multi-source API Ingestion
│   ├── data_preprocessing_v3.py    # MICE Imputation & Feature Engineering
│   ├── regime_discovery.py         # GMM Clustering & Markov Transitions
│   ├── dl_hybrid_modeling.py       # CNN-LSTM Implementation
│   ├── policy_simulation.py        # DML Causal Evaluation (EconML)
│   ├── health_impact_model.py      # WHO CRF Epidemiological Metrics
│   ├── xai_interpretability.py     # SHAP Gradient Explanations
│   └── requirements.txt            # Dependency Freeze
├── plots/                  # Visual Evidence (Organized by methodology)
├── reports/                # Intermediary Benchmarking & Strategy Results
└── models/                 # (Ignored) Trained .keras model files
```

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

### 2. Execution Order
To reproduce the findings from scratch:
1. `python src/data_acquisition_v2.py` (Enter your OpenAQ API Key in config).
2. `python src/data_preprocessing_v3.py`.
3. `python src/regime_discovery.py`.
4. `python src/policy_simulation.py`.
5. `python src/health_impact_model.py`.
6. `python src/generate_pro_pdf.py` (To re-render the final manuscript).

## 👤 Author & Licensing
- **buddywhitman**
- This project is developed for undergraduate capstone excellence and Q1 academic submission.
