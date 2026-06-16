# Framework Implementation Plan: Pollution Regime Discovery & Causal Simulation

## Objective
Implement the "Pollution Regime Discovery Framework" to identify latent pollution states, model their transitions, and simulate policy interventions across 7 Indian cities.

## Phase 1: Advanced Feature Engineering
- **New Features:** PM2.5/PM10 ratio, NO2/CO ratio, O3/NO2 ratio.
- **Temporal Lags:** 1h, 3h, 6h, 24h for all key pollutants and weather variables.
- **Rolling Stats:** 3h and 24h rolling mean, std, and max.
- **Data Quality:** Flagging actual vs. imputed values.

## Phase 2: Pollution Regime Discovery (Clustering)
- **Dimensionality Reduction:** PCA (95% variance).
- **Clustering:** HDBSCAN and GMM.
- **Regime Labeling:** Traffic-dominated, Industrial, Stagnation, Dust, Mixed Severe.
- **Validation:** Silhouette Score, Davies–Bouldin.

## Phase 3: Markov Regime Transition Analysis
- **Sequence Mapping:** Convert hourly data to regime sequences.
- **Transition Matrices:** Calculate probabilities (e.g., P(Traffic -> Stagnation)).
- **Persistence:** Quantify how long each regime typically lasts per city.

## Phase 4: Enhanced Predictive Benchmarking
- **Models:** LightGBM, CatBoost, Temporal Fusion Transformer (TFT).
- **Evaluation:** 5-fold CV with RMSE, MAE, R², MAPE.
- **Explainability:** Global and Local SHAP on the best model.

## Phase 5: Causal Inference & Policy Simulation
- **Algorithm:** Double Machine Learning (DML) and Causal Forests (EconML).
- **Treatments:** NO2 (Traffic proxy), SO2 (Industrial proxy).
- **Simulation:** "What-if" scenarios (e.g., -30% industrial emissions).
- **Outcomes:** Predicted AQI reduction, Health Benefit (YLL), Economic Benefit ($).

## Phase 6: Automated Manuscript Update
- Incorporate all new results into the 30-page manuscript.
- Add specific sections for "Regime Transition Probabilities" and "Policy Counterfactuals."

## Technical Dependencies
- `hdbscan`, `gmm`, `econml`, `lightgbm`, `catboost`, `pytorch-forecasting` (for TFT).
