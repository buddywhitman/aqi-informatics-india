## 3. Materials and Methods

### 3.1 Data Acquisition and High-Resolution Synthesis
This study utilizes a high-resolution, multi-source dataset synthesized from the **OpenAQ API v3** (aggregating CPCB and SAFAR ground sensors) and the **Open-Meteo Historical Archive**. We targeted seven Indian metropolises: Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, and Ahmedabad. These cities were selected to represent a diverse array of climatic zones (from tropical wet and dry, to semi-arid), topographies (coastal, deltaic, plateau, and landlocked plains), and primary economic drivers.

#### 3.1.1 Temporal Resolution Transition
The primary methodological innovation in our data acquisition was the transition from the standard daily ('D') frequency to an hourly ('h') frequency. This resulted in a dataset of approximately **17,520 observations per city** over the two-year focus period (2024-2026). This resolution is critical for capturing the diurnal atmospheric dynamics that daily averages fundamentally mask.

### 3.2 Advanced Data Engineering and the "Anti-NaN" Protocol
Developing nation sensor networks are frequently plagued by data sparsity caused by power outages, sensor drift, and maintenance downtime. Initial auditing of the raw OpenAQ data revealed significant missingness in certain multi-pollutant combinations. To transform this sparse data into a SOTA informatics product, we implemented a highly robust, dual-stage imputation protocol.

#### 3.2.1 Stage 1 & 2: Interpolation and Multivariate Imputation
For gaps less than or equal to 3 hours, we utilized linear interpolation to preserve local temporal momentum. For larger gaps, we utilized the `IterativeImputer` (Multivariate Imputation by Chained Equations - MICE). MICE operates under the assumption that pollutants are not independent; for example, high PM2.5 levels are highly correlated with high PM10 and low wind speed. The algorithm iteratively solves a series of Bayesian Ridge regression models, ensuring that the local chemical and meteorological correlations of each specific metropolis were preserved in the imputed values.

### 3.3 Unsupervised Regime Discovery (PCA & GMM)
To identify latent "Pollution Regimes", we reduced the dimensionality of the 9-variable pollutant-weather matrix using **Principal Component Analysis (PCA)**, retaining enough components to explain **95% of the total variance**. 

We then applied a **Gaussian Mixture Model (GMM)** with 5 components. Unlike K-Means, which assumes spherical clusters, GMM allows for ellipsoidal clusters with different covariances, which more accurately reflects the skewed distributions of environmental data. The regimes were validated using the **Silhouette Score** and **Davies-Bouldin Index**, and their structural integrity was confirmed via **Bootstrap Stability Analysis** utilizing the Adjusted Rand Index. Finally, we modeled the transitions between these regimes using first-order Markov Chains to calculate the probability of atmospheric state shifts.

### 3.4 Modeling Architecture: Stacking Ensembles and TFT
To predict PM2.5 concentrations at high resolution, we deployed two distinct State-of-the-Art architectures.

#### 3.4.1 Weighted Stacking Ensemble
Our primary predictive engine is a Stacking Ensemble. Layer 1 consists of diverse, highly tuned gradient boosting algorithms: LightGBM (optimized for leaf-wise growth and speed), CatBoost (optimized for categorical handling and oblivious trees), and RandomForest (providing robust variance reduction). The outputs of these models are fed into a Layer 2 Ridge Regression Meta-Learner, which intelligently weights the base models to prevent overfitting and maximize generalization across the diverse Indian climate.

#### 3.4.2 Temporal Fusion Transformer (TFT)
To provide deep temporal memory and, crucially, uncertainty quantification, we implemented a Temporal Fusion Transformer. The TFT utilizes multi-head attention to weigh the importance of historical lags (up to 168 hours, or 7 days) and employs Quantile Loss functions (P10, P50, P90) to generate prediction intervals, allowing policymakers to understand the confidence bounds of the forecast.

### 3.5 Causal Evaluation: Double Machine Learning (DML)
To isolate the causal effect of specific emissions (Treatment $T$) on PM2.5 (Outcome $Y$), while rigorously controlling for weather (Confounders $X$), we utilized the **DML** framework and **Causal Forests** from the `EconML` library.
DML uses a 'Residual-on-Residual' approach:
1.  **Residualize Y:** Train a model to predict $Y$ from $X$ and calculate the residual. This removes the weather-driven part of the pollution.
2.  **Residualize T:** Train a model to predict $T$ from $X$ and calculate the residual. This identifies the 'anomalous' emission signal.
3.  **Causal Fit:** Regress the residuals to find the true Marginal Causal Elasticity—exactly how much PM2.5 will drop for every 1 unit reduction in NO2 (traffic proxy) or SO2 (industrial proxy).

### 3.6 Constrained Policy Optimization
To move from causal estimation to formal policy recommendation, we utilized `cvxpy` to define a convex optimization problem. The objective was to minimize the total 'Economic Friction' (represented by the percentage reduction required in various sectors) subject to the strict constraint that the expected causal reduction in PM2.5 must meet specific, aggressive health targets.

### 3.7 Public Health Epidemiology (Concentration-Response)
Finally, we applied the standard **Concentration-Response Function (CRF)** to quantify attributable mortality. We utilized a Relative Risk (RR) of **1.06** for every 10 µg/m³ increase in PM2.5 above a baseline WHO guideline of **15 µg/m³**, integrating this with municipal census data for the seven target cities to calculate the economic Value of a Statistical Life (VSL).
