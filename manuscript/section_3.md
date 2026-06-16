## 3. Materials and Methods

### 3.1 Data Acquisition and High-Resolution Synthesis
This study utilizes a high-resolution, multi-source dataset synthesized from the **OpenAQ API v3** and the **Open-Meteo Historical Archive**. We targeted seven Indian metropolises (Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, and Ahmedabad), which collectively represent a population of over 100 million people and a diverse range of climatic zones.

### 3.2 Advanced Data Engineering and the "Anti-NaN" Protocol
Developing nation sensor networks are frequently plagued by data sparsity caused by power outages, sensor drift, and maintenance downtime. To transform this sparse data into a SOTA informatics product, we implemented a dual-stage imputation protocol.

#### 3.2.1 Stage 1 & 2: Interpolation and Multivariate Imputation (MICE/KNN)
For gaps less than or equal to 3 hours, we utilized linear interpolation to preserve local temporal momentum. For larger gaps, we utilized the `IterativeImputer` (MICE) and `KNNImputer`. MICE operates under the assumption that pollutants are not independent; for example, high PM2.5 levels are highly correlated with high PM10 and low wind speed. 

### 3.3 Modeling Architecture: From Hybrid CNN-LSTM to TFT
The core predictive engine of our informatics system initially deployed a **Hybrid Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)** architecture. However, to achieve absolute SOTA performance, we advanced the architecture to a **Stacking Ensemble** (LightGBM, CatBoost, RandomForest with Ridge Meta-Learner) and a **Temporal Fusion Transformer (TFT)** for uncertainty quantification.

#### 3.3.1 Spatial Feature Extraction (CNN Layer)
The 1D-CNN layer acts as a 'local motif detector'. It slides a series of 64 filters over a 3-hour window of the multivariate stream to identify short-term 'fronts' or 'spikes'.

#### 3.3.2 Temporal Memory (LSTM Layer)
The output of the CNN is passed into an LSTM layer with 50 units. The LSTM uses a gating mechanism to manage its internal memory cell, allowing it to 'remember' that a regional smog front identified 48 hours ago is still relevant. We used a **168-hour (7-day)** sliding window.

### 3.4 Unsupervised Regime Discovery (GMM & HDBSCAN)
To identify latent "Pollution Regimes", we reduced the dimensionality of the 9-variable pollutant-weather matrix using **Principal Component Analysis (PCA)**, retaining enough components to explain **95% of the total variance**. We then applied a **Gaussian Mixture Model (GMM)** with 5 components, validated by Bootstrap Stability Analysis (Adjusted Rand Index).

### 3.5 Causal Evaluation: Double Machine Learning (DML) and Causal Forests
To isolate the causal effect of emissions (Treatment $T$) on PM2.5 (Outcome $Y$), while controlling for weather (Confounders $X$), we utilized the **DML** and **Causal Forest** frameworks from `EconML`.
This provides the **Marginal Causal Elasticity**—exactly how much PM2.5 will drop for every 1 unit reduction in NO2 or SO2 emissions.

### 3.6 Constrained Policy Optimization
To move from causal estimation to formal policy recommendation, we utilized `cvxpy` to define a convex optimization problem. The objective was to minimize the total 'Economic Friction' (percentage reduction in sectors) subject to the constraint that the expected causal reduction in PM2.5 must meet WHO interim targets.

### 3.7 Public Health Epidemiology (Concentration-Response)
Finally, we applied the standard **Concentration-Response Function (CRF)** to quantify attributable mortality ($\Delta M$). We utilized a Relative Risk (RR) of **1.06** for every 10 µg/m³ increase in PM2.5, a baseline WHO guideline of **15 µg/m³**.
