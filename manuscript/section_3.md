## 3. Materials and Methods

### 3.1 Data Acquisition and High-Resolution Synthesis
This study utilizes a high-resolution, multi-source dataset synthesized from the **OpenAQ API v3** and the **Open-Meteo Historical Archive**. We targeted seven Indian metropolises (Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, and Ahmedabad), which collectively represent a population of over 100 million people and a diverse range of climatic zones (from tropical wet to semi-arid).

#### 3.1.1 Temporal Resolution Transition
The primary methodological innovation in our data acquisition was the transition from the standard daily ('D') frequency to an hourly ('h') frequency. This resulted in a dataset of approximately **17,520 observations per city** over the two-year focus period (2024-2026). This resolution is critical for capturing the diurnal atmospheric dynamics that daily averages mask.

### 3.2 Advanced Data Engineering and the "Anti-NaN" Protocol
Developing nation sensor networks are frequently plagued by data sparsity caused by power outages, sensor drift, and maintenance downtime. Initial auditing of the raw OpenAQ data revealed a missingness rate of up to **96%** in certain multi-pollutant combinations. To transform this sparse data into a SOTA informatics product, we implemented a dual-stage imputation protocol.

#### 3.2.1 Stage 1: Local Linear Interpolation
For gaps less than or equal to 3 hours, we utilized linear interpolation to preserve the local temporal momentum:
$$ \hat{x}_{t} = x_{t-1} + \frac{x_{t+k} - x_{t-1}}{k+1} $$
Where $k$ is the gap size. This handles minor sensor glitches without introducing significant synthetic bias.

#### 3.2.2 Stage 2: Multivariate Imputation by Chained Equations (MICE)
For larger gaps, we utilized the `IterativeImputer` (MICE). MICE operates under the assumption that pollutants are not independent; for example, high PM2.5 levels are highly correlated with high PM10 and low wind speed. 
The algorithm iteratively solves a series of regression models:
$$ X_j | X_{-j} \sim f(X_1, ..., X_{j-1}, X_{j+1}, ..., X_p) $$
We implemented this using a **Bayesian Ridge** estimator with 5-10 iterations per city, ensuring that the local chemical and meteorological correlations of each metropolis were preserved in the imputed values.

### 3.3 The Hybrid CNN-LSTM Architecture
The core predictive engine of our informatics system is a **Hybrid Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)** architecture.

#### 3.3.1 Spatial Feature Extraction (CNN Layer)
The 1D-CNN layer acts as a 'local motif detector'. It slides a series of 64 filters over a 3-hour window of the multivariate stream.
The convolution operation for a filter $w$ is defined as:
$$ z_{i} = \sigma(w \cdot x_{i:i+k-1} + b) $$
Where $x$ is the input vector of pollutants and weather variables, $k$ is the kernel size (3), and $\sigma$ is the ReLU activation function. This identifies short-term 'fronts' or 'spikes' in the data.

#### 3.3.2 Temporal Memory (LSTM Layer)
The output of the CNN is passed into an LSTM layer with 50 units. The LSTM uses a gating mechanism (Input, Forget, and Output gates) to manage its internal memory cell ($c_t$), allowing it to 'remember' that a regional smog front identified 48 hours ago is still relevant to the current prediction.
**Mathematical Gates:**
- Forget Gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- Input Gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- Output Gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

We used a **168-hour (7-day)** sliding window, providing the model with a comprehensive weekly context of atmospheric history.

### 3.4 Unsupervised Regime Discovery (GMM & PCA)
To identify latent "Pollution Regimes", we reduced the dimensionality of the 9-variable pollutant-weather matrix using **Principal Component Analysis (PCA)**, retaining enough components to explain **95% of the total variance**.

We then applied a **Gaussian Mixture Model (GMM)** with 5 components. Unlike K-Means, which assumes spherical clusters, GMM allows for ellipsoidal clusters with different covariances, which more accurately reflects the skewed distributions of environmental data.
The probability of a data point $x$ belonging to regime $k$ is:
$$ P(x) = \sum_{k=1}^{5} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) $$
Where $\pi_k$ is the mixing weight, $\mu_k$ is the mean pollutant profile, and $\Sigma_k$ is the covariance matrix of the regime.

### 3.5 Causal Evaluation: Double Machine Learning (DML)
To isolate the causal effect of emissions (Treatment $T$) on PM2.5 (Outcome $Y$), while controlling for weather (Confounders $X$), we utilized the **DML** framework from `EconML`.
DML uses a 'Residual-on-Residual' approach:
1.  **Residualize Y:** Train a model to predict $Y$ from $X$ and calculate $Y_{res} = Y - \hat{Y}(X)$. This removes the weather-driven part of the pollution.
2.  **Residualize T:** Train a model to predict $T$ from $X$ and calculate $T_{res} = T - \hat{T}(X)$. This identifies the 'anomalous' emission signal.
3.  **Causal Fit:** Regress $Y_{res}$ on $T_{res}$ to find the true causal parameter $\theta$:
$$ Y_{res} = \theta(X) T_{res} + \epsilon $$
This provides the **Marginal Causal Elasticity**—exactly how much PM2.5 will drop for every 1 unit reduction in NO2 or SO2 emissions.

### 3.6 Public Health Epidemiology (Concentration-Response)
Finally, we applied the standard **Concentration-Response Function (CRF)** to quantify attributable mortality ($\Delta M$):
$$ \Delta M = Y_0 \times \left(1 - e^{-\beta \cdot \Delta X}\right) \times Pop $$
We utilized a Relative Risk (RR) of **1.06** for every 10 µg/m³ increase in PM2.5, a baseline WHO guideline of **15 µg/m³**, and the most recent municipal census data for the seven target cities.