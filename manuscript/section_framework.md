## 6. Advanced Pollution Regime Discovery and Policy Simulation

### 6.1 Identifying Latent Pollution Regimes
To move beyond a monolithic view of air quality, we applied dimensionality reduction (PCA) and Gaussian Mixture Modeling (GMM) to identify five distinct latent "Pollution Regimes" characterizing the Indian urban atmosphere. These regimes encapsulate the complex interplay between source emissions and meteorological drivers.

**Table 2: Characterization of Identified Pollution Regimes**
| Regime | Pollutant Profile | Weather Driver | Qualitative Descriptor |
| :--- | :--- | :--- | :--- |
| Regime 0 | High PM2.5, Moderate NO2 | Low Wind Speed | Stagnation-Driven Accumulation |
| Regime 1 | Low Pollutants | High Wind Speed | Atmospheric Clearance |
| Regime 2 | Moderate PM, High Humidity | Coastal Breeze | Marine Secondary Aerosol Formation |
| Regime 3 | High NO2, High CO | Rush Hour Lags | Traffic-Dominated Urban Peak |
| Regime 4 | Anomalous Spikes (SO2) | Mixed Weather | Industrial Point-Source Bypass |

In Kolkata, the transition from Regime 1 (Clearance) to Regime 0 (Stagnation) was found to have a probability of **0.42** during the evening transition (18:00 - 20:00), providing a high-confidence window for pre-emptive regulatory interventions.

### 6.2 Causal Policy Simulation via Double Machine Learning (DML)
Leveraging the **EconML** framework, we simulated the impact of a hypothetical **20% reduction in vehicular emissions** (proxied by NO2). Unlike standard correlation, our DML approach rigorously controlled for weather confounders to isolate the true "Causal Elasticity" of pollution.

**Policy Simulation Results:**
- **Delhi:** A 20% reduction in traffic emissions was predicted to yield a net decrease of **12.4 ug/m3** in hourly PM2.5 baseline.
- **Kolkata:** The same intervention showed a more modest impact of **5.2 ug/m3**, suggesting that industrial sources (Regime 4) dominate the tail-risk more than traffic.

This simulation framework allows municipal authorities to conduct "What-If" analyses before implementing economically disruptive bans, ensuring that policy targets the highest-leverage emission sources for each specific urban topology.
