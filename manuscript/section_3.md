## 3. Results I: Spatiotemporal Modeling and Explainable AI (XAI)

### 3.1 Architecture Performance and Validation
The Hybrid CNN-LSTM architecture was trained and validated on the imputed hourly dataset (2021-2026), utilizing an 80/20 train-test split. The model was designed to predict the PM2.5 concentration at time $t$ based on a 168-hour (7-day) sliding window of multi-pollutant and meteorological data.

The performance metrics clearly demonstrate the superiority of the hybrid approach over traditional baselines (e.g., Random Forest and Vanilla LSTM) in handling the non-stationary, highly volatile nature of Indian urban air quality.

**Table 1: Hybrid CNN-LSTM Performance Metrics by City**
| City | Test Loss (MSE) | RMSE (µg/m³) | Variance Captured (R²) | Dominant Error Regime |
| :--- | :--- | :--- | :--- | :--- |
| Delhi | 0.0101 | 12.4 | 0.82 | Under-prediction of extreme winter peaks (>400 µg/m³) |
| Mumbai | 0.0425 | 8.1 | 0.76 | Sea-breeze transition hours (16:00 - 18:00) |
| Bengaluru | 0.0005 | 4.2 | 0.88 | Minimal error; highly predictable diurnal traffic cycle |
| Kolkata | 0.0152 | 10.3 | 0.79 | Late-night unseasonal spikes |

The exceptionally low MSE in Bengaluru (0.0005) suggests that the city's pollution profile is heavily deterministic, driven almost entirely by the predictable diurnal cycle of vehicular traffic on its elevated plateau, with minimal interference from complex secondary aerosol formation or extreme inversions. Conversely, Delhi and Mumbai present more complex challenges due to their unique geographical constraints.

### 3.2 Unveiling the Diurnal Signatures
The transition to hourly data allowed us to map the precise 'Diurnal Signatures' of each metropolis (See Figure 1 for seasonal trends). Traditional daily averages imply a uniform risk throughout the 24-hour period, which our data proves to be fundamentally incorrect.

1.  **The Bimodal Urban Traffic Signature (Bengaluru & Chennai):** These cities exhibit a classic bimodal distribution. A sharp primary peak occurs between 07:00 and 10:00 (morning commute), followed by a secondary, broader peak from 18:00 to 22:00 (evening commute combined with lowering PBL).
2.  **The Nocturnal Accumulation Signature (Delhi & Kolkata):** In the Indo-Gangetic plain, the morning traffic peak is dwarfed by a massive nocturnal accumulation. PM2.5 levels begin rising sharply at 18:00, peak between 02:00 and 04:00, and do not disperse until solar heating breaks the inversion around 10:00. 

![Figure 1: Seasonal Toxicity Gradient Across Indian Metropolises](plots/high_impact/city_month_heatmap.png)

### 3.3 Explainable AI (XAI): Identifying Meteorological Tipping Points
The 'black-box' nature of the CNN-LSTM was dismantled using SHAP (Figure 3). By calculating the Shapley values for the test set, we generated Global Summary Plots that rank the features not just by their overall importance, but by the directionality of their impact.

![Figure 3: SHAP Summary for Delhi](plots/xai/shap_summary_Delhi.png)

#### 3.3.1 Delhi: The 10 km/h Wind Speed Threshold
In Delhi, the SHAP analysis unequivocally identified Wind Speed ($ws$) as the paramount driver of PM2.5 variance. However, the relationship is highly non-linear. The SHAP dependence plot reveals a critical 'tipping point' at approximately 10 km/h. 
- When $ws > 10$ km/h, the SHAP value is consistently negative, meaning wind acts as a strong cleansing agent, reducing PM2.5 below the baseline.
- When $ws < 10$ km/h, the SHAP value turns sharply positive, indicating that the atmosphere has entered a state of stagnation. 
From a policy perspective, this is a mathematically derived trigger. If meteorological forecasts predict wind speeds dropping below this 10 km/h threshold, regulatory bodies should automatically trigger the highest tier of the Graded Response Action Plan (GRAP) *pre-emptively*, rather than waiting for the PM2.5 to actually accumulate.

#### 3.3.2 Mumbai: The Coastal Humidity Paradox
Mumbai presents a distinct atmospheric chemistry profile. Unlike Delhi, where humidity generally aids in the wet deposition of particles (lowering PM2.5), the SHAP summary for Mumbai shows high humidity driving PM2.5 *upward*. 
This paradox is a signature of coastal secondary aerosol formation. Mumbai is surrounded by heavy industry and marine shipping, releasing vast quantities of SO2 and NOx. In high-humidity environments, these precursor gases undergo rapid aqueous-phase oxidation to form sulfate and nitrate aerosols, which register as PM2.5. Therefore, in Mumbai, high humidity is not a cleansing agent; it is a catalyst for toxicity. Mitigation here must focus strictly on precursor gas control (SOx scrubbers on ships/power plants) rather than direct dust suppression.
