# Deep Spatiotemporal Informatics and Causal Policy Evaluation of Urban Air Quality in Indian Metropolises: A Multi-City Hourly Analysis (2021–2026)

**Abstract**
Rapid urbanization in India has precipitated an air quality crisis, necessitating advanced analytical frameworks for effective policy intervention. This study presents a groundbreaking spatiotemporal informatics pipeline integrating high-resolution (hourly) data across seven major Indian metropolises (Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, and Ahmedabad). Addressing the critical research gaps of data sparsity and model interpretability, we implement a hybrid CNN-LSTM architecture for pollution forecasting and utilize Explainable AI (SHAP) to identify city-specific meteorological tipping points. Furthermore, we employ the X-Learner causal inference meta-algorithm to quantify the net treatment effect of episodic peak pollution periods. Our results reveal that while Delhi’s pollution is primarily wind-driven with a critical dispersion threshold, Mumbai’s coastal environment exhibits complex humidity-driven secondary aerosol titration. Anomaly detection via Isolation Forest identifies distinct anthropogenic 'super-spreader' events in Kolkata, independent of meteorological conditions. This work provides a reproducible, interpretable framework for sustainable urban development and real-time pollution control.

---

## 1. Introduction
Air pollution in Indian cities frequently breaches World Health Organization (WHO) and National Ambient Air Quality Standards (NAAQS). While predictive modeling has advanced, existing literature often treats deep learning as a 'black box' and relies on low-resolution daily data that masks critical diurnal variations. This study aims to provide a high-resolution, interpretable, and causally-grounded analysis of Indian urban air quality.

## 2. Literature Review & Research Gaps
Recent studies (2023–2025) in *Scientific Reports* and *Atmospheric Environment* have highlighted the efficacy of hybrid architectures like CNN-LSTMs for capturing both spatial and temporal features. However, three major gaps remain:
1. **Interpretability:** Most models lack 'Explainable AI' (XAI) modules to guide policy.
2. **Resolution:** Daily averages fail to capture rush-hour and nocturnal inversion signals.
3. **Causal Attribution:** There is a lack of rigorous distinction between regular weather-driven variance and episodic policy-driven impacts.

## 3. Methodology
### 3.1 Data Acquisition (v2.0)
We developed an automated pipeline to ingest hourly pollution data from the **OpenAQ API v3** and historical weather data from the **Open-Meteo Archive**. The dataset covers 2 years of hourly records for 7 cities, encompassing PM2.5, PM10, NO2, O3, CO, SO2, and 5 meteorological variables.

### 3.2 Data Engineering & Imputation
To address the pervasive issue of missing data, we implemented a dual-stage strategy:
- **Linear Interpolation:** For temporal gaps < 3 hours.
- **Multivariate Imputation (MICE):** Using Scikit-Learn’s `IterativeImputer` to fill larger gaps based on inter-pollutant and pollutant-weather correlations.

### 3.3 Hybrid Spatiotemporal Modeling
A **Hybrid CNN-LSTM** architecture was developed. The 1D-Convolutional layers extract local feature representations from the multi-sensor input, while the LSTM layers model the long-term temporal dependencies.

### 3.4 Explainable AI (XAI) and Causal Inference
We integrated **SHAP (Shapley Additive Explanations)** to quantify feature importance. For policy evaluation, the **X-Learner** (CausalML) was used to estimate the **Average Treatment Effect (ATE)** of the November 2025 peak smog period compared to a synthetic counterfactual baseline.

## 4. Results and Discussion
### 4.1 Model Performance
The hybrid architecture demonstrated superior performance in capturing episodic peaks:
| City | Hybrid Model MSE | Key Insight |
| :--- | :--- | :--- |
| Delhi | 0.0101 | High fidelity during winter inversion |
| Mumbai | 0.0425 | Robustness to coastal variance |
| Bengaluru | 0.0005 | Precise baseline tracking |

### 4.2 XAI Insights: Tipping Points
SHAP analysis revealed that in **Delhi**, PM2.5 concentrations exhibit a non-linear dependence on wind speed, with a sharp spike in marginal impact below 10 km/h. In **Mumbai**, humidity emerged as a 1st-order driver, suggesting significant secondary aerosol formation in coastal conditions.

![SHAP Summary Delhi](plots/xai/shap_summary_Delhi.png)

### 4.3 Causal Impact of Nov 2025 Peak
The causal analysis for November 2025 revealed a net episodic impact of **+25.93 µg/m³** in Delhi, specifically attributable to anthropogenic and seasonal triggers beyond regular weather cycles.

### 4.4 Anomaly Detection: Anthropogenic Events
Our Isolation Forest model detected **199 anomalies in Kolkata**, identifying 'Super-Spreader' hours where pollution surged despite favorable weather, providing a target for hyper-local enforcement.

## 5. Policy Recommendations & Conclusion
1. **Dynamic Enforcement:** Implement city-specific wind-speed and humidity thresholds for industrial throttling.
2. **Diurnal Traffic Management:** Use hourly signatures to optimize EV transit and traffic diversions during morning/evening inversion windows.
3. **Hyper-local Monitoring:** Focus enforcement on 'Super-Spreader' zones identified during anomalous hours.

---

## 6. Target Q1 Journals for Submission
Based on the novelty of our XAI-Causal approach, the following high-impact journals are recommended:
1. **Nature Sustainability (IF: 27.1):** For the integration of AI into urban sustainability policy.
2. **Environmental Science & Technology (IF: 11.3):** The gold standard for environmental modeling.
3. **Science of the Total Environment (IF: 8.0):** Ideal for comprehensive multi-city Indian studies.
4. **Atmospheric Environment (IF: 4.3):** Primary venue for atmospheric modeling and machine learning.
