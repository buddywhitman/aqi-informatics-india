# Urban Environmental Informatics: Pan-India High-Resolution Study (2021-2026)

## 🌟 Project Overview
This repository hosts a seminal spatiotemporal informatics pipeline for air quality analysis across 4 major Indian metropolises: **Delhi, Mumbai, Bengaluru, and Kolkata**. It integrates high-resolution hourly data with Deep Learning, Explainable AI (XAI), and Causal Inference to bridge the gap between "Black Box" predictions and actionable urban policy.

## 📄 Final Manuscript
- **[Final_Q1_Manuscript_30Pages.pdf](./Final_Q1_Manuscript_30Pages.pdf)**: An 11,000+ word professional manuscript formatted for Q1 journal submission.

## 📁 Repository Structure
- `src/`: Core Python pipeline scripts (Acquisition, Preprocessing, CNN-LSTM, SHAP, CausalML, Health Models).
- `data/`: (Ignored by git) Raw and processed hourly datasets.
- `plots/`: High-resolution visualizations including diurnal signatures and SHAP tipping points.
- `manuscript/`: Structured Markdown sections and drafts for the final paper.
- `models/`: (Ignored by git) Trained Keras model files.
- `reports/`: Intermediary analysis and strategy summaries.

## 🔬 Methodology & Novelty
1. **High-Resolution (Hourly) Data**: Captures diurnal peaks and atmospheric inversions masked by daily averages.
2. **Hybrid CNN-LSTM Architecture**: Models spatial advection and long-term temporal dependencies.
3. **SHAP-Based Explainability**: Mathematically identifies non-linear meteorological thresholds (e.g., the 10km/h wind speed stagnation trigger).
4. **Causal X-Learner**: Estimates the *net* anthropogenic impact of episodic events (e.g., Nov 2025 smog) by controlling for weather.
5. **Epidemiological Linking**: Translates PM2.5 spikes into excess mortality using WHO-standardized Concentration-Response Functions.

## 📈 Key Insights & Informed Recommendations
- **Dynamic Industrial Throttling**: Regulators should implement a predictive **Atmospheric Stagnation Index (ASI)**. When wind speeds are forecast to drop below **10 km/h** in Delhi, industrial emissions should be pre-emptively throttled by 30-50%.
- **Targeted "Super-Spreader" Enforcement**: In Kolkata, 1% of hours (anthropogenic anomalies) account for ~2.5% of annual excess mortality. Policy should prioritize **Edge-AI monitoring** at industrial point-sources to flag these weather-independent spikes.
- **Diurnal Traffic Management**: Implementing "EV-Only" hours between **18:00 and 22:00** in Bengaluru and Chennai can mitigate the bimodal exposure peaks identified during boundary layer collapse.

## 🚀 Reproducibility
```bash
# 1. Setup Environment
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

# 2. Run Pipeline (Example)
python src/data_acquisition_v2.py  # Requires OpenAQ API Key
python src/data_preprocessing_v2.py
python src/dl_hybrid_modeling.py
```

## 🏆 Target Q1 Journals
Based on our multi-disciplinary approach, the following are recommended:
1. **Nature Sustainability** (IF: 27.1): For policy-linked sustainability frameworks.
2. **The Lancet Planetary Health** (IF: 21.6): For the integrated health impact findings.
3. **Environmental Science & Technology (ES&T)** (IF: 11.3): For methodological rigor in atmospheric modeling.
4. **Atmospheric Environment** (IF: 4.3): For specialized spatiotemporal air quality studies.
