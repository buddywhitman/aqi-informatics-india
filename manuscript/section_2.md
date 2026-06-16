## 2. Literature Review: The SOTA Frontier in Environmental Informatics

### 2.1 Evolution of Predictive Paradigms (2021-2025)
The landscape of air quality forecasting has transitioned through three distinct technological epochs over the last half-decade. The 'Classical Period' (pre-2021) was dominated by linear statistical models such as ARIMA (Auto-Regressive Integrated Moving Average) and land-use regression (LUR). The second epoch, the 'Deep Learning Boom' (2021-2023), saw the widespread adoption of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) cells. 

We are currently entering the third epoch: 'Regime-Aware Causal Informatics'. SOTA research in 2024 and 2025 emphasizes the integration of physical constraints and explainable modules into deep architectures, moving toward Transformer-based models (TFTs) that inherently quantify uncertainty. This study builds on this epochal shift by introducing a framework that is predictive, causally evaluative, and optimized for policy.

### 2.2 Spatial Dependencies and the CNN-LSTM Hybrid
Air quality is a spatiotemporal field. A pollutant emitted at an industrial source in Ahmedabad does not stay at the point of origin; it is a vector quantity transported by the prevailing wind regime. Convolutional Neural Networks (CNNs), traditionally utilized for computer vision, can be repurposed as high-dimensional spatial feature extractors. By treating a network of monitoring stations as a 1D or 2D grid, CNN layers can identify 'Spatial Signatures'—patterns of rising CO or NO2 that precede a regional smog front. 

### 2.3 The Interpretability Crisis and the Rise of XAI
Explainable AI (XAI) represents the technological bridge across the 'Black Box' gap. SHAP (Shapley Additive Explanations), grounded in cooperative game theory, has emerged as the most mathematically rigorous XAI framework. Unlike LIME, SHAP satisfies the 'Axioms of Fairness'—including consistency and local accuracy. Recent studies have used SHAP to identify the 'importance' of pollutants, but very few have utilized it to derive 'Tipping Points' for policy. Our work fills this gap by using SHAP dependence analysis to identify the exact meteorological thresholds where the atmospheric risk profile shifts from 'Stable' to 'Severe' for each of the seven cities in our study.

### 2.4 Causal Inference: The Next Frontier
The final, and perhaps most critical, gap in current environmental literature is the lack of rigorous causal attribution. To achieve SOTA status, research must move toward 'Counterfactual Reasoning'. The use of Double Machine Learning (DML) and Causal Forests (EconML) allows researchers to estimate the 'Average Treatment Effect' (ATE) of a policy by training models to 'imagine' the world without the policy, given the same weather conditions. 

### 2.5 Summary of Research Gaps
In conclusion, while predictive accuracy is high, the literature currently lacks Regime-Awareness, Explicit Tipping Point Logic, Net Anthropogenic Attribution, and Integrated Health-Economic Feedback. Our framework is explicitly designed to close these four gaps simultaneously across a diverse cross-section of Indian geography.
