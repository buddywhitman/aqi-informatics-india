## 2. Literature Review: The SOTA Frontier in Environmental Informatics

### 2.1 Evolution of Predictive Paradigms (2021-2025)
The landscape of air quality forecasting has transitioned through three distinct technological epochs over the last half-decade. The 'Classical Period' (pre-2021) was dominated by linear statistical models such as ARIMA (Auto-Regressive Integrated Moving Average) and land-use regression (LUR). While these models provided foundational insights into the steady-state concentrations of pollutants, they were fundamentally unable to capture the non-linear, stochastic spikes characteristic of rapid urban transitions in the Global South.

The second epoch, the 'Deep Learning Boom' (2021-2023), saw the widespread adoption of Recurrent Neural Networks (RNNs) and particularly Long Short-Term Memory (LSTM) cells. Researchers like Barthwal & Goel (2024) demonstrated that LSTMs could successfully mitigate the 'vanishing gradient' problem, allowing for the modeling of temporal autocorrelation in pollution data. However, as noted by Pande et al. (2024), these models often treated the monitoring stations as isolated points in time, neglecting the inherently spatial nature of atmospheric advection.

We are currently entering the third epoch: 'Regime-Aware Causal Informatics'. SOTA research in 2024 and 2025 emphasizes the integration of physical constraints and explainable modules into deep architectures. Models like AirSense-X (2025) have pioneered the use of Physics-Informed Neural Networks (PINNs), where atmospheric dispersion equations are integrated directly into the model's loss function. This study builds on this epochal shift by introducing a framework that is not just predictive, but causally evaluative.

### 2.2 Spatial Dependencies and the CNN-LSTM Hybrid
Air quality is a spatiotemporal field. A pollutant emitted at an industrial source in Ahmedabad does not stay at the point of origin; it is a vector quantity transported by the prevailing wind regime. Recent literature has highlighted that standalone temporal models suffer from 'Advection Blindness'. To counteract this, hybrid architectures have emerged as the gold standard. 

Aruna Rani & Sampathkumar (2025) demonstrated that Convolutional Neural Networks (CNNs), traditionally utilized for computer vision, can be repurposed as high-dimensional spatial feature extractors. By treating a network of monitoring stations as a 1D or 2D grid, CNN layers can identify 'Spatial Signatures'—patterns of rising CO or NO2 that precede a regional smog front. Our study utilizes this CNN-LSTM synergy to create a model that 'looks' at neighboring city signals to predict local spikes, a significant advance over the single-station models that still dominate much of the Indian literature.

### 2.3 The Interpretability Crisis and the Rise of XAI
As models become more complex, their adoption by regulatory bodies like the Central Pollution Control Board (CPCB) has slowed due to a lack of transparency. The 'Interpretability Crisis' is a major research gap identified in multiple 2025 Q1 review papers. Explainable AI (XAI) represents the technological bridge across this gap. 

SHAP (Shapley Additive Explanations), grounded in cooperative game theory, has emerged as the most mathematically rigorous XAI framework. Unlike LIME (Local Interpretable Model-agnostic Explanations), which provides locally faithful approximations, SHAP satisfies the 'Axioms of Fairness'—including consistency and local accuracy. Recent studies have used SHAP to identify the 'importance' of pollutants, but very few have utilized it to derive 'Tipping Points' for policy. Our work fills this gap by using SHAP dependence analysis to identify the exact meteorological thresholds (e.g., specific wind speeds or humidity levels) where the atmospheric risk profile shifts from 'Stable' to 'Severe'.

### 2.4 Causal Inference: The Next Frontier
The final, and perhaps most critical, gap in current environmental literature is the lack of rigorous causal attribution. Most researchers report that "AQI dropped by 20% during the firecracker ban," but as Pearl et al. (2023) argue, such statements are observational, not causal. Natural weather variance often accounts for more variance in pollution than any single policy intervention. 

To achieve SOTA status, research must move toward 'Counterfactual Reasoning'. The use of Double Machine Learning (DML) and Causal Forests (EconML) allows researchers to estimate the 'Average Treatment Effect' (ATE) of a policy by training models to 'imagine' the world without the policy, given the same weather conditions. This methodology is virtually non-existent in Indian air quality studies published before 2024. This research represents one of the first applications of the X-Learner meta-algorithm to quantify the net efficacy of the North Indian winter mitigation plans, providing a peer-reviewable basis for multi-billion dollar urban infrastructure decisions.

### 2.5 Summary of Research Gaps
In conclusion, while predictive accuracy is high, the literature currently lacks:
1.  **Regime-Awareness:** Models do not adapt to different atmospheric states (Stagnation vs. Clearance).
2.  **Explicit Tipping Point Logic:** Models do not provide actionable 'if-then' triggers for regulators.
3.  **Net Anthropogenic Attribution:** Research fails to rigorously separate human impact from meteorological noise.
4.  **Integrated Health-Economic Feedback:** There is a disconnect between model output and the economic value of lives saved.

Our framework is explicitly designed to close these four gaps simultaneously.
