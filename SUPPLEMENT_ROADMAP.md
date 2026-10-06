# Authoritative Supplementary Material Roadmap

**Paper Title:** Reliable Causal Estimation under Latent Markov Confounding  
**Conference:** AISTATS 2027 (Under Review)  
**Author:** Anonymous Author(s)  
**Target Document Length:** Strictly 24 Pages (`paper/main.pdf`)

---

## 1. Document Structure & Page Allocation

The submitted manuscript adheres strictly to the conference page layout and numbering constraints:

| Section | Content | Page Range | Authoritative Target |
| :--- | :--- | :--- | :--- |
| **Title & Abstract** | Title, Abstract, Keywords | Page 1 | Page 1 |
| **Main Paper Body** | Sections 1 through 6 | Pages 1–8 | Ends strictly at bottom of Page 8 |
| **AI Statement & References** | AI Use Statement, Bibliographic Citations | Page 9 | Page 9 |
| **Reproducibility Checklist** | AISTATS Formal Checklist | Page 10 | Page 10 |
| **Mathematical Appendices** | Proofs of Theorems 1–6 (Sections A–F) | Pages 11–20 | Pages 11–20 |
| **Empirical Appendices** | Diagnostic Extensions, Latent Profiles (Section G) | Pages 20–21 | Pages 20–21 |
| **LatentRegimeBench & Zoo** | Benchmarks, Representation Zoo, 3 Decisive Experiments (Section H) | Pages 21–24 | Pages 21–24 |
| **Macro Financial Stress Test** | Financial transfer & 3-way decision regret (Section I) | Page 24 | Ends at bottom of Page 24 |

---

## 2. Theoretical Statements & Mathematical Proofs Mapping

All theoretical guarantees stated in the main paper possess complete, rigorous proofs in the appendix:

| Result | Main Text Location | Appendix Location | Key Technique / Bound |
| :--- | :--- | :--- | :--- |
| **Theorem 1** (Frisch-Waugh Singularity) | Section 2 (Page 4) | **Appendix A** (Page 11) | Omitted regime bias $\mathcal{B}_{\mathrm{FW}} = (\theta_0 - \theta_1) \cdot \frac{\Delta_T \rho_S \mathrm{Var}(S)}{1 - \rho_S^2}$ |
| **Theorem 2** (Graceful Degradation) | Section 4 (Page 5) | **Appendix B** (Page 13) | $\|\boldsymbol{\theta}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})}$ |
| **Theorem 3** (Spectral Bias-Variance) | Section 4 (Page 5) | **Appendix C** (Page 15) | $\mathrm{Var}(\hat{\boldsymbol{\theta}}_\lambda) \le \frac{\sigma^2}{(\lambda_{\min} + \lambda)^2}$, $\mathrm{Bias}^2 \le \lambda^2 \|\boldsymbol{\theta}^*\|_2^2$ |
| **Theorem 4** (Observable Entropy Bridge)| Section 4 (Page 6) | **Appendix D** (Page 16) | $\varepsilon_\gamma \le \sqrt{2 \ln 2 \cdot \bar{H}(\boldsymbol{\gamma})}$ via Pinsker's inequality and Gini impurity |
| **Theorem 5** (Uniform Asymptotic Normality)| Section 4 (Page 7) | **Appendix E** (Page 17) | $\sqrt{N}(\hat{\boldsymbol{\theta}}_\lambda - \boldsymbol{\theta}_\lambda) \xrightarrow{d} \mathcal{N}(0, \boldsymbol{V})$ under $\beta$-mixing and purged cross-fitting |
| **Theorem 6** (Long-Memory Risk Bound) | Section 4 (Page 7) | **Appendix F** (Page 19) | Multi-rate spectral decomposition separating nuisance, proxy, and spectral inflation |

---

## 3. Empirical Benchmark Suite & Artifact Provenance

Every empirical table and figure in the manuscript is generated deterministically by the underlying code and serialized in `reports/`:

| Manuscript Object | Physical Artifact | Generating Script | Core Finding |
| :--- | :--- | :--- | :--- |
| **Figure 1** (Causal Graph) | `paper/plots/fig1_bias_amplification.png` | TikZ / `fig1.py` | Structural DAG with unobserved Markov confounder $S_t$ |
| **Figure 2** (Singularity Frontier) | `paper/plots/fig2_difficulty_frontier.png` | `src/plot_figures.py` | Frisch-Waugh singularity as $\lambda_{\min}(\boldsymbol{J}) \to 0$ |
| **Figure 3** (Dynamic IRF) | `paper/plots/fig3_dynamic_irf.png` | `src/empirical_evaluation.py` | Regime-specific exposure trajectories across 4 megacities |
| **Figure 4** (LatentRegimeBench) | `paper/plots/fig4_latent_regime_bench.png` | `src/latent_regime_bench.py` | Latent recovery vs. downstream causal regret |
| **Figure 5** (Financial Transfer) | `paper/plots/fig5_financial_transfer.png` | `src/financial_regime_transfer.py` | Market coverage frontier and 98.74% decision regret reduction |
| **Table 1** (Difficulty Frontier) | Inline / `reports/latent_regime_bench_results.csv` | `src/latent_regime_bench.py` | 500 Monte Carlo runs over $(\varepsilon_\gamma, \lambda_{\min})$ grid |
| **Table 2** (Empirical Causal Effects)| `reports/empirical_or_dml_results.csv` | `src/empirical_evaluation.py` | Multi-season hourly estimates across Delhi, Mumbai, Bengaluru, Kolkata |
| **Table 3** (Meteorological Profiles)| `reports/meteorological_regime_profiles.csv` | `src/empirical_evaluation.py` | Physical cluster profiles (inversion vs. convective dispersion) |
| **Table 4** (Kolkata Regularization Grid)| `reports/kolkata_lambda_sensitivity.csv` | `src/or_dml_regularization_frontier.py`| 13-point regularization sweep under ill-conditioning ($\lambda_{\min}=0.0024$) |
| **Table 5** (Imputation Sensitivity) | `reports/data_imputation_sensitivity.csv` | `src/data_sensitivity.py` | Complete-case deletion vs. causal forward-fill invariance |
| **Table 6** (OR-DML Benchmark Summary)| `reports/or_dml_benchmark_summary.csv` | `src/latent_regime_bench.py` | Comparison across Standard DML, Block DML, and OR-DML |
| **Table 7** (LatentRegimeBench Results)| `reports/latent_regime_bench_results.csv` | `src/latent_regime_bench.py` | 5 diagnostic configurations ($N=1,200$, 30 seeds each) |
| **Table 8** (Zoo Shifts & Reliability)| `reports/representation_zoo_disentangled_shifts.csv` & `reports/representation_zoo_reliability_auc.csv` | `src/train_real_representation_zoo.py` | Panel A: Disentangled shifts; Panel B: Downstream reliability |
| **Table 9** (Multi-Horizon Forecasting)| `reports/multidomain_failure_forecasting.csv` | `src/train_real_representation_zoo.py` | Worst-decile downstream failure forecasting across $h \in [1, 24]$ |
| **Table 10** (Financial Risk Quartiles) | `reports/financial_regime_risk_quartiles.csv` | `src/financial_regime_transfer.py` | Dynamic risk quartiles on macroeconomic trading days ($N=1,500$) |
| **Table 11** (3-Way Decision Regret) | `reports/financial_abstention_policy.csv` | `src/financial_regime_transfer.py` | Adaptive risk policy slashing regret to 0.0025 at 99% coverage |

---

## 4. Verification and Reproducibility Protocol

To reproduce and verify every scientific claim, execute:

```bash
# 1. Verify 9/9 fundamental mathematical and scientific invariances:
python verify_science.py

# 2. Verify all 25+ artifact existence, float-exact agreement, and absence of placeholders:
python verify_artifacts.py

# 4. Execute the representation zoo calibration intervention:
python src/calibration_intervention_zoo.py

# 5. Execute hierarchical multi-world regression and LOWO cross-validation:
python src/hierarchical_reliability_regression.py

# 6. Execute empirical pre-treatment placebo checks:
python src/empirical_falsification_checks.py
```
