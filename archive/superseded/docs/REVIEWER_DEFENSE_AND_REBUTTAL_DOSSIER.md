# AISTATS 2027 Reviewer Defense & Author Rebuttal Dossier

**Manuscript Title**: *Reliable Causal Estimation under Latent Markov Confounding*  
**Tracking Branch**: `feature/or-dml-phase2-pivot`  
**Purpose**: Pre-computed, mathematically anchored, and empirically referenced defenses for anticipated reviewer objections during the AISTATS 2027 peer-review cycle.

---

## 🎯 Strategic Framing & Mindset

At elite machine learning venues (AISTATS, NeurIPS, ICML), competitive papers do not survive by asserting that a proposed method dominates every baseline on every synthetic benchmark. They succeed by:
1. Identifying a fundamental structural problem that current methods mishandle (latent state uncertainty interacting multiplicatively with downstream task conditioning).
2. Providing rigorous, finite-sample theoretical characterizations (Theorems 1, 3, 4, 5).
3. Exhibiting complete scientific transparency regarding failure modes, trade-offs, and falsification tests.
4. Delivering flawless, byte-for-byte reproducible artifacts.

Below are the 5 hardest objections that seasoned Area Chairs and reviewers will raise, along with the authoritative mathematical and empirical refutations.

---

## 🛡️ Objection 1: Soft OR-DML vs. Hard Regime FE Bias Trade-Off

### Reviewer Formulation
> *"On continuous mixture benchmarks and synthetic DGPs, naive hard clustering (`Regime FE DML`) achieves equal or lower finite-sample point bias than Soft OR-DML (e.g. bias $2.81$ vs. $3.71$). Why should practitioners adopt Soft OR-DML when naive hard assignment is simpler and performs better in point estimation?"*

### Core Mathematical Defense
1. **The Multicollinearity Origin**:
   In soft coupled estimation (OR-DML), posterior beliefs $\gamma_{tk} \in (0, 1)$ induce off-diagonal Gram cross-terms:
   $$J_{01} = \frac{1}{N}\sum_{t=1}^N \gamma_{t0}\gamma_{t1}\tilde{T}_{t0}\tilde{T}_{t1} > 0$$
   When regimes have large baseline outcome/treatment shifts ($g_1 - g_0 = 30$, $b_1 - b_0 = 4$), slight posterior misclassification leaks a massive product ($164.0$) into the score vector $\boldsymbol{S}$. Hard clustering sets $\hat{S}_t \in \{0, 1\}$, forcing $\gamma_{t0}\gamma_{t1} \equiv 0$ and diagonalizing $\boldsymbol{J}$ by decree, which prevents cross-talk between regime scores.
2. **The Fatal Cost of Hard Clustering**:
   - **Destruction of Asymptotic Distribution Theory**: Hard clustering turns continuous beliefs into discontinuous indicator steps. Under Markov temporal dependence, standard M-estimation asymptotics and sandwich variance estimators collapse; hard FE provides **no valid standard errors or confidence intervals**.
   - **Inference Failure**: In Table 1, Hard FE achieves 0% valid coverage when uncertainty is unaccounted for, whereas OR-DML yields valid delta-method sandwich coverage ($92.0\%/97.0\%$, matching the Oracle).
   - **Inability to Screen for Failure**: Hard FE completely discards belief uncertainty, making it impossible to compute posterior entropy $H_t$ or the difficulty index $\mathcal{D}_t$. It cannot decide when to abstain.
3. **Resolution**:
   OR-DML preserves probabilistic belief uncertainty and valid asymptotic inference, resolving Gram ill-conditioning through spectral shrinkage $\lambda > 0$ (Theorem 4).

### 200-Word Rebuttal Snippet
> "We thank the reviewer for highlighting this critical comparison. The difference in point bias stems directly from Gram matrix geometry: soft posterior weights $\gamma_{tk} \in (0,1)$ generate positive off-diagonal Gram terms $J_{01} = \frac{1}{N}\sum_t \gamma_{t0}\gamma_{t1}\tilde{T}_{t0}\tilde{T}_{t1}$, leaking residual treatment variance across regimes under baseline shifts. Hard clustering sets $\hat{S}_t \in \{0,1\}$, zeroing cross-terms by decree. 
> 
> However, hard clustering incurs severe inferential costs: it breaks continuous semiparametric regularity under dependence, invalidating sandwich covariance and destroying confidence interval coverage. In contrast, OR-DML preserves full posterior beliefs and admits valid delta-method HAC sandwich inference ($92.0\%/97.0\%$ empirical coverage, Table 1). Furthermore, hard clustering cannot evaluate posterior entropy $H_t$ or compute the operational difficulty index $\mathcal{D}_t = H_t/\lambda_{\min}(\boldsymbol{J}_t)$, blinding practitioners to when estimation is compromised. Soft OR-DML addresses ill-conditioning through principled spectral shrinkage $\lambda > 0$ (Theorem 4), balancing finite-sample bias against bounded asymptotic variance."

---

## 🛡️ Objection 2: Universal Failure under Severe Overlap ($\Delta_Z \le 1.0$)

### Reviewer Formulation
> *"Table 1 shows that for $\Delta_Z \le 0.5$, all methods fail, with biases exceeding $6.5$. Even at $\Delta_Z = 1.0$, all methods suffer substantial bias ($\ge 2.64$). Doesn't this mean OR-DML fails precisely in the regime overlap setting it claims to address?"*

### Core Mathematical Defense
1. **Theorem 1 (Non-Identification under Unconstrained Overlap)**:
   Theorem 1 establishes that when proxies provide insufficient distributional separation between regimes, regime-specific causal effects are **fundamentally non-identified from observables**. No estimator can point-identify the effects without additional untestable assumptions; claiming low bias at $\Delta_Z \le 0.5$ would be mathematically fraudulent.
2. **The Real Value Proposition: Selective Abstention**:
   OR-DML does not claim to manufacture information where none exists. Instead, it provides the **computable difficulty metric** $\mathcal{D}_{\mathrm{operational}}(t) = H(\boldsymbol{\gamma}_t)/\lambda_{\min}(\boldsymbol{J}_t)$ (Proposition 1).
   - As shown in Frontier A ([`reports/frontier_a_selective_estimation_summary.csv`](file:///D:/OneDrive/Documents/repos/aqi-informatics-india/reports/frontier_a_selective_estimation_summary.csv)), abstaining on high-difficulty periods drops causal error by **$23.8\times$** ($3.286$ at 100% coverage $\to 0.138$ at 50% coverage).
   - Standard DML and hard clustering have no such diagnostic and blindly output biased estimates with $0\%$ coverage.

### 200-Word Rebuttal Snippet
> "The reviewer makes a profound observation that directly validates Theorem 1. When $\Delta_Z \le 0.5$, the proxy distributions overlap almost entirely ($\varepsilon_\gamma \approx 0.78$), rendering regime-specific causal effects mathematically non-identified from observables. Any method producing confident estimates in this regime is fitting noise.
> 
> Rather than claiming impossible point identification, our framework's core contribution is providing a mathematically grounded, computable diagnostic: the operational difficulty index $\mathcal{D}_t = H(\boldsymbol{\gamma}_t)/\lambda_{\min}(\boldsymbol{J}_t)$ (Proposition 1). In our selective estimation evaluation (Frontier A, reports/frontier_a_selective_estimation_summary.csv), gating estimation via $\mathcal{D}_t$ reduces causal error by $23.8\times$ ($3.286 \to 0.138$). Crucially, filtering by task conditioning alone ($1/\lambda_{\min}$) achieves zero error reduction. OR-DML transforms latent overlap from an invisible hazard into an explicit, actionable abstention trigger."

---

## 🛡️ Objection 3: Apparent Inactivity of Spectral Regularization ($\lambda$) in Table 1

### Reviewer Formulation
> *"In Table 1, the 'Spectral' and 'Unregularized' rows report nearly identical biases ($0.0812$ vs. $0.0823$ at $\Delta_Z = 2.0$, and $0.0541$ vs. $0.0543$ at $\Delta_Z = 4.0$). Why is spectral shrinkage highlighted in Theorem 4 if it produces negligible differences in the headline benchmark?"*

### Core Mathematical Defense
1. **Conditioning Regimes**:
   Table 1 evaluates a DGP with healthy residual variation ($\lambda_{\min}(\boldsymbol{J}) \approx 0.45$, $\kappa \approx 2.0$). In well-conditioned problems, $\lambda_{\min}$ is bounded safely away from 0, so $(J + \lambda I)^{-1} \approx J^{-1}$; regularization is inactive because it is not needed.
2. **Empirical Validation in Ill-Conditioned Regimes**:
   Theorem 4 specifically governs the ill-conditioned regime $\lambda_{\min}(\boldsymbol{J}) \to 0$. In our deep evaluation suite ([`reports/deep_eval_exploration2_ill_conditioned_summary.csv`](file:///D:/OneDrive/Documents/repos/aqi-informatics-india/reports/deep_eval_exploration2_ill_conditioned_summary.csv)):
   - When $\lambda_{\min} \le 0.015$ and $\kappa \ge 70$, unregularized coupled estimation suffers variance explosion ($\Var = 0.032\text{--}0.045$, coverage dropping to $40\%$).
   - Spectral OR-DML ($\lambda = 0.01\text{--}0.05$) reduces variance by **$2.8\times$ to $9.8\times$**, cutting RMSE in half and restoring coverage to $80\%$.
   - In Kolkata's real airshed ($\lambda_{\min} = 0.0025$), unregularized SE explodes to $1.11$, while spectral shrinkage stabilizes SE to $0.060$ (Table 2).

### 200-Word Rebuttal Snippet
> "We appreciate this astute observation. In Table 1, the DGP exhibits healthy residual variation ($\lambda_{\min} \approx 0.45, \kappa \approx 2.0$), so the Gram matrix is already well-conditioned. In this regime, Theorem 4 predicts that optimal shrinkage $\lambda^* \to 0$, acting purely as inert variance insurance.
> 
> However, spectral regularization becomes decisive when downstream task geometry is ill-conditioned ($\lambda_{\min}(\boldsymbol{J}) \approx 0$). In our ill-conditioning suite (reports/deep_eval_exploration2_ill_conditioned_summary.csv; $\lambda_{\min} \le 0.015, \kappa \ge 70$), unregularized estimation explodes in variance ($\Var = 0.045$, coverage $40\%$). Setting $\lambda \in [0.01, 0.05]$ reduces variance by up to $9.8\times$ and restores coverage to $80\%$. In our real-world Kolkata deployment ($\lambda_{\min} = 0.0025$), spectral regularization stabilizes standard errors from $1.11$ down to $0.060$ (Table 2). Spectral shrinkage is not intended to alter well-conditioned point estimates, but to prevent catastrophic variance collapse under collinear treatment geometries."

---

## 🛡️ Objection 4: Delhi's Pre-Treatment Placebo Rejection

### Reviewer Formulation
> *"Table 2 and Appendix G show that Delhi fails pre-treatment falsification placebos at $h \in \{-6, -3, -1\}$ ($p < 0.01$). If pre-treatment treatment values 'predict' current outcomes, does this invalidate the causal interpretation of the real-world sensor study?"*

### Core Mathematical Defense
1. **Transparent Disclosure vs. Overclaiming**:
   We deliberately designed and ran the lead placebo falsification suite ([`src/empirical_falsification_checks.py`](file:///D:/OneDrive/Documents/repos/aqi-informatics-india/src/empirical_falsification_checks.py)) and reported the exact p-values in Section 7 and Appendix G.
2. **Observational Stress-Test Framing**:
   The manuscript explicitly states (Section 7, Page 8):
   *"...treating the sensor study as an observational stress test of the estimator's diagnostics rather than independent ground truth for physical causal effects... We therefore interpret its dynamic curve as an exposure-response diagnostic rather than a validated causal impulse response."*
3. **Contrast with Other Cities**:
   Mumbai, Bengaluru, and Kolkata pass immediate pre-treatment placebos ($p > 0.05$). Delhi's unique airshed (intense shallow winter nocturnal inversion trapping emissions with diurnal boundary-layer feedback) illustrates the exact diagnostic role of falsification checks in sequential causal workflows.

### 200-Word Rebuttal Snippet
> "We welcome this point, as it underscores our commitment to transparent causal reporting. Rather than concealing pre-treatment lead correlations, we explicitly implemented the placebo falsification suite (Appendix G) and disclosed Delhi's failure ($p < 0.01$) directly in the main text (Section 7, Page 8).
> 
> Delhi's boundary layer dynamics and agricultural burning involve severe unmeasured diurnal confounding that violates strict unconfoundedness. Crucially, the manuscript does not claim a validated physical causal impulse response for Delhi; we explicitly characterize it as an 'observational stress test of the estimator's diagnostics.' In contrast, Mumbai, Bengaluru, and Kolkata pass immediate pre-treatment placebos, confirming that OR-DML reliably separates regimes where atmospheric transport permits causal interpretation. Transparent falsification reporting is essential for credible empirical causal inference."

---

## 🛡️ Objection 5: Scalability to Multi-Regime Settings ($K > 2$)

### Reviewer Formulation
> *"The empirical experiments focus almost exclusively on binary regimes ($K = 2$). Does the coupled estimator scale computationally and statistically to settings with $K = 3, 4, 5$ states?"*

### Core Mathematical Defense
1. **Mathematical Scalability (Appendix D.2)**:
   Appendix D.2 derives the computational and statistical properties for arbitrary $K \ge 2$. Matrix Bernstein concentration under geometric $\alpha$-mixing demonstrates that $\hat{\boldsymbol{J}}$ concentrates with error scaling as $\mathcal{O}(\sqrt{K \log K / N})$.
2. **Empirical Benchmark (`EXP-14`)**:
   We evaluated OverlapAwareRegimeDML across $K \in \{2, 3, 4, 5\}$ on dependent Markov processes ($N=1,200$, [`reports/multiregime_scalability_summary.csv`](file:///D:/OneDrive/Documents/repos/aqi-informatics-india/reports/multiregime_scalability_summary.csv)):
   - $K=2$: Bias $= 0.0608$, $\lambda_{\min} = 0.7581$, $\kappa = 1.31$, Runtime $= 3.63$s.
   - $K=3$: Bias $= 0.0772$, $\lambda_{\min} = 0.5061$, $\kappa = 1.22$, Runtime $= 5.72$s.
   - $K=4$: Bias $= 0.0887$, $\lambda_{\min} = 0.3609$, $\kappa = 1.43$, Runtime $= 6.89$s.
   - $K=5$: Bias $= 0.1400$, $\lambda_{\min} = 0.2380$, $\kappa = 2.00$, Runtime $= 9.26$s.
   - As $K$ grows, $\lambda_{\min}(\boldsymbol{J})$ scales gracefully ($0.758 \to 0.238$) reflecting simplex mass partitioning ($1/K$), while condition numbers remain bounded ($\le 2.0$) and solve times remain under 10 seconds.

### 200-Word Rebuttal Snippet
> "We thank the reviewer for raising the question of scalability. As formalized in Appendix D.2, OR-DML is structurally dimension-agnostic: the coupled Gram matrix $\hat{\boldsymbol{J}} \in \mathbb{R}^{K \times K}$ requires solving a $K$-dimensional linear system, taking $\mathcal{O}(K^3)$ Cholesky operations ($<1$ ms for $K \le 20$).
> 
> To empirically verify this, our multi-regime benchmark (EXP-14, reports/multiregime_scalability_summary.csv) evaluates $K \in \{2, 3, 4, 5\}$ on dependent Markov processes ($N=1,200$). As $K$ scales, runtime increases linearly ($3.6\text{s} \to 9.3\text{s}$), condition numbers remain tightly bounded ($\kappa \in [1.2, 2.0]$), and estimation bias scales gracefully ($0.0608 \to 0.1400$), tracking the expected simplex mass partitioning $\sum_k \gamma_k = 1$. The coupled regularized solver operates cleanly across arbitrary finite state spaces."

---

## 📋 Rebuttal Checklist & Verification Cross-References

| Topic | Key Formula / Bound | Supporting Report | LaTeX Section |
| :--- | :--- | :--- | :--- |
| **Soft vs. Hard Weighting** | $J_{01} = \frac{1}{N}\sum_t \gamma_{t0}\gamma_{t1}\tilde{T}_{t0}\tilde{T}_{t1}$ | `reports/deep_eval_exploration2_ill_conditioned_summary.csv` | Appendix D.1 |
| **Theorem 1 Non-Identification** | $P_1(Y, T, X, Z) = P_2(Y, T, X, Z)$ | `reports/or_dml_benchmark_summary.csv` | Section 5, Theorem 1 |
| **Theorem 3 Graceful Degradation** | $\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})}$ | `reports/factorial_reliability_correlations.csv` | Section 5, Theorem 3 |
| **Theorem 4 Spectral Risk Bound** | $\E\|\hat{\boldsymbol{\theta}}_\lambda - \boldsymbol{\theta}^*\|_2^2 \le \frac{(C\varepsilon_\gamma + \lambda M)^2 + \frac{\mathrm{Tr}(\boldsymbol{\Omega})}{N}}{(\lambda_{\min} + \lambda)^2}$ | `reports/deep_eval_exploration2_ill_conditioned_summary.csv` | Section 5, Theorem 4 |
| **Theorem 5 Asymptotic Normality** | $\sqrt{N}(\hat{\boldsymbol{\theta}}_\lambda - \boldsymbol{\theta}^* - \mathrm{Bias}) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\lambda)$ | `reports/empirical_or_dml_results.csv` | Section 5, Theorem 5 |
| **Proposition 1 Surrogate Bridge** | $\mathcal{D}_{\mathrm{causal}} \le C_K \overline{c}_J \E[\mathcal{D}_{\mathrm{operational}}(t)]$ | `reports/factorial_risk_coverage.csv` | Section 5, Prop 1 |
| **Proposition 2 Delta Variance** | $\sigma_{\mathrm{PATE}}^2 = \boldsymbol{\pi}^{*\top} \boldsymbol{\Sigma}_\lambda \boldsymbol{\pi}^* + \boldsymbol{\theta}^{*\top} \boldsymbol{\Sigma}_\pi \boldsymbol{\theta}^*$ | `verify_science.py` (Assert 1-10) | Section 5, Prop 2 |
| **Selective Abstention** | $\mathcal{D}_t = H(\boldsymbol{\gamma}_t)/\lambda_{\min}(\boldsymbol{J}_t) \le \tau$ | `reports/frontier_a_selective_estimation_summary.csv` | Appendix I, Table 12 |
| **Multi-Regime Scalability** | $\kappa(\boldsymbol{J}) \le 2.0$ across $K \in \{2, 3, 4, 5\}$ | `reports/multiregime_scalability_summary.csv` | Appendix D.2, EXP-14 |
