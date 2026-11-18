# Replacement text for the AISTATS 2027 paper (Submission 1823)

Purpose: paste-ready fixes for the clarity complaints (BBdA, the Program Chairs' AI review, area chair). Nothing here changes a number. Where a citation is needed it is marked **[CITE]**; do not invent one. No revised PDF can be uploaded during the rebuttal, so use these in the rebuttal replies (BBdA block) and, if a revision window opens, in `paper/main.tex`.

## 1. Title

Current: *Reliable Causal Estimation under Latent Markov Confounding* (OpenReview metadata also reads "Casual", which must be corrected in the submission page if editable).

Proposed: ***When Does Causal Estimation Remain Reliable under Latent Markov Confounding?***

## 2. Abstract (about 190 words; the current one is about 330)

> Observational time series are often driven by an unobserved, persistent state, or *regime* (for example the mixing regime of the atmosphere), that affects both a treatment and an outcome. A common remedy infers the regime from noisy auxiliary measurements (proxies) with a hidden Markov model (HMM) and then applies Double Machine Learning (DML). Practitioners judge such pipelines by two signs: the inferred regime looks confident, and the matrix in the estimating equation looks well conditioned. We show that both signs can mislead. When the inferred regime is wrong, the conditioning can improve while the estimate gets worse ("phantom causal resolution"). We derive the exact bias: it depends on the uncertainty about the regime that remains after inference, not on how confident the posterior looks. We also show that using the inferred regime as an adjustment variable, rather than as a weight, makes the estimator exactly Neyman-orthogonal for any fixed posterior (OR-DML). On simulated data, a semi-synthetic benchmark calibrated to four Indian cities, and real sensor data, OR-DML is the most accurate regime-aware DML variant, but no proxy-based estimator is unbiased when the proxies are weak. We therefore give a sensitivity bound and do not claim nominal coverage for the causal effect.

Defined at first use: *causal error* = distance between estimated and true regime-specific effects; *regime-aware* = the estimator has a separate effect for each latent regime.

## 3. Introduction, opening (replaces the first three paragraphs; no numbers, example first)

> **A motivating example.** Does reducing nitrogen dioxide (NO2) reduce fine particulate matter (PM2.5) in a city? Both are driven by the state of the atmosphere, for instance whether air is stagnant or well ventilated, and nobody measures that state directly. When stagnant air raises both pollutants, a naive comparison attributes the joint rise to NO2. Adjusting for observed weather helps only partly, because the unobserved state is not a function of the weather we observe.
>
> **The usual remedy and the usual check.** A standard remedy has two steps. First, infer the unobserved state from auxiliary measurements (here, meteorological variables) with a hidden Markov model (HMM), which yields for each time point a probability that the state is each of its possible values (the *posterior*). Second, use these probabilities in Double Machine Learning (DML; Chernozhukov et al., 2018), a method that estimates a causal effect after removing the part of treatment and outcome that observed variables explain. In practice such pipelines are judged by two signs: the posterior is confident, and the matrix that DML inverts to obtain the effects (the *score Jacobian*) is well conditioned, meaning its smallest eigenvalue is not near zero. **[CITE: where these diagnostics are reported; do not state "common" without sources.]**
>
> **What we show.** These two signs can be misleading. When the inferred state is wrong, errors in the posterior do not only add noise: they leak the difference in treatment between states into the quantity DML treats as exogenous variation. That can make the matrix look *better* conditioned while the causal estimate gets *worse*. We call this *phantom causal resolution*. Section 2 describes the setting, Section 3 the estimator, Section 4 the bias, and Sections 5 and 6 the experiments.

(Removes: "state that nobody measured", "geometry of the equation", "The mechanism has a closed form...", "it locates the damage in the nuisance step", the unreferenced "its circularity", the abrupt "The mechanism..." paragraph, and all numbers from the introduction.)

## 4. Contributions in plain terms (replaces the list that cites theorem numbers first)

> 1. **When standard methods fail.** An exact decomposition of the bias of standard DML when a regime is omitted (Theorem 1), and a proof that the regime-specific effects cannot be recovered when the proxies carry no regime information (Theorem 2).
> 2. **Why the usual check misleads.** Exact formulas showing that the conditioning of the estimating equation can improve as the posterior gets worse (Proposition 1, Corollary 1), and a bias law whose size is set by the *residual* uncertainty about the regime, multiplied by how much treatment and outcome shift between regimes (Theorem 4).
> 3. **An estimator.** OR-DML: nuisance regressions are fitted with the posterior as an input feature (posterior-adjusted), and the posterior is used as a weight only to define the regime-specific target. This is exactly Neyman-orthogonal for any fixed posterior (Lemma 1).
> 4. **Evidence.** A factorial design verified against the closed form, learned-posterior experiments, a semi-synthetic benchmark built from real data in four Indian cities with known causal effects, and an observational sensor study with falsification tests.

## 5. Definitions to add at first use

| Term | Definition to insert |
|---|---|
| Regime | The value of the unobserved discrete state that persists over time (not a confounder in general). |
| Proxy | An auxiliary measurement that depends on the regime but not otherwise on treatment or outcome. |
| Posterior | gamma_tk = P(S_t = k given proxies), computed by the HMM. |
| Posterior-weighted | One nuisance regression per regime, each weighted by gamma_tk. |
| Posterior-adjusted | One shared nuisance regression whose inputs include the posterior (X, gamma, gamma times X). |
| State confusion | The posterior assigns mass to the wrong regime. |
| Bartlett bands | Confidence bands for residual autocorrelation, used to choose how many time steps to leave out (the embargo) around each fold. |
| Asymmetric factorial design | A grid that crosses proxy error with treatment-innovation scale (with different scales per regime) so the closed form can be checked cell by cell. |
| Lead placebo | A test that the treatment at a *later* time has no effect on the outcome now; a nonzero estimate signals confounding or reverse timing. |

## 6. Remarks to add in Section 2 and after Theorem 4

**Remark (role of Assumption 3).** Assumption 3 (proxy sufficiency) is an idealised benchmark, not a condition for Theorems 3 and 4 (App. A). In the model T = m(X) + a D + V it implies a v = 0: if T carried information about the regime beyond the proxies, then P(S = k given X, T, Z) would depend on T unless a = 0 or the posterior were already degenerate. The Theorem 4 bias a b v / (a^2 v + sigma^2) then vanishes. Our simulations violate Assumption 3 on purpose, to measure what happens when it fails. *(Linear-Gaussian setting of the bias-law appendix only.)*

**Remark (heteroscedastic treatment innovations).** If the innovation variance depends on the regime, Var(V given S = k) = sigma_k^2, the pooled law is unchanged except that sigma^2 is replaced by its posterior average: bias = a b v / (a^2 v + E[sigma_S^2]), with E[sigma_S^2] = E[gamma sigma_1^2 + (1 - gamma) sigma_0^2]. *Proof.* With a calibrated posterior, tilde T = a(D - gamma) + V and E[V given D] = 0, so E[tilde T (b tilde D + U)] = a b v and E[tilde T^2] = a^2 v + E[sigma_S^2]. The observable plug-in is unchanged because Var(tilde T) - a^2 v still equals E[sigma_S^2]. *Numerical check* (`src/rebuttal/hetero_law_check.py`, `reports/rebuttal/hetero_law_check.csv`, N = 300,000, six configurations, prior 0.2): relative error of the law 0.07% to 1.7%; the naive law with the average variance is off by 42% to 55%. *The pooled estimator only; extension to the K-regime coupled estimator is not proved.*

**Remark (tuning rule).** In the frontier and semi-synthetic experiments the spectral penalty is the fixed rule lambda = 0.05 tr(J-hat) / (K sqrt(N)), not a selected value. It is positive in every run. On the frontier DGP it is about 7e-4 to 9e-4 (median over 20 datasets), so sqrt(N) lambda is about 0.025 to 0.030: a constant, **not** o(1), so the condition lambda = o(N^{-1/2}) in Theorem 5 is not met by this rule. The shrinkage bias bound lambda M_theta / (lambda_min + lambda) is about 0.004 there (M_theta = 2.6). The data-driven selector `AdaptiveSpectralOptimizer` is used only in the legacy real-data benchmark script. (`reports/rebuttal/lambda_rule_check.csv`.)

## 7. Discussion and Outlook (rewrite; keep it to the main message)

> **Main message.** A confident inferred state and a well-conditioned estimating equation do not show that a causal estimate is reliable. The inferred state enters the estimating equation twice: in the estimate and in the apparent information available to the estimator. What matters is the uncertainty about the state that remains after inference, multiplied by how much treatment and outcome shift between states.
>
> **Practical rules.** (i) Use the inferred state as an adjustment input to the nuisance regressions and only as a weight for the target. (ii) Report the residual state variance, the regime shifts it multiplies, and a robustness value, and compare against hard-regime and posterior-feature baselines. (iii) Test temporal validity with placebos aligned to the outcome's information set.
>
> **What we do not claim.** We do not claim that the effects are identified from weak proxies, that OR-DML is unbiased, that confidence intervals have nominal structural coverage, or that the four-city study measures emissions effects. When a correctly specified joint model is available it should be preferred (App. Table 10): a joint Markov-switching MLE and proximal two-stage least squares are nearly unbiased there, while OR-DML and the bias law matter when such a model is not credible.
>
> **Outlook.** Nothing in the mechanism is specific to HMMs: any inferred state (a neural encoder's embedding, a cluster label, a segmentation) enters the estimating equation as both an adjustment and an apparent source of information.

## 8. Notation fixes (from the Program Chairs' AI review; reconcile before any revision)

- Regime index set: {0, 1} in Proposition 1 and {1, 2} in Proposition 2. Pick one.
- `v` is used for both the random variable v_t = gamma_t(1 - gamma_t) and its expectation; use v_t and v-bar.
- `D_t` is both the regime indicator and a smoother-derivative envelope (App. G, Lemma 3); rename the latter.
- `B_lambda` is both a matrix (App. H) and a scalar bound (Theorem 6); rename the scalar b_lambda.
- Assumption 4 writes (mu_D, mu_Y) where the text uses (m, mu).

## 9. Data inconsistencies found in this repo (verify before replying)

- **ROC-AUC for the HMM**: 0.615 in `reports/representation_zoo_reliability_auc.csv` (used by `\ZooAucHMM` in the text and Appendix N) vs. 0.5895 in `reports/representation_zoo_calibration_intervention.csv` (used in Table 17). Different files, possibly different runs. Pick one source or explain both.
- **Tables 3 vs. 20** (Newey-West lag 24): Mumbai matches (-4.012, 2.757); Kolkata differs (Table 3 screened: 0.191, 6.916; Table 20: -0.027, 7.084); Bengaluru differs slightly (Table 3: 0.185, -0.156; Table 20: 0.182, -0.158). The likely cause is a different screening specification; this has **not** been traced.
- **Table 10** (joint MLE and proximal 2SLS) is not referenced from the main text.
