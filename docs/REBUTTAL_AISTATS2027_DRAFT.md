# AISTATS 2027 rebuttal draft, Submission 1823 (v2)

Status: DRAFT, nothing posted. Corrected reviewer mapping after the author's clarification. Only three actions exist on OpenReview (Rebuttal, Official Comment, Flag Review), so **no revised PDF can be uploaded**: every promised fix must be backed by concrete text or numbers shown in the reply. Each block is under about 5,000 characters.

## 0. Where the points are

| Reviewer | Rating (conf.) | Gist | Movable? |
|---|---|---|---|
| BBdA | 2 (3) | Presentation too poor to check the math | **Yes**: show rewritten text; a 3 is plausible |
| Qg1H | 2 (5) | Identification, assumption 3, inference, gains, baselines | Hard, but a 3 is possible if the claim scope is conceded |
| mC7v | 3 (2) | Calibration assumption; Theorem 5 vs. implemented procedure | Hold |
| wRgp | 4 (2) | Coverage, heteroscedasticity, T in HMM, lambda | Hold |

Current average is 2.75. **One reviewer moving up one point gives 3.0.** BBdA is the best bet because the complaint is about writing, which can be fixed visibly inside the reply. Qg1H is the second.

The "typo audit" (Appendix G/I proofs, Table 3 vs 20, etc.) was the **Program Chairs' AI review**, not a human reviewer. It does not carry a rating. Fix what is real, but do not spend rebuttal space answering it item by item.

## 1. Verified facts (checked in this repo)

- `verify_submission.py --fast`: all checks pass (PDF, archive and sources consistent).
- **Assumption 3 is not used by Theorem 3** (App. A, line 51 says so). Where it matters: in the bias-law setting (App. `biaslaw`) `T = m(X) + a D + V` with `V` independent of `(S,Z,X)`. If `T` carried no regime information beyond the posterior (Assumption 3), then `a v = 0`, so the Theorem 4 bias `a b v / (a^2 v + sigma^2)` vanishes. That is the missing link the area chair asked for. **The linear-model argument is mine, not in the paper; check before posting.**
- **Table 10 (`paper/generated/biaslaw/tab_ext.tex`)**: in the correctly specified HMM case, posterior-DML bias is +0.20 (Delta_T = 1) and +0.40 (Delta_T = 4), while joint Markov-switching MLE is +0.00 and proximal 2SLS is -0.01 / +0.02. Under a continuous AR(1) latent state, MLE is also biased (+0.77 to +0.89), 2SLS is not (about -0.02). A negative-control leak breaks 2SLS (-0.45) but not the MLE.
  - **Consequence:** reviewers are right that the strong baselines beat posterior-DML when their assumptions hold. The honest message is "which assumptions let `b` be learned", not "OR-DML wins". Concede it.
- **Frontier (`tab_frontier_designs.tex`, Delta_Z = 1)**: OR-DML (posterior-adjusted) ATE bias 2.134 vs. posterior-feature DML 2.164 and hard regime FE 2.575; weighted nuisances 4.236. At Delta_Z = 2: 0.129 vs. 0.141 vs. 0.141. At Delta_Z <= 0.5 everything fails (5.9 to 8.6). **The gain of posterior-adjusted over posterior-feature DML is essentially zero for the ATE.** Do not claim otherwise. The real gap is conditioning vs. weighting (weighted is worst) and, for regime effects, 2.954 vs. 3.652 (hard FE) at Delta_Z = 1.
- Coverage: OR-DML 0% at Delta_Z = 1, 91% at Delta_Z = 2, 94% at Delta_Z = 4 (`claims.csv`: FDCovOursOne/Two/Four). Oracle is 94 to 96%.
- Regularization: best lambda = 0.1 on the ill-conditioned DGP (MSE 0.0143 vs. 0.785 at lambda = 0); in the well-conditioned frontier the penalty is immaterial (App. `surrogate`). **The data-driven selector's selection frequency (wRgp Q4) is not recorded in the reports.** Do not invent it.

## 2. Posting order

1. **Rebuttal** to Qg1H (largest lever, conf. 5).
2. **Rebuttal** to BBdA, with the rewritten text.
3. **Rebuttal** to mC7v and wRgp (short; point to the scope statement).
4. **Official Comment** to the area chair summarising the scope changes (block 5).
5. Do not use Flag Review.

## 3. Block A: Qg1H (rating 2, confidence 5)

> We thank the reviewer. Your comments identify real gaps and we agree with most of them. We state what the paper does and does not claim, then answer each question.
>
> **Scope (W1, W5).** The paper does not claim that theta* is identified from imperfect proxies, nor that the city results are causal. The contribution is (i) a diagnostic failure ("phantom causal resolution": lambda_min(J-hat) can improve while error grows), (ii) an exact bias law for posterior-conditioned estimators, and (iii) a design principle (condition nuisances on the posterior; use it to weight only the target). We will retitle to remove "Reliable", e.g. "When Does Causal Estimation Remain Reliable under Latent Markov Confounding?", and say in the abstract that no proxy-based estimator is unbiased when proxies are weak.
>
> **Q2, identification.** Theorem 2 gives non-identification when proxies carry no regime information. For informative proxies the answer is Theorem 4, not an identification theorem: theta-hat -> theta + J^{-1} b_gamma, with b_gamma proportional to a b E[v], where a is the treatment regime shift, b the outcome shift, v = gamma(1-gamma) the residual state variance. So theta* is recovered when a = 0 or v -> 0. Assumption 3 (proxy sufficiency) is where a = 0 comes from: if T carries no information about S beyond Z, there is no residual treatment shift. It is therefore an idealised benchmark that our simulations violate deliberately; Theorem 3 does not require it (App. A). We will add this as a remark after Theorem 4.
>
> **W2.** Correct: the theory assumes a fixed transition matrix and stationary latent law. We will say so and remove the "dynamic environments" framing.
>
> **W3, Q1.** "Posterior-weighted" fits one nuisance per regime k and weights it by gamma_k. "Posterior-adjusted" fits a single nuisance pair on F = (X, gamma, gamma (x) X), so the posterior enters as a feature. Lemma 1 (exact orthogonality) holds for any fixed posterior; App. H states that this does not extend to the fitted HMM parameters, which is what the sentence you quote means. We will say this in Sec. 3.
>
> **Q3, R-learner.** Eq. (3) is an R-learner-style residual regression with effect basis gamma(F) and a ridge penalty. The differences are that the basis is a generated posterior (its error enters Theorems 3 to 5) and that the nuisances are conditioned on it. We will add this sentence and cite Nie and Wager (2021).
>
> **W4, Q4.** Agreed that Assumption 3 fails in our DGPs, and meteorology is both control and proxy in the real data (delta_X > 0, App. `biaslaw`), so (T, Y) independent of Z given (X, S) cannot be established there. The city analysis is labelled an exposure association; only Delhi survives the lead placebo, and we do not interpret the other cities.
>
> **W6.** Agreed. For the pooled ATE the margin of posterior-adjusted over posterior-feature DML is negligible (2.13 vs. 2.16 at Delta_Z = 1; 0.129 vs. 0.141 at Delta_Z = 2). The effect is conditioning vs. weighting: weighted nuisances give 4.24 at Delta_Z = 1. We will state this in the abstract and unify the Figure 3 scale.
>
> **Q5.** We agree they should be main baselines. Table 10 already shows the result: with a correctly specified HMM, the joint MLE and proximal 2SLS are essentially unbiased (+0.00 and -0.01 at Delta_T = 1), whereas posterior-DML has bias +0.20. So when a parametric joint model is credible it should be preferred. OR-DML and the bias law matter when it is not (continuous latent state: MLE bias +0.79 at SNR 1) and as a diagnostic. We will move Table 10 to the main text and reposition the claim accordingly.

## 4. Block B: BBdA (rating 2, confidence 3)

> Thank you. We accept that the presentation was not at publication standard. As we cannot upload a revision, we show the rewritten abstract here and describe the other changes; the full text will be revised.
>
> **Revised abstract.** *Observational time series are often driven by an unobserved, persistent state (a "regime", such as the boundary-layer regime of the atmosphere) that affects both a treatment and an outcome. A common remedy infers the state from noisy proxies with a hidden Markov model (HMM) and then applies Double Machine Learning (DML). We show that the usual reliability check for this recipe can fail: when the inferred state is wrong, the conditioning of the estimating equation can improve while the causal estimate gets worse ("phantom causal resolution"). We derive the exact bias, which depends on the residual uncertainty about the state, not on how confident the posterior looks. Using the posterior as an adjustment feature, rather than as a weight, makes the estimator exactly Neyman-orthogonal for any fixed posterior (OR-DML). Experiments on simulated data, on a semi-synthetic benchmark calibrated to four Indian cities, and on real sensors show that OR-DML is the best regime-aware DML variant, but that no proxy-based estimator is unbiased when the proxies are weak. We give a sensitivity bound for this case.*
>
> Terms: *causal error* is the distance between estimated and true regime-specific effects; *regime-aware* means the estimator has a separate effect for each latent regime.
>
> **Other changes.** (1) Remove all draft-history language (p. 2) and the numbers in the introduction; use the space for motivation. (2) Replace unnatural phrases (e.g. "state that nobody measured" becomes "unobserved state"; "geometry of the equation" becomes "conditioning of the estimating equation"; "locates the damage in the nuisance step" becomes "the bias comes from the nuisance estimation step"). (3) Resolve "it/its circularity" and "the mechanism" references, and present NO2/PM2.5 as an example. (4) Add intuition before Theorems 1 and 2 and split Corollary 1 into result and interpretation. (5) Define Bartlett bands, asymmetric factorial design, state confusion and placebos at first use.
>
> **W2.** The implicit rule is the practice of reporting posterior confidence and the conditioning of the score matrix as evidence of reliability. We will cite where it appears. "Regime" is the value of the latent discrete state, not of a confounder in general.
>
> **Q1.** Pearl (2010) shows that adding observed controls can amplify omitted bias. Phantom resolution is a diagnostic failure for a generated state: lambda_min(J-hat) grows (70x in the factorial design) while error is non-monotone in proxy error. Same algebra, new object.
>
> **Q2.** We audited the hourly grid against raw timestamps and found half-hour stamps rounded ties-to-even, collapsing onto a two-hour grid. We rebuilt it; Delhi's single-index estimate moved from 0.58 to 0.37.
>
> **Q3.** Bartlett bands choose the embargo length from residual autocorrelation; the asymmetric factorial design crosses proxy error with treatment-innovation scale so the closed form can be checked cell by cell; state confusion means the posterior puts mass on the wrong regime; placebos are leads of the treatment aligned to the outcome's information set (zero effect expected if the temporal reading holds).

## 5. Block C: mC7v (3) and wRgp (4), short

> We agree the diagnostic depends on posterior calibration and that Theorem 5 covers the proxy target, not the structural effect (stated after Theorem 5; Table 16 shows 0% structural coverage for proxy-based methods at weak separation). We will state in the abstract that nominal structural coverage is **not** claimed, and that the released software conditions on the fitted HMM (full-sample smoothing).
>
> For heteroscedastic innovations, Proposition 1 already carries regime-specific sigma_s; we expect Theorem 4 to hold with v weighted by regime variances but have not proved it. Including T in the HMM changes the conditioning set of Lemma 1 and is outside the theory (Table 15); we will say so. For lambda, we report the MSE-minimising value on the ill-conditioned design (lambda = 0.1; MSE 0.014 vs. 0.785 at lambda = 0) and note that the penalty is immaterial in the well-conditioned frontier. **We did not record how often a data-driven selector picks lambda > 0.**

## 6. Block D: Official Comment to the AC

> We summarise the changes we commit to. (1) Claims narrowed: remove "reliable"; no structural coverage is claimed; no identification beyond Theorem 4. (2) Assumption 3 relabelled as an idealisation; its role (a = 0, hence no residual bias) is added after Theorem 4. (3) Table 10 moved to the main text with the honest message that a correctly specified joint MLE or proximal 2SLS beats posterior-DML, and OR-DML and the bias law matter when those assumptions fail. (4) Abstract, introduction and notation rewritten (see our reply to BBdA). (5) The gain over posterior-feature DML on the pooled ATE is stated as negligible; the claim is conditioning vs. weighting.

## 7. Do not claim

- That OR-DML clearly beats posterior-feature DML on the ATE (it does not).
- Any statistic on lambda selection frequency.
- Any new experiment result: none was run for this draft.
- A proof of "Assumption 3 implies a v = 0" beyond the linear setting of App. `biaslaw`.

## 8. Still open

- Retitle: yes (user said no objection). Suggested title above.
- If you want a stronger answer to wRgp Q1/Q4, I can run bias-aware intervals and record lambda selection, and add real numbers to Block C. Not done yet.
