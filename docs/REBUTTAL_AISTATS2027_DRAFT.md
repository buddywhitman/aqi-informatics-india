# AISTATS 2027 rebuttal draft, Submission 1823

Status: DRAFT. Nothing has been posted. Items marked **[VERIFY]** need a number or statement checked against `paper/appendix.tex` and `reports/` before posting. Do not post an unverified claim. The OpenReview comment limit is usually about 5,000 characters, so each block below is written to fit.

## 0. Triage

| Reviewer | Rating (conf.) | Main issue | Stance |
|---|---|---|---|
| BBdA | 2 (3) | Clarity; could not check the math | Concede and revise |
| (unnamed in screenshot, W1-W7/Q1-Q5) | 2 (5) | Identification, assumptions vs. experiments, small gains, baselines | Partly concede, clarify scope |
| wRgp | 3 (2) | Calibration assumption; Thm 5 vs. implemented procedure | Concede, clarify claim |
| Qg1H | 4 (2) | Coverage, heteroscedasticity, T in HMM, lambda | Answer directly |
| mC7v | not visible | Typo and inconsistency audit, no errors found | Thank, fix all |

Area chair's four concerns: (1) which inferential claim is supported; (2) identification and the role of Assumption 3; (3) clarity, "posterior-adjusted" and R-learner; (4) modest gains and missing baselines. Most of these are **valid, not misreadings**. The credible strategy is to narrow the claims and fix the writing, not to argue.

## 1. Response to Area Chair (post first)

> We thank the AC and reviewers. We agree most concerns are valid and have scoped the claims accordingly.
>
> **1. Which inferential claim is supported.** Theorem 5 gives asymptotic normality for the *proxy target* theta_lambda under a separated HMM-calibration window. It does **not** give nominal coverage for the structural theta*, which additionally requires proxy displacement o(N^-1/2); Table 16 shows this fails (0% coverage for all proxy-based estimators), as stated after Theorem 5. We will state this in the abstract and Sec. 1 and drop "reliable" claims about structural inference. The paper's inferential contribution is the bias law, the robustness value b*, and bias-aware intervals (below), not nominal structural coverage. We will also state that the released software conditions on the fitted HMM (full-sample smoothing), so Theorem 5 does not cover it.
>
> **2. Identification and Assumption 3.** Theorem 2 shows non-identification when proxies carry no regime information. Theorem 4 shows that for informative but imperfect proxies, theta* is recovered only up to the residual-state term J^-1 b_gamma, which vanishes as the residual state variance v -> 0. We will add a proposition stating when theta* is identified (correctly specified, identifiable HMM emissions, Allman et al. 2009; proxy separation Delta_Z > 0; calibrated posterior) and say plainly that outside this regime it is not. Assumption 3 (proxy sufficiency) is an idealised benchmark used only for Lemma 1. Theorems 3-4 require only posterior calibration given F_t. Our simulations violate Assumption 3 on purpose, to test robustness. We will relabel it and mark where each result uses it. **[VERIFY against App. A that Thm 3/4 do not use Assumption 3]**
>
> **3. Clarity.** We will rewrite the abstract and introduction in plain language, define "regime", "posterior-adjusted" and "causal error" at first use, state that Lemma 1 holds for a *fixed* posterior and not for estimated HMM parameters (App. H), and add the relation between Eq. (3) and the R-learner (see reply to the reviewer who asked).
>
> **4. Evidence.** We agree the gains are modest and design-dependent. The claim is that conditioning the nuisances on the posterior (rather than weighting by it) is the better design. In the real-data-calibrated benchmark, posterior-feature DML is better for the pooled ATE in Mumbai and Bengaluru (Table 2), and this is already stated in the text. We will move the Table 10 baselines (joint Markov-switching MLE, proximal 2SLS, HMM(Z,T), bias-adjusted three-step) into the main comparison.

## 2. Reviewer BBdA (clarity)

> Thank you. We accept the presentation criticism and will revise.
>
> - **Title typo:** the OpenReview title reads "Casual" instead of "Causal". We will correct the metadata.
> - **Abstract:** we will shorten it and define causal error (error of the estimated regime effects) and regime-aware (the estimator allows a separate effect in each latent regime).
> - **"Earlier drafts", AI traces, unnatural phrasing:** we will remove all draft-history language (e.g. p. 2) and rewrite the phrases you cite ("state that nobody measured" to "unobserved state", "geometry of the equation" to "conditioning of the estimating equation", etc.). We will keep the AI Use Statement.
> - **Forward-references and numbers in the introduction:** we will replace them with motivation and a worked intuition, and present NO2/PM2.5 as an example.
> - **Added intuition** before Theorems 1 and 2, and Corollary 1 split into result and interpretation.
> - **W2, "implicit reliability rule":** this is the practice of reporting posterior confidence and the conditioning of the score matrix as evidence of reliability. We will cite the sources that report these diagnostics. **[VERIFY: add concrete citations; do not claim "common" without them]**
> - **W3, why two-stage:** it is the natural, widely used pipeline; joint latent-model likelihoods are a different class of method. We will say this and expand the baseline descriptions.
>
> **Q1 (Pearl).** Related but distinct. Pearl (2010) shows that conditioning on more observed controls can amplify omitted bias. Phantom resolution is a *diagnostic* failure: when the state is *generated*, lambda_min(J) grows while bias is non-monotone in proxy error. They share the same algebra (see Related Work); the new element is the generated-state, regime-coupled form and its effect on the usual conditioning diagnostic.
>
> **Q2 (audit).** We checked the hourly grid against raw timestamps and found the earlier panel rounded half-hour stamps with ties-to-even, collapsing onto a two-hour grid, so "hourly" lags were two hours. We rebuilt the grid, which changed Delhi's single-index estimate from 0.58 to 0.37 (App. O). We will describe this audit in a short paragraph.
>
> **Q3.** We will define Bartlett bands (a rule for choosing the embargo length from the autocorrelation of the residuals), the asymmetric factorial design, state confusion (the posterior assigns mass to the wrong regime) and placebos (leads of the treatment, aligned to the outcome's information set, that should have zero effect if the temporal reading holds).

## 3. Reviewer with W1-W7 / Q1-Q5 (rating 2, conf. 5; name not visible in screenshot)

> Thank you for a careful review.
>
> **W1/Q2 (identification).** See the AC response, item 2. We will add a proposition on identification and will soften "Reliable" in the title/abstract to match what is proved. **[Decide: retitle, e.g. "When Does Causal Estimation Remain Reliable under Latent Markov Confounding?"]**
>
> **W2 (nonstationarity).** Correct. The theory assumes a fixed transition matrix and stationary law. We will state this as a limitation and remove the "dynamic environments" framing.
>
> **W3/Q1 (what is adjusted).** Posterior-weighted designs fit regime-k nuisances on X with weights gamma_k. Posterior-adjusted designs fit one shared nuisance pair on F_t = (X_t, gamma_t, gamma_t (x) X_t), so the posterior enters as a *feature*; it is also used as a weight for the regime-effect target. Lemma 1 (exact orthogonality) holds for any fixed posterior; Appendix H states that orthogonality in (m, mu) does not carry over to the fitted representation. We will say this in the main text.
>
> **Q3 (R-learner).** Eq. (3) is an R-learner-type residual-on-residual regression with effect basis gamma(F) and a ridge penalty. The difference is the use of a *generated* basis: the basis is itself an estimated posterior, so its error enters bias (Theorem 4) and variance (Theorem 5). We will state the connection explicitly and cite Nie and Wager (2021). **[VERIFY wording against Sec. 3]**
>
> **W4/Q4 (assumption 3; (T,Y) independent of Z given (X,S)).** In the synthetic DGPs this holds by construction in the main design, but the frontier DGP deliberately violates proxy sufficiency (App. G.1). In the real data, meteorology enters as both control and proxy (delta_X > 0), so the condition is an assumption we cannot establish. This is why the city analysis is reported as an exposure association with falsification checks, not an emissions effect.
>
> **W5.** Agreed. The real-data section is a stress test; only Delhi survives the lead placebo, and we say it is not a causal effect.
>
> **W6.** Agreed that gains are small except at weak-to-moderate observability. Figure 3 and the text are on different scales (log vs. linear); we will unify them.
>
> **W7.** We will unify notation (regime indices {0,1} vs {1,2}, v as process vs. expectation, etc.) and add a notation table.
>
> **Q5.** We will promote these baselines to the main comparison. **[VERIFY results in Table 10 before characterising them]**

## 4. Reviewer wRgp (rating 3)

> Thank you.
>
> **W1.** Agreed: the diagnostic is only as good as the posterior calibration. We will state that a misspecified HMM can give confident posteriors and an over-reassuring v-hat and b*, and recommend checking calibration (e.g. held-out likelihood, restarts).
>
> **W2.** Agreed; see AC response, item 1. Theorem 5 covers the proxy target; the application conditions on the fitted HMM. We will state this where results are reported and show the Table 16 coverage in the main text.

## 5. Reviewer Qg1H (rating 4)

> Thank you for the positive and precise review.
>
> **Q1 (bias-aware intervals).** Yes. We will report intervals theta-hat +/- (|a-hat| v-bar b-bar / Var(T-tilde)-hat) +/- 1.96 SE, indexed by a sensitivity budget b-bar, and show them against the Table 16 setting. **[Only promise this if you will run it. Otherwise say "we will add the construction and discuss coverage"]**
>
> **Q2.** Theorem 4 assumes homoscedastic treatment innovations. With sigma_0 != sigma_1 the pooled formula changes (Proposition 1 already carries sigma_s). We expect the same structure with v weighted by regime-specific innovation variances but have not proved it, and we will say so.
>
> **Q3.** Including T in the HMM uses information that is correlated with the state; Lemma 1 conditions on F, so adding T to the posterior changes the conditioning set, and collider bias is possible if T is a descendant of S and Y. We have no theory for this case and will say so in the main text next to Table 15.
>
> **Q4.** **[VERIFY from `reports/` and App. I: values of lambda tried, selected value per sweep, and share of runs with lambda > 0.]** Currently the paper reports the grid minimum on one ill-conditioned benchmark (Table reg) and states lambda is immaterial in the well-conditioned frontier.

## 6. Reviewer mC7v (typo and inconsistency audit)

> Thank you for an unusually thorough audit. We will fix every item: the beta-mixing block-coupling for Step 3 of the pooled-law proof, the O(N eps) scaling and the r_T^2 term, the shared-target phrasing in Theorem 3, the missing sqrt(K) factor, the missing + sign in the Theorem 5 score expansion, the C_K statement (C_2 = 1/ln 2, C_K = 2), the interior-minimiser condition, the B_lambda collision, Appendix J (legacy design), notation collisions, and the Table 3 vs Table 20 and Appendix N vs Table 17 mismatches. We will add the suggested references (Ogburn-VanderWeele, Kuroki-Pearl, Murphy-Topel / Escanciano-Perez-Izquierdo) and the callout for Table 10. **[VERIFY each item against the PDF/appendix before this reply; the numeric mismatches must be reconciled, not just documented]**

## 7. Revision work list (in priority order)

1. Rewrite abstract and introduction (plain language, remove forward-references, remove "earlier drafts").
2. Narrow claims: title/abstract "reliable"; state Theorem 5 scope and the 0% coverage in the main text.
3. Add identification proposition and relabel Assumption 3.
4. Define terms at first use; add notation table; harmonise indices and symbols.
5. Promote Table 10 baselines to the main comparison.
6. Reconcile numeric inconsistencies (Table 3 vs 20, Appendix N vs Table 17, Figure 3 scales).
7. Apply mC7v's proof fixes; rerun `verify_submission.py`.
8. Fix the OpenReview title typo ("Casual").
9. Optional: bias-aware intervals (Qg1H Q1).

## 8. Open questions for the authors

- Is a revised PDF allowed during the rebuttal period, or only comments? This changes how many items can be promised versus shown.
- Do you want to retitle?
- Which of items 5 and 9 will you actually run before the deadline?
