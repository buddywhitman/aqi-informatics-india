# v2 changelog and errata

v2 re-centres the paper on a closed-form **residual-state bias law** for latent-regime DML and replaces claims that did not survive re-verification.

## Withdrawn / corrected
- **Exp. 29/30 (`src/three_decisive_experiments.py`)**: nuisances were fitted on X only, so the regime shift in T leaked into residuals. The reported calibration/representation effects were artefacts. File kept, marked SUPERSEDED.
- **Old Table 10 Panels A/B**: same cause. Regime fixed effects (hard labels) *tie* OR-DML in the corrected setting; the spectral-penalty ridge is inert. The paper no longer claims otherwise.
- **Appendix G.4**: statements about lag-stable SEs were unsupported; replaced by measured Newey–West sensitivity (`reports/bias_law/e6_hac_lag_sensitivity.csv`).
- **tau\*-threshold / LightGBM claims** in README removed.
- **Mumbai**: results are sensitive to frozen-sensor screening; both versions reported (`e5_real_city_sensitivity.csv`).

## New
- `src/bias_law/`: closed forms (K=2, K-state matrix, heterogeneous theta), plug-in estimators (v̂, â, robustness value b*), simulations E1–E4, E7, real-city sensitivity E5, HAC sensitivity E6, figure/table generator.
- `paper/main.tex` (v2); previous manuscript kept as `paper/main_v1.tex`. Tables/figures in `paper/generated`, `paper/plots/bl_*`.
- `verify_science.py` (7 simulation-vs-theory tests) and `verify_artifacts.py` (file/number consistency) rewritten.

## Known limitations (be honest in review)
- The law is an omitted-variable-bias identity in the spirit of Cinelli–Hazlett / Chernozhukov et al.; novelty is the calibrated-HMM-posterior form, dependent-data setting and the conditioning-diagnostic critique.
- Learned-parameter + mixing case is proven only as an outline (filter-stability locality assumed).
- v̂ ignores information in X (X-aware version not done). b* is not identified from data; it is a sensitivity yardstick.
- Title/abstract differ substantially from the submitted abstract (flag risk, see docs/bestpaper.md).
- Bibliographic details of newly added citations should be checked manually.
- An OpenAQ API key exists in git history: revoke it at the provider; history rewrite alone is not sufficient.

## Extension round (joint / proximal / continuous-latent)
Scripts: `src/bias_law/ext_joint_proximal.py` (x1,x2), `ext_contlat.py` (x3), `ext_selfnc.py` (x4, x4b), `ext_phase.py` (x5), `make_ext_tables.py`.
Findings (honest):
- A correctly specified joint Markov-switching MLE is unbiased and lowest-variance, even with uninformative Z (Gaussian mixture identifiable): the modular posterior-DML is dominated there; the law is the price of modularity. Old "non-identification" claims hold only nonparametrically.
- Continuous AR(1) latent confounder: posterior-DML and joint HMM MLE both biased ~+0.8 (theta=1); the law with v=Var(u|X,gamma_hat) predicts it within ~4%. Proximal 2SLS (external negative control W, instrument = HMM posterior) has bias <0.03.
- Proximal bias under negative-control leakage l is -l*b/c (matches simulation, -0.45); the identification result itself is known (Miao et al. 2018; Tchetgen Tchetgen et al.) -- no novelty claimed.
- NEGATIVE results: lagged outcome Y_{t-h} as self-contained negative control is not robust (AR shocks: bias 0.07-0.21); smoothed posterior is an invalid instrument with lagged-outcome NC (bias ~0.2 at N=40k), raw Z_t is valid.
- No valid negative control/excluded proxy exists in the four-city data (meteorology affects PM2.5 directly), so no proximal real-data claim.
- Not best-paper-level: contributions remain a calibrated-HMM OVB law + honest comparison map. Unverified: author lists of arXiv:2208.00105 (not cited in paper).

## Round 3: primitives, sieve, real-data joint check
- `bias_law.wiener_smoother_var`: closed-form residual variance v = (1-phi^2)/sqrt((1+phi^2+iota(1-phi^2))^2-4phi^2) for an AR(1) latent smoothed from sensors with information iota = h'R^-1 h. Matches the Kalman smoother to 2e-4; predicts DML bias before seeing (T,Y) (corr 0.9987, max err 0.021 over 18 configs; `x6_kalman.csv`). Regimes: v ~ 0.5*sqrt((1-phi^2)/iota) for 0.02<<iota<<1/(1-phi^2) (worst-case bias ~ iota^-1/4), v ~ 1/iota beyond. Wiener-Kolmogorov theory is classical; the causal-bias consequence is the contribution; not derived for finite-state chains.
- K-state HMM posterior as a sieve for the continuous latent (`x6_sieve.csv`): bias 0.82/0.52/0.39/0.40 for K=2/3/4/6 vs law 0.78/0.49/0.37/0.38.
- Real cities, joint Markov-switching MLE vs posterior-DML (`x7_real_joint.csv`): law-implied corrections are negligible (v_hat 0.005-0.035 and small a_hat), so law-adjusted theta ~ theta_DML; joint MLE differs (Delhi 0.47 vs 0.58, Mumbai 2.35 vs 1.90, Kolkata 0.42 vs 0.12, Bengaluru ~0), likely Gaussian/shared-variance misspecification and data quality (Kolkata Var(T~)=0.006). Not used in the paper as evidence of confounding.

## Round 4: finite-state bound, X-aware v
- `ext_xaware.py`: for a persistent binary chain, v = MMSE <= linear Wiener error 0.25*v_W(2rho-1, iota) (`x8_finite_state.csv`). Tight for weak signals (ratio 1.04-1.4 at dz<=0.5, rho=.9), very loose for separated regimes (v=1e-4 vs bound 0.017 at dz=2): discrete regimes -> exponentially small v, continuous latent -> polynomial. Bound is for known parameters.
- X-aware v (`x9_xaware.csv`): Z-only v_hat predicted bias within 3% when X0 informs the regime (0.301 vs 0.308); an HMM refit on (Z,X) *under*-predicts (0.242) because linear nuisances cannot use that information. v is relative to the learner class; do NOT refit the HMM on X for v_hat. Z-only v_hat slightly under-predicts without informative X (0.301 vs 0.324, posterior overconfidence).

## Round 5: learned-parameter statement
App. A now states precisely what is reduced to cited results (Leroux 1992; Douc-Moulines-Olsson-van Handel 2011; Cappe et al. 2005 for filter continuity/forgetting) and what is not proved (rate; independence of the fitted HMM from the held-out fold). Citation details are from memory (not re-verified online): check before camera-ready.

## Round 6: law-regression (environment route)
`ext_envreal.py`, `ext_envreg.py`: theta_e = theta + b_e k_e with observable k_e = a_e v_e / Var(T~_e). Cross-environment regression intercept recovers theta when b_e is invariant or independent of k_e (sim: bias +0.03/0.00 vs pooled +0.30; violation b_e ~ a_e: -0.14 vs +0.36; `x10_envreg.csv`). Real cities (monthly environments): no leverage (|k|<=0.06) and unstable monthly theta; no real-data claim (`x11_*`). Related in spirit to invariance/ICP-style identification; novelty limited to the exact linear form supplied by the law; unvalidated beyond simulation.
Literature check (search only, not read in full): environment-based identification with hidden confounding exists (arXiv:2506.11756 "Causal Effect Identification in Heterogeneous Environments from Higher-Order Moments"; "Estimating causal effects with hidden confounding using instrumental variables and environments"). Treat law-regression as a variant, not a new identification principle; read both before any novelty claim. No HMM-posterior + DML residual-variance closed form turned up in two searches (not exhaustive).

## Round 7: change-point scaling of v (discrete regimes)
`ext_switchrate.py` (`x12_switchrate.csv`): exact-HMM v = (1-rho)*tau(d), tau nearly independent of rho (rho 0.8-0.99), tau ~ 0.68*exp(-0.158 d^2) for d^2 >~ 6 (d = Mahalanobis separation of emission means); universal in d^2 across 1-D and 2-D proxy geometries (tau 0.1185 vs 0.118 at d^2=11.04; 0.0456 vs 0.0453 at 17.25). Overestimates v at dz=2 (v<1e-3). A-priori law prediction vs learned-HMM E1 cells: dz=0.8 v_prior 0.0223 = v_hat 0.0223, bias 0.220 pred vs 0.226 obs at dT=8; dz=1.2: 0.110 vs 0.117 at dT=16. Empirical fit only; no proof; symmetric 2-state Gaussian only. Exponent 0.158 vs Bhattacharyya 1/8 (0.125) suggests a log(1/(1-rho)) correction (ρ=0.99 deviates +20%).
Round 7b (`ext_switchrate_k3.py`, `x13_switchrate_ext.csv`): scaling form v ~ (1-rho) tau(d^2_min) carries to K=3 and t5 proxies, constants differ. NOTE: the script's K=2 column is sum_k gamma_k(1-gamma_k) = 2*gamma(1-gamma) and the K=3 column is half of the sum, so compare after doubling the K=3 column (K=3 ~0.8x K=2; t5 1.1-1.5x Gaussian at d^2 in [11,17]).

## Round 8: parameter-free change-point closed form (supersedes the fitted 0.68*exp(-0.158 d^2))
The fitted exponent 0.158 was e^{-d^2/8}/d seen over d^2 in [6,25]. Closed form v ~ sqrt(8/pi) (1-rho) e^{-d^2/8}/d = (1-rho) * 2*Phi(-d/2) (Mills asymptotic): within +-10% of exact HMM v for rho<=0.95 and +-14% at rho=0.97, d^2 in [2.8,25] (`x12_switchrate.csv`; audit corrected the earlier '+-10% to 0.97' claim), 0.89-1.11 for a 1-D proxy at equal d^2 (`x17_oned_check.csv`; earlier '4%' was unstored), +23% at rho=0.99, noise-dominated at dz=2. As an a priori predictor of learned-HMM bias (E1, dz in {0.8,1.2}, 16 cells) RMSE 0.0059 vs 0.0080 for the data-based v_hat. Heuristic derivation only (single-change-point limit; ~2*Phi(-d/2) variance mass per switch); no proof; log(1/(1-rho)) correction not derived. `bias_law.v_changepoint`; verify_science test (9/9).

## Round 9: asymmetric chains, derivation sketch, corrected constant
- `ext_asym.py` (`x14_asym.csv`): v ~ c * r * e^{-d^2/8}/d with r = pi0*eps0 + pi1*eps1 (switch rate) holds for asymmetric chains (ratio 0.87-1.02, pi1 down to 0.1). Relevant quantity is the switch rate, not rho.
- `ext_singlechange.py` (`x15_singlechange.csv`), `/tmp`-run HMM limit (`x16_hmm_limit.csv`): analytic nearest-neighbour calculation gives asymptotic constant sqrt(pi/2)=1.2533 (E[sigma(W)sigma(-W)], W~N(-d^2/2,d^2); int dw/(4cosh(w/2)) = pi/2). Isolated-change simulation approaches it slowly (nearest-neighbour constant 0.71,0.92,1.02,1.08,1.11 for d=2..6, still rising); exact HMM constant falls 1.63 -> 1.23 (d=4) as eps 0.1 -> 1e-3. The earlier claim "2*Phi(-d/2) per switch" was a numerical coincidence over the practical range and has been REMOVED; sqrt(8/pi) is kept only as an effective constant for rho in [0.9,0.97]. Still a derivation sketch (remainder and consecutive-switch interaction not controlled).


## Round 10 -- audit remediation (paper text vs stored evidence)
Independent audit found claims not backed by stored results; fixes in `paper/main.tex`:
- H1 baseline numbers (X-only 0.387, soft 0.125, hard 0.142, oracle 0.004); H2/H3 change-point accuracy (+-10% rho<=0.95, +-14% at 0.97; 1-D 0.89-1.11, x17); H4 t5 0.9-1.7x (increasing in d^2), K3/K2 0.74-0.84 (x13, corrected normalisation sum_k g_k(1-g_k)).
- H5 lagged-outcome negative control: only stored x4/x4b/x4c reported. N=20000,h=4,dT=1,4 reps: iid dml 0.111, prox(gamma) 0.094, prox(Z) 0.018; AR shocks dml 0.110, prox(gamma) 0.148, prox(Z) 0.098. The earlier 'raw Z valid' and 'bias 0.2 at N=40000' claims were REMOVED (x4b has no generating script).
- H6 continuous-latent law: within 1.4-3.2% (x3b: 0.808/0.834, 0.750/0.774, 0.888/0.901, 0.861/0.874).
- H7 a priori RMSE across groups (x18): smooth/lin 0.0059, smooth/gbm 0.0155, filter/lin 0.0362; constant fitted on same-generator exact-HMM v, so a consistency check, not out-of-sample.
- H8 CE needs labels (abstract, Alg. step 1); H9 v defined relative to the learner (new Remark); H10 learned-parameter result stated as assumed, not proved.
- M1 underestimate 10-32%; M2 K=3 median 9.8%; M5 Delhi '1,200x' softened (c_Y is a conditional coefficient); M6 Section ref; M7 AI statement; M10 'spectral property' softened.
- x15 vs x16 unreconciled: isolated-change plateau ~1.11 vs asymptotic sqrt(pi/2)=1.2533; x16 now has a script (`ext_hmm_limit.py`): d=4 1.63,1.456,1.296,1.227,1.225; d=3 1.555,1.484,1.361,1.272,1.227.
- New scripts: ext_apriori_check, ext_contlat_law, ext_oned_check, ext_selfnc_check, ext_hmm_limit.
Still open: M3 replication counts everywhere, M4 degenerate bootstrap CIs (Kolkata unscreened, Bengaluru screened), M8-M9, M11-M13, M15, L-items; no proof of change-point formula, learned-HMM rate or cross-fit independence.

Round 10b: closed M4 (Bengaluru screened CI excludes point estimate; noted), M11 (means), M12 (diag-cov HMM on real data), M13 (complete cases), L5 (block 72, 300 draws), softened Discussion 'exact' wording.

## Round 11 -- change-point constant reconciled; learned-HMM rate and cross-fit argument
- `ext_nn_quad.py` -> `x20_nn_quad.csv`: nearest-neighbour constant by quadrature, c_NN(d)=sqrt(pi/2)(1-pi^2/(2d^2)+...); 0.91,1.01,1.08,1.12,1.17 at d=3,4,5,6,8; 1.2501 at d=40. This RECONCILES x15 (1.02 at d=4) with the sqrt(pi/2) derivation: the slow O(d^-2) approach, not a different limit. verify_science test added (10/10).
- `ext_hmm_window.py` -> `x19_hmm_window.csv`: variance-reduced exact-HMM single-switch windows, eps<=1e-3: const 1.19 (d=3), 1.12 (d=4), 1.12 (d=5), SE 0.004. Total exceeds NN by +10% (d=4), +4% (d=5), not bounded. CORRECTION: x16's d=4 value 1.23 was Monte Carlo noise (rare-event tail, ~1500 switches); x16 kept for the eps>=0.01 rows only (windows with eps>=0.03 are not comparable to the stationary chain: blip prior).
- `ext_hmm_rate.py` -> `x21_hmm_rate.csv`: learned HMM, well-specified emissions: E|gamma_hat-gamma*| slope -0.49 in N, |v_hat-v*| slope -0.55 (N^-1/2). Misspecified diag-covariance HMM (correlated proxies): plateau, v_hat understates v* by ~9% (pseudo-true gap). Prop. (rate) added to App. A with hypotheses H1-H3 (sqrt(N)-consistency of ML, Lipschitz posterior, X uninformative given Z); the projection argument gives CE <= E(gamma_hat-gamma*)^2.
- `ext_cond_decay.py` -> `x22_cond_decay.csv`: smoothed chain given Z decorrelates geometrically; |corr| < 0.003 at lag 24 in all six designs, supporting the embargo argument. Constants not bounded in general.
- Not proved: H1-H3 for EM as implemented; remainder terms of the change-point formula; non-NN site contribution.

## Round 12 -- second independent audit (agent) and fixes
Verified against CSVs; fixed: K=3 claim (2 of 8 cells pass |bias|>0.05; all-cell median 43%, law(v_N) 10.4%), 'exact' Remark (GBM law(v_N) 14.2%, E4 +26% overshoot; Limitation 7 mislabelled numbers), soft-beats-hard scope (smoother+lin 24/24, GBM 23/24, filter 0/24), Prop. (rate) restated for smoother/unrestricted v_N with a.s. derivative bound, expectation-level claims, non-expansive projection, v_N-v* = ||gamma*-c||^2 <= E(gamma_hat-gamma*)^2; cross-fit paragraph (needs joint alpha-mixing of noise; E[D~|Z,X]=gamma*-c gives O_P(N^-1/2); no 'bounded LR' hypothesis), change-point range (rho in {0.9,0.95}; +32% at 0.8; x19 at eps=1e-4; quadrature values vs first-order series), x16 low-eps rows retired, proximal identifies b/c, Delhi wording, Bengaluru 0.20 SD / Kolkata unscreened CI disclosed, E4 CE caveat + caption params, 'non-monotone in Delta_T', abstract 5% qualifiers, bib label width, compute statement.
Open (not fixed): Kolkata coverage dates / Mumbai missing hours not re-checked in text; purged-CV/Newey-West/Davydov references missing; bib entries typed from memory and unverified online; Prop. (rate) does not cover linear/GBM learners or the filter; misspecified-emission law with pseudo-true v untested.

## Round 13 -- Kolkata/Mumbai data audit, pipeline bug, EM hypotheses, bibliography
- **Pipeline bug (data_pipeline_clean.py)**: OpenAQ timestamps are hh:30Z; `.dt.round('h')` rounds ties to even, so every reading maps to an even hour, bins average two readings, and the inner join with hourly weather keeps only even hours. `combined_hourly_clean.csv` is a 2-HOURLY series (lag_1h = 2 h, roll_3h = 6 h; consecutive complete rows 2 h apart). Paper had called it hourly. Not overwritten; `ext_real_hourly_fix.py` rebuilds a true hourly grid (floor to hour, causal pairing with the weather hour that began 30 min earlier, ffill <=2 h): `x23_real_hourly_fix.csv`. Delhi theta 0.37 [0.29,0.44], b* 82 SD [54,144] (legacy 0.58, 175); Mumbai 2.23 unscreened / 0.31 screened (legacy 1.90 / 4.89); Kolkata 1.79 / 4.53 [1.48,7.42] (legacy 0.12 / 0.41) with only 16 distinct NO2 values; Bengaluru ~0. Paper main table/figure now use x23; legacy table kept in the appendix (`tab:cities2h`). HAC (e6) numbers still from legacy series.
- Kolkata ends 2025-06-23 (not 2026); Mumbai 749 missing hours; stated.
- **EM** (`ext_em_check.py`, `ext_em_real.py`, `ext_real_bestem.py`): hmmlearn absent -> NumPy EM, n_iter=25, fixed quantile init. Simulations: converged (gap<=4e-4 ll; same v_hat; 5 restarts same optimum). Real: Delhi/Bengaluru/Kolkata unimodal; **Mumbai multimodal** (restart spread 1182 ll, default fit 118 below best). Best-of-8 restarts (x27): Mumbai v_hat 0.021->0.025, b* 11.9->7.1 SD; others unchanged. H2 finite differences (x25): rms flat in N, max drifts slowly. H3 fails on real data: R^2 of gamma_hat on X 0.73-0.75 ridge / 0.89-0.96 GBM (Mumbai 0.19/0.28) (x26).
- **Bibliography verified against web sources (titles/venues/volumes/pages by IDEAS, PubMed, arXiv/Project Euclid search results)**: two errors fixed: Tchetgen Tchetgen et al. arXiv id is 2009.10982 (was 2009.10835, wrong); Pearl's paper is 'On a class of bias-amplifying variables that endanger effect estimates', UAI 2010 (was 'defy common inference', 2011). All other 15 entries exist with matching titles/authors/journals; volume/issue/page numbers confirmed for Cinelli-Hazlett, Ding et al., Greenland, Hamilton, Miao et al., Wang-Blei; Bickel et al. (26(4):1614-1635) also confirmed via the HUJI repository record; page numbers for Douc et al. (474-513), Leroux (127-143), Chernozhukov et al. (C1-C68) and the two books (Cappe et al.; Anderson-Moore) were NOT confirmed from a source this session. Crossref (429/garbage) and arXiv API (robots.txt) were unavailable.

## Round 14 -- closing round (change-point total constant, misspecified v, fold-wise HMM, third audit)
- `ext_cp_is.py` -> `x28_cp_total.csv`: importance-sampled total constant for one isolated change (steps tilted only in the first two sites; exact weights). d=2..6: 1.39, 1.18, 1.11, 1.11, 1.13 (SE <= 0.015); total/NN-1 = +88%, +30%, +9.7%, +3.2%, +0.8%. d=8 too noisy (SE 0.1; flagged invalid); d>=10 fail through weight degeneracy (Gauss-Hermite on the first two sites also breaks down there), rows removed. Agrees with x19 (exact HMM windows) and x15.
- `ext_misspec_crossfit.py` -> `x29_misspec_v*.csv`, `x30_foldwise_hmm*.csv` (24 reps, N=3000, dT=1, dg=3, dz=0.5).
  - x29: diagonal-Gaussian HMM on correlated t5 proxies: v_hat = pseudo-true v(theta_KL) (gap -1e-4, sd 0.002), but both are HALF the realised residual variance E(S-gamma)^2 (0.059 vs 0.122). Law with realised variance 0.326 vs bias 0.308; plug-in law 0.168. Well-specified: 0.063 vs 0.073 (-14%); law 0.204 vs bias 0.194; plug-in 0.178. => plug-in v_hat is a lower bound under misspecification. Paper qualified (abstract, contribution iii, algorithm, discussion, rate paragraph).
  - x30: HMM fitted on training folds only vs on all of Z: corr(bias) 0.999, mean diff 6e-4, v_hat diff 2e-4. HMM cross-fitting does not drive the bias here (evidence, not proof).
- Third independent audit (agent): found the change-point remainder percentages were the Gauss-Hermite J=2 values, not total/NN (fixed to x28 remainder_ratio); stale 'did not store sign' clause removed; v_hat gap wording (sd 0.002, max 0.005) and x30 numbers corrected; lower-bound caveat added to abstract/contribution/algorithm/discussion.
- Tests: verify_science 10/10, verify_artifacts passes. Paper 16 pages.
- Still open: H1-H3 unproved for EM; change-point remainder numerical only; Newey-West appendix on legacy 2h series; four bib page ranges (Douc, Leroux, Chernozhukov, books) unconfirmed; OpenAQ key must be revoked by the owner.
