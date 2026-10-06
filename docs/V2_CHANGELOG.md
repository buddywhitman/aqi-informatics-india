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
