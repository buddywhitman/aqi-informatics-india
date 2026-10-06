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
