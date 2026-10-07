# Scientific corrections and feedback assessment

The correction changes mathematical validity, variance computation, and claim scope. It does not guarantee conference acceptance or establish causal identification of observational pollution effects.

## References checked against primary sources

| Entry | Correction | Primary source |
|---|---|---|
| Allman, Matias and Rhodes (2009) | Title: *Identifiability of parameters in latent structure models with many observed variables*. | https://arxiv.org/abs/0809.5032 ; doi:10.1214/09-AOS689 |
| Crump et al. (2009) | Fourth author: Oscar A. Mitnik. | https://academic.oup.com/biomet/article-abstract/96/1/187/235329 |
| D'Amour (2019) | Title: *On Multi-Cause Causal Inference with Unobserved Confounding: Counterexamples, Impossibility, and Alternatives*. | https://proceedings.mlr.press/v89/d-amour19a/d-amour19a.pdf |
| Hatt and Feuerriegel (2021) | Title: *Sequential Deconfounding for Causal Inference with Unobserved Confounders*. | https://arxiv.org/abs/2104.09323 |
| Tchetgen Tchetgen et al. (2020) | Correct ID: arXiv:2009.10982; retain the preprint title *An Introduction to Proximal Causal Learning*. | https://arxiv.org/abs/2009.10982 |
| Lewis and Syrgkanis | Use the full current title ending *via g-Estimation*. The original 2020 v1 had the shorter title, so the old entry was not fabricated. | https://arxiv.org/abs/2002.07285 ; https://arxiv.org/abs/2002.07285v1 |

D'Amour's proceedings landing page and its PDF have different titles. The correction follows the actual paper PDF, which agrees with the supplied feedback. Current bibliography sources are corrected; archived historical manuscripts remain historical.

## Feedback decisions

| Feedback | Assessment and implementation |
|---|---|
| Posterior weighting is a strawman | The scope needed clarification. Add primary context on stepwise latent-model methods (Di Mari, Oberski and Vermunt, 2016; doi:10.1080/10705511.2016.1191015). Do not equate the exact ablation with all standard HMM regressions or joint likelihood methods. Proposition 1 uses oracle statewise nuisances; Lemma 1 handles nuisance estimation; Theorem 4 handles residual bias after posterior adjustment. |
| Misspecification implies an optimistic lower bound | Reject the universal mathematical claim. Posterior Gini can err in either direction. Mumbai's restart instability motivates caution but does not prove model misspecification, the asymptotic failure of every optimizer output, or a particular bias direction. Add a Section 6 footnote. |
| Bartlett bands implement the theoretical exponent | Reject that interpretation. They are finite-lag operational diagnostics and do not estimate or certify the beta-mixing exponent. The inference theorem now assumes absolute regularity of the augmented process. Fix the code so the ACF search cap cannot truncate its chosen logarithmic embargo floor. |
| Delhi 0.37 versus 0.234 | Clarify single-index posterior-feature versus coupled posterior-occupation estimates. Both are observational exposure associations, not known true ATEs. |
| Mumbai loses roughly 23% | Explain that the longest 277-hour run is one episode; the screen removes all qualifying runs, costing 1,857 complete cases (22.9%). It is a sensitivity exclusion, not proof of hardware failure or validation of remaining calibration. Counts are generated from the result ledger. |
| Short-horizon confounding | Preserve the distinction between residual regime confounding and other temporal confounding. A failed lead placebo prevents a causal emissions interpretation even when regime-channel sensitivity appears favorable. |
| Posterior-feature DML wins scalar ATEs | State this directly for Mumbai and Bengaluru. OR-DML's observed advantage concerns heterogeneous regime effects; no universal scalar-ATE superiority is claimed. |

## Corrections to theory and implementation

- The non-identification construction now satisfies proxy sufficiency: independent randomized treatment, independent memoryless states, and uninformative proxies. Its two binary outcome models have identical complete observable laws and different regime effect vectors. Their average effect is shared and identified.
- The inference appendix is rewritten for posterior-adjusted coupled moments. Its sufficient conditions include bounded clipped nuisances, an absolutely regular augmented process, separated calibration and nuisance samples, and an additional stochastic differentiability condition for estimated representations. It does not claim to prove inference for full-sample smoothing.
- Canonical HAC now uses the observation moments actually solved, including off-diagonal terms, matched-diagonal adjustments, and decoupled modes. Scores are centered at their regularized proxy target. The residual bootstrap uses the same moment matrices as the point solve.
- Empirical-share aggregation includes the effect--occupation cross-covariance through a combined HAC influence. The compatibility property `pate_se_` does not implement transition-MLE stationary inference.
- Implemented HAC and residual bootstrap uncertainty excludes HMM fitting uncertainty and structural latent bias. The appendix explains the additional representation derivative and calibration variance that full inference would require.
- Legacy uncertainty and rejection claims from the incorrect canonical variance implementation are retired. Historical point-estimation results remain as provenance. Current affected uncertainty results are regenerated by the scientific-reproduction workflow.
- Entropy and conditioning thresholds are abstention heuristics, not proofs of structural non-identification. A singular proxy-score system is not equivalent to failure of identification by every observable moment.

## Verification

Five numerical regression tests cover four score/bread mode combinations, score-mean invariance of HAC, occupation cross-covariance, bootstrap moment consistency, and the uncapped logarithmic embargo floor. Each defect was reproduced against the earlier implementation. The exact non-identification probability tables are enumerated by `src/synthesis/nonidentification_check.py`; `verify_submission.py` includes these checks and checks archive code/manuscript freshness.

The submission workflow compiles and checks the manuscript. The scientific-reproduction workflow independently recomputes the analytical identities and affected frontier, semi-synthetic, city and HAC-sensitivity results, then rebuilds and verifies the PDF and supplement. The registered abstract is preserved.
