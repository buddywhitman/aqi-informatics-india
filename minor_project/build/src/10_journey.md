# Project Journey: From Applied Study to a Research Contribution

A doctoral-style report is expected to say not only what was found but how the finding was reached and what was discarded on the way. This chapter does so. The project passed through three phases, and the audits at the phase boundaries are as much a result as the final theorems.

## 10.1 Phase 1: the applied air-quality study (March–September 2026)

The project began as an applied spatio-temporal air-quality informatics study across Indian cities. The codebase of this phase (preserved in `archive/pre-pivot/`) assembled hourly OpenAQ/CPCB pollution and Open-Meteo weather data, explored seasonal and spatial structure, and benchmarked forecasting models: LSTM and CNN-LSTM hybrids, a Temporal Fusion Transformer, random forests, LightGBM and stacking ensembles. It then attempted policy simulation (emission-reduction scenarios) and drafted a long manuscript aimed at a multi-disciplinary environmental journal.

The phase produced real engineering value: a robust ingestion and cleaning pipeline, a model benchmarking harness, and a clear empirical finding that tree ensembles matched or beat the deep sequence models on lagged meteorological tables. It also produced the claims that did not survive scrutiny.

## 10.2 The audit

Before submission, a forensic audit of the draft, performed by the team the way a skeptical reviewer would read it, found four problems serious enough to make the manuscript unpublishable at a top machine-learning venue.

@tab audit | Findings of the forensic audit of the applied draft
| Finding | What was wrong | Consequence |
|---|---|---|
| Decorated arithmetic | Health and economic benefits were computed by multiplying a pollution change by an arbitrary scalar, with a code comment that contradicted the constant used | headline economic claims had no derivation |
| A sign-flipping `abs()` | In Mumbai a naive model estimated a *negative* effect; taking the absolute value turned a worsening into thousands of "lives saved" | the policy simulation was invalid |
| Internal inconsistencies | MSE and RMSE in one table were reported in different units; a README cited a model score that was absent from the results; stage directions from an AI draft survived in the text | credibility of the whole table set |
| Baseline paradox | The paper advertised deep hybrids although its own benchmark showed tree ensembles winning | central claim undercut |

The most instructive item is the first. A downstream "causal" number computed as pollution change times a scalar is not causal inference; it is arithmetic wearing a causal label. The remedy was not to patch the scalar but to retire every health and economic figure and to ask a harder question: *what does it take to estimate a causal effect from this kind of data at all?*

## 10.3 Phase 2: the methodological pivot (late September 2026)

The audit redirected the project towards a genuine open problem in causal machine learning: environmental time series violate standard DML in two ways, **latent non-stationary confounding** and **temporal autocorrelation** in cross-fitting. Phase 2 formulated *Regime-Conditional DML* (RC-DML): a coupled, multi-regime orthogonal score solved by inverting the full $K\times K$ Jacobian rather than regime-by-regime, purged block cross-fitting, and a variance decomposition for stationary averages. It produced the first versions of the omitted-regime bias decomposition, the non-identification result and the coupled-score theory used in this report, the first synthetic benchmarks (naive OLS bias of +525% against near-zero for RC-DML in the early design), and a real-data evaluation across three cities.

The pivot was documented as an architecture decision record, and the repository was restructured around the new contribution; the 30-page narrative was discarded.

## 10.4 Phase 3: consolidation and correction (October 2026)

Phase 3 is where most of the intellectual progress of the project occurred, and it came from *trying to break its own results*. Several development lines were merged on `master`: the submitted draft (which supplied the registered abstract and estimator theory), a residual-state bias-law analysis with a time-grid audit, and a research ledger on task-relative reliability (phantom resolution, directional reliability). Audits of the merged work changed the claims as follows.

@tab corr | Claims that changed during consolidation
| # | Earlier claim or artifact | Audit finding | Resolution |
|---|---|---|---|
| 1 | "Hourly" real-data lags | the panel sat on a two-hour grid | all real-data numbers recomputed (Section 6.3) |
| 2 | Placebo leads | not aligned to the outcome's information set; one-hour lead mechanically zero | aligned placebos |
| 3 | $\varepsilon_\gamma/\lambda_{\min}$ as a general reliability predictor | the factorial's 0.995 is algebraic; learned posteriors give 0.517 | Proposition 1 and Corollary 1 |
| 4 | Posterior-*weighted* nuisances as the method of choice | hard fixed effects beat it in every benchmark | posterior-*adjusted* design; weighted design kept as ablation |
| 5 | Semi-synthetic benchmark with the true state a function of the proxies | circular | replaced by real-data-calibrated benchmark with sampled states |
| 6 | Financial transfer, imputation sensitivity, multi-horizon forecasting tables, policy simulation | built on the legacy grid or unsupported | retired; code kept in the archive |
| 7 | Regularization numbers | stale | recomputed from the committed path |

### Scientific corrections after external feedback (7 October 2026)

A second round of review identified errors of mathematics and of claim scope, summarized below. Each was reproduced against the earlier implementation before it was fixed.

* **Non-identification proof.** The earlier construction did not satisfy proxy sufficiency. The replacement uses independent randomized treatment, independent memoryless states and uninformative proxies, with identical observable laws and different regime effects (Theorem 2 as stated here).
* **Inference appendix.** Rewritten for posterior-adjusted coupled moments under explicit sufficient conditions (bounded clipped nuisances, an absolutely regular augmented process, separated calibration and nuisance samples); full-sample smoothing is excluded.
* **HAC variance.** The canonical HAC now uses the moments actually solved, including off-diagonal terms, with scores centered at their regularized proxy target; the bootstrap uses the same moment matrices. Legacy uncertainty claims from the incorrect implementation were retired.
* **Embargo.** A search cap in the ACF routine could truncate the logarithmic floor; this was fixed and tested.
* **Bartlett bands** were reclassified as finite-lag operational diagnostics, not an estimate of the mixing exponent.
* **Bibliography.** Six references were checked against primary sources and corrected (titles, authors, arXiv identifiers).
* **Feedback assessed and partly rejected.** A suggestion that misspecification implies an optimistic lower bound was rejected as a universal claim (the posterior's Gini can err in either direction). The Mumbai restart instability motivated a footnote, not a theorem.

Five numerical regression tests were added, and the verifier now checks code–manuscript freshness.

## 10.5 What we learned

1. **Verification beats persuasion.** Every real improvement in the project was the result of an attempt to falsify it, whether a grid audit, an aligned placebo or an exact law.
2. **A good diagnostic is an object to be tested, not trusted.** The most widely used reliability rule was shown to fail in a precisely characterized way.
3. **Simple baselines are mandatory.** Hard fixed effects and posterior-feature DML were the competitors that forced the design change; the project's first estimator lost to them.
4. **State the scope of every result.** Delhi is an association, three other cities are inconclusive, and every estimator fails at weak separation. These sentences cost nothing and protect the whole.
5. **Single sources of truth for numbers** remove an entire class of error, and the cost is a few hundred lines of generating code.

## 10.6 Research output: the AISTATS conference

The research output required of this minor project is a peer-reviewed submission. Ours is the manuscript *Reliable Causal Estimation under Latent Markov Confounding*, submitted to the 30th **International Conference on Artificial Intelligence and Statistics (AISTATS 2027)**. This section explains what that venue is and how it is rated, because in computer science and artificial intelligence the standing of a conference differs from the convention in electrical and other engineering disciplines, where journals are the usual benchmark.

**Conferences as primary publication venues.** In machine learning and statistics, the top conferences are full-length, archival, peer-reviewed publication venues, not presentation forums. A complete paper is submitted, assessed in a double-blind review (typically three or more independent reviewers, then meta-review and an author rebuttal), and the accepted papers are printed in the conference proceedings, which are indexed in bibliographic databases such as DBLP and Google Scholar. The pace of the field is the reason: a journal review cycle of a year or more is too slow for it, and the leading results appear first at conferences.

**What AISTATS is.** AISTATS began as a small workshop series on artificial intelligence and statistics in Princeton, USA, in 1985, and grew into an annual conference. It is organised by the research community in the area where machine learning meets statistics. Its proceedings are published in the *Proceedings of Machine Learning Research* (PMLR), the series associated with the *Journal of Machine Learning Research*. The venue is known for rewarding theoretically grounded work with careful empirical validation, which is the character of this project.

@tab venue | How AISTATS is rated and evaluated
| Measure | What it means | AISTATS |
|---|---|---|
| CORE rank (Computing Research and Education Association of Australasia, ICORE portal) | Community-run classification of conferences: A* (flagship), A (excellent), B (good), C (other) | **A**, in the 2017, 2018, 2020, 2021 and 2023 rankings |
| Acceptance rate | Share of submissions accepted; a lower share means a more selective review | about **31%** in 2025 (583 of 1,861 submissions); about 29% in 2015; historically 25-31% |
| Submission volume | Size of the competing field | roughly 1,900 full-paper submissions per year |
| h5-index (Google Scholar Metrics) | Largest number *h* such that *h* papers from the last five complete years each have at least *h* citations; the standard citation-based venue score | 101 (Google Scholar Metrics, as read by the authors; five-year window of 2020 to 2024) |
| Publication record | Where accepted papers appear and are indexed | PMLR; indexed by DBLP and Google Scholar |

**Reading these numbers.** CORE A places AISTATS in the second tier of the four-tier scheme, directly below the handful of flagship conferences (such as NeurIPS, ICML and ICLR), and it is commonly treated as comparable in standing to a good journal in the field. An acceptance rate near 30% means that roughly seven of ten submissions, many of them strong, are rejected. The h5-index is the venue-level analogue of a journal's impact factor, and a value of 101 means that its top 101 papers of the last five years each have at least 100 citations.

**Status of our submission.** The manuscript, with its eight-page main body, appendices A-Q, and a reproducible supplementary archive, has been submitted to AISTATS 2027 and is under peer review. Its title and abstract match the registered submission, which the verifier checks. Acceptance is not asserted anywhere in this report; the outcome will be decided by the reviewers. The work was therefore formulated to the standard of a peer-reviewed venue of this rank and has entered its review process.
