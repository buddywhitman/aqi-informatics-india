# Supplementary Material: Reliable Causal Estimation under Latent Markov Confounding

This archive holds everything needed to rebuild and check every number, table and figure in the
manuscript. It contains the code, the raw and processed data, all result files, the LaTeX sources and a verifier.

## Quick start

```bash
pip install -r requirements.txt
python verify_submission.py --fast     # ~1 minute: checks every number in the PDF against the result files
bash reproduce.sh                      # ~12 minutes on 2 CPU cores: reruns the core experiments, rebuilds PDF + archive
bash reproduce.sh --full               # also reruns the long legacy benchmarks (frontier, bias-law suite, zoo; hours, needs torch)
```

## How numbers reach the paper

1. Experiments write CSV/JSON files under `reports/`.
2. `src/synthesis/make_tables_numbers.py` reads only those files. It writes `paper/generated/numbers.tex`, one LaTeX macro per number quoted in the text, plus every results table `paper/generated/tab_*.tex`, and a ledger `paper/generated/claims.csv` (macro, value, source file, derivation).
3. The manuscript uses only these macros and tables for quoted results.
4. `verify_submission.py` regenerates them into a temporary directory and fails unless they are byte-identical to the committed copies. It also:
   - re-evaluates the closed form of Proposition 1 against the factorial data and recomputes its symbolic identities (unless `--fast`);
   - runs estimator invariance tests;
   - checks the page budget and that the PDF is not stale;
   - scans the PDF and this archive for de-anonymizing strings.

## Map from claims to code

| Claim (manuscript) | Script | Output |
|---|---|---|
| Figure 1a-b, Proposition 1, Appendix F (exact regime-weighted score law vs 4,900-run factorial) | `src/factorial_reliability_experiment.py`, `src/synthesis/closed_form_factorial.py` | `reports/factorial_reliability_*.csv`, `reports/synthesis/closed_form_factorial_*` |
| Lemma 1, Theorem 4, Appendix E (exact orthogonality; residual-state law for posterior-adjusted designs) | `src/synthesis/coupled_law_check.py` | `reports/synthesis/orthogonality_check.csv`, `coupled_law_check.csv` |
| Figure 1c, Table 1 (learned HMM posteriors, phantom inflation, estimator comparison) | `src/synthesis/learned_posterior_coupled.py` | `reports/synthesis/learned_posterior_coupled_*` |
| Figure 2, Appendix G (pooled bias law with learned HMMs) | `src/bias_law/run_experiments.py`, `src/bias_law/ext_*.py` | `reports/bias_law/e*.csv`, `x*.csv` |
| Figure 3a-b, Table 2 left, Appendix L (observability x persistence frontier, 8 estimators) | `src/synthesis/frontier_designs.py` (estimators in `src/synthesis/estimators.py`) | `reports/synthesis/frontier_designs_*` |
| Appendix L (legacy 11-method benchmark, Frontier B size, regularization path) | `src/synthetic_dgp_benchmark.py` | `reports/or_dml_benchmark_summary.csv`, `reports/frontier_*`, `reports/or_dml_regularization_frontier.csv` |
| Figure 3c, Table 2 right, Appendix M (real-data-calibrated semi-synthetic benchmark) | `src/synthesis/semisynthetic_hourly.py` | `reports/synthesis/semisynthetic_hourly_*` |
| Table 3, Appendix O (four-city stress test, aligned placebos, audit, HAC lags, robustness values) | `src/synthesis/city_stress_hourly.py`, `src/bias_law/ext_real_hourly_fix.py`, `src/bias_law/hac_sensitivity.py` | `reports/synthesis/city_*`, `reports/bias_law/x23_real_hourly_fix.csv`, `e6_hac_lag_sensitivity.csv` |
| Appendix N (representation zoo, calibration, LOWO) | `src/train_real_representation_zoo.py`, `src/calibration_intervention_zoo.py`, `src/hierarchical_reliability_regression.py` | `reports/representation_zoo_*` |
| Appendices K-L (continuous latent mixture; numerator scale) | `src/test_soft_vs_hard_continuous_mixture.py`, `src/deep_empirical_evaluations.py` | `reports/synthesis/soft_vs_hard_*`, `reports/deep_eval_*` |
| Appendix J (K = 2..5 scalability) | `src/test_multiregime_scalability.py` | `reports/multiregime_scalability_summary.csv` |

## Data

- `data/raw_hourly/`: raw pollutant readings (CPCB stations via OpenAQ, timestamps at hh:30 UTC) and Open-Meteo weather.
- `data/processed_clean/`: the legacy processed panel. Its timestamps were rounded with ties-to-even, which collapses the series onto a two-hour grid; see Appendix O. It is kept only for provenance, and all real-data results in the paper rebuild a corrected hourly grid from `data/raw_hourly` with `src/bias_law/ext_real_hourly_fix.py`.

## Provenance

`archive/manuscript_lineages/` holds the earlier manuscripts this paper consolidates: the submitted overlap-aware DML draft, the residual-state bias-law manuscript (with full proofs of the change-point and Wiener results summarized in Appendix G) and the pre-synthesis integrated draft. `research/` is the research ledger behind the phantom-resolution and directional-reliability results. Appendix Q of the paper lists every claim that changed during consolidation and why.

Some result files in `reports/` come from legacy analyses that the paper no longer uses as evidence: the financial transfer, the policy simulation, failure forecasting and the imputation table on the two-hour grid. They are kept for transparency and are not referenced by any macro.

## Corrected inference scope

The current non-identification proof uses independent randomized treatment and verifies the entire observable probability table, including proxy sufficiency. It proves that the regime effect vector is unidentified; its average effect is shared and identified. Run `python src/synthesis/nonidentification_check.py`.

The inference appendix now proves the posterior-adjusted coupled case under explicit sufficient boundedness, absolute-regularity and calibration conditions. The canonical HAC score matches the normal equation, is centered at the regularized proxy target, and includes effect--occupation cross-covariance for empirical-share aggregation. Run `python -m unittest discover -s src -p test_inference_regression.py -v`.

Implemented intervals exclude HMM fitting uncertainty and structural latent bias. They do not certify structural coverage for full-sample smoothing. Compact benchmark ATE intervals treat empirical shares as fixed and their simulated coverage is a diagnostic. Legacy uncertainty columns and null-rejection percentages produced by the earlier canonical score calculation are retired; historical CSV files remain for provenance. No universal lower-bound claim is made for misspecified posterior Gini, and entropy/conditioning thresholds are abstention heuristics rather than identification tests.
