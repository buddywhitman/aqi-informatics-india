# AISTATS 2027 Supplementary Material

This anonymous archive accompanies the submission **Reliable Causal Estimation under Latent Markov Confounding**.

## Reproduction order

1. Install the pinned dependencies in `src/requirements.txt`.
2. Run `python verify_science.py` for estimator identities and scientific invariance checks.
3. Run `python verify_artifacts.py` for report/figure/manuscript synchronization.
4. The primary implementation is `src/or_dml.py`; experiment scripts and their committed CSV outputs are listed in `SUPPLEMENT_ROADMAP.md`.
5. `paper/main.tex` and `paper/aistats2027.sty` reproduce the anonymous manuscript.

## Evidence hierarchy

- The central claim is phantom causal resolution: generated latent states can distort apparent downstream causal information while residual confounding remains.
- `reports/` contains canonical numerical outputs cited by the manuscript.
- `research/` contains selected post-audit stress tests included to delimit claim scope and document negative results.
- Observational sensor analyses are stress tests, not causal ground truth. `data/processed_clean/combined_hourly_clean.csv` is retained for legacy/canonical experiment reproducibility; the corrected-hourly bias-law audit is separately reproduced by `src/bias_law/real_cities_sensitivity.py` and associated audit outputs, because the earlier processed file is affected by the timestamp-rounding issue documented in `docs/V2_CHANGELOG.md`.
- Synthetic oracle experiments are labeled as such.

## Anonymity

The archive intentionally contains no author names, affiliations, acknowledgements, repository URLs, commit URLs, or other identifying links. Public code/data release information can be added after the double-blind review period.

## AI disclosure

The manuscript's mandatory AI Use Statement describes the hybrid human-AI research and writing workflow. The authors take responsibility for all code, mathematical statements, empirical results, citations, and conclusions.
