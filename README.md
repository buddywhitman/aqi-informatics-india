# Reliable Causal Estimation under Latent Markov Confounding

Code, data and submission artifacts for the AISTATS 2027 manuscript of this title.

**Submission artifacts (authoritative, on `master`):**

| File | What it is |
|---|---|
| `AISTATS2027_Main_Paper.pdf` | The manuscript (identical to `paper/main.pdf`): 8-page main body, AI-use statement, references, checklist, appendices A–Q |
| `AISTATS2027_Supplementary_Material.zip` | The anonymized supplementary archive: code, data, every result file, LaTeX sources, verifier |

The title and abstract match the registered submission (`paper/LOCKED_ABSTRACT.txt`); `verify_submission.py` checks this.

## The paper in one paragraph

The standard recipe for regime-confounded time series is to infer the latent state with an HMM and then run regime-aware Double ML. It trusts the estimate when the posterior is confident and the score Jacobian is well conditioned. We show that this reliability rule fails in a specific, checkable way. A generated state inflates the apparent causal information: the smallest eigenvalue of the generated-state Gram matrix *grows* as state error grows. We call this **phantom causal resolution**.

We derive two exact laws, and they say how to build the estimator:
- the population displacement of the coupled regime score under state confusion (Proposition 1, which reproduces a 4,900-run factorial experiment to within 2%). It locates the failure in *weighting* regime nuisances by the posterior;
- conditioning the nuisances on the posterior instead makes the coupled score exactly Neyman-orthogonal for every posterior (Lemma 1), and leaves only the residual-state bias law (Theorem 4, exact for both the pooled and the regime-coupled estimator).

The laws explain the non-monotone error, the failure of ε/λ_min once posteriors are learned (rank correlation 0.995 → 0.52), and why soft posteriors help as features but hurt as weights. OR-DML with posterior-adjusted nuisances is the most accurate regime-aware estimator in every synthetic frontier cell and in every city of the semi-synthetic benchmark (regime effects); the posterior-weighted design of earlier drafts is kept as an ablation. The evidence includes:
- a real-data-calibrated semi-synthetic benchmark on four Indian cities, on a corrected hourly grid;
- an audited observational stress test with aligned placebos;
- a sensitivity analysis with robustness values.

## Reproduce and verify

```bash
pip install -r requirements.txt
python verify_submission.py --fast   # every number in the PDF vs the result files, page budget, anonymity
bash reproduce.sh                    # rerun core experiments, rebuild figures/tables/PDF/archive, verify (~12 min, 2 cores)
bash reproduce.sh --full             # also rerun the long legacy benchmarks
```

Every number quoted in the manuscript is a LaTeX macro in `paper/generated/numbers.tex`, and every results table is in `paper/generated/tab_*.tex`. Both are written only by `src/synthesis/make_tables_numbers.py` from `reports/`. The ledger `paper/generated/claims.csv` maps each number to its source file. CI (`.github/workflows/submission.yml`) reruns the verifier and the LaTeX build on every push; it never commits.

## Layout

```text
paper/                 main.tex, appendix_src.tex (+ build_appendix.py), checklist.tex, references, generated/, plots/
src/or_dml.py          OR-DML estimator (posterior inference, posterior-adjusted or -weighted nuisances, purged
                       cross-fitting, spectral regularization, HAC); nuisance_mode='posterior' is the paper's design
src/synthesis/         consolidation experiments: closed form, orthogonality and coupled-law checks, observability x
                       persistence frontier, learned posteriors, semi-synthetic, city stress test, figures, tables/numbers
src/bias_law/          residual-state bias-law experiments and real-data sensitivity analysis (corrected hourly grid)
src/*.py               factorial, frontier, representation zoo, scalability, soft-vs-hard and legacy analyses
reports/               all result files (reports/synthesis, reports/bias_law)
data/raw_hourly/       raw OpenAQ/CPCB pollutant and Open-Meteo weather files
archive/               earlier manuscript lineages, legacy verifiers and submission documents, legacy drafts
research/              research ledger (phantom resolution, directional reliability, etc.)
```

## Branch history

`master` consolidates the following branches. The submitted draft (`feature/or-dml-phase2-pivot`) supplied the locked abstract and the estimator theory. `v2-bias-law` supplied the residual-state bias law and the time-grid audit. `research/task-relative-reliability` supplied phantom resolution and directional reliability. `v2` is an ancestor of the submitted draft. The other branches hold the original applied study (`agent-*`), a stale snapshot (`imgbot`) and earlier submission integrations (`submission/*`).

Appendix Q of the paper records every claim that changed during consolidation: the two-hour time grid, the placebo alignment, the scope of ε/λ_min, the estimator design and ranking, and the analyses retired from the evidence.
