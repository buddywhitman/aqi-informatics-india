# Engineering, Reproducibility and Verification

## 9.1 Repository architecture

The project is a research codebase of 496 commits (23 March 2026 to 7 October 2026). Its structure is designed so that a reviewer can move from a sentence in the paper to the file that produced it.

@tab repo | Repository layout
| Path | Contents |
|---|---|
| `paper/` | `main.tex`, appendix sources, checklist, bibliography; `generated/` holds `numbers.tex`, `claims.csv` and all `tab_*.tex` |
| `src/or_dml.py` | the OR-DML estimator (HMM, purged folds, coupled score, spectral solve, HAC) |
| `src/synthesis/` | consolidation experiments: closed form, orthogonality and coupled-law checks, frontier, learned posteriors, semi-synthetic, city stress test, figures, tables |
| `src/bias_law/` | residual-state bias-law experiments and real-data sensitivity analysis |
| `src/*.py` | factorial, frontier, representation zoo, scalability, soft-versus-hard, legacy analyses |
| `reports/` | all result files (CSV/JSON) |
| `data/raw_hourly/` | raw OpenAQ/CPCB and Open-Meteo hourly files |
| `archive/` | earlier manuscript lineages, legacy verifiers and drafts, kept for transparency |
| `research/` | research ledger (phantom resolution, directional reliability, and others) |

@svg commits | timeline | **Project timeline.** Three phases of the project (Chapter 10) on a calendar axis, from the first commit to this report.

## 9.2 A single source for every number

Every numeric claim in the manuscript's main text is a LaTeX macro defined in `paper/generated/numbers.tex` (247 macros), and every results table is an `\input` of `paper/generated/tab_*.tex`. Both are written *only* by `src/synthesis/make_tables_numbers.py` from committed CSV/JSON outputs. The ledger `paper/generated/claims.csv` maps each macro to its value, source file and derivation. As an example, the headline Spearman correlation is traceable as follows.

```text
macro,value,source,how
FacRhoD,0.995,reports/factorial_reliability_correlations.csv,Spearman_Run_Difficulty
```

This design eliminates the most common source of error in research writing: a number typed into a paper that no longer matches the experiment. The numbers quoted in this report were taken from the same files.

## 9.3 The verifier

`verify_submission.py` performs five checks, and `python verify_submission.py --fast` runs them in minutes.

1. Regenerate all macros and tables into a temporary directory and require *byte-identical* output.
2. Re-evaluate the closed form and its symbolic identities.
3. Check that every macro used in the manuscript is defined and every defined macro is used or listed.
4. Check that the PDF is newer than its sources and that the main body is eight pages.
5. Scan the PDF and the supplementary archive for de-anonymizing strings (the submission is double-blind).

`verify_science.py` independently recomputes the analytical identities and the affected frontier, semi-synthetic, city and HAC-sensitivity results.

## 9.4 One-command reproduction

```bash
pip install -r requirements.txt
python verify_submission.py --fast   # PDF numbers vs result files
bash reproduce.sh                    # core experiments + rebuild (~12 min)
bash reproduce.sh --full             # plus long legacy benchmarks
```

`reproduce.sh` reruns the synthesis experiments (about ten minutes on two CPU cores), the factorial experiment (about 20 seconds), regenerates figures, tables and `numbers.tex`, and then runs the verifier. Two GitHub Actions workflows, `submission.yml` and `scientific-reproduction.yml`, rerun the verifier and the LaTeX build on every push; neither commits to the repository.

## 9.5 Architecture decision records

Seven architecture decision records in `docs/adr/` document why the project's direction changed, including the methodological pivot, the RC-DML formulation, the overlap-aware pivot, the empirical diagnostics and figure standards, the oracle score-difference audit, the editorial compression of the paper, and the repository purge that consolidated the project's scientific identity. They are summarized in Chapter 10.

## 9.6 Supplementary material

The anonymized supplementary archive (`AISTATS2027_Supplementary_Material.zip`) contains the code, data, every result file, the LaTeX sources and the verifier, so that a reviewer can regenerate the paper from the archive alone.
