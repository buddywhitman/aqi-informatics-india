#!/usr/bin/env bash
# Reproduce every number, table and figure of the manuscript, rebuild the PDF and the supplementary archive,
# and verify. Default: fast path (~12 minutes on 2 CPU cores). --full also reruns the long legacy benchmarks.
set -euo pipefail
cd "$(dirname "$0")"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1

if [[ $FULL -eq 1 ]]; then
  python src/synthetic_dgp_benchmark.py              # difficulty frontier, Frontier B/C, regularization path
  python src/bias_law/run_experiments.py             # residual-state bias-law suite (E1-E7)
  python src/bias_law/ext_real_hourly_fix.py         # corrected-hourly sensitivity analysis (robustness values)
  python src/train_real_representation_zoo.py        # representation zoo
  python src/calibration_intervention_zoo.py
  python src/hierarchical_reliability_regression.py
  python src/test_multiregime_scalability.py
fi

python src/factorial_reliability_experiment.py       # 7x7 factorial (4,900 runs)
python src/synthesis/closed_form_factorial.py        # Proposition 1 vs factorial + symbolic checks
python src/synthesis/coupled_law_check.py            # Lemma 1 orthogonality + Theorem 4 law (N = 1e6 draws)
python src/synthesis/frontier_designs.py             # observability x persistence frontier, 8 estimators (900 datasets)
python src/synthesis/learned_posterior_coupled.py    # learned-HMM posteriors (720 runs)
python src/synthesis/semisynthetic_hourly.py         # real-data-calibrated semi-synthetic benchmark
python src/synthesis/city_stress_hourly.py           # four-city stress test + aligned placebos
python src/bias_law/hac_sensitivity.py               # HAC lag sensitivity, posterior-adjusted OR-DML, corrected grid
python src/test_soft_vs_hard_continuous_mixture.py   # soft vs hard, continuous mixture

python src/synthesis/make_figures.py
python src/synthesis/make_tables_numbers.py

( cd paper && python build_appendix.py && python prune_bib.py \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null )
cp paper/main.pdf AISTATS2027_Main_Paper.pdf
python src/build_supplementary_archive.py
python verify_submission.py
