# A Calibrated-Posterior Residual-Confounding Law for Latent-Regime DML

Code and reproducibility artefacts for the AISTATS 2027 submission (v2 manuscript: `paper/main.tex`; v1: `paper/main_v1.tex`).

**Core result.** With a persistent latent state S that shifts treatment (by ΔT) and outcome (by Δg), conditioning on a calibrated HMM posterior γ̂ leaves a residual-state bias

    θ̂ − θ → aᵀΣb / (aᵀΣa + σ²)        (K = 2:  ΔT·Δg·v / (ΔT²·v + σ²)),   v = E Var(S | X, γ̂),

estimable from data (v̂ = mean γ̂(1−γ̂)), non-monotone in the treatment shift (peak at |ΔT| = σ/√v), monotone in v, and bounded under miscalibration by √CE + CE. It yields a robustness value b* for real data. It is an omitted-variable-bias identity specialised to HMM posteriors and dependent data; see `docs/V2_CHANGELOG.md` for scope, errata and limitations.

## Layout
```
paper/            main.tex (v2), main_v1.tex (previous), main.pdf, generated/ (auto tables), plots/ (bl_* auto figures)
src/or_dml.py     OverlapAwareRegimeDML (HMM posterior + purged block cross-fitting + HAC)
src/bias_law/     bias_law.py (closed forms, plug-ins) | sim.py | run_experiments.py (E1-E4,E7)
                  real_cities_sensitivity.py (E5) | hac_sensitivity.py (E6) | make_figures_tables.py
reports/bias_law/ CSV/JSON outputs consumed by the paper
data/processed_clean/combined_hourly_clean.csv   14,122 hourly records, 4 Indian cities
verify_science.py   simulation-vs-theory tests (~1-2 min)
verify_artifacts.py file / number consistency checks
docs/             V2_CHANGELOG.md, bestpaper.md
```
`src/three_decisive_experiments.py` and other pre-v2 scripts are kept for provenance; Exp. 29/30 are superseded (buggy).

## Reproduce
```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1                         # avoids OpenMP oversubscription with joblib workers
python src/bias_law/run_experiments.py           # E1-E4, E7   (~minutes)
python src/bias_law/real_cities_sensitivity.py   # E5
python src/bias_law/hac_sensitivity.py           # E6
python src/bias_law/make_figures_tables.py       # figures, tables, summary_numbers.json
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python verify_science.py && python verify_artifacts.py
```
Data access keys must be supplied through environment variables; none are stored in the repo.
