# A Calibrated-Posterior Residual-Confounding Law for Latent-Regime DML

Code and reproducibility artefacts for the AISTATS 2027 submission (v2 manuscript `paper/main.tex`, compiled `paper/main.pdf`, 16 pages; v1: `paper/main_v1.tex`). Branch `v2-bias-law`.

## Result
With a persistent latent state S that shifts treatment (ΔT) and outcome (Δg), conditioning on an HMM posterior γ̂ leaves

    θ̂ − θ → aᵀΣb / (aᵀΣa + σ²)        (K = 2:  ΔT·Δg·v / (ΔT²·v + σ²)),   v = E Var(S | X, γ̂)

The bias is linear in Δg, monotone in v, and non-monotone in ΔT (peak at |ΔT| = σ/√v, maximum |Δg|√v/(2σ)). So the Gram-matrix conditioning λ_min of the residualised treatment is not a monotone reliability index. Δg is the one unidentified parameter, which gives a one-parameter sensitivity analysis and a robustness value b*.

**Honest framing.** This is an omitted-variable-bias identity with a partially observed confounder, not a new estimator. The additions are (i) identifying the residual variance with the calibrated HMM posterior under purged cross-fitting, (ii) the non-monotone-conditioning consequence, (iii) a mostly observable sensitivity analysis for latent Markov regimes. The rate result covers the smoother and the unrestricted v only, under assumptions H1-H3.

## What the final rounds found (details in `docs/V2_CHANGELOG.md`, Rounds 10-14)
| Topic | Finding |
|---|---|
| Plug-in v̂ | Exact only for the realised residual variance E(S−γ̂)². With well-specified emissions v̂ understates it by about 14%. With misspecified emissions (diagonal Gaussian HMM, correlated t5 proxies) v̂ is HALF of it (0.059 vs 0.122): the plug-in law predicts 0.168, the realised-variance law 0.326, observed 0.308 (x29). v̂ is a lower bound, not an estimate, under misspecification. |
| Cross-fitting the HMM | Fitting it on training folds only vs on all of Z gives bias correlation 0.999 (x30). Evidence, not a proof of independence. |
| Change-point scaling | v ≈ √(8/π)·r·e^(−d²/8)/d is a fitted constant, accurate to ±10% (ρ ≤ 0.95). The asymptotic constant √(π/2) = 1.2533 is approached as d⁻². Total isolated-change constant: 1.39, 1.18, 1.11, 1.11, 1.13 for d = 2..6 (x28; matches exact-HMM x19). Remainder is numerical, not proved. d ≥ 8 not estimable by importance sampling. |
| Learned parameters | Rate proposition under H1-H3 (smoother, unrestricted v). EM checks (x24-x27): converged in simulation; Mumbai multimodal; H3 fails on the real series (γ̂ is predictable from X, R² ≈ 0.74). |
| Real data | A pipeline bug (`.dt.round('h')` ties-to-even on hh:30Z timestamps) created a 2-hourly series. The corrected hourly rerun is primary; Delhi θ̂ 0.58 → 0.37, with robustness value about 82 SD. Mumbai, Bengaluru and Kolkata are not informative (Kolkata: 16 distinct NO2 values, ends June 2025). The old 2-hourly tables are kept as a comparison. |
| Bibliography | Each entry checked against web sources; two errors fixed (Tchetgen arXiv id, Pearl title/year). |
| Audits | Three independent skeptical audits; all findings fixed (K=3 claim, "exact" remark, remainder percentages, stale text). |

## Open gaps (not closed)
H1-H3 unproved for EM as implemented. The change-point remainder and the cross-fit independence are supported numerically only. The Newey-West appendix (e6) still uses the legacy 2-hourly series. Page ranges for four bibliography entries (Douc 474-513, Leroux 127-143, Chernozhukov C1-C68, two books) are unconfirmed. **The OpenAQ API key that was used in earlier history must be revoked by its owner.**

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
`src/bias_law/ext_*.py` write `reports/bias_law/x1..x30*.csv` (extensions, checks and audits; see changelog).

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
