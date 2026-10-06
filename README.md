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
| Change-point scaling | v ≈ √(8/π)·r·e^(−d²/8)/d is a fitted constant at moderate switch rates, accurate to ±10% (ρ ≤ 0.95). **Proved (Prop. cp):** for an isolated switch M = √(π/2)(q/d)(1 + O(d⁻² + d·q)), q = e^(−d²/8). The remainder beyond the nearest-neighbour term is rigorously O(d·q) via an exact tilting identity (x31 checks it); measured remainder 30%, 10%, 3% at d = 3, 4, 5. Cor. cp gives the r → 0 stationary limit (proof sketch). Total constants 1.39, 1.18, 1.11, 1.11, 1.13 for d = 2..6 (x28; match exact-HMM x19). |
| Learned parameters | **H2 is now a lemma** (Doeblin/Dobrushin + Fisher identity: moment, not almost-sure, derivative bound; x32). **H3 is replaced** by an explicit term δ_X (information X carries about the regime beyond Z), so the rate is N^(−1/2) + √δ_X. H1 is a hypothesis on the EM *output* (consistent root exists by the MLE theory; EM may land elsewhere, as in Mumbai) and is checked by multistart (x24-x27). δ_X > 0 on the real series (γ̂ predictable from X, R² ≈ 0.74). |
| Real data | A pipeline bug (`.dt.round('h')` ties-to-even on hh:30Z timestamps) created a 2-hourly series. The corrected hourly rerun is primary; Delhi θ̂ 0.58 → 0.37, with robustness value about 82 SD. Mumbai, Bengaluru and Kolkata are not informative (Kolkata: 16 distinct NO2 values, ends June 2025). The Newey-West lag sensitivity (e6) is recomputed on the hourly grid: Delhi stays significant at every lag (p ≤ 0.010); Bengaluru regime 2 is not robust (p 0.017 → 0.087 from lag 12 to 168). The old 2-hourly tables are kept as a comparison. |
| Bibliography | Each entry checked against web sources; two errors fixed (Tchetgen arXiv id, Pearl title/year). |
| Audits | Three independent skeptical audits; all findings fixed (K=3 claim, "exact" remark, remainder percentages, stale text). |

## Remaining limits (stated in the paper)
- (H1) cannot be proved for a local optimiser such as EM; it is a checked property of the output (fails for Mumbai).
- δ_X = 0 (the old H3) is a property of the data-generating process; it fails on the real series, where the rate degrades by √δ_X.
- Cor. cp (stationary r → 0 limit) is a proof sketch at the level of finite-window expansions; the isolated-switch Prop. cp is proved with unoptimised absolute constants.
- The plug-in v̂ is a lower bound under emission misspecification (factor 2 in x29).
- The cross-fitting independence argument is supported by x22/x30 and an outline, not a full proof.

## Final deliverables (this branch, `v2-bias-law`)
| File | What it is |
|---|---|
| `paper/main.tex` | Final manuscript source (17 pages with appendices; body ends on p. 8). Bibliography inline (`thebibliography`, 19 entries). |
| `paper/main.pdf` | Compiled final manuscript. |
| `AISTATS2027_OR_DML_Supplementary_Material.zip` | Supplementary archive (187 files): PDF, LaTeX source, code, data, `reports/`, manifests. Rebuilt by `python src/build_supplementary_archive.py`. |
| `docs/V2_CHANGELOG.md` | Round-by-round log (Rounds 10-15) of every audit, fix and finding. |

## Session work log (what was done, in order)
1. **Audit remediation and change-point reconciliation.** Closed the NN-vs-total gap in the change-point constant (nearest-neighbour term -> √(π/2) with relative gap π²/(2d²); total constant 1.12-1.19 at d = 3-5), replaced a Monte-Carlo-noise value, qualified the abstract and contributions (K = 3, learned-parameter rate, observability).
2. **Learned-HMM theory.** Rate proposition for learned parameters, cross-fit decorrelation argument, and EM checks on the fit actually used.
3. **Real-data pipeline bug.** `.dt.round('h')` ties-to-even on hh:30Z timestamps produced a 2-hourly series; the corrected hourly rerun is primary and the 2-hourly tables are kept for comparison. Kolkata/Mumbai coverage re-checked.
4. **Bibliography.** Every entry verified against web sources; two errors fixed; Wu (1983) added.
5. **Closing round.** Total isolated-change constant by importance sampling (x28); misspecified-emission study showing plug-in v̂ is a lower bound (x29); fold-wise vs global HMM (x30).
6. **Closing open items.** Newey-West rerun on the hourly grid; H2 proved as a lemma; H3 replaced by an explicit δ_X term; H1 restated as a checked hypothesis on the EM output; isolated-switch change-point proposition with a proved O(d·q) remainder; numerical checks (x31, x32); four independent audits with all findings applied.

## Index of new or changed files
**Manuscript and docs:** `paper/main.tex`, `paper/main.pdf`, `paper/generated/tab_cities*.tex`, `tab_cities_2h*.tex`, `tab_hac.tex`, `paper/plots/bl_fig*.pdf`, `README.md`, `docs/V2_CHANGELOG.md`.

**Library and drivers (changed):**
- `src/bias_law/bias_law.py` - closed forms plus Wiener smoother variance, sensor information, change-point v.
- `src/bias_law/real_cities_sensitivity.py` - `analyse(..., gamma=None)` accepts an external posterior; block = 72, 300 bootstrap draws.
- `src/bias_law/hac_sensitivity.py` - now on the hourly grid with the frozen-reading screen.
- `src/bias_law/make_figures_tables.py` - hourly city tables (x23 primary, e5 legacy), HAC table.
- `verify_science.py` - 12 checks (new: NN-limit, isolated-switch identity and bounds, Doeblin/envelope); `verify_artifacts.py` unchanged and passing.
- Unchanged and documented: `src/data_pipeline_clean.py` (bug documented, not overwritten), `src/or_dml.py`, `src/bias_law/sim.py`.

**Experiment scripts (`src/bias_law/ext_*.py`) -> outputs (`reports/bias_law/`):**
| Script | Output | Purpose |
|---|---|---|
| `ext_joint_proximal.py` | x1 | joint Markov-switching MLE and proximal 2SLS |
| `ext_contlat.py`, `ext_contlat_law.py`, `ext_kalman.py` | x3, x3b, x6 | continuous latent, Wiener/Kalman closed form |
| `ext_selfnc.py`, `ext_selfnc_check.py` | x4, x4b, x4c | self-contained proximal regime-DML |
| `ext_phase.py`, `ext_xaware.py` | x5, x8, x9 | phase diagram, finite-state bound, X-aware v |
| `ext_envreg.py`, `ext_envreal.py`, `ext_real_joint.py` | x10, x11, x7 | environment regression, real-data joint MLE |
| `ext_switchrate.py`, `ext_switchrate_k3.py`, `ext_asym.py`, `ext_oned_check.py`, `ext_apriori_check.py` | x12-x14, x17, x18 | change-point scaling and a priori law |
| `ext_singlechange.py`, `ext_hmm_limit.py`, `ext_hmm_window.py`, `ext_nn_quad.py` | x15, x16, x19, x20 | isolated-switch constant, exact-HMM windows, nearest-neighbour quadrature |
| `ext_hmm_rate.py`, `ext_cond_decay.py` | x21, x22 | learned-HMM rate, decorrelation of the smoothed chain |
| `ext_real_hourly_fix.py` | x23 | corrected hourly real-data analysis |
| `ext_em_check.py`, `ext_em_real.py`, `ext_real_bestem.py` | x24-x27 | EM convergence, multimodality, best-restart rerun |
| `ext_cp_is.py` | x28 | importance-sampled total isolated-switch constant (valid for d <= 6) |
| `ext_misspec_crossfit.py` | x29, x30 (+ `_wellspec`) | misspecified-emission v, fold-wise vs global HMM |
| `ext_cp_proof_check.py` | x31 | checks of the change-point proof ingredients |
| `ext_h2_lemma_check.py` | x32 | Doeblin/Dobrushin and derivative-envelope check |

Legacy: `reports/bias_law/e5_*` (2-hourly city results), `e6_hac_lag_sensitivity.csv` (now hourly).

## Key findings (summary)
See "What the final rounds found" below and the changelog. In one line each: the bias law is an OVB identity with v tied to the calibrated posterior; plug-in v̂ is a lower bound under misspecification; the isolated-switch scaling is proved with an O(d·q) remainder; the learned-parameter rate is N^(-1/2) + √δ_X; Delhi is the only informative city and its Newey-West conclusions are stable.

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
`src/bias_law/ext_*.py` write `reports/bias_law/x1..x32*.csv` (extensions, checks and audits; see changelog).

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
