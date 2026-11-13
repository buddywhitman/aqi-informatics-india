# !Annexure A: Team and Project Administration

@tab team | Project team and supervision
| Name | Registration No. | School | Role |
|---|---|---|---|
| Pulkit Kumar | 230906192 | Electrical Engineering | team member |
| Utsav Verma | 230906526 | Electrical Engineering | team member |
| Nishant Kumar Upadhyay | 230906522 | Electrical Engineering | team member |
| Dr. Savitha G | n/a | Computer Engineering | project guide |

**Programme.** Minor in Data Science, 8 credits, 7th semester, academic year 2026-27, towards the Bachelor of Technology degree in Electrical and Electronics Engineering of Manipal Institute of Technology. **Required output.** A peer-reviewed research submission (the AISTATS 2027 manuscript).

# !Annexure B: AI-Use Statement

Large language models assisted with code, derivations, experiment scripting and drafting. Every number in the manuscript is generated from committed artifacts by a single script (`src/synthesis/make_tables_numbers.py`) and checked by `verify_submission.py`, which re-derives all tables and macros and fails on any mismatch. The closed forms are verified symbolically, and the errors found in audit in earlier drafts (time grid, placebo alignment, overstated estimator claims) are documented in Chapter 10 and in Appendix Q of the manuscript. The authors reviewed all arguments and take full responsibility for the content. This report was also prepared with AI assistance; its numbers are drawn from the same generated files and its figures from the committed plots.

# !Annexure C: Key Code Listings

**C.1 Embargo selection** (`src/or_dml.py`). Finite-lag autocorrelation bands choose the embargo, floored at $C\log N$.

```python
def estimate_optimal_embargo(X, Z, max_tau=24, c_log=2.0):
    N = len(X); crit = 1.96 / np.sqrt(N)
    V = np.column_stack([X, Z]); V_centered = V - np.mean(V, axis=0)
    var_V = np.var(V, axis=0) + 1e-12
    search_tau = min(max_tau, max(12, int(N // 10)))
    significant_lags = [0]
    for h in range(1, search_tau + 1):
        cov_h = np.mean(V_centered[:-h] * V_centered[h:], axis=0)
        if np.any(np.abs(cov_h / var_V) >= crit):
            significant_lags.append(h)
    bartlett_tau = max(significant_lags) + 1
    min_log_tau = int(np.ceil(c_log * np.log(max(N, 2))))
    return int(max(bartlett_tau, min_log_tau, 1))
```

**C.2 Purged block cross-validation.**

```python
class PurgedBlockKFold:
    def split(self, N):
        indices = np.arange(N)
        sizes = np.full(self.n_splits, N // self.n_splits, dtype=int)
        sizes[:N % self.n_splits] += 1
        current = 0
        for size in sizes:
            test_idx = indices[current:current + size]
            train_mask = np.ones(N, dtype=bool)
            lo = max(0, current - self.embargo_tau)
            hi = min(N, current + size + self.embargo_tau)
            train_mask[lo:hi] = False          # purge + embargo on both sides
            yield indices[train_mask], test_idx
            current += size
```

**C.3 Coupled solve with spectral penalty** (condensed from `OverlapAwareRegimeDML.fit`).

```python
eigvals = np.linalg.eigvalsh(J_mat)            # diagnose conditioning first
try:
    inv_J_reg = np.linalg.inv(J_mat + eff_lambda * np.eye(K))
except np.linalg.LinAlgError:
    inv_J_reg = np.linalg.pinv(J_mat + eff_lambda * np.eye(K))
theta_vec = inv_J_reg @ S_vec
```


**C.4 Planned public API** (`docs/FUTURE_WORK_AND_FRONTIERS.md`; not yet released).

```python
estimator = OverlapAwareRegimeDML(
    regime_model=GaussianHMMRegimeModel(n_regimes=2),
    model_y=LGBMRegressor(), model_t=LGBMRegressor(),
    spectral_penalty="adaptive_plugin", embargo_buffer="auto_bartlett", n_splits=5)
estimator.fit(X=X_env, T=treatment_t, Y=outcome_y, Z=proxy_z)
```

# !Annexure D: Reproduction Commands

```bash
git clone <repository> && cd aqi-informatics-india
pip install -r requirements.txt            # or: conda env create -f environment.yml
python verify_submission.py --fast         # numbers, macros, page budget
bash reproduce.sh                          # ~12 min on 2 cores
bash reproduce.sh --full                   # also the long legacy benchmarks
```

Raw data are in `data/raw_hourly/`; all result files in `reports/`; generated artifacts in `paper/generated/`.

# !Annexure E: Additional Results

@tex frontier | tab_frontier_full.tex | Synthetic frontier, full estimator list: mean absolute ATE bias, median, RMSE and smallest eigenvalue at each proxy separation $\Delta_Z$ (100 datasets per row). | sm long

# !Annexure F: Map of the Manuscript Appendices

The AISTATS manuscript's supplementary material is in appendices A–Q. They are the formal back-up for this report.

@tab appmap | Manuscript appendices and where this report uses them
| App. | Title | Used in |
|---|---|---|
| A | Assumptions | Chapter 3 |
| B | Proof of Theorem 1 (omitted regime bias) | Section 3.2 |
| C | Proof of Theorem 2 (non-identification) | Section 3.3 |
| D | Proof of the graceful-degradation theorem | Section 5.2 |
| E | Posterior-adjusted nuisances | Sections 4.2, 5.1 |
| F | The coupled score under state confusion | Section 5.3 |
| G | The residual-state bias law | Section 5.4 |
| H | Inference | Section 5.5 |
| I | Operational surrogate, screening and regularization | Sections 4.2, 7.6 |
| J | Algorithm and scalability | Sections 4.2, 7.7 |
| K | Factorial and learned-posterior experiments | Sections 7.1, 7.2 |
| L | Synthetic frontiers | Section 7.3 |
| M | Real-data-calibrated semi-synthetic benchmark | Section 7.4 |
| N | Representations, calibration and reliability | Section 7.5 |
| O | Sensor data, audit and full city results | Chapter 8, Section 6.3 |
| P | Reproduction and verification | Chapter 9 |
| Q | Lineage, corrections and scope | Chapter 10 |
