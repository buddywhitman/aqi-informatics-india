# Reliable Causal Estimation under Latent Markov Confounding

Official code repository and reproducibility artifact suite for the manuscript **"Reliable Causal Estimation under Latent Markov Confounding"** (AISTATS 2027).

---

## 🌟 Overview & Scientific Problem

Causal inference from dependent observational time series is challenging when an unobserved, persistent state jointly affects treatment assignment and outcomes. Standard Double Machine Learning (DML) controls for observed covariates through orthogonal residualization, but does not resolve confounding induced by an imperfect proxy for latent state dynamics.

We study causal estimation under latent Markov confounding and characterize how latent-state uncertainty and regime overlap propagate into causal estimation error:
1. **Non-Identification under Unconstrained Overlap (Theorem 1)**: Proves that when proxies contain zero information distinguishing latent regimes, regime-specific causal effects cannot be point-identified from observables.
2. **Graceful Degradation Error Bound (Theorem 3)**: Separates posterior proxy error $\varepsilon_\gamma$ from task conditioning $\lambda_{\min}(\boldsymbol{J})$, establishing an explicit error bound:
   $$\|\hat{\boldsymbol{\theta}}_\gamma - \boldsymbol{\theta}^*\|_2 \le \frac{C \cdot \varepsilon_\gamma}{\lambda_{\min}(\boldsymbol{J})} + \mathcal{O}_P(N^{-1/2}).$$
3. **Surrogate Entropy-Difficulty Bridge (Proposition 1)**: Demonstrates that under Bayesian calibration, posterior entropy bounds latent-state error, yielding the computable operational difficulty index $\mathcal{D}_{\mathrm{operational}}(t) = \frac{H(\boldsymbol{\gamma}_t)}{\lambda_{\min}(\boldsymbol{J}_t)}$.
4. **Spectral Regularization & Asymptotics (Theorems 4 & 5, Proposition 2)**: Derives leading-order risk bounds for ridge-regularized inversion $(\boldsymbol{J} + \lambda \boldsymbol{I})^{-1}$ and establishes centered asymptotic normality with purged block cross-fitting under dependent data.
5. **Representation-to-Task Reliability**: A corrected 7x7 factorial intervention (4,900 runs) independently varies proxy error and post-residualization task conditioning; the joint ratio $\varepsilon_\gamma/\lambda_{\min}(J)$ ranks causal error far better than either constituent alone. Across 15 shifted representation worlds, NLL improves leave-one-world-out reliability prediction beyond $F_1$, while temperature scaling cuts neural ECE by up to 60% without improving downstream failure AUC, showing calibration is informative but insufficient.
6. **Real-World Observational Stress Test**: Evaluates four Indian megacities ($14,122$ complete analysis hours across Delhi, Mumbai, Bengaluru, and Kolkata), demonstrating how disparate task geometries illuminate distinct operating regimes on the difficulty frontier.

---

## 📁 Repository Structure

```text
├── paper/
│   ├── main.tex                    # Authoritative LaTeX manuscript (exact 8-page main text, strictly 24 pages total)
│   ├── main.pdf                    # Compiled PDF submission (24 pages, 0 warnings)
│   ├── aistats2027.sty             # Official AISTATS conference style file
│   └── plots/                      # Publication figures embedded in manuscript
├── src/                            # Active, self-contained Python codebase
│   ├── or_dml.py                   # Canonical Overlap-Aware Regime DML (OR-DML) implementation
│   ├── calibration_intervention_zoo.py # Post-hoc temperature-scaling intervention on Representation Zoo
│   ├── factorial_reliability_experiment.py # Corrected 7x7 representation-by-task mechanism experiment
│   ├── empirical_falsification_checks.py # Pre-treatment lead placebo falsification suite (h in {-6, -3, -1})
│   ├── synthetic_dgp_benchmark.py  # 500-draw Monte Carlo difficulty frontier simulation
│   ├── empirical_evaluation.py     # 4-city sensor evaluation, conditioning diagnostics, dynamic IRFs
│   ├── train_real_representation_zoo.py # HMM, GRU, Transformer, SSM benchmark under distribution shifts
│   ├── hierarchical_reliability_regression.py # 15-world fixed-effects regressions & LOWO cross-validation
│   ├── financial_regime_transfer.py # Multi-domain synthetic transfer & selective abstention policy
│   ├── data_pipeline_clean.py      # Clean data engineering pipeline for 14,122 hourly records
│   ├── regime_intelligence.py      # Meteorological regime validation and profiling
│   └── requirements.txt            # Minimal pip environment dependencies
├── reports/                        # Synchronized CSV report artifacts cited in manuscript
├── plots/                          # Generated high-resolution publication figures
├── data/
│   └── processed_clean/            # Cleaned analysis dataset (14,122 hourly rows)
├── docs/
│   └── adr/                        # Architectural Decision Records (ADRs 0001–0007)
├── experiments_manifest.json       # Machine-readable experiment and theorem manifest
├── SUPPLEMENT_ROADMAP.md           # Authoritative supplementary roadmap and table of contents
├── verify_artifacts.py             # Automated artifact and byte-for-byte manuscript synchronization check
└── verify_science.py               # Automated verification of 10 core mathematical/algebraic invariances
```

---

## 🔬 Canonical Implementation: OR-DML (`src/or_dml.py`)

The primary estimator is implemented in [`src/or_dml.py`](./src/or_dml.py):
* **Class**: `OverlapAwareRegimeDML`
* **Posterior Modes**: Supports causal forward filtering ($\boldsymbol{\gamma}_t = P(S_t \mid \mathcal{F}_t)$) and retrospective smoothing ($\boldsymbol{\gamma}_t = P(S_t \mid Z_{1:N})$).
* **Nuisance Estimation**: Cross-fitted with purged temporal blocks and temporal embargo buffers (Ridge regressors in the 4-city observational study; Ridge and Gradient Boosting regressors in the synthetic benchmarks).
* **Inversion**: Spectrally regularized coupled Jacobian inversion $\hat{\boldsymbol{\theta}}_\lambda = (\hat{\boldsymbol{J}} + \lambda \boldsymbol{I})^{-1} \hat{\boldsymbol{S}}$.
* **Conditioning Diagnostics**: Automatically computes $\lambda_{\min}(\hat{\boldsymbol{J}})$, condition number $\kappa(\hat{\boldsymbol{J}})$, and sample state occupancy.

---

## 🚀 Reproducibility Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution

To reproduce all numerical results, figures, and benchmark tables from scratch:

```bash
# Step 1: Preprocess raw atmospheric records into clean analysis format (14,122 rows)
python src/data_pipeline_clean.py

# Step 2: Run 500-draw Monte Carlo difficulty frontier benchmark
python src/synthetic_dgp_benchmark.py

# Step 3: Run corrected 7x7 representation-by-task factorial mechanism test
python src/factorial_reliability_experiment.py

# Step 4: Run real-world megacity evaluation and dynamic exposure-response functions
python src/empirical_evaluation.py

# Step 5: Run representation zoo evaluation (HMM, GRU, Transformer, SSM)
python src/train_real_representation_zoo.py

# Step 6: Run hierarchical reliability regressions across 15 benchmark worlds
python src/hierarchical_reliability_regression.py

# Step 7: Run multi-domain transfer and selective abstention policy
python src/financial_regime_transfer.py

# Step 8: Run post-hoc calibration interventions on representation zoo
python src/calibration_intervention_zoo.py

# Step 9: Run pre-treatment lead placebo falsification checks
python src/empirical_falsification_checks.py

```

### 3. Automated Dual-Layer Verification

The repository includes two automated verification layers to ensure reproducibility and scientific soundness:

```bash
# Layer 1: Artifact & Manuscript Synchronization
# Checks report existence, figure existence, absence of mock tokens, and byte-for-byte synchronization
python verify_artifacts.py

# Layer 2: Mathematical & Scientific Invariances
# Asserts normal equations, oracle recovery, K=1 reduction, permutation covariance, simplex, and temporal disjointness
python verify_science.py
```

### 4. Compiling the Manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
* **Page Budget**: Strictly 8 pages of main text; exactly 24 pages total including AI statement, references, checklist, and Appendices A–I.
* **Compilation Status**: 0 errors, 0 warnings.


## Exploratory task-relative reliability research

The isolated branch `research/task-relative-reliability` contains a large adversarial follow-up program. It is **not part of the canonical AISTATS submission evidence chain** unless a result is explicitly promoted after verification.

Start here:

- `research/README.md` — chronological result ledger (R1 onward), including negative results.
- `research/RESEARCH_INDEX.md` — study-to-code/result index.
- `research/REMAINING_RESEARCH_PLAN.md` — work packages and saturation criteria.
- `research/SATURATION_STATUS.md` — completion status and unresolved requirements.
- `research/FINAL_CANDIDATE_SYNTHESIS.md` — compressed candidate contribution set.
- `research/NOVELTY_COLLISION_AUDIT.md` — conservative prior-art collision audit.
- `research/FINAL_REAL_DATA_COLLISION.md` — observational sensor-data stress test.
- `research/REPRESENTATION_ZOO_REANALYSIS.md` — aggregate zoo reinterpretation.
- `research/FINAL_AUDIT_CHECKLIST.md` — claim/reproducibility/limitation audit.
- `research/UNIFIED_THEOREM_SKETCH.md` — working local reliability expansion.

The strongest surviving research themes are phantom causal resolution, self-canceling proxy diagnostics, target/directional generated-representation reliability, conservative posterior-moment/sensitivity diagnostics, and the distinction between fine adjustment resolution and scientifically justified target resolution. Broad weak-identification, task-aware-calibration, generated-covariate, latent-class correction, and Neyman-orthogonality ideas are explicitly treated as prior art rather than claimed as new.

The research workflow `.github/workflows/research-reliability.yml` executes the committed exploratory experiments and publishes their result artifacts.


## AISTATS 2027 final synthesis branch

The branch `submission/aistats2027-final-synthesis` is the submission-oriented synthesis built from the fully audited research branch. It preserves the title and abstract registered before the AISTATS abstract deadline and revises the body/supplement only to improve factual scope, falsification transparency, reproducibility, and compliance.

Submission-critical files:

- `paper/main.tex` — anonymous AISTATS manuscript source.
- `paper/aistats2027.sty` — AISTATS 2027 style used by the manuscript.
- `paper/main.pdf` — compiled manuscript produced by the final verification workflow.
- `AISTATS2027_OR_DML_Supplementary_Material.zip` — canonical supplementary archive.
- `SUPPLEMENT_ROADMAP.md` — map of proofs, protocols, reports, and post-audit evidence.
- `verify_science.py` — mathematical/scientific invariance checks.
- `verify_artifacts.py` — manuscript/report/figure synchronization checks.
- `.github/workflows/submission-final.yml` — compile, verification, anonymity, AI-statement-ordering, and packaging workflow.
- `SUBMISSION_SYNTHESIS_CHANGELOG.md` — exact scientific/editorial changes promoted from the audit.

The final synthesis deliberately does **not** promote every exploratory result. Results enter the submission only when they are reproducible, consistent with adversarial falsification, and compatible with the frozen title/abstract and the main-paper evidence chain.


## AISTATS 2027 final synthesis branch

The branch `submission/aistats2027-final-synthesis` is the submission-candidate integration branch. It preserves the registered title/abstract framing while incorporating only post-audit qualifications that are supported by committed evidence.

Canonical submission artifacts:

- `paper/main.tex` — anonymous AISTATS manuscript source.
- `paper/aistats2027.sty` — AISTATS 2027 style used by the manuscript.
- `paper/main.pdf` — generated by the final verification workflow; do not hand-edit.
- `AISTATS2027_OR_DML_Supplementary_Material.zip` — generated anonymous supplementary archive.
- `SUPPLEMENT_README.md` and `SUPPLEMENT_ROADMAP.md` — reproduction and evidence maps.
- `verify_science.py`, `verify_artifacts.py`, and `verify_submission_synthesis.py` — scientific, synchronization, and promoted-claim gates.
- `.github/workflows/submission-final.yml` — compiles the manuscript, builds the supplement, checks page placement/anonymity, and publishes final artifacts.

Final-synthesis editorial policy:

- main text is limited to eight pages before the AI Use Statement/references;
- the mandatory AI Use Statement precedes references;
- observational sensor estimates are explicitly stress tests rather than causal ground truth;
- negative/falsification results are retained where they delimit interpretation;
- exploratory research is promoted only when supported by committed evidence and consistent with the final novelty audit;
- the supplement contains the code/results needed to interrogate promoted claims while excluding identifying repository links.
