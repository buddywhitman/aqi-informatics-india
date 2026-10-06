# Reliable Causal Estimation under Latent Markov Confounding

Official code repository and reproducibility artifact suite for the manuscript **"Reliable Causal Estimation under Latent Markov Confounding"** (AISTATS 2027).

---

## 🌟 Overview & Scientific Problem

The submission is organized around one failure mode: **phantom causal resolution**. When an unobserved sequential confounder is replaced by a learned posterior, representation error can simultaneously leave residual confounding and make the downstream score geometry appear better conditioned. A generated-state conditioning diagnostic can therefore become more reassuring while the causal analysis becomes less trustworthy.

The evidence chain has four layers: (1) an exact residual-confounding sensitivity law; (2) analytic and synthetic demonstrations of phantom geometry; (3) controlled factorial and higher-dimensional task-relative reliability tests; and (4) conservative inference and abstention procedures with negative results retained. OR-DML is one operational response, not the sole contribution or an unconditionally dominant estimator.

The four-city sensor analysis is an observational stress test, not causal ground truth. The corrected-hourly bias-law audit and the canonical 14,122-row processed dataset are retained with separate provenance because the earlier processed file was affected by a timestamp-rounding issue documented in `docs/V2_CHANGELOG.md`.

---

## 📁 Repository Structure

```text
├── paper/
│   ├── main.tex                    # Authoritative anonymous manuscript source; main scientific text is pages 1-8
│   ├── main.pdf                    # Verified compiled submission (28 pages total; AI Use Statement begins page 9)
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
* **Page Budget**: The scientific main text occupies pages 1–8; the AI Use Statement begins on page 9, before references.
* **Verified artifact**: `paper/main.pdf` is 28 pages total and is generated only after the end-to-end submission gate passes.


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


## AISTATS 2027 authoritative submission

The **`master` branch is the sole authoritative submission source of truth**. Earlier research, bias-law, phase-2, and final-synthesis branches remain provenance only; they must not be submitted independently.

Canonical submission artifacts:

- `paper/main.tex` — anonymous AISTATS manuscript source.
- `paper/main.pdf` — verified compiled manuscript; main scientific text pages 1–8, AI Use Statement from page 9.
- `AISTATS2027_OR_DML_Supplementary_Material.zip` — deterministic anonymous supplementary archive.
- `SUPPLEMENT_README.md`, `SUPPLEMENT_ROADMAP.md`, and `SUBMISSION_COMPLIANCE.md` — reproduction/evidence/compliance maps.
- `experiments_manifest.json` — machine-readable theorem/experiment evidence map.
- `verify_science.py`, `verify_artifacts.py`, and `verify_submission_synthesis.py` — scientific, synchronization, and promoted-claim gates.
- `.github/workflows/submission-final.yml` — end-to-end release verification for `master`.

The final editorial policy is conservative: phantom causal resolution is the central contribution; the exact residual-confounding law supplies its sensitivity mechanism; task/directional geometry defines the reliability scope; observational sensor analyses are stress tests rather than causal ground truth; negative/falsification results remain visible; and superseded leakage-prone experiments are provenance only.
