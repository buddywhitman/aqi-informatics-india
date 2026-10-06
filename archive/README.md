# Repository Archive & Legacy Artifacts

This directory contains historical, pre-pivot, and superseded materials preserved for full scientific provenance and auditability. None of the files in this directory are part of the active AISTATS 2027 submission or verification pipeline.

---

## Directory Taxonomy

```text
archive/
├── legacy_nature_drafts/       # Original drafts and SI from the early Nature-style manuscript
├── pre-pivot/                  # Pre-pivot materials before the Overlap-Aware Regime DML formulation
│   ├── lightning_logs/         # PyTorch Lightning training runs and checkpoints
│   ├── models/                 # Pre-pivot Keras deep learning weights (.keras)
│   ├── packs/                  # Legacy Overleaf submission zip archives
│   ├── legacy_plots/           # Exploratory EDA, SHAP, and legacy empirical plots
│   ├── paper_legacy_plots/     # Unused figure assets from earlier manuscript iterations
│   ├── legacy_reports/         # Text summaries, TFT benchmarks, and exploratory reports
│   ├── manuscript/             # Pre-pivot markdown draft chapters
│   └── src/                    # Pre-pivot scripts (data acquisition, dl_modeling, etc.)
├── superseded/                 # Code and documents replaced by refined Phase 2 implementations
│   ├── src/                    # Scripts replaced by canonical counterparts (e.g. three_decisive_experiments.py -> factorial_reliability_experiment.py)
│   └── docs/                   # Brainstorming notes, implementation plans, and interim feedback
└── references/                 # Foundational literature PDFs and references
```

---

## Active vs. Archived Mapping

| Archived / Superseded File | Active Canonical Replacement | Reason for Transition |
| :--- | :--- | :--- |
| `archive/superseded/src/three_decisive_experiments.py` | `src/factorial_reliability_experiment.py` | Corrected nuisance residualization so regime shifts do not contaminate residuals. |
| `archive/superseded/src/latent_regime_bench_fast.py` | `src/latent_regime_bench.py` | Consolidated into full benchmark script. |
| `archive/superseded/src/experiment_failure_*.py` | `src/advanced_methodological_frontiers.py` | Formalized into Frontier A (Selective Estimation) and Frontier B (Size Distortion). |
| `archive/pre-pivot/models/*.keras` | `src/train_real_representation_zoo.py` | Replaced legacy black-box forecast models with transparent representation zoo (HMM, GRU, Transformer, SSM). |
| `archive/pre-pivot/legacy_plots/*` | `plots/fig1` through `fig5` | Standardized on the 5 canonical publication figures. |
