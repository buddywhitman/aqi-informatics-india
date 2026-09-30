# ADR 0007: Repository Scientific Purge, Identity Consolidation, and Exact Title/Abstract Alignment

## Status
Accepted

## Context
Following a full repository audit benchmarking our submission against AISTATS 2027 excellence standards:
1. **Title and Abstract Synchronization**:
   - The user mandated the exact title: `Reliable Causal Estimation under Latent Markov Confounding`.
   - The user specified the exact revised abstract establishing the formalization: causal inference under dependent observational time series, regime-aware orthogonal estimation, purged cross-fitting, regularized inversion, real-data-calibrated semi-synthetic time series from Indian megacities, and difficulty frontier diagnostics.
2. **Restoration of Style Header / Watermark**:
   - Reverted `paper/aistats2027.sty` line 78 to the original conference template string:
     `\newcommand{\Notice@String}{Preliminary work. Under review by AISTATS \@conferenceyear. Do not distribute.}`
   - Preserved exact official conference geometry and styling.
3. **Repository-Wide Consistency Purge**:
   - The repository historically accumulated legacy artifacts from earlier iterations (CNN-LSTM/TFT forecasting, generic AQI prediction, XAI SHAP scripts, economic policy simulation, lives-saved/VSL scripts, old X-Learner, and obsolete Nature manuscript generators).
   - These legacy files posed a risk of confusing reviewers and diluting the central scientific thesis.
   - All legacy files were moved to `archive/pre-pivot/` via `git mv`, preserving history while maintaining a lean, focused active `src/` directory containing strictly the 14 core scientific scripts.
4. **Second-Layer Scientific Verification**:
   - Created `verify_science.py` to complement `verify_artifacts.py`.
   - Asserts 9 mathematical and algebraic invariances: normal equations, oracle recovery, $K=1$ standard DML equivalence, permutation equivariance, posterior simplex conservation, Gram conditioning, regularization monotonicity, temporal fold disjointness, and out-of-sample evaluation.

## Decision
1. **Title & Abstract**:
   - Updated `paper/main.tex` title to `Reliable Causal Estimation under Latent Markov Confounding` and `\runningtitle{Reliable Causal Estimation under Latent Markov Confounding}`.
   - Updated `\begin{abstract}` to the exact user-specified text.
   - Inlined foundational assumptions and tightened Section 1 related work, maintaining the main text within exactly 8 pages.
2. **Page Budget**:
   - Main text: Pages 1–8 (ends cleanly at `proof of live-trading alpha.`).
   - AI statement & References: Page 9.
   - Checklist: Page 10.
   - Appendices: Pages 11–24 (ends at Table 12).
   - Total document length: **EXACTLY 24 PAGES** with 0 warnings.
3. **Repository Architecture**:
   - `archive/pre-pivot/src/`: Contains all legacy modeling, preprocessing, and policy simulation scripts.
   - `archive/pre-pivot/manuscript/`: Contains legacy Nature/Q1 manuscript markdown drafts.
   - `src/`: Contains strictly the active, self-contained codebase for the AISTATS submission.
   - `verify_science.py`: Added as automated second-layer scientific validator.

## Consequences
- The repository now presents a single, unified scientific story: latent Markov confounding, task conditioning, and sequential reliability.
- All 25 checks in `verify_artifacts.py` and all 9 checks in `verify_science.py` pass with 0 errors.
- Document compiles to exactly 24 pages with 0 warnings.
