# ADR 0006: Editorial Compression, Representation Elevation, and Exact 24-Page Budget Compliance

## Status
Accepted

## Context
Following an in-depth editorial audit comparing the manuscript to AISTATS award-winning exemplars (*EventFlow* and *Optimization*), several architectural, presentation, and structural enhancements were required:
1. **Core Scientific Thesis**: Crystallizing the primary insight: *"A model can be confident about a hidden state and still be unreliable for the downstream task because downstream task geometry governs failure."*
2. **Elevation of Representation Zoo**: In prior drafts, the Representation Zoo comparison (HMM vs. GRU vs. Causal Transformer vs. Linear SSM) was confined to the appendix. To position the paper as an impactful contribution to both causal inference and sequential representation learning, Table 3 (`tab:main_rep_zoo`) was promoted into Section 6 of the main text.
3. **Streamlined Main Text Tables**:
   - Table 1 (`tab:benchmark`): Compressed to 6 informative columns ($\Delta_Z$, $\kappa(J)$, Abs Bias, RMSE, 95% Coverage, Runtime) with `\renewcommand{\arraystretch}{0.80}`.
   - Table 2 (`tab:empirical`): Streamlined to 6 key contrast rows (`Delhi`, `Mumbai R1/R2`, `Bengaluru R1/R2`, `Kolkata Overall`) with `\renewcommand{\arraystretch}{0.76}`, relocated to float at the top of Page 8.
4. **Strict AISTATS 2027 Page Budget and Section Allocation**:
   - **Main Text (Pages 1–8)**: Exactly 8 pages. Ends cleanly on Page 8 at line: `proof of live-trading alpha.`
   - **References & AI Use Statement (Page 9)**: Contains the compliant `\subsection*{AI Use Statement}` immediately preceding `\begin{thebibliography}{28}`. All 28 cited landmark references are retained; 5 un-cited bibitems were pruned. Fits 100% on Page 9 with zero spillover.
   - **Paper Checklist (Page 10)**: Exactly 1 page. Fits 100% on Page 10.
   - **Mathematical Appendices (Pages 11–24)**: Exactly 14 pages across Appendices A through I.
   - **Total Document Length**: **EXACTLY 24 PAGES**.

## Decision
1. **P0 Page Budget Compliance**:
   - Main text compressed to 8 pages via concise Section 7 conditioning diagnostics and a unified 9-line Section 8 conclusion.
   - References and AI Use Statement consolidated onto Page 9 using `\setlength{\itemsep}{-4.0pt}` and `\footnotesize`.
   - Appendix display math skips adjusted to `\setlength{\abovedisplayskip}{3pt plus 1pt minus 1pt}` and `\setlength{\belowdisplayskip}{3pt plus 1pt minus 1pt}`.
   - Appendix section headings tightened with `\vspace{-2mm}`.
   - Appendix tables and figures compacted:
     - Table 4 (`tab:regime_profiles`): `\renewcommand{\arraystretch}{0.78}`.
     - Table 5 (`tab:kolkata_lambda_grid`): Streamlined from 13 rows to 7 strategic grid rows ($0.0000, 0.0005, 0.0024^*, 0.0100, 0.0500, 0.1000, 1.0000$) with `\renewcommand{\arraystretch}{0.75}`.
     - Table 6 (`tab:imputation_sensitivity`): Compacted with in-line summary text replacing bullet points.
     - Table 7 (`tab:bench_results`) & Table 8 (`tab:bench_models`): `\renewcommand{\arraystretch}{0.78}`.
     - Figure 3, Figure 4, and Figure 5: Scaled to `height=1.65cm` / `height=1.60cm`.
     - Inlined Section H three-regime enumeration.
     - Reordered Figure 5 inside Appendix I, placing Table 11 and Table 12 compactly on Page 24.
2. **Zero-Warning Compilation**:
   - Resolved all undefined references (`eq:reg_identity` labeled, `tab:empirical_full` redirected to `app:empirical_sensitivity`).
   - Achieved 0 LaTeX warnings and 0 compilation errors across 24 pages.
3. **Artifact and Benchmark Integrity**:
   - All verified numbers (`0.6153`, `0.8481`, `0.0025`, `0.7038`, `0.9817`, `-0.0653`) preserved byte-for-byte in the text.
   - All 25 checks in `python verify_artifacts.py` pass with 0 errors.

## Consequences
- The manuscript satisfies every AISTATS 2027 formatting mandate and editorial excellence standard.
- The paper cleanly establishes the double-descent / difficulty frontier narrative: sequence encoders can excel at latent state tracking while catastrophically failing at downstream sequential decision tasks due to uncalibrated confidence and task ill-conditioning.
- Clean PDF build compiles to exactly 24 pages with zero overflow.
